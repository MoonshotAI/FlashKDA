import torch
from flash_kda_C import fwd as _fwd_raw, get_workspace_size


def get_intermediate_state_shape(q, cu_seqlens=None):
    """Return the required ``[H, tile_capacity, D, D]`` snapshot-buffer shape.

    In ragged mode, ``tile_capacity`` is an upper bound with at most one unused
    tile per sequence. Use ``get_intermediate_state_tile_prefix`` to locate the
    compact valid chunk states without synchronizing to the host.
    """
    B, T_seq, H, D = q.shape
    if cu_seqlens is None:
        total_tiles = B * ((T_seq + 15) // 16)
    else:
        total_tiles = (B * T_seq + 15) // 16 + cu_seqlens.numel() - 1
    return H, total_tiles, D, D


def get_intermediate_state_tile_prefix(q, cu_seqlens=None):
    """Return device-resident offsets for the compact chunk-state layout.

    The state after chunk ``t`` of sequence ``n`` is at
    ``intermediate_state[head, tile_prefix[n] + t]``.
    ``tile_prefix[-1]`` is the number of valid snapshot slots, so this helper
    locates ragged snapshots without host synchronization. Snapshots are useful
    for inspection or a serial handoff, but cannot alone implement exact CP
    prefix composition: KDA's input-state dependence requires the affine
    transition prototype in flash_kda.cp.
    """
    B, T_seq = q.shape[:2]
    if cu_seqlens is None:
        tiles_per_sequence = (T_seq + 15) // 16
        return torch.arange(B + 1, dtype=torch.long, device=q.device) * tiles_per_sequence
    if cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the same device as q")
    sequence_tiles = torch.div(cu_seqlens[1:] - cu_seqlens[:-1] + 15, 16,
                               rounding_mode="floor")
    return torch.cat((torch.zeros(1, dtype=torch.long, device=cu_seqlens.device),
                      torch.cumsum(sequence_tiles, dim=0)))


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None, intermediate_state=None):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.
        intermediate_state (torch.Tensor, optional): BF16 output buffer for the
            recurrent state after each 16-token chunk. Its shape must equal
            ``get_intermediate_state_shape(q, cu_seqlens)``. In ragged mode, use
            ``get_intermediate_state_tile_prefix(q, cu_seqlens)``: chunk ``t``
            of sequence ``n`` is stored at ``[head, tile_prefix[n] + t]``;
            slots from ``tile_prefix[-1]`` onward are left untouched.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be CUDA, contiguous, and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    workspace = torch.empty(get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)

    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens,
             intermediate_state=intermediate_state)
