import math

import torch


CHUNK_SIZE = 16


def _get_sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _calc_cp_seqs(cu_seqlens, num_heads, chunk_size=CHUNK_SIZE):
    """Split sequences into sub-segments for context parallelism.

    Mirrors FlashQLA's _calc_cp_seqs logic adapted for FlashKDA's chunk_size=16.

    Returns:
        use_cp: bool
        cp_cu_seqlens: Tensor[int64] or None
        seq_map_r2c: Tensor[int32] or None  (raw_batch_size+1, maps raw seq → CP seg range)
        seq_map_c2r: Tensor[int32] or None  (cp_batch_size, maps CP seg → raw seq idx)
    """
    sm_count = _get_sm_count()
    device = cu_seqlens.device
    seqlen_dtype = cu_seqlens.dtype

    cu_seqlens_list = cu_seqlens.tolist()
    raw_batch_size = len(cu_seqlens_list) - 1
    seqlens = [cu_seqlens_list[i + 1] - cu_seqlens_list[i] for i in range(raw_batch_size)]
    num_chunks = [(s + chunk_size - 1) // chunk_size for s in seqlens]

    H = num_heads
    total_chunks = sum(num_chunks)

    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H * total_chunks / sm_count) * 3)
    )
    max_local_chunks = max(max_local_chunks, 4)

    max_local_tokens = max_local_chunks * chunk_size

    cp_cu_seqlens = []
    seq_map_c2r = []
    seq_map_r2c = [0]

    for i, c in enumerate(num_chunks):
        s = cu_seqlens_list[i]
        e = cu_seqlens_list[i + 1]
        if c > max_local_chunks:
            while s < e:
                cp_cu_seqlens.append(s)
                seq_map_c2r.append(i)
                s += max_local_tokens
        else:
            cp_cu_seqlens.append(s)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(cu_seqlens_list[-1])

    Be = total_chunks / max(num_chunks) if max(num_chunks) > 0 else raw_batch_size
    # FlashKDA uses a 2-pass strategy, so CP only helps when SM utilization
    # is very low. The non-CP grid is (N, H) = Be*H blocks.
    # With 2 passes, break-even requires Be*H < SM_COUNT/4 approximately.
    use_cp = Be * H <= sm_count // 4

    if not use_cp:
        return False, None, None, None

    cp_cu_seqlens = torch.tensor(cp_cu_seqlens, dtype=seqlen_dtype, device=device)
    seq_map_r2c = torch.tensor(seq_map_r2c, dtype=torch.int32, device=device)
    seq_map_c2r = torch.tensor(seq_map_c2r, dtype=torch.int32, device=device)

    return True, cp_cu_seqlens, seq_map_r2c, seq_map_c2r


def _estimate_warmup_converges(A_log, min_seg_len, threshold=-5.0):
    """Check if gate decay ensures the initial state contribution is negligible.

    For KDA, per-head decay rate per token is approximately A_log[h] (negative).
    After L tokens, initial state is scaled by exp(L * A_log[h]).
    If L * max(A_log) < threshold, decay is sufficient.
    """
    max_decay_per_token = A_log.max().item()
    total_decay = min_seg_len * max_decay_per_token
    return total_decay < threshold


def fwd_cp(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
           initial_state=None, final_state=None, cu_seqlens=None, auto_cp=True):
    """FlashKDA forward with automatic intra-card context parallelism.

    Wraps flash_kda.fwd() with a 2-pass CP strategy:
      Pass 1: Run all sub-segments with h0=0 to capture final states.
      Pass 2: Run all sub-segments with corrected initial states.

    Falls back to standard fwd() when CP is not beneficial or gate decay
    is insufficient for the simplified (no transition matrix) approach.

    Args: Same as flash_kda.fwd(), plus:
        auto_cp (bool): Enable automatic CP. Default True.
    """
    from flash_kda import fwd

    B, T_seq, H, D = q.shape

    if not auto_cp or B > 1:
        return fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                   initial_state, final_state, cu_seqlens)

    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T_seq], dtype=torch.int64, device=q.device)

    raw_N = cu_seqlens.numel() - 1

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r = _calc_cp_seqs(
        cu_seqlens, H, CHUNK_SIZE
    )

    if not use_cp:
        return fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                   initial_state, final_state, cu_seqlens)

    cp_N = cp_cu_seqlens.numel() - 1

    # Check if all sub-segments are long enough for warmup convergence
    cp_seqlens_list = cp_cu_seqlens.tolist()
    min_seg_len = min(
        cp_seqlens_list[i + 1] - cp_seqlens_list[i] for i in range(cp_N)
    )

    if not _estimate_warmup_converges(A_log, min_seg_len):
        return fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                   initial_state, final_state, cu_seqlens)

    # --- Pass 1: Run all sub-segments with h0=0, capture final_state ---
    ht_buffer = torch.empty(cp_N, H, D, D, dtype=torch.float32, device=q.device)
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=ht_buffer, cu_seqlens=cp_cu_seqlens)

    # --- Build corrected initial states ---
    cp_h0 = torch.zeros(cp_N, H, D, D, dtype=torch.float32, device=q.device)
    seq_map_r2c_cpu = seq_map_r2c.cpu().tolist()

    for raw_idx in range(raw_N):
        seg_start = seq_map_r2c_cpu[raw_idx]
        seg_end = seq_map_r2c_cpu[raw_idx + 1]
        if initial_state is not None:
            cp_h0[seg_start] = initial_state[raw_idx]
        for i in range(seg_start, seg_end - 1):
            cp_h0[i + 1] = ht_buffer[i]

    # --- Pass 2: Run with corrected initial states ---
    cp_final_state = None
    if final_state is not None:
        cp_final_state = torch.empty(cp_N, H, D, D, dtype=torch.float32, device=q.device)

    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=cp_h0, final_state=cp_final_state, cu_seqlens=cp_cu_seqlens)

    # --- Extract final states for original sequences ---
    if final_state is not None:
        for raw_idx in range(raw_N):
            last_seg = seq_map_r2c_cpu[raw_idx + 1] - 1
            final_state[raw_idx].copy_(cp_final_state[last_seg])
