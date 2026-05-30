import math

import torch
from flash_kda_C import get_warmup_chunks as _get_warmup_chunks_cuda


CHUNK_SIZE = 16


def _get_sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_warmup_chunks_cuda(g, A_log, dt_bias, lower_bound, cu_seqlens, chunk_size=CHUNK_SIZE, threshold=-10.0):
    """CUDA implementation of get_warmup_chunks. Much faster than Python version."""
    device = cu_seqlens.device
    N = cu_seqlens.numel() - 1
    num_warmup = torch.empty(N, dtype=torch.int32, device=device)
    fallback = torch.empty(N, dtype=torch.bool, device=device)
    _get_warmup_chunks_cuda(g, A_log, dt_bias, lower_bound, cu_seqlens,
                            chunk_size, threshold, num_warmup, fallback)
    return num_warmup, fallback


def get_warmup_chunks(g, A_log, dt_bias, lower_bound, cu_seqlens, chunk_size=CHUNK_SIZE, threshold=-10.0):
    """Determine warmup chunk count per segment by scanning gate decay from end.

    For each segment, scans backwards from the last chunk accumulating the
    per-chunk decay. Stops when cumulative decay exceeds threshold (meaning
    initial state contribution has decayed sufficiently).

    Args:
        g: Gate input, bf16, shape [1, T, H, D].
        A_log: Log-gate parameter, fp32, shape [H].
        dt_bias: Gate bias, fp32, shape [H, D].
        lower_bound: Gate lower bound scalar.
        cu_seqlens: Cumulative sequence lengths, int64, [N+1].
        chunk_size: Chunk size (default 16).
        threshold: Decay threshold in log-space. Default -10.0 (exp(-10)~4.5e-5).

    Returns:
        num_warmup_chunks: int32, shape [N]. Number of trailing chunks to process.
        fallback_mask: bool, shape [N]. True if warmup covers entire segment (need mt).
    """
    device = cu_seqlens.device
    N = cu_seqlens.numel() - 1

    gate_scale = lower_bound * 1.4426950408889634
    a_log_exp = A_log.exp()
    dt_bias_mean = dt_bias.float().mean(dim=-1)

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (seqlens + chunk_size - 1) // chunk_size

    eos_offsets = cu_seqlens[1:]
    last_token_indices = eos_offsets - 1

    g_flat = g.reshape(-1, A_log.shape[0], g.shape[-1]).float()
    g_at_end = g_flat[last_token_indices].mean(dim=-1)
    per_seg_decay = gate_scale * torch.sigmoid(a_log_exp.unsqueeze(0) * (g_at_end + dt_bias_mean.unsqueeze(0))) * chunk_size

    cum_decay = per_seg_decay.max(dim=-1).values
    converges_in_1 = cum_decay < threshold

    num_warmup = torch.ones(N, dtype=torch.int32, device=device)
    fallback = torch.zeros(N, dtype=torch.bool, device=device)

    not_converged = ~converges_in_1
    if not_converged.any().item():
        idx_list = not_converged.nonzero(as_tuple=True)[0].cpu().tolist()
        cu_cpu = cu_seqlens.cpu().tolist()
        for seg_idx in idx_list:
            bos = cu_cpu[seg_idx]
            eos = cu_cpu[seg_idx + 1]
            seg_len = eos - bos
            nc = (seg_len + chunk_size - 1) // chunk_size

            g_cumsum = torch.zeros(A_log.shape[0], device=device)
            warmup = nc
            found = False
            for c in range(nc):
                chunk_end = eos - c * chunk_size - 1
                if chunk_end < bos:
                    chunk_end = bos
                g_val = g_flat[chunk_end].mean(dim=-1)
                g_activated = gate_scale * torch.sigmoid(a_log_exp * (g_val + dt_bias_mean))
                g_cumsum += g_activated * chunk_size
                if g_cumsum.max().item() < threshold:
                    warmup = c + 1
                    found = True
                    break

            num_warmup[seg_idx] = warmup
            if not found:
                fallback[seg_idx] = True

    return num_warmup, fallback


def correct_initial_states(raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c):
    """Serial correction: h0[i+1] = mt[i] @ h0[i] + ht[i].

    Chains the transition matrices and accumulated states to compute exact
    initial states for each CP sub-segment.

    Args:
        raw_h0: Initial state for original sequences, fp32, [raw_N, H, D, D] or None.
        ht_buffer: State-only output (h0=0), fp32, [cp_N, H, D, D].
        mt_buffer: Transition matrix, fp32, [cp_N, H, D, D] or None.
        fallback_mask: bool, [cp_N]. True means need mt correction.
        seq_map_r2c: int32, [raw_N+1]. Maps raw seq idx to CP segment range.

    Returns:
        cp_h0: Corrected initial states, fp32, [cp_N, H, D, D].
    """
    cp_N, H, D, _ = ht_buffer.shape
    device = ht_buffer.device

    need_mt = fallback_mask.any().item()
    seq_map_cpu = seq_map_r2c.cpu().tolist()
    raw_N = len(seq_map_cpu) - 1

    if not need_mt and raw_h0 is None:
        cp_h0 = torch.zeros(cp_N, H, D, D, dtype=torch.float32, device=device)
        for raw_idx in range(raw_N):
            seg_start = seq_map_cpu[raw_idx]
            seg_end = seq_map_cpu[raw_idx + 1]
            if seg_end > seg_start + 1:
                cp_h0[seg_start + 1:seg_end] = ht_buffer[seg_start:seg_end - 1]
        return cp_h0

    if not need_mt and raw_h0 is not None:
        cp_h0 = torch.zeros(cp_N, H, D, D, dtype=torch.float32, device=device)
        for raw_idx in range(raw_N):
            seg_start = seq_map_cpu[raw_idx]
            seg_end = seq_map_cpu[raw_idx + 1]
            cp_h0[seg_start] = raw_h0[raw_idx]
            if seg_end > seg_start + 1:
                cp_h0[seg_start + 1:seg_end] = ht_buffer[seg_start:seg_end - 1]
        return cp_h0

    cp_h0 = torch.zeros(cp_N, H, D, D, dtype=torch.float32, device=device)
    for raw_idx in range(raw_N):
        seg_start = seq_map_cpu[raw_idx]
        seg_end = seq_map_cpu[raw_idx + 1]

        if raw_h0 is not None:
            h = raw_h0[raw_idx].clone()
        else:
            h = torch.zeros(H, D, D, dtype=torch.float32, device=device)

        cp_h0[seg_start] = h

        for i in range(seg_start, seg_end - 1):
            if fallback_mask[i]:
                h = torch.einsum('hdk,hkv->hdv', mt_buffer[i], h) + ht_buffer[i]
            else:
                h = ht_buffer[i]
            cp_h0[i + 1] = h

    return cp_h0


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

    Uses a warmup + mt + serial correction strategy:
      1. Determine warmup chunk count per segment (scan gate decay from end).
      2. Run state_only on warmup suffix → ht (and mt for full-segment cases).
      3. Serial correction: h0[i+1] = mt[i] @ h0[i] + ht[i].
      4. Single forward pass with corrected initial states.

    Falls back to standard fwd() when CP is not beneficial.

    Args: Same as flash_kda.fwd(), plus:
        auto_cp (bool): Enable automatic CP. Default True.
    """
    from flash_kda import fwd, state_only

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

    # --- Step 1: Determine warmup chunks per segment ---
    # Fast path: if 1 chunk of decay is guaranteed to exceed threshold, skip scanning.
    # Conservative estimate: per-chunk decay >= lower_bound * 1.4427 * 0.25 * CHUNK_SIZE
    # (sigmoid min is ~0.25 for typical A_log range; using 0.25 is conservative)
    per_chunk_decay_bound = lower_bound * 1.4426950408889634 * 0.25 * CHUNK_SIZE
    if per_chunk_decay_bound < -10.0:
        num_warmup = torch.ones(cp_N, dtype=torch.int32, device=q.device)
        fallback_mask = torch.zeros(cp_N, dtype=torch.bool, device=q.device)
    else:
        num_warmup, fallback_mask = get_warmup_chunks_cuda(
            g, A_log, dt_bias, lower_bound, cp_cu_seqlens, CHUNK_SIZE
        )

    # Determine if any segment needs mt computation
    need_mt = fallback_mask.any().item()

    # --- Step 2: State-only kernel (warmup suffix only) → ht + mt ---
    if need_mt:
        ht_buffer, mt_buffer = state_only(
            k, v, g, beta, A_log, dt_bias, lower_bound,
            cp_cu_seqlens, num_warmup, calc_mt=True
        )
    else:
        ht_buffer = state_only(
            k, v, g, beta, A_log, dt_bias, lower_bound,
            cp_cu_seqlens, num_warmup, calc_mt=False
        )
        mt_buffer = None

    # --- Step 3: Serial correction → corrected h0 for each sub-segment ---
    cp_h0 = correct_initial_states(
        initial_state, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c
    )

    # --- Step 4: Single forward pass with corrected initial states ---
    cp_final_state = None
    if final_state is not None:
        cp_final_state = torch.empty(cp_N, H, D, D, dtype=torch.float32, device=q.device)

    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=cp_h0, final_state=cp_final_state, cu_seqlens=cp_cu_seqlens)

    # --- Extract final states for original sequences ---
    if final_state is not None:
        seq_map_r2c_cpu = seq_map_r2c.cpu().tolist()
        for raw_idx in range(raw_N):
            last_seg = seq_map_r2c_cpu[raw_idx + 1] - 1
            final_state[raw_idx].copy_(cp_final_state[last_seg])
