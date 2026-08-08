"""KDA training-path operators (CUDA), replicating fla's Triton semantics.

Each function mirrors its fla counterpart so outputs can be compared
stage-by-stage against `fla.ops.kda` / `fla.ops.utils` Triton kernels.
"""

import torch

try:
    from flash_kda_train_C import (
        chunk_gated_delta_rule_bwd_dhu as _chunk_gated_delta_rule_bwd_dhu,
    )
    from flash_kda_train_C import (
        chunk_gated_delta_rule_fwd_h as _chunk_gated_delta_rule_fwd_h,
    )
    from flash_kda_train_C import chunk_gla_fwd_o_gk as _chunk_gla_fwd_o_gk
    from flash_kda_train_C import chunk_kda_bwd_dAv as _chunk_kda_bwd_dAv
    from flash_kda_train_C import chunk_kda_bwd_intra as _chunk_kda_bwd_intra
    from flash_kda_train_C import (
        chunk_kda_bwd_wy_dqkg_fused as _chunk_kda_bwd_wy_dqkg_fused,
    )
    from flash_kda_train_C import (
        chunk_kda_fwd_inter_solve_fused as _chunk_kda_fwd_inter_solve_fused,
    )
    from flash_kda_train_C import (
        chunk_kda_fwd_intra_sub_chunk as _chunk_kda_fwd_intra_sub_chunk,
    )
    from flash_kda_train_C import (
        chunk_kda_fwd_intra_token_parallel as _chunk_kda_fwd_intra_token_parallel,
    )
    from flash_kda_train_C import chunk_local_cumsum as _chunk_local_cumsum
    from flash_kda_train_C import kda_gate_bwd as _kda_gate_bwd
    from flash_kda_train_C import kda_gate_chunk_cumsum as _kda_gate_chunk_cumsum
    from flash_kda_train_C import recompute_w_u_fwd as _recompute_w_u_fwd
except ImportError as e:
    raise ImportError(
        "The flash_kda_train_C CUDA extension is not available. "
        "Build it with `pip install -e . --no-build-isolation` from the FlashKDA repo root."
    ) from e


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Pure-torch equivalent of fla.ops.utils.index.prepare_chunk_indices.

    Returns an int64 `[NT, 2]` tensor of `(sequence index, chunk index within sequence)`.
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (lens + chunk_size - 1) // chunk_size
    total = int(n_chunks.sum())
    device = cu_seqlens.device
    seq_ids = torch.repeat_interleave(torch.arange(len(lens), device=device), n_chunks)
    prefix = torch.cumsum(n_chunks, 0) - n_chunks
    chunk_ids = torch.arange(total, device=device) - prefix[seq_ids]
    return torch.stack([seq_ids, chunk_ids], dim=1).to(torch.int64).contiguous()


def _resolve(cu_seqlens, chunk_indices, chunk_size):
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int64).contiguous()
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        else:
            chunk_indices = chunk_indices.to(torch.int64).contiguous()
    return cu_seqlens, chunk_indices


def _resolve_offsets(cu_seqlens, chunk_size):
    """fla.ops.utils.index.prepare_chunk_offsets equivalent plus total chunk count.

    The chunk_h CUDA kernels index chunks via per-sequence chunk offsets
    instead of FLA-style chunk_indices.
    """
    if cu_seqlens is None:
        return None, None, 0
    cu_seqlens = cu_seqlens.to(torch.int64).contiguous()
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (lens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.cat([cu_seqlens.new_zeros(1), n_chunks.cumsum(0)]).contiguous()
    return cu_seqlens, chunk_offsets, int(chunk_offsets[-1])


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.bfloat16 else t.bfloat16()


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fla.ops.kda.gate.kda_gate_chunk_cumsum equivalent."""
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    out = torch.empty_like(g, dtype=torch.float32)
    _kda_gate_chunk_cumsum(
        g.contiguous(), A_log.contiguous(), dt_bias,
        out,
        float(scale) if scale is not None else 0.0, scale is not None,
        float(lower_bound) if lower_bound is not None else 0.0, lower_bound is not None,
        chunk_size, cu_seqlens, chunk_indices,
    )
    return out if output_dtype == torch.float32 else out.to(output_dtype)


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fla.ops.utils.chunk_local_cumsum equivalent (4D vector variant)."""
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    out = torch.empty_like(g, dtype=torch.float32)
    _chunk_local_cumsum(
        g.contiguous(), out,
        float(scale) if scale is not None else 0.0, scale is not None,
        reverse, chunk_size, cu_seqlens, chunk_indices,
    )
    return out if output_dtype == torch.float32 else out.to(output_dtype)


def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dyg: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """fla.ops.kda.gate.kda_gate_bwd equivalent. Returns (dg, dA, dbias)."""
    B, T, H, D = g.shape
    NT = (B * T + 31) // 32
    dg = torch.empty_like(g, dtype=torch.float32)
    dA_partial = torch.empty(NT, H, dtype=torch.float32, device=g.device)
    _kda_gate_bwd(
        g.contiguous(), A_log.contiguous(), dt_bias,
        dyg.contiguous(), dg, dA_partial,
        float(lower_bound) if lower_bound is not None else 0.0, lower_bound is not None,
    )
    dg = dg.type_as(g)
    dA = dA_partial.sum(0).type_as(A_log)
    dbias = dg.view(-1, H * D).sum(0).to(dt_bias) if dt_bias is not None else None
    return dg, dA, dbias


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fla.ops.kda.chunk_intra.chunk_kda_fwd_intra equivalent (recompute path).

    Returns (w, u, kg, Aqk, Akk); qg is not computed here (q=None path).
    """
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    B, T, _, _ = k.shape
    HV = gk.shape[2]
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gk = _f32(gk).contiguous()
    beta = beta.contiguous()

    Aqk = torch.empty(B, T, HV, chunk_size, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - the kernel only writes the lower triangle
    Akk = torch.zeros(B, T, HV, chunk_size, device=k.device, dtype=k.dtype)
    # fp32 buffer for the diagonal 16x16 blocks (precision in the tril solve)
    Akkd = torch.empty(B, T, HV, 16, device=k.device, dtype=torch.float32)

    if safe_gate:
        _chunk_kda_fwd_intra_sub_chunk(
            q, k, gk, beta, Aqk, Akkd, scale, chunk_size, cu_seqlens, chunk_indices,
        )
    else:
        _chunk_kda_fwd_intra_token_parallel(
            q, k, gk, beta, Aqk, Akkd, scale, chunk_size, cu_seqlens,
        )
    _chunk_kda_fwd_inter_solve_fused(
        q, k, gk, beta, Aqk, Akkd, Akk, scale, chunk_size, safe_gate,
        cu_seqlens, chunk_indices,
    )
    w, u, _, kg = _recompute_w_u_fwd(k, v, beta, Akk, gk, None, cu_seqlens, chunk_indices)
    return w, u, kg, Aqk, Akk


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """fla.ops.kda.chunk_intra.recompute_w_u_fwd equivalent. Returns (w, u, qg, kg)."""
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, A.shape[-1])
    gk = _f32(gk).contiguous()
    w, u, qg, kg = _recompute_w_u_fwd(
        k.contiguous(), v.contiguous(), beta.contiguous(),
        A.to(k.dtype).contiguous() if A.dtype != k.dtype else A.contiguous(),
        gk, q.contiguous() if q is not None else None,
        cu_seqlens, chunk_indices,
    )
    return w, u, qg, kg


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> torch.Tensor:
    """fla chunked GLA forward output (gk variant) equivalent."""
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    return _chunk_gla_fwd_o_gk(
        q.contiguous(), v.contiguous(), _f32(g).contiguous(),
        A.to(q.dtype).contiguous() if A.dtype != q.dtype else A.contiguous(),
        h.to(q.dtype).contiguous() if h.dtype != q.dtype else h.contiguous(),
        scale, state_v_first, cu_seqlens, chunk_size, chunk_indices,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_h equivalent.

    Returns (h, v_new, final_state). chunk_indices is accepted for signature
    parity; the kernel consumes chunk_offsets derived from cu_seqlens.
    """
    cu_seqlens, chunk_offsets, nt_total = _resolve_offsets(cu_seqlens, chunk_size)
    h, v_new, final_state = _chunk_gated_delta_rule_fwd_h(
        kg=_bf16(k).contiguous(), w=_bf16(w).contiguous(), u=_bf16(u).contiguous(),
        gk=_f32(gk).contiguous(),
        initial_state=_f32(initial_state).contiguous() if initial_state is not None else None,
        output_final_state=output_final_state,
        chunk_size=chunk_size, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets, nt_total=nt_total,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    do: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_bwd_dhu equivalent.

    Returns (dh, dh0, dv). chunk_indices is accepted for signature parity;
    the kernel consumes chunk_offsets derived from cu_seqlens.
    """
    cu_seqlens, chunk_offsets, nt_total = _resolve_offsets(cu_seqlens, chunk_size)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    dh, dh0, dv = _chunk_gated_delta_rule_bwd_dhu(
        qg=_bf16(q).contiguous(), kg=_bf16(k).contiguous(), w=_bf16(w).contiguous(),
        gk=_f32(gk).contiguous(), do_=_bf16(do).contiguous(), dv=_bf16(dv).contiguous(),
        h0=_f32(h0).contiguous() if h0 is not None else None,
        dht=_f32(dht).contiguous() if dht is not None else None,
        scale=scale, chunk_size=chunk_size, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets, nt_total=nt_total,
    )
    return dh, dh0, dv


def chunk_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fla.ops.kda.chunk_bwd.chunk_kda_bwd_dAv equivalent. Returns (dAqk fp32, dv)."""
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    return _chunk_kda_bwd_dAv(
        q, k, v.contiguous(), do.contiguous(),
        A.to(do.dtype).contiguous() if A.dtype != do.dtype else A.contiguous(),
        scale, cu_seqlens, chunk_indices, chunk_size,
    )


def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fla.ops.kda.chunk_bwd.chunk_kda_bwd_wy_dqkg_fused equivalent.

    Returns (dq, dk, dv, db, dg, dAkk); the dv output is the kernel's dv2.
    """
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)

    def _like_q(t):
        return t.to(q.dtype).contiguous() if t.dtype != q.dtype else t.contiguous()

    dq, dk, dv2, db, dg, dAkk = _chunk_kda_bwd_wy_dqkg_fused(
        q=q.contiguous(), k=k.contiguous(), v=_like_q(v), v_new=_like_q(v_new),
        g=_f32(g).contiguous(), beta=_f32(beta).contiguous(),
        A=_like_q(A), h=_like_q(h), do=_like_q(do), dh=_like_q(dh), dv=_like_q(dv),
        scale=scale, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size,
    )
    return dq, dk, dv2, db, dg, dAkk


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fla.ops.kda.chunk_intra.chunk_kda_bwd_intra equivalent.

    Returns (dq, dk, db, dg), all fp32; db already includes the upstream db.
    """
    cu_seqlens, chunk_indices = _resolve(cu_seqlens, chunk_indices, chunk_size)
    return _chunk_kda_bwd_intra(
        q.contiguous(), k.contiguous(), _f32(g).contiguous(), _f32(beta).contiguous(),
        _f32(dAqk).contiguous(), _f32(dAkk).contiguous(),
        _f32(dq).contiguous(), _f32(dk).contiguous(),
        _f32(db).contiguous(), _f32(dg).contiguous(),
        safe_gate, cu_seqlens, chunk_indices, chunk_size,
    )

from flash_kda.train.pipeline import (  # noqa: E402
    chunk_kda_train_bwd,
    chunk_kda_train_fwd,
)

__all__ = [
    'chunk_gated_delta_rule_bwd_dhu',
    'chunk_gated_delta_rule_fwd_h',
    'chunk_gla_fwd_o_gk',
    'chunk_kda_bwd_dAv',
    'chunk_kda_bwd_intra',
    'chunk_kda_bwd_wy_dqkg_fused',
    'chunk_kda_fwd_intra',
    'chunk_kda_train_bwd',
    'chunk_kda_train_fwd',
    'chunk_local_cumsum',
    'kda_gate_bwd',
    'kda_gate_chunk_cumsum',
    'prepare_chunk_indices',
    'recompute_w_u_fwd',
]
