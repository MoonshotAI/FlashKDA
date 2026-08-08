"""End-to-end KDA training forward/backward on the CUDA kernels.

Mirrors `fla.ops.kda.chunk_fwd.chunk_kda_fwd` and
`fla.ops.kda.chunk_bwd.chunk_kda_bwd` for the default recompute path
(`disable_recompute=False`): the forward only returns `o`, `final_state`,
the gate cumsum (or `None` when `use_gate_in_kernel=True`), `Aqk` and `Akk`;
everything else is recomputed during the backward.

Stage wrappers are imported lazily from `flash_kda.train` so this module
stays importable while individual stages are still under development.
"""

import torch

RCP_LN2 = 1.4426950408889634


def chunk_kda_train_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
):
    from flash_kda.train import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gla_fwd_o_gk,
        chunk_kda_fwd_intra,
        chunk_local_cumsum,
        kda_gate_chunk_cumsum,
    )

    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g, A_log=A_log, dt_bias=dt_bias, chunk_size=chunk_size, scale=RCP_LN2,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
        )
        g_out = None
    else:
        g = chunk_local_cumsum(
            g=g, chunk_size=chunk_size, scale=RCP_LN2,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
        g_out = g

    w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, state_v_first=state_v_first,
    )

    o = chunk_gla_fwd_o_gk(
        q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return o, final_state, g_out, Aqk, Akk


def chunk_kda_train_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    g: torch.Tensor | None = None,
    g_org: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
):
    from flash_kda.train import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
        chunk_kda_bwd_dAv,
        chunk_kda_bwd_intra,
        chunk_kda_bwd_wy_dqkg_fused,
        chunk_local_cumsum,
        kda_gate_bwd,
        kda_gate_chunk_cumsum,
        recompute_w_u_fwd,
    )

    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g_org, A_log=A_log, dt_bias=dt_bias, chunk_size=chunk_size, scale=RCP_LN2,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
        )

    w, u, qg, kg = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=Akk, gk=g, q=q,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        initial_state=initial_state, output_final_state=False,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=chunk_size, state_v_first=state_v_first,
    )

    dAqk, dv = chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg, k=kg, w=w, gk=g, h0=initial_state, dht=dht, do=do, dv=dv, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )

    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=Akk, h=h, do=do, dh=dh, dv=dv,
        scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
        chunk_indices=chunk_indices, state_v_first=state_v_first,
    )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, db=db, dg=dg,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    dg = chunk_local_cumsum(
        dg, chunk_size=chunk_size, reverse=True,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    dA, dbias = None, None
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(
            g=g_org, A_log=A_log, dt_bias=dt_bias, dyg=dg, lower_bound=lower_bound,
        )
    return dq, dk, dv, db, dg, dh0, dA, dbias
