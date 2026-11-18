"""End-to-end pipeline test: flash_kda.train CUDA pipeline vs fla Triton hosts.

Compares `chunk_kda_train_fwd`/`chunk_kda_train_bwd` against
`fla.ops.kda.chunk_fwd.chunk_kda_fwd` / `fla.ops.kda.chunk_bwd.chunk_kda_bwd`
stage-composed references on identical inputs (recompute path only).
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda  # noqa: F401

fla = pytest.importorskip("fla")

from fla.ops.kda.chunk_bwd import chunk_kda_bwd  # noqa: E402
from fla.ops.kda.chunk_fwd import chunk_kda_fwd  # noqa: E402

from flash_kda.train.pipeline import RCP_LN2, chunk_kda_train_bwd, chunk_kda_train_fwd  # noqa: E402

device = "cuda"


def _make_inputs(B, T, H, D, N, use_gate_in_kernel, safe_gate, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    if not use_gate_in_kernel:
        # g must be a non-positive log-decay (fla test_chunk uses logsigmoid);
        # raw randn gates make exp2(cumsum) overflow and even the Triton reference NaNs
        g = F.logsigmoid(g)
        if safe_gate:
            g = g.clamp(-5, 0)
        g = g.to(dtype)
    else:
        g = g.to(dtype)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    if use_gate_in_kernel:
        A_log = torch.log(torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16))
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    else:
        A_log = dt_bias = None
    do = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dht = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    return q, k, v, g, beta, h0, A_log, dt_bias, do, dht


def _run_pair(B, T, H, D, use_gate_in_kernel, safe_gate, cu_seqlens=None):
    lower_bound = -5.0 if safe_gate else None
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    q, k, v, g, beta, h0, A_log, dt_bias, do, dht = _make_inputs(
        B, T, H, D, N, use_gate_in_kernel, safe_gate)
    scale = D ** -0.5
    common = dict(
        scale=scale, initial_state=h0, cu_seqlens=cu_seqlens,
        safe_gate=safe_gate, lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias,
        chunk_size=64,
    )

    ref_o, ref_ht, ref_g, ref_Aqk, ref_Akk, *_ = chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta, output_final_state=True, **common)
    cuda_o, cuda_ht, cuda_g, cuda_Aqk, cuda_Akk = chunk_kda_train_fwd(
        q=q, k=k, v=v, g=g, beta=beta, output_final_state=True, **common)

    bwd_common = dict(
        q=q, k=k, v=v, beta=beta, scale=scale, initial_state=h0, do=do, dht=dht,
        cu_seqlens=cu_seqlens, safe_gate=safe_gate, lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias,
        chunk_size=64,
    )
    ref_out = chunk_kda_bwd(
        Aqk=ref_Aqk, Akk=ref_Akk, g=ref_g, g_org=g if use_gate_in_kernel else None, **bwd_common)
    cuda_out = chunk_kda_train_bwd(
        Aqk=cuda_Aqk, Akk=cuda_Akk, g=cuda_g, g_org=g if use_gate_in_kernel else None, **bwd_common)
    return (ref_o, ref_ht, *ref_out), (cuda_o, cuda_ht, *cuda_out)


def _assert_pair(ref, cuda, use_gate_in_kernel):
    names = ["o", "ht", "dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias"]
    # o/ht carry the bf16 Aqk storage-order diff (~4e-3) straight through;
    # dbias is a full-T reduction of dg with cancellation, so its relative
    # error is amplified (fla's own test warns on dA instead of asserting)
    ratios = [0.008, 0.008, 0.008, 0.008, 0.008, 0.02, 0.02, 0.008, 0.02, 0.02]
    for name, r, c, ratio in zip(names, ref, cuda, ratios):
        if r is None and c is None:
            continue
        r, c = r.float(), c.float()
        denom = r.abs().max().clamp(min=1e-6)
        err = (r - c).abs().max() / denom
        assert err < ratio, f"{name}: relative max err {err:.2e} >= {ratio}"


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "use_gate_in_kernel", "safe_gate"),
    [
        (2, 512, 4, 128, False, False),
        (2, 1000, 4, 128, False, True),
        (1, 1024, 4, 128, True, True),
        (2, 256, 2, 128, True, False),
    ],
)
def test_pipeline_dense(B, T, H, D, use_gate_in_kernel, safe_gate):
    ref, cuda = _run_pair(B, T, H, D, use_gate_in_kernel, safe_gate)
    _assert_pair(ref, cuda, use_gate_in_kernel)


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens"),
    [
        (4, 128, [0, 256, 500, 1000]),
        (4, 128, [0, 100, 300, 1200, 2000]),
    ],
)
def test_pipeline_varlen(H, D, cu_seqlens):
    cu = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    ref, cuda = _run_pair(1, T, H, D, True, True, cu_seqlens=cu)
    _assert_pair(ref, cuda, True)


def test_pipeline_no_initial_state():
    B, T, H, D = 2, 512, 4, 128
    lower_bound = -5.0
    q, k, v, g, beta, h0, A_log, dt_bias, do, dht = _make_inputs(B, T, H, D, B, True, True)
    scale = D ** -0.5
    common = dict(
        scale=scale, initial_state=None, cu_seqlens=None,
        safe_gate=True, lower_bound=lower_bound,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, chunk_size=64,
    )
    ref_o, _, ref_g, ref_Aqk, ref_Akk, *_ = chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta, output_final_state=False, **common)
    cuda_o, _, cuda_g, cuda_Aqk, cuda_Akk = chunk_kda_train_fwd(
        q=q, k=k, v=v, g=g, beta=beta, output_final_state=False, **common)
    bwd_common = dict(
        q=q, k=k, v=v, beta=beta, scale=scale, initial_state=None, do=do, dht=None,
        cu_seqlens=None, safe_gate=True, lower_bound=lower_bound,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, chunk_size=64,
    )
    ref_out = chunk_kda_bwd(Aqk=ref_Aqk, Akk=ref_Akk, g=ref_g, g_org=g, **bwd_common)
    cuda_out = chunk_kda_train_bwd(Aqk=cuda_Aqk, Akk=cuda_Akk, g=cuda_g, g_org=g, **bwd_common)
    names = ["dq", "dk", "dv", "db", "dg", "dA", "dbias"]
    ratios = [0.008, 0.008, 0.008, 0.02, 0.02, 0.02, 0.02]
    for name, r, c, ratio in zip(names, ref_out[:5] + ref_out[6:], cuda_out[:5] + cuda_out[6:], ratios):
        r, c = r.float(), c.float()
        err = (r - c).abs().max() / r.abs().max().clamp(min=1e-6)
        assert err < ratio, f"{name}: relative max err {err:.2e} >= {ratio}"
    err = (ref_o.float() - cuda_o.float()).abs().max() / ref_o.float().abs().max().clamp(min=1e-6)
    assert err < 0.008, f"o: relative max err {err:.2e}"
