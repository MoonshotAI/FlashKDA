import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv as tri_dav
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import kda_gate_chunk_cumsum, prepare_chunk_indices
from flash_kda.train._dev import load_stage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_dav = load_stage('bwd_dav', ['csrc/train/bwd_dav.cu', 'csrc/train/bwd_dav_binding.cpp'])


def check(name, ref, out, rtol=5e-3, atol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    tol = atol + rtol * ref.abs()
    ratio = (err / tol).max().item()
    print(f"{name}: max abs err {err.max().item():.3e}, worst err/tol {ratio:.3f}")
    assert ratio < 1.0, f"{name}: max abs err {err.max().item():.3e}"


def make_inputs(T, B=2, H=2, HV=4, K=128, V=64, cu_seqlens=None, seed=0):
    torch.manual_seed(seed)
    dev = 'cuda'
    chunk_size = 64
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None

    g_raw = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(HV * K, dtype=torch.float32, device=dev)
    g = kda_gate_chunk_cumsum(
        g_raw, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    # q/k are l2-normalized as in the real KDA path (fla.layers.kda applies l2norm);
    # raw randn values explode the WY state recursion and produce NaNs in v_new.
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=dev), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=dev), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=dev)
    scale = K ** -0.5

    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
    )
    do = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    return q, k, v_new, do, Aqk, scale, cu_seqlens, chunk_indices


def run_case(T, B=2, H=2, HV=4, K=128, V=64, lens=None, seed=0):
    if lens is not None:
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device='cuda')
        T = int(cu[-1])
        B = 1
    else:
        cu = None
    q, k, v_new, do, Aqk, scale, cu, ci = make_inputs(T, B=B, H=H, HV=HV, K=K, V=V, cu_seqlens=cu, seed=seed)

    ref_dA, ref_dv = tri_dav(
        q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale,
        cu_seqlens=cu, chunk_size=64, chunk_indices=ci,
    )
    dA, dv = _dav.chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do_=do, A=Aqk, scale=scale,
        cu_seqlens=cu, chunk_indices=ci, chunk_size=64,
    )
    check("dA", ref_dA, dA)
    check("dv", ref_dv, dv, rtol=1e-2, atol=1e-2)  # bf16 storage: 1-ulp wiggle allowed


def test_dense_multichunk():
    run_case(T=256, V=64)


def test_dense_non_multiple_of_64():
    run_case(T=200, V=64)


def test_dense_two_v_tiles():
    run_case(T=128, V=128)


def test_varlen():
    run_case(T=None, lens=[70, 200, 16], V=64)


def test_varlen_single_token_seq():
    run_case(T=None, lens=[1, 64, 100], V=128)
