import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv as tri_dav
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as tri_intra
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import kda_gate_chunk_cumsum, prepare_chunk_indices
from flash_kda.train._dev import load_stage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_intra = load_stage('bwd_intra', ['csrc/train/bwd_intra.cu', 'csrc/train/bwd_intra_binding.cpp'])


def check(name, ref, out, rtol=5e-3, atol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    tol = atol + rtol * ref.abs()
    ratio = (err / tol).max().item()
    print(f"{name}: max abs err {err.max().item():.3e}, worst err/tol {ratio:.3f}")
    assert ratio < 1.0, f"{name}: max abs err {err.max().item():.3e}"


def run_case(safe_gate, T, B=2, H=2, HV=4, K=128, V=64, lens=None, seed=0):
    torch.manual_seed(seed)
    dev = 'cuda'
    chunk_size = 64
    if lens is not None:
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device=dev)
        T = int(cu[-1])
        B = 1
    else:
        cu = None
    ci = prepare_chunk_indices(cu, chunk_size) if cu is not None else None

    g_raw = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(HV * K, dtype=torch.float32, device=dev)
    g = kda_gate_chunk_cumsum(
        g_raw, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
        cu_seqlens=cu, chunk_indices=ci,
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
        cu_seqlens=cu, chunk_size=chunk_size, chunk_indices=ci,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        cu_seqlens=cu, chunk_size=chunk_size, chunk_indices=ci,
    )
    do = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)

    # dAqk from the Triton dAv kernel (fp32); dAkk stands in for the fused kernel's
    # output with values only in its strict lower triangle, as produced upstream.
    dAqk, _ = tri_dav(
        q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale,
        cu_seqlens=cu, chunk_size=chunk_size, chunk_indices=ci,
    )
    dAkk = torch.randn(B, T, HV, chunk_size, dtype=torch.float32, device=dev) * 0.1
    t_in_chunk = torch.arange(T, device=dev) % chunk_size
    mask = t_in_chunk[:, None] > torch.arange(chunk_size, device=dev)[None, :]  # strict lower
    dAkk = dAkk * mask[None, :, None, :]

    # Upstream (fused kernel) gradients; the intra kernel accumulates on top of these.
    dq = torch.randn(B, T, HV, K, dtype=torch.float32, device=dev)
    dk = torch.randn(B, T, HV, K, dtype=torch.float32, device=dev)
    db = torch.randn(B, T, HV, dtype=torch.float32, device=dev)
    dg = torch.randn(B, T, HV, K, dtype=torch.float32, device=dev)

    ref = tri_intra(
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dk=dk, db=db, dg=dg,
        cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size, safe_gate=safe_gate,
    )
    out = _intra.chunk_kda_bwd_intra(
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dk=dk, db=db, dg=dg,
        safe_gate=safe_gate, cu_seqlens=cu, chunk_indices=ci, chunk_size=chunk_size,
    )
    names = ["dq", "dk", "db", "dg"]
    for name, r, o in zip(names, ref, out):
        tol = 2e-2 if name in ("db", "dg") else 5e-3
        check(name, r, o, rtol=tol, atol=tol)


@pytest.mark.parametrize("safe_gate", [False, True])
def test_dense(safe_gate):
    run_case(safe_gate, T=256)


@pytest.mark.parametrize("safe_gate", [False, True])
def test_dense_non_multiple_of_64(safe_gate):
    run_case(safe_gate, T=200)


@pytest.mark.parametrize("safe_gate", [False, True])
def test_varlen(safe_gate):
    run_case(safe_gate, T=None, lens=[70, 200, 16])


@pytest.mark.parametrize("safe_gate", [False, True])
def test_varlen_tiny_seqs(safe_gate):
    run_case(safe_gate, T=None, lens=[1, 8, 64], V=128)


@pytest.mark.parametrize("safe_gate", [False, True])
def test_k64_single_chunk(safe_gate):
    run_case(safe_gate, T=64, K=64)
