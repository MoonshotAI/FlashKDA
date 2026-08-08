import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum as tri_gate_cumsum
from fla.ops.kda.wy_fast import recompute_w_u_fwd as tri_recompute
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import prepare_chunk_indices
from flash_kda.train._dev import load_stage

_mod = load_stage(
    "wy_fast",
    ["/root/FlashKDA/csrc/train/wy_fast.cu", "/root/FlashKDA/csrc/train/wy_fast_binding.cpp"],
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_close(name, ref, out, rtol=5e-3, atol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    denom = ref.abs().clamp_min(1.0)
    ratio = (err / denom).max().item()
    max_abs = err.max().item()
    print(f"{name}: max rel err {ratio:.3e}, max abs err {max_abs:.3e}")
    assert ratio < rtol or max_abs < atol, (
        f"{name}: max rel err {ratio:.3e}, max abs err {max_abs:.3e}"
    )


def build_inputs(B, T, H, HV, K, V, lens=None, seed=42):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
    # match the real pipeline (use_qk_l2norm_in_kernel): unnormalized k makes
    # the Akk inverse ill-conditioned and the reference chain itself blows up
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, HV, V, device=dev, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, HV, device=dev)).float()
    g_raw = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
    A_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV * K, device=dev, dtype=torch.float32)
    cu = None
    ci = None
    if lens is not None:
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device=dev)
        ci = prepare_chunk_indices(cu, 64)
    g = tri_gate_cumsum(
        g_raw, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64,
        cu_seqlens=cu, chunk_indices=ci,
    )
    # Akk from the Triton intra chain: bf16 full lower-triangular inverse
    *_, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=K ** -0.5,
        cu_seqlens=cu, chunk_size=64, chunk_indices=ci,
    )
    return q, k, v, beta, g, Akk, cu, ci


def run_case(B, T, H, HV, K, V, lens=None, store_qg=True, seed=42):
    q, k, v, beta, g, Akk, cu, ci = build_inputs(B, T, H, HV, K, V, lens, seed)

    ref_w, ref_u, ref_qg, ref_kg = tri_recompute(
        k=k, v=v, beta=beta, A=Akk, gk=g,
        q=q if store_qg else None, cu_seqlens=cu, chunk_indices=ci,
    )
    w, u, qg, kg = _mod.recompute_w_u_fwd(
        k, v, beta, Akk, g,
        q if store_qg else None, cu, ci,
    )

    assert_close("w", ref_w, w)
    assert_close("u", ref_u, u)
    assert_close("kg", ref_kg, kg)
    if store_qg:
        assert_close("qg", ref_qg, qg)
    else:
        assert qg is None


@pytest.mark.parametrize("store_qg", [True, False])
def test_wy_fast_dense(store_qg):
    # T=200 is not a multiple of the 64 chunk size
    run_case(B=2, T=200, H=2, HV=4, K=128, V=128, store_qg=store_qg)


def test_wy_fast_single_chunk():
    run_case(B=1, T=64, H=1, HV=2, K=64, V=64)


def test_wy_fast_irregular_dims():
    run_case(B=2, T=100, H=1, HV=2, K=100, V=68)


@pytest.mark.parametrize("store_qg", [True, False])
def test_wy_fast_varlen(store_qg):
    lens = [70, 200, 16]
    run_case(B=1, T=sum(lens), H=2, HV=4, K=128, V=128, lens=lens, store_qg=store_qg)


def test_wy_fast_varlen_exact_chunks():
    lens = [64, 128]
    run_case(B=1, T=sum(lens), H=1, HV=2, K=128, V=64, lens=lens)
