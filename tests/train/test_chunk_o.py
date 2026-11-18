import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as tri_chunk_o
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum as tri_gate_cumsum
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import prepare_chunk_indices
from flash_kda.train._dev import load_stage

_mod = load_stage(
    "chunk_o",
    ["/root/FlashKDA/csrc/train/chunk_o.cu", "/root/FlashKDA/csrc/train/chunk_o_binding.cpp"],
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


def run_case(B, T, H, HV, K, V, state_v_first, lens=None, seed=42):
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
    scale = K ** -0.5
    # Triton reference chain: intra (Aqk, w, u, kg) then h/v_new
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu, chunk_size=64, chunk_indices=ci,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, state_v_first=state_v_first,
        chunk_size=64, cu_seqlens=cu, chunk_indices=ci,
    )

    ref_o = tri_chunk_o(
        q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale,
        state_v_first=state_v_first, cu_seqlens=cu, chunk_size=64, chunk_indices=ci,
    )
    o = _mod.chunk_gla_fwd_o_gk(
        q, v_new, g, Aqk, h, scale, state_v_first, cu, 64, ci,
    )
    assert_close("o", ref_o, o)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_chunk_o_dense(state_v_first):
    # T=200 is not a multiple of the 64 chunk size
    run_case(B=2, T=200, H=2, HV=4, K=128, V=128, state_v_first=state_v_first)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_chunk_o_varlen(state_v_first):
    lens = [70, 200, 16]
    run_case(B=1, T=sum(lens), H=2, HV=4, K=128, V=128, state_v_first=state_v_first, lens=lens)


def test_chunk_o_single_chunk():
    run_case(B=1, T=64, H=1, HV=2, K=64, V=64, state_v_first=False)


def test_chunk_o_deep_heads():
    run_case(B=1, T=150, H=2, HV=2, K=256, V=64, state_v_first=True)
