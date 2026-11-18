import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv as tri_bwd_dAv
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused as tri_fused
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import kda_gate_chunk_cumsum as cuda_gate_cumsum
from flash_kda.train._dev import load_stage

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_mod = load_stage(
    "bwd_wy_dqkg",
    ["csrc/train/bwd_wy_dqkg.cu", "csrc/train/bwd_wy_dqkg_binding.cpp"],
)
cuda_fused = _mod.chunk_kda_bwd_wy_dqkg_fused

K = V = 128
CHUNK = 64


def check(name, ref, out, rtol=5e-3, atol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    tol = atol + rtol * ref.abs()
    max_abs = err.max().item()
    max_rel = (err / ref.abs().clamp_min(1e-3)).max().item()
    worst = (err / tol).max().item()
    print(f"  {name}: max abs err {max_abs:.3e}, max rel err {max_rel:.3e}, err/tol {worst:.3f}")
    assert worst <= 1.0, f"{name}: max abs err {max_abs:.3e} exceeds tol (rtol={rtol}, atol={atol})"


def make_inputs(lens, H, HV, state_v_first, seed=42):
    """Generate realistic intermediates with fla's Triton forward/backward stages."""
    torch.manual_seed(seed)
    if isinstance(lens, int):
        B, T = 2, lens
        cu_seqlens = None
        chunk_indices = None
    else:
        B = 1
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device="cuda")
        T = int(cu[-1])
        cu_seqlens = cu
        chunk_indices = prepare_chunk_indices(cu, CHUNK)

    # input distributions follow fla's own tests/ops/test_kda.py to keep the
    # WY representation solve (Akk) and the chunk state well-conditioned
    dtype = torch.bfloat16
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda"), p=2, dim=-1).to(dtype)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device="cuda"), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, HV, V, dtype=dtype, device="cuda")
    do = torch.randn(B, T, HV, V, dtype=dtype, device="cuda")
    beta = torch.randn(B, T, HV, dtype=torch.float32, device="cuda").sigmoid()

    # fp32 log2-domain chunk cumsum gate (validated CUDA implementation)
    g_org = torch.randn(B, T, HV, K, dtype=torch.float32, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV * K, dtype=torch.float32, device="cuda")
    g = cuda_gate_cumsum(
        g_org, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=CHUNK,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    scale = K ** -0.5
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q, k, v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=CHUNK, chunk_indices=chunk_indices,
        disable_recompute=True,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, initial_state=None, output_final_state=False,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_size=CHUNK, chunk_indices=chunk_indices,
    )
    _, dv_intra = tri_bwd_dAv(
        q, k, v_new, do, A=Aqk, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=CHUNK, chunk_indices=chunk_indices,
    )
    dh, _, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg, k=kg, w=w, do=do, dv=dv_intra, gk=g, h0=None, dht=None, scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_size=CHUNK, chunk_indices=chunk_indices,
    )
    args = dict(
        scale=scale, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=CHUNK,
    )
    return (q, k, v, v_new, g, beta, Akk, h, do, dh, dv), args


def run_case(lens, H, HV, state_v_first, seed=42):
    tensors, args = make_inputs(lens, H, HV, state_v_first, seed)
    ref = tri_fused(*tensors, **args)
    out = cuda_fused(*tensors, **args)
    names = ["dq", "dk", "dv2", "db", "dg", "dAkk"]
    for name, r, o in zip(names, ref, out):
        check(name, r, o)


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("T", [256, 200])  # 200 is not a multiple of the chunk size
def test_bwd_wy_dqkg_dense(state_v_first, T):
    print(f"dense T={T} state_v_first={state_v_first} (GVA: H=2, HV=4)")
    run_case(T, 2, 4, state_v_first)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_bwd_wy_dqkg_dense_no_gva(state_v_first):
    print(f"dense T=130 state_v_first={state_v_first} (H=HV=2)")
    run_case(130, 2, 2, state_v_first)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_bwd_wy_dqkg_varlen(state_v_first):
    print(f"varlen lens=[100, 1, 64, 300] state_v_first={state_v_first}")
    run_case([100, 1, 64, 300], 2, 4, state_v_first)
