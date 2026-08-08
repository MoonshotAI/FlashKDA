import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu as tri_bwd_dhu,
)
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h as tri_fwd_h,
)
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train._dev import load_stage

chunk_h = load_stage(
    "chunk_h",
    ["csrc/train/chunk_h.cu", "csrc/train/chunk_h_binding.cpp"],
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_close(name, ref, out, tol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    denom = ref.abs().clamp_min(1.0)
    ratio = (err / denom).max().item()
    print(f"{name}: max abs err {err.max().item():.3e}, max rel(clamp1) err {ratio:.3e}")
    assert ratio < tol or err.max().item() < tol, (
        f"{name}: max rel err {ratio:.3e}, max abs err {err.max().item():.3e}"
    )


def make_intra_inputs(T, H, K, V, B=1, lens=None, seed=42):
    """Build realistic (w, u, qg, kg, gk) through fla's Triton intra pipeline."""
    torch.manual_seed(seed)
    dev = "cuda"
    HV = H
    cu = None
    ci = None
    if lens is not None:
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device=dev)
        ci = prepare_chunk_indices(cu, 64)
        B, T = 1, int(cu[-1])
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    # tame k so Akk_inv (and hence w/u) stays well-conditioned for large K
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev) * 0.5
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32, device=dev))
    # natural-log domain per-token decay, turned into fp32 log2 chunk cumsum
    g_raw = -torch.rand(B, T, HV, K, dtype=torch.float32, device=dev) * 4
    gk = chunk_local_cumsum(
        g_raw, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci,
        output_dtype=torch.float32,
    )
    scale = K ** -0.5
    w, u, qg, kg, _, _ = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=gk, beta=beta, scale=scale,
        cu_seqlens=cu, chunk_indices=ci, chunk_size=64,
        safe_gate=False, disable_recompute=True,
    )
    return w, u, qg, kg, gk, scale, cu, ci


def make_state(N, HV, K, V, state_v_first, seed=0):
    torch.manual_seed(seed)
    shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    return torch.randn(shape, dtype=torch.float32, device="cuda")


def varlen_meta(cu):
    if cu is None:
        return None, 0
    coff = prepare_chunk_offsets(cu, 64)
    return coff, int(coff[-1])


@pytest.mark.parametrize("K", [64, 128, 256])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("use_h0", [False, True])
def test_fwd_h_dense(K, state_v_first, use_h0):
    B, T, H, V = 2, 200, 4, 64  # T not a multiple of 64
    w, u, _, kg, gk, _, _, _ = make_intra_inputs(T, H, K, V, B=B)
    h0 = make_state(B, H, K, V, state_v_first, seed=1) if use_h0 else None

    h_ref, vn_ref, fs_ref = tri_fwd_h(
        k=kg, w=w, u=u, g=None, gk=gk, initial_state=h0,
        output_final_state=True, chunk_size=64, save_new_value=True,
        state_v_first=state_v_first,
    )
    h, vn, fs = chunk_h.chunk_gated_delta_rule_fwd_h(
        kg=kg, w=w, u=u, gk=gk, initial_state=h0,
        output_final_state=True, chunk_size=64, state_v_first=state_v_first,
    )
    assert_close("h", h_ref, h)
    assert_close("v_new", vn_ref, vn)
    assert_close("final_state", fs_ref, fs)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_fwd_h_varlen(state_v_first):
    H, K, V = 4, 128, 64
    lens = [70, 200, 16]
    w, u, _, kg, gk, _, cu, ci = make_intra_inputs(0, H, K, V, lens=lens)
    N = len(lens)
    h0 = make_state(N, H, K, V, state_v_first, seed=2)
    coff, nt_total = varlen_meta(cu)

    h_ref, vn_ref, fs_ref = tri_fwd_h(
        k=kg, w=w, u=u, g=None, gk=gk, initial_state=h0,
        output_final_state=True, chunk_size=64, save_new_value=True,
        state_v_first=state_v_first, cu_seqlens=cu, chunk_indices=ci,
    )
    h, vn, fs = chunk_h.chunk_gated_delta_rule_fwd_h(
        kg=kg, w=w, u=u, gk=gk, initial_state=h0,
        output_final_state=True, chunk_size=64, state_v_first=state_v_first,
        cu_seqlens=cu, chunk_offsets=coff, nt_total=nt_total,
    )
    assert_close("h_varlen", h_ref, h)
    assert_close("v_new_varlen", vn_ref, vn)
    assert_close("final_state_varlen", fs_ref, fs)


@pytest.mark.parametrize("K", [64, 128, 256])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("use_dht", [False, True])
def test_bwd_dhu_dense(K, state_v_first, use_dht):
    B, T, H, V = 2, 200, 4, 64
    w, _, qg, kg, gk, scale, _, _ = make_intra_inputs(T, H, K, V, B=B)
    torch.manual_seed(7)
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    dv = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
    h0 = make_state(B, H, K, V, state_v_first, seed=3)
    dht = make_state(B, H, K, V, state_v_first, seed=4) if use_dht else None

    dh_ref, dh0_ref, dv2_ref = tri_bwd_dhu(
        q=qg, k=kg, w=w, do=do, dv=dv, g=None, gk=gk, h0=h0, dht=dht,
        scale=scale, state_v_first=state_v_first, chunk_size=64,
    )
    dh, dh0, dv2 = chunk_h.chunk_gated_delta_rule_bwd_dhu(
        qg=qg, kg=kg, w=w, gk=gk, do_=do, dv=dv, h0=h0, dht=dht,
        scale=scale, chunk_size=64, state_v_first=state_v_first,
    )
    assert_close("dh", dh_ref, dh)
    assert_close("dh0", dh0_ref, dh0)
    assert_close("dv2", dv2_ref, dv2)


@pytest.mark.parametrize("state_v_first", [False, True])
def test_bwd_dhu_varlen(state_v_first):
    H, K, V = 4, 128, 64
    lens = [70, 200, 16]
    w, _, qg, kg, gk, scale, cu, ci = make_intra_inputs(0, H, K, V, lens=lens)
    N = len(lens)
    T_total = int(cu[-1])
    torch.manual_seed(8)
    do = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device="cuda")
    dv = torch.randn(1, T_total, H, V, dtype=torch.bfloat16, device="cuda")
    h0 = make_state(N, H, K, V, state_v_first, seed=5)
    dht = make_state(N, H, K, V, state_v_first, seed=6)
    coff, nt_total = varlen_meta(cu)

    dh_ref, dh0_ref, dv2_ref = tri_bwd_dhu(
        q=qg, k=kg, w=w, do=do, dv=dv, g=None, gk=gk, h0=h0, dht=dht,
        scale=scale, state_v_first=state_v_first, chunk_size=64,
        cu_seqlens=cu, chunk_indices=ci,
    )
    dh, dh0, dv2 = chunk_h.chunk_gated_delta_rule_bwd_dhu(
        qg=qg, kg=kg, w=w, gk=gk, do_=do, dv=dv, h0=h0, dht=dht,
        scale=scale, chunk_size=64, state_v_first=state_v_first,
        cu_seqlens=cu, chunk_offsets=coff, nt_total=nt_total,
    )
    assert_close("dh_varlen", dh_ref, dh)
    assert_close("dh0_varlen", dh0_ref, dh0)
    assert_close("dv2_varlen", dv2_ref, dv2)
