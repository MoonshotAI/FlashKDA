import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as tri_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum as tri_gate_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2

_REPO = "/root/FlashKDA"

# Load the dev loader by path to avoid importing flash_kda.train (__init__
# pulls in the main extension, which this stage does not depend on).
_spec = importlib.util.spec_from_file_location(
    "flash_kda_train_dev", os.path.join(_REPO, "flash_kda", "train", "_dev.py")
)
_dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dev)

cuda_intra = _dev.load_stage(
    "intra", ["csrc/train/intra.cu", "csrc/train/intra_binding.cpp"], verbose=False
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def check_close(name, ref, out, rtol=5e-3, atol=5e-3):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    denom = ref.abs().clamp_min(1.0)
    rel = (err / denom).max().item()
    abs_max = err.max().item()
    print(f"{name}: max abs err {abs_max:.3e}, max rel err {rel:.3e}")
    assert rel < rtol or abs_max < atol, (
        f"{name}: max rel err {rel:.3e}, max abs err {abs_max:.3e}"
    )


def make_inputs(B, T, H, HV, K, safe_gate, chunk_size, cu_seqlens=None, seed=42):
    torch.manual_seed(seed)
    q = F.normalize(torch.rand(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).cuda()
    k = F.normalize(torch.rand(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).cuda()
    v = torch.rand(B, T, HV, K, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda").sigmoid()
    g_raw = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(HV * K, dtype=torch.float32, device="cuda")
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    )
    g = tri_gate_cumsum(
        g_raw, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
        lower_bound=-5.0 if safe_gate else None,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    return q, k, v, g, beta, chunk_indices


def run_case(B, T, H, HV, K, safe_gate, chunk_size, scale=None, lens=None, seed=42):
    if scale is None:
        scale = K ** -0.5
    if lens is not None:
        assert B == 1
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device="cuda")
        T = int(cu[-1])
        seq_spans = [[(int(cu[i]), int(cu[i + 1])) for i in range(len(lens))]]
    else:
        cu = None
        seq_spans = [[(0, T)] for b in range(B)]  # per-batch local span

    q, k, v, g, beta, chunk_indices = make_inputs(
        B, T, H, HV, K, safe_gate, chunk_size, cu_seqlens=cu, seed=seed
    )

    # Triton reference (also runs recompute_w_u_fwd, which we ignore here)
    *_, ref_Aqk, ref_Akk = tri_intra(
        q, k, v, gk=g, beta=beta, scale=scale,
        cu_seqlens=cu, chunk_size=chunk_size, chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    BT = chunk_size
    Aqk = torch.empty(B, T, HV, BT, dtype=k.dtype, device=k.device)
    # Akk must be zero-initialized like fla's host code: kernels only write
    # the lower triangle.
    Akk = torch.zeros(B, T, HV, BT, dtype=k.dtype, device=k.device)
    Akkd = torch.empty(B, T, HV, 16, dtype=torch.float32, device=k.device)
    if safe_gate:
        cuda_intra.chunk_kda_fwd_intra_sub_chunk(
            q, k, g, beta, Aqk, Akkd, scale, BT, cu, chunk_indices
        )
    else:
        cuda_intra.chunk_kda_fwd_intra_token_parallel(
            q, k, g, beta, Aqk, Akkd, scale, BT, cu
        )
    cuda_intra.chunk_kda_fwd_inter_solve_fused(
        q, k, g, beta, Aqk, Akkd, Akk, scale, BT, safe_gate, cu, chunk_indices
    )
    torch.cuda.synchronize()

    tag = f"safe={safe_gate} B{B} T{T} H{H} HV{HV} K{K} BT{BT}"
    check_close(f"Akk [{tag}]", ref_Akk, Akk)

    # Aqk: compare only entries written by both sides. A token at
    # sequence-local position t has chunk-local row r = t % BT; the written
    # region is columns 0..r (chunk-local lower triangle incl. diagonal).
    for b, spans in enumerate(seq_spans):
        for bos, eos in spans:
            t_loc = torch.arange(eos - bos, device=q.device)
            col = torch.arange(BT, device=q.device)
            m = col[None, :] <= (t_loc % BT)[:, None]  # [seq_len, BT]
            ref_s = ref_Aqk[b, bos:eos].float()[:, :, :][m[:, None, :].expand(-1, HV, -1)]
            out_s = Aqk[b, bos:eos].float()[:, :, :][m[:, None, :].expand(-1, HV, -1)]
            check_close(f"Aqk [{tag}] b{b} seq[{bos}:{eos}]", ref_s, out_s)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_intra_dense(safe_gate):
    run_case(B=2, T=300, H=2, HV=4, K=128, safe_gate=safe_gate, chunk_size=64)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_intra_ragged_T(safe_gate):
    # T not a multiple of 64 and a final partial sub-chunk
    run_case(B=1, T=100, H=2, HV=2, K=128, safe_gate=safe_gate, chunk_size=64)
    run_case(B=2, T=17, H=1, HV=2, K=64, safe_gate=safe_gate, chunk_size=64)
    run_case(B=1, T=1, H=1, HV=1, K=64, safe_gate=safe_gate, chunk_size=64)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_intra_chunk32(safe_gate):
    run_case(B=2, T=200, H=2, HV=4, K=128, safe_gate=safe_gate, chunk_size=32)
    run_case(B=1, T=33, H=1, HV=1, K=64, safe_gate=safe_gate, chunk_size=32)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_intra_nonpow2_K(safe_gate):
    run_case(B=1, T=150, H=2, HV=2, K=60, safe_gate=safe_gate, chunk_size=64)
    run_case(B=1, T=150, H=2, HV=2, K=100, safe_gate=safe_gate, chunk_size=64)


@pytest.mark.parametrize("safe_gate", [True, False])
def test_intra_varlen(safe_gate):
    lens = [100, 1, 64, 300]
    run_case(B=1, T=None, H=2, HV=4, K=128, safe_gate=safe_gate, chunk_size=64, lens=lens)


def test_intra_gva_scale():
    run_case(B=2, T=256, H=2, HV=8, K=128, safe_gate=True, chunk_size=64, scale=1.0)
    run_case(B=2, T=256, H=2, HV=8, K=128, safe_gate=False, chunk_size=64, scale=1.0)
