import pytest
import torch

from conftest import _FLA_REPO  # noqa: F401  (ensures sys.path setup runs)

from fla.ops.kda.gate import kda_gate_bwd as tri_gate_bwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum as tri_gate_cumsum
from fla.ops.utils import chunk_local_cumsum as tri_cumsum
from fla.ops.utils.constant import RCP_LN2

from flash_kda.train import chunk_local_cumsum as cuda_cumsum
from flash_kda.train import kda_gate_bwd as cuda_gate_bwd
from flash_kda.train import kda_gate_chunk_cumsum as cuda_gate_cumsum
from flash_kda.train import prepare_chunk_indices

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_close(name, ref, out, rtol=1e-4, atol=1e-5):
    ref = ref.float()
    out = out.float()
    err = (ref - out).abs()
    denom = ref.abs().clamp_min(1.0)
    ratio = (err / denom).max().item()
    assert ratio < rtol or err.max().item() < atol, (
        f"{name}: max rel err {ratio:.3e}, max abs err {err.max().item():.3e}"
    )


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gate_cumsum_dense(lower_bound, has_bias, dtype):
    torch.manual_seed(42)
    B, T, H, K = 2, 1000, 4, 128
    g = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
    A_log = torch.randn(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(H * K, dtype=torch.float32, device="cuda") if has_bias else None

    ref = tri_gate_cumsum(g, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64, lower_bound=lower_bound)
    out = cuda_gate_cumsum(g, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64, lower_bound=lower_bound)
    assert_close("g_cumsum", ref, out)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_gate_cumsum_varlen(lower_bound):
    torch.manual_seed(42)
    H, K = 4, 128
    lens = [100, 1, 64, 300]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device="cuda")
    T_total = int(cu[-1])
    g = torch.randn(1, T_total, H, K, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(H * K, dtype=torch.float32, device="cuda")

    chunk_indices = prepare_chunk_indices(cu, 64)
    ref = tri_gate_cumsum(
        g, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64,
        cu_seqlens=cu, chunk_indices=chunk_indices, lower_bound=lower_bound,
    )
    out = cuda_gate_cumsum(
        g, A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64,
        cu_seqlens=cu, chunk_indices=chunk_indices, lower_bound=lower_bound,
    )
    assert_close("g_cumsum_varlen", ref, out)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_local_cumsum(reverse, dtype):
    torch.manual_seed(42)
    B, T, H, K = 2, 500, 4, 128
    g = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
    scale = RCP_LN2 if not reverse else None

    ref = tri_cumsum(g, chunk_size=64, scale=scale, reverse=reverse, output_dtype=torch.float32)
    out = cuda_cumsum(g, chunk_size=64, scale=scale, reverse=reverse)
    assert_close("cumsum", ref, out)


def test_local_cumsum_varlen():
    torch.manual_seed(42)
    H, K = 4, 128
    lens = [70, 200, 16]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.long, device="cuda")
    T_total = int(cu[-1])
    g = torch.randn(1, T_total, H, K, dtype=torch.float32, device="cuda")
    chunk_indices = prepare_chunk_indices(cu, 64)

    ref = tri_cumsum(
        g, chunk_size=64, reverse=True, cu_seqlens=cu,
        chunk_indices=chunk_indices, output_dtype=torch.float32,
    )
    out = cuda_cumsum(g, chunk_size=64, reverse=True, cu_seqlens=cu, chunk_indices=chunk_indices)
    assert_close("cumsum_varlen", ref, out)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 1e-4, 1e-5),
    (torch.bfloat16, 2e-2, 1e-2),
])
def test_gate_bwd(lower_bound, has_bias, dtype, rtol, atol):
    torch.manual_seed(42)
    B, T, H, K = 2, 300, 4, 128
    g = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
    A_log = torch.randn(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(H * K, dtype=torch.float32, device="cuda") if has_bias else None
    dyg = torch.randn(B, T, H, K, dtype=torch.float32, device="cuda")

    ref_dg, ref_dA, ref_dbias = tri_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=lower_bound)
    dg, dA, dbias = cuda_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg, lower_bound=lower_bound)

    assert_close("dg", ref_dg, dg, rtol=rtol, atol=atol)
    assert_close("dA", ref_dA, dA, rtol=rtol, atol=atol)
    if has_bias:
        assert_close("dbias", ref_dbias, dbias, rtol=rtol, atol=atol)
