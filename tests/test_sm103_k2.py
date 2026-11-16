"""Opt-in SM103 parity checks; build V1a and V1aE before enabling these tests."""

import os

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_KDA_TEST_SM103") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3),
    reason="requires FLASH_KDA_TEST_SM103=1 and an SM103 GPU",
)


@pytest.mark.parametrize(
    "B,T,H", [(1, 16, 64), (1, 17, 64), (4, 2048, 64), (8, 1024, 64), (1, 2048, 32)]
)
@torch.inference_mode()
def test_sm103_k2_exact(B, T, H, monkeypatch):
    import flash_kda

    torch.manual_seed(42)
    D = 128
    shape = (B, T, H, D)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn_like(v)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    initial = torch.randn((B, H, D, D), dtype=torch.bfloat16, device="cuda")
    initial_copy = initial.clone()

    for mode in ("baseline", "v1a", "v1ae", "auto"):
        monkeypatch.setenv("FLASH_KDA_K2_IMPL", mode)
        out = torch.full_like(q, float("nan"))
        final = torch.full_like(initial, float("nan"))
        flash_kda.fwd(q, k, v, g, beta, D ** -0.5, out,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                      initial_state=initial, final_state=final)
        torch.cuda.synchronize()
        assert torch.equal(initial, initial_copy), f"{mode}: initial_state changed"
        if mode == "baseline":
            assert torch.isfinite(out).all() and torch.isfinite(final).all()
            baseline_out, baseline_final = out, final
        else:
            assert torch.equal(out, baseline_out), f"{mode}: output mismatch"
            assert torch.equal(final, baseline_final), f"{mode}: final_state mismatch"
