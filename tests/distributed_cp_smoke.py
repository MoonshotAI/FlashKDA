"""Two-GPU NCCL and end-to-end FlashKDA smoke tests for CP transitions."""

import importlib.util
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

import flash_kda
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_ref import LOG2E, fp32_ex2_ftz, l2_normalize_kernel_match, sigmoid_ext


_CP_PATH = _REPO / "flash_kda" / "cp.py"
_SPEC = importlib.util.spec_from_file_location("flash_kda_cp_distributed", _CP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cp
_SPEC.loader.exec_module(cp)


def make_segment(seed: int, tokens: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    heads, key_dim, value_dim = 2, 128, 128
    key = F.normalize(
        torch.randn((tokens, heads, key_dim), generator=generator), p=2, dim=-1
    ).to(device=device, dtype=torch.bfloat16)
    value = torch.randn(
        (tokens, heads, value_dim), generator=generator
    ).to(device=device, dtype=torch.bfloat16)
    log_decay = (
        -0.1 * torch.rand((tokens, heads, key_dim), generator=generator)
    ).to(device)
    beta = (
        0.1 + 0.8 * torch.rand((tokens, heads), generator=generator)
    ).to(device)
    return key, value, log_decay, beta


def make_raw_inputs(seed: int, tokens: int, heads: int, dim: int, device):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, tokens, heads, dim)
    tensors = (
        torch.randn(shape, generator=generator).to(torch.bfloat16),
        torch.randn(shape, generator=generator).to(torch.bfloat16),
        torch.randn(shape, generator=generator).to(torch.bfloat16),
        torch.randn(shape, generator=generator).to(torch.bfloat16),
        torch.randn((1, tokens, heads), generator=generator).to(torch.bfloat16),
        torch.rand((heads,), generator=generator),
        torch.randn((heads, dim), generator=generator),
    )
    return tuple(tensor.to(device) for tensor in tensors)


def slice_segment(inputs, start: int, end: int):
    q, k, v, g, beta, a_log, dt_bias = inputs
    return (
        q[:, start:end].contiguous(),
        k[:, start:end].contiguous(),
        v[:, start:end].contiguous(),
        g[:, start:end].contiguous(),
        beta[:, start:end].contiguous(),
        a_log,
        dt_bias,
    )


def transition_from_raw_inputs(inputs, lower_bound: float):
    _, key, value, gate, beta, a_log, dt_bias = inputs
    tokens, heads, dim = key.shape[1:]
    key = l2_normalize_kernel_match(key.reshape(tokens, heads, dim))
    gate = gate.reshape(tokens, heads, dim).float() + dt_bias.unsqueeze(0)
    a_log_exp = fp32_ex2_ftz(a_log * LOG2E).view(1, heads, 1)
    log_decay = lower_bound * LOG2E * sigmoid_ext.sigmoid_tanh_fp32(
        a_log_exp * gate
    )
    beta = sigmoid_ext.sigmoid_tanh_fp32(beta.reshape(tokens, heads).float())
    return cp.kda_segment_transition(
        key, value.reshape(tokens, heads, dim), log_decay, beta
    )


def run_flash_kda(inputs, initial_state, lower_bound: float, snapshots=None):
    q, k, v, g, beta, a_log, dt_bias = inputs
    output = torch.empty_like(q)
    final_state = torch.empty_like(initial_state)
    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(q.shape[-1]),
        output,
        a_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        intermediate_state=snapshots,
    )
    return output, final_state


def check_flash_kda_end_to_end(rank: int, device: torch.device) -> None:
    """Use the distributed prefix as the real FlashKDA initial state."""
    heads, dim, tokens_per_rank = 2, 128, 128
    lower_bound = -5.0
    global_inputs = make_raw_inputs(
        20260821, 2 * tokens_per_rank, heads, dim, device
    )
    start = rank * tokens_per_rank
    end = start + tokens_per_rank
    local_inputs = slice_segment(global_inputs, start, end)

    local_transition = transition_from_raw_inputs(local_inputs, lower_bound)
    prefix = cp.distributed_exclusive_prefix_transition(local_transition)
    zero_state = torch.zeros((heads, dim, dim), device=device, dtype=torch.float32)
    cp_start = cp.apply_transition(zero_state, prefix)
    cp_output, _ = run_flash_kda(
        local_inputs, cp_start.unsqueeze(0), lower_bound
    )

    snapshots = torch.empty(
        flash_kda.get_intermediate_state_shape(global_inputs[0]),
        device=device,
        dtype=torch.bfloat16,
    )
    serial_output, _ = run_flash_kda(
        global_inputs, zero_state.unsqueeze(0), lower_bound, snapshots
    )
    torch.cuda.synchronize(device)
    expected_output = serial_output[:, start:end]
    expected_start = (
        zero_state
        if rank == 0
        else snapshots[:, tokens_per_rank // cp.KDA_CHUNK_SIZE - 1].float()
    )

    if rank == 0:
        torch.testing.assert_close(cp_start, expected_start, rtol=0, atol=0)
        torch.testing.assert_close(cp_output, expected_output, rtol=0, atol=0)
    else:
        # The affine scan stays in FP32, while the legacy serial kernel rounds
        # its recurrent state to BF16 after each chunk.
        torch.testing.assert_close(
            cp_start, expected_start, rtol=5e-2, atol=3e-2
        )
        torch.testing.assert_close(
            cp_output, expected_output, rtol=5e-3, atol=1e-3
        )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("this smoke test expects exactly two ranks")

    # Build the independent oracle before the first NCCL operation. CUDA work
    # after rank-divergent P2P can otherwise be ordered behind a collective on
    # the peer and turn a test-only synchronization pattern into a deadlock.
    if rank == 0:
        expected = cp.identity_transition(
            2, 128, 128, device=device, dtype=torch.float32
        )
    else:
        expected = cp.kda_segment_transition(*make_segment(20260820, 32, device))
    local = cp.kda_segment_transition(*make_segment(20260820 + rank, 32, device))
    prefix = cp.distributed_exclusive_prefix_transition(local)

    torch.testing.assert_close(prefix.matrix, expected.matrix, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(prefix.bias, expected.bias, rtol=2e-5, atol=2e-5)

    initial_state = torch.zeros((2, 128, 128), device=device, dtype=torch.bfloat16)
    actual_start = cp.apply_transition(initial_state, prefix)
    expected_start = cp.apply_transition(initial_state, expected)
    torch.testing.assert_close(actual_start, expected_start, rtol=2e-5, atol=2e-5)
    check_flash_kda_end_to_end(rank, device)
    dist.barrier()
    if rank == 0:
        print("distributed CP + FlashKDA end-to-end smoke test passed on 2 GPUs")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
