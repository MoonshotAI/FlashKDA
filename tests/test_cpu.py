import torch
import pytest

import flash_kda


def _make_inputs(batch=2, sequence_length=4, query_heads=2, value_heads=2, key_dim=3, value_dim=4):
    torch.manual_seed(42)
    return (
        torch.randn(batch, sequence_length, query_heads, key_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(batch, sequence_length, query_heads, key_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(batch, sequence_length, value_heads, value_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(batch, sequence_length, value_heads, key_dim, dtype=torch.float64, requires_grad=True),
        torch.randn(batch, sequence_length, value_heads, dtype=torch.float64, requires_grad=True),
        torch.randn(value_heads, dtype=torch.float64, requires_grad=True),
        torch.randn(value_heads, key_dim, dtype=torch.float64, requires_grad=True),
    )


def _run(inputs, initial_state=None, **kwargs):
    q, k, v, g, beta, A_log, dt_bias = inputs
    return flash_kda.torch_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        A_log=A_log,
        dt_bias=dt_bias,
        **kwargs,
    )


def test_torch_kda_cpu_forward_and_backward():
    inputs = _make_inputs()
    initial_state = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)

    output, final_state = _run(inputs, initial_state=initial_state)
    loss = output.square().mean() + final_state.square().mean()
    loss.backward()

    assert output.shape == (2, 4, 2, 4)
    assert final_state.shape == (2, 2, 3, 4)
    for tensor in (*inputs, initial_state):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    "use_cpp_backend",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not flash_kda.has_cpu_extension(),
                reason="compiled CPU extension is unavailable",
            ),
        ),
    ],
)
def test_torch_kda_cpu_gradcheck(use_cpp_backend):
    inputs = _make_inputs(batch=1, sequence_length=2, query_heads=1, value_heads=1, key_dim=2, value_dim=2)
    initial_state = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)

    def function(*arguments):
        return _run(arguments[:-1], initial_state=arguments[-1], use_cpp_backend=use_cpp_backend)

    assert torch.autograd.gradcheck(function, (*inputs, initial_state), fast_mode=True)


@pytest.mark.skipif(not flash_kda.has_cpu_extension(), reason="compiled CPU extension is unavailable")
def test_cpp_backend_matches_pytorch_forward_and_gradients():
    inputs = _make_inputs(batch=1, sequence_length=3, query_heads=1, value_heads=2, key_dim=2, value_dim=3)
    initial_state = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)

    def evaluate(use_cpp_backend):
        cloned_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)
        cloned_state = initial_state.detach().clone().requires_grad_(True)
        output, final_state = _run(
            cloned_inputs,
            initial_state=cloned_state,
            use_cpp_backend=use_cpp_backend,
        )
        gradients = torch.autograd.grad(
            output.square().sum() + final_state.square().sum(),
            (*cloned_inputs, cloned_state),
        )
        return output, final_state, gradients

    cpp_output, cpp_state, cpp_gradients = evaluate(True)
    torch_output, torch_state, torch_gradients = evaluate(False)

    torch.testing.assert_close(cpp_output, torch_output)
    torch.testing.assert_close(cpp_state, torch_state)
    for cpp_gradient, torch_gradient in zip(cpp_gradients, torch_gradients):
        torch.testing.assert_close(cpp_gradient, torch_gradient)


def test_torch_kda_cpu_supports_fla_gate_and_grouped_value_heads():
    inputs = _make_inputs(batch=1, query_heads=1, value_heads=2, key_dim=3, value_dim=2)
    q, k, v, g, beta, A_log, dt_bias = inputs

    output, final_state = flash_kda.torch_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    (output.sum() + final_state.sum()).backward()

    assert output.shape == (1, 4, 2, 2)
    assert final_state.shape == (1, 2, 3, 2)
    assert A_log.grad is not None
    assert dt_bias.grad is not None


def test_torch_kda_cpu_varlen_matches_individual_sequences():
    inputs = _make_inputs(batch=1, sequence_length=5, query_heads=1, value_heads=1, key_dim=2, value_dim=3)
    initial_state = torch.randn(2, 1, 3, 2, dtype=torch.float64)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.long)

    output, final_state = _run(
        inputs,
        initial_state=initial_state,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )

    expected_outputs = []
    expected_states = []
    for sequence_index, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        sliced_inputs = tuple(
            tensor[:, start:end] if tensor.ndim >= 3 and tensor.shape[0] == 1 else tensor
            for tensor in inputs
        )
        sequence_output, sequence_state = _run(
            sliced_inputs,
            initial_state=initial_state[sequence_index:sequence_index + 1],
            state_v_first=True,
        )
        expected_outputs.append(sequence_output)
        expected_states.append(sequence_state)

    assert torch.allclose(output, torch.cat(expected_outputs, dim=1))
    assert torch.allclose(final_state, torch.cat(expected_states, dim=0))


def test_legacy_fwd_uses_cpu_backend():
    inputs = _make_inputs(batch=1, sequence_length=3, query_heads=1, value_heads=1, key_dim=2, value_dim=2)
    q, k, v, g, beta, A_log, dt_bias = inputs
    initial_state = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        0.5,
        output,
        A_log,
        dt_bias,
        -5.0,
        initial_state=initial_state,
        final_state=final_state,
    )
    expected_output, expected_state = _run(
        inputs,
        initial_state=initial_state,
        state_v_first=True,
    )

    assert torch.allclose(output, expected_output)
    assert torch.allclose(final_state, expected_state)
