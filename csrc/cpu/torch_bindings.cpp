#include <torch/extension.h>

#include <vector>

namespace {

void check_recurrent_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state
) {
    TORCH_CHECK(q.device().is_cpu(), "q must be a CPU tensor");
    TORCH_CHECK(k.device().is_cpu() && v.device().is_cpu() && g.device().is_cpu(),
                "k, v, and g must be CPU tensors");
    TORCH_CHECK(beta.device().is_cpu() && initial_state.device().is_cpu(),
                "beta and initial_state must be CPU tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kFloat64,
                "CPU KDA supports float32 and float64 compute tensors");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                g.scalar_type() == q.scalar_type() && beta.scalar_type() == q.scalar_type() &&
                initial_state.scalar_type() == q.scalar_type(),
                "all CPU KDA tensors must use the same dtype");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4,
                "q, k, v, and g must be 4D tensors");
    TORCH_CHECK(beta.dim() == 3 && initial_state.dim() == 4,
                "beta must be 3D and initial_state must be 4D");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == g.sizes(),
                "q, k, and g must have the same shape");

    const auto batch = q.size(0);
    const auto sequence_length = q.size(1);
    const auto heads = q.size(2);
    const auto key_dim = q.size(3);
    const auto value_dim = v.size(3);
    TORCH_CHECK(v.size(0) == batch && v.size(1) == sequence_length && v.size(2) == heads,
                "v must match q's batch, sequence, and head dimensions");
    TORCH_CHECK(beta.size(0) == batch && beta.size(1) == sequence_length && beta.size(2) == heads,
                "beta must match q's batch, sequence, and head dimensions");
    TORCH_CHECK(initial_state.size(0) == batch && initial_state.size(1) == heads &&
                initial_state.size(2) == key_dim && initial_state.size(3) == value_dim,
                "initial_state must have shape [B, H, K, V]");
}

std::vector<torch::Tensor> recurrent_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state
) {
    check_recurrent_inputs(q, k, v, g, beta, initial_state);
    TORCH_CHECK(q.size(1) > 0, "the C++ CPU backend requires a non-empty sequence");

    auto state = initial_state;
    std::vector<torch::Tensor> outputs;
    outputs.reserve(q.size(1));
    for (int64_t token_index = 0; token_index < q.size(1); ++token_index) {
        const auto q_token = q.select(1, token_index);
        const auto k_token = k.select(1, token_index);
        const auto v_token = v.select(1, token_index);
        const auto gate_token = g.select(1, token_index);
        const auto beta_token = beta.select(1, token_index);

        state = state * gate_token.exp().unsqueeze(-1);
        const auto predicted_value = (k_token.unsqueeze(-1) * state).sum(-2);
        const auto value_delta = beta_token.unsqueeze(-1) * (v_token - predicted_value);
        state = state + k_token.unsqueeze(-1) * value_delta.unsqueeze(-2);
        outputs.push_back((q_token.unsqueeze(-1) * state).sum(-2));
    }

    return {torch::stack(outputs, 1), state};
}

std::vector<torch::Tensor> recurrent_backward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_final_state
) {
    check_recurrent_inputs(q, k, v, g, beta, initial_state);
    TORCH_CHECK(grad_output.sizes() == v.sizes(), "grad_output must match v's shape");
    TORCH_CHECK(grad_final_state.sizes() == initial_state.sizes(),
                "grad_final_state must match initial_state's shape");

    std::vector<torch::Tensor> states;
    states.reserve(q.size(1) + 1);
    auto state = initial_state;
    states.push_back(state);
    for (int64_t token_index = 0; token_index < q.size(1); ++token_index) {
        const auto k_token = k.select(1, token_index);
        const auto v_token = v.select(1, token_index);
        const auto gate_token = g.select(1, token_index);
        const auto beta_token = beta.select(1, token_index);

        const auto decayed_state = state * gate_token.exp().unsqueeze(-1);
        const auto predicted_value = (k_token.unsqueeze(-1) * decayed_state).sum(-2);
        const auto value_delta = beta_token.unsqueeze(-1) * (v_token - predicted_value);
        state = decayed_state + k_token.unsqueeze(-1) * value_delta.unsqueeze(-2);
        states.push_back(state);
    }

    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);
    auto dg = torch::zeros_like(g);
    auto dbeta = torch::zeros_like(beta);
    auto dstate = grad_final_state;

    for (int64_t token_index = q.size(1) - 1; token_index >= 0; --token_index) {
        const auto q_token = q.select(1, token_index);
        const auto k_token = k.select(1, token_index);
        const auto v_token = v.select(1, token_index);
        const auto gate_token = g.select(1, token_index);
        const auto beta_token = beta.select(1, token_index);
        const auto output_gradient = grad_output.select(1, token_index);
        const auto previous_state = states[token_index];
        const auto current_state = states[token_index + 1];
        const auto decay = gate_token.exp();
        const auto decayed_state = previous_state * decay.unsqueeze(-1);
        const auto predicted_value = (k_token.unsqueeze(-1) * decayed_state).sum(-2);
        const auto residual = v_token - predicted_value;
        const auto value_delta = beta_token.unsqueeze(-1) * residual;

        const auto dq_token = (current_state * output_gradient.unsqueeze(-2)).sum(-1);
        auto total_state_gradient = dstate + q_token.unsqueeze(-1) * output_gradient.unsqueeze(-2);
        const auto dk_from_update = (total_state_gradient * value_delta.unsqueeze(-2)).sum(-1);
        const auto dvalue_delta = (total_state_gradient * k_token.unsqueeze(-1)).sum(-2);
        const auto dbeta_token = (dvalue_delta * residual).sum(-1);
        const auto dresidual = dvalue_delta * beta_token.unsqueeze(-1);
        const auto dv_token = dresidual;
        const auto dpredicted_value = -dresidual;
        const auto dk_from_prediction =
            (decayed_state * dpredicted_value.unsqueeze(-2)).sum(-1);
        const auto ddecayed_state =
            total_state_gradient + k_token.unsqueeze(-1) * dpredicted_value.unsqueeze(-2);
        const auto dg_token = (ddecayed_state * previous_state).sum(-1) * decay;
        dstate = ddecayed_state * decay.unsqueeze(-1);

        dq.select(1, token_index).copy_(dq_token);
        dk.select(1, token_index).copy_(dk_from_update + dk_from_prediction);
        dv.select(1, token_index).copy_(dv_token);
        dg.select(1, token_index).copy_(dg_token);
        dbeta.select(1, token_index).copy_(dbeta_token);
    }

    return {dq, dk, dv, dg, dbeta, dstate};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("recurrent_forward", &recurrent_forward, "KDA recurrent forward (CPU)");
    module.def("recurrent_backward", &recurrent_backward, "KDA recurrent backward (CPU)");
}
