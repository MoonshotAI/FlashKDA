#pragma once

#include <cstdint>
#include <string_view>

namespace flash_kda {

constexpr int64_t kK2ChunkSize = 16;
constexpr int64_t kV1AAutoMinChunks = 128;

enum class K2Mode { Baseline, V1A, V1AE, Auto, Invalid };
enum class K2Implementation { Baseline, V1A, V1AE, Unsupported };

struct K2DispatchConfig {
    bool v1a_compiled = false;
    bool v1ae_compiled = false;
    int compute_major = 0;
    int compute_minor = 0;
    bool is_varlen = false;
    bool has_state_in = false;
    bool has_state_out = false;
    bool state_fp32 = false;
    int64_t total_tokens = 0;
    int64_t sequences = 0;
    int64_t heads = 0;
};

constexpr K2Mode parse_k2_mode(std::string_view value) {
    return value == "baseline" ? K2Mode::Baseline
         : value == "v1a" ? K2Mode::V1A
         : value == "v1ae" ? K2Mode::V1AE
         : value == "auto" ? K2Mode::Auto
         : K2Mode::Invalid;
}

constexpr bool v1a_supported(K2DispatchConfig const& config) {
    return config.v1a_compiled && config.compute_major == 10 &&
           config.compute_minor == 3 && !config.is_varlen &&
           config.has_state_in && config.has_state_out &&
           !config.state_fp32 && config.sequences > 0;
}

constexpr int64_t chunks_per_sequence(K2DispatchConfig const& config) {
    if (config.is_varlen || config.sequences <= 0) return 0;
    const int64_t sequence_length = config.total_tokens / config.sequences;
    return (sequence_length + kK2ChunkSize - 1) / kK2ChunkSize;
}

constexpr bool v1ae_supported(K2DispatchConfig const& config) {
    return config.v1ae_compiled && config.compute_major == 10 &&
           config.compute_minor == 3 && !config.is_varlen &&
           config.has_state_in && config.has_state_out &&
           !config.state_fp32 && config.sequences > 0;
}

// Empirical B300 H=64 policy, not a universal grid-size rule.
constexpr K2Implementation h64_auto_candidate(int64_t batch, int64_t chunks) {
    using Impl = K2Implementation;
    switch (batch) {
    case 1:
        return chunks < 80 ? Impl::Baseline : chunks < 160 ? Impl::V1AE : Impl::V1A;
    case 2:
        return chunks < 32 ? Impl::Baseline : chunks < 96 ? Impl::V1AE : Impl::V1A;
    case 3:
        // Below 24 is a conservative, unmeasured lower fallback.
        return chunks < 24 ? Impl::Baseline : chunks < 320 ? Impl::V1AE : Impl::V1A;
    case 4:
        return chunks < 32 ? Impl::Baseline : chunks < 256 ? Impl::V1AE : Impl::V1A;
    case 5:
    case 6:
    case 7:
        // V1aE wins through 320 inclusive. Neither <24 nor >320 was mapped;
        // >320 -> V1a is a conservative fallback, NOT a measured crossover.
        return chunks < 24 ? Impl::Baseline : chunks <= 320 ? Impl::V1AE : Impl::V1A;
    case 8:
        return chunks < 24 || chunks >= 320 ? Impl::Baseline : Impl::V1AE;
    default:
        // B>8 is unvalidated; do not extrapolate B8 behavior.
        return Impl::Baseline;
    }
}

constexpr K2Implementation select_auto_k2(K2DispatchConfig const& config) {
    const auto chunks = chunks_per_sequence(config);
    if (config.heads != 64) {
        return v1a_supported(config) && chunks >= kV1AAutoMinChunks
            ? K2Implementation::V1A : K2Implementation::Baseline;
    }
    const auto candidate = h64_auto_candidate(config.sequences, chunks);
    if (candidate == K2Implementation::V1AE && v1ae_supported(config)) return candidate;
    if (candidate == K2Implementation::V1A && v1a_supported(config)) return candidate;
    // Unsupported/uncompiled winners fall back to baseline.
    return K2Implementation::Baseline;
}

constexpr K2Implementation select_k2_implementation(
    K2Mode mode, K2DispatchConfig const& config) {
    switch (mode) {
    case K2Mode::Baseline:
        return K2Implementation::Baseline;
    case K2Mode::V1A:
        return v1a_supported(config)
            ? K2Implementation::V1A : K2Implementation::Unsupported;
    case K2Mode::V1AE:
        return v1ae_supported(config)
            ? K2Implementation::V1AE : K2Implementation::Unsupported;
    case K2Mode::Auto:
        return select_auto_k2(config);
    default:
        return K2Implementation::Unsupported;
    }
}

}  // namespace flash_kda
