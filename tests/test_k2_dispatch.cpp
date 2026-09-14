#include "k2_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using flash_kda::K2DispatchConfig;
using flash_kda::K2Implementation;
using flash_kda::K2Mode;
using flash_kda::parse_k2_mode;
using flash_kda::select_k2_implementation;
using Impl = K2Implementation;

static int checks = 0;

static void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}

// Keep checks active even when the compiler defines NDEBUG.
#define CHECK(expression) check((expression), #expression, __LINE__)

static K2DispatchConfig supported(int64_t batch, int64_t chunks, int64_t heads = 64) {
    K2DispatchConfig config;
    config.v1a_compiled = config.v1ae_compiled = true;
    config.compute_major = 10;
    config.compute_minor = 3;
    config.has_state_in = config.has_state_out = true;
    config.sequences = batch;
    config.total_tokens = batch * chunks * 16;
    config.heads = heads;
    return config;
}

int main() {
    CHECK(parse_k2_mode("baseline") == K2Mode::Baseline);
    CHECK(parse_k2_mode("v1a") == K2Mode::V1A);
    CHECK(parse_k2_mode("v1ae") == K2Mode::V1AE);
    CHECK(parse_k2_mode("auto") == K2Mode::Auto);
    for (auto value : {"unknown", "", "V1A"}) {
        CHECK(parse_k2_mode(value) == K2Mode::Invalid);
        CHECK(select_k2_implementation(parse_k2_mode(value), supported(1, 160)) == Impl::Unsupported);
    }

    struct Boundary { int batch, chunks; Impl expected; };
    const Boundary boundaries[] = {
        {1, 79, Impl::Baseline}, {1, 80, Impl::V1AE}, {1, 159, Impl::V1AE}, {1, 160, Impl::V1A},
        {2, 31, Impl::Baseline}, {2, 32, Impl::V1AE}, {2, 95, Impl::V1AE}, {2, 96, Impl::V1A},
        {3, 23, Impl::Baseline}, {3, 24, Impl::V1AE}, {3, 319, Impl::V1AE}, {3, 320, Impl::V1A},
        {4, 31, Impl::Baseline}, {4, 32, Impl::V1AE}, {4, 255, Impl::V1AE}, {4, 256, Impl::V1A},
        {5, 23, Impl::Baseline}, {5, 24, Impl::V1AE}, {5, 320, Impl::V1AE}, {5, 321, Impl::V1A},
        {6, 23, Impl::Baseline}, {6, 24, Impl::V1AE}, {6, 320, Impl::V1AE}, {6, 321, Impl::V1A},
        {7, 23, Impl::Baseline}, {7, 24, Impl::V1AE}, {7, 320, Impl::V1AE}, {7, 321, Impl::V1A},
        {8, 23, Impl::Baseline}, {8, 24, Impl::V1AE}, {8, 319, Impl::V1AE}, {8, 320, Impl::Baseline},
        {9, 24, Impl::Baseline}, {9, 512, Impl::Baseline}, {16, 512, Impl::Baseline},
    };
    for (const auto& test : boundaries) {
        auto config = supported(test.batch, test.chunks);
        CHECK(flash_kda::chunks_per_sequence(config) == test.chunks);
        CHECK(select_k2_implementation(K2Mode::Auto, config) == test.expected);
        // Explicit requests ignore the empirical AUTO thresholds.
        CHECK(select_k2_implementation(K2Mode::V1A, config) == Impl::V1A);
        CHECK(select_k2_implementation(K2Mode::V1AE, config) == Impl::V1AE);
        CHECK(select_k2_implementation(K2Mode::Baseline, config) == Impl::Baseline);
        for (bool v1a : {false, true}) for (bool v1ae : {false, true}) {
            config.v1a_compiled = v1a;
            config.v1ae_compiled = v1ae;
            auto expected = test.expected;
            if ((expected == Impl::V1A && !v1a) || (expected == Impl::V1AE && !v1ae))
                expected = Impl::Baseline;
            CHECK(select_k2_implementation(K2Mode::Auto, config) == expected);
            CHECK(select_k2_implementation(K2Mode::V1A, config) ==
                  (v1a ? Impl::V1A : Impl::Unsupported));
            CHECK(select_k2_implementation(K2Mode::V1AE, config) ==
                  (v1ae ? Impl::V1AE : Impl::Unsupported));
        }
    }

    for (int heads : {1, 32, 65, 128}) for (int chunks : {127, 128}) {
        auto config = supported(4, chunks, heads);
        CHECK(select_k2_implementation(K2Mode::Auto, config) ==
              (chunks == 128 ? Impl::V1A : Impl::Baseline));
        config.v1a_compiled = false;
        CHECK(select_k2_implementation(K2Mode::Auto, config) == Impl::Baseline);
    }

    // Exercise unsupported runtime configurations at both kinds of AUTO winner.
    for (int chunks : {128, 256}) for (int condition = 0; condition < 10; ++condition) {
        auto config = supported(4, chunks);
        if (condition == 0) config.is_varlen = true;
        if (condition == 1) config.state_fp32 = true;
        if (condition == 2) config.has_state_in = false;
        if (condition == 3) config.has_state_out = false;
        if (condition == 4) config.compute_minor = 0; // SM100
        if (condition == 5) { config.compute_major = 12; config.compute_minor = 0; }
        if (condition == 6) { config.compute_major = 9; config.compute_minor = 0; }
        if (condition == 7) { config.compute_major = 8; config.compute_minor = 9; }
        if (condition == 8) config.sequences = 0;
        if (condition == 9) config.sequences = -1;
        CHECK(select_k2_implementation(K2Mode::V1A, config) == Impl::Unsupported);
        CHECK(select_k2_implementation(K2Mode::V1AE, config) == Impl::Unsupported);
        CHECK(select_k2_implementation(K2Mode::Auto, config) == Impl::Baseline);
        CHECK(select_k2_implementation(K2Mode::Baseline, config) == Impl::Baseline);
    }
    std::printf("K2 dispatch tests passed (%d checks)\n", checks);
}
