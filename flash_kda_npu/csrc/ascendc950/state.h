#ifndef FLASH_KDA_950_STATE_H
#define FLASH_KDA_950_STATE_H

// Persistent state updates and optional residual packing.
#include "kernel_operator.h"
#include "reg_compute/kernel_reg_compute_intf.h"

namespace kda950 {
#if defined(KDA_950_RESIDUAL_VF)
// Read contiguous value rows, subtract and round once, then scatter directly
// into transposed NZ. This avoids the gather prototype's strided input reads.
template<int Rows>
__simd_vf__ inline void ResidualNzVF(__ubuf__ float* u, __ubuf__ float* d1,
                                   __ubuf__ bfloat16_t* nz) {
    using namespace AscendC;
    using namespace AscendC::Reg;
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::SAT,
        MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    uint32_t lanes32 = 64, lanes16 = 64;
    auto mask = UpdateMask<float>(lanes32);
    auto mask16 = UpdateMask<bfloat16_t>(lanes16);
    RegTensor<int16_t> index;
    RegTensor<float> a, b, difference;
    RegTensor<bfloat16_t> rounded, dense, unused;
    Arange(index, int16_t(0)); Muls(index, index, int16_t(16), mask16);
    for (uint16_t row = 0; row < Rows; ++row) {
        LoadAlign(a, u + row * 64); LoadAlign(b, d1 + row * 64);
        Sub(difference, a, b, mask);
        Cast<bfloat16_t, float, cast>(rounded, difference, mask);
        DeInterleave(dense, unused, rounded, rounded);
        Scatter(nz + (row / 16) * 64 * 16 + row % 16, dense,
                (RegTensor<uint16_t>&)index, mask16);
    }
}
#endif

// Separate intrinsic Mul and Add preserve the existing two FP32 operations.
// Decay is reused in registers across all value rows. No BF16 rounding moves.
__simd_vf__ inline void UpdateStateVF(__ubuf__ float* state, __ubuf__ float* decay,
                                     __ubuf__ float* delta_low, __ubuf__ float* delta_high) {
    using namespace AscendC::Reg;
    uint32_t lanes = 64;
    auto mask = UpdateMask<float>(lanes);
    RegTensor<float> value, factor, delta;
    for (uint16_t part = 0; part < 2; ++part) {
        LoadAlign(factor, decay + part * 64);
        for (uint16_t row = 0; row < 64; ++row) {
            auto ptr = state + row * 128 + part * 64;
            auto update = row < 32 ? delta_low + row * 128 + part * 64
                                   : delta_high + (row - 32) * 128 + part * 64;
            LoadAlign(value, ptr);
            LoadAlign(delta, update);
            Mul(value, value, factor, mask);
            Add(value, value, delta, mask);
            StoreAlign(ptr, value, mask);
        }
    }
}

// Write the FP32 update and its rounded NZ publication in one register pass.
// RINT places BF16 in even halfwords; unzip packs 64 values before VSSTB.
// NZ column-block stride: 64 value rows * 16 BF16 = 64 blocks of 32 bytes.
template<bool Update>
__simd_vf__ inline void UpdateStatePackVF(
    __ubuf__ float* state, __ubuf__ float* decay,
    __ubuf__ float* delta_low, __ubuf__ float* delta_high,
    __ubuf__ bfloat16_t* packed) {
    using namespace AscendC;
    using namespace AscendC::Reg;
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::SAT,
        MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    uint32_t lanes32 = 64, lanes16 = 64;
    auto mask32 = UpdateMask<float>(lanes32);
    auto mask16 = UpdateMask<bfloat16_t>(lanes16);
    RegTensor<float> value, factor, delta;
    RegTensor<bfloat16_t> rounded, dense, unused;
    for (uint16_t part = 0; part < 2; ++part) {
        if constexpr (Update) LoadAlign(factor, decay + part * 64);
        for (uint16_t row = 0; row < 64; ++row) {
            auto ptr = state + row * 128 + part * 64;
            LoadAlign(value, ptr);
            if constexpr (Update) {
                auto update = row < 32 ? delta_low + row * 128 + part * 64
                                       : delta_high + (row - 32) * 128 + part * 64;
                LoadAlign(delta, update);
                Mul(value, value, factor, mask32);
                Add(value, value, delta, mask32);
                StoreAlign(ptr, value, mask32);
            }
            Cast<bfloat16_t, float, cast>(rounded, value, mask32);
            DeInterleave(dense, unused, rounded, rounded);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY>(
                packed + part * 4096 + row * 16, dense, 64, mask16);
        }
    }
}
}

#endif
