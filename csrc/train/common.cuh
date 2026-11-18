#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

// Numeric helpers matching fla's Triton semantics (fla/ops/utils/softplus.py,
// fla/ops/utils/op.py). All gate math is done in fp32.

__device__ __forceinline__ float to_f32(cutlass::bfloat16_t x) {
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

__device__ __forceinline__ float to_f32(cutlass::half_t x) {
    return float(x);
}

__device__ __forceinline__ float to_f32(float x) {
    return x;
}

// softplus with threshold 20, identical PTX to fla's softplus_nv
__device__ __forceinline__ float softplus_f32(float x) {
    float out;
    asm(
        "{\n"
        ".reg .pred p;\n"
        "setp.gt.f32  p, %1, 20.;\n"
        "@p  mov.f32  %0, %1;\n"
        "@!p mul.f32            %0, %1, 1.4426950408889634;\n"
        "@!p ex2.approx.ftz.f32 %0, %0;\n"
        "@!p add.f32            %0, %0, 1.0;\n"
        "@!p lg2.approx.ftz.f32 %0, %0;\n"
        "@!p mul.f32            %0, %0, 0.6931471805599453;\n"
        "}\n"
        : "=f"(out)
        : "f"(x));
    return out;
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}
