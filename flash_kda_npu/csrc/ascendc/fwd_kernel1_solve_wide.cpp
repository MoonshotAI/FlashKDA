// Batch independent diagonal solves along the vector repeat axis to amortize
// instruction overhead. Broadcasting each coefficient across lanes avoids
// materializing a full coefficient tile. Zero-padded L blocks solve to
// identity, allowing full launch groups without a separate tail kernel.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
#ifndef KDA_SOLVE_WIDE_SUBB
#define KDA_SOLVE_WIDE_SUBB 1
#endif
constexpr int32_t SB = KDA_SOLVE_WIDE_SUBB;
constexpr int32_t PC = KDA_CHUNK;
constexpr int32_t M = PC / SB;
constexpr int32_t MM = M * M;
#ifndef KDA_SOLVE_WIDE_NCHUNK
#define KDA_SOLVE_WIDE_NCHUNK 32
#endif
constexpr int32_t NC = KDA_SOLVE_WIDE_NCHUNK;
// Splitting the parent triangle reduces vector substitution work by roughly
// SB squared; Cube handles the coupling product between diagonal blocks.
constexpr int32_t NCH = NC / SB;
// Use [row, instance, lane] order to keep repeat strides within 8 bits.
// A whole 64-row tile in chunk-major order would need a 512-block stride,
// exceeding the hardware limit of 255.
constexpr int32_t RW = NC * M;
constexpr int32_t CH = NC * MM;
constexpr int32_t CM = 64 * 255;
constexpr int32_t CE = MM;
// Keep L21 chunk-contiguous so its negated export needs no repacking.

extern "C" __global__ __aicore__ void kda_solve_wu_wide(
    GM_ADDR pL, GM_ADDR pEye, GM_ADDR pA16,
    GM_ADDR pLneg, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c0 = GetBlockIdx() * NCH;
    if (c0 >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> bLraw, bCexp, bAf, bAb, bEye;
    pipe.InitBuffer(bLraw, CH * 4);
#if KDA_SOLVE_WIDE_SUBB > 1
    // Cube reads the full inverse, so the otherwise unwritten upper block
    // must be zero. Allocate a full source tile: a zero DMA gap means
    // contiguous rows, not broadcasting a single row.
    TBuf<TPosition::VECCALC> bZf, bZb;
    pipe.InitBuffer(bZf, MM * 4);
    pipe.InitBuffer(bZb, MM * 2);
#endif
    pipe.InitBuffer(bCexp, NC * M * 8 * 4);
    pipe.InitBuffer(bAf, CH * 4);
    pipe.InitBuffer(bAb, CH * 2);
    pipe.InitBuffer(bEye, MM * 4);
#if KDA_SOLVE_WIDE_SUBB > 1
    TBuf<TPosition::VECCALC> bL21f, bL21b;
    pipe.InitBuffer(bL21f, NCH * CE * 4);
    pipe.InitBuffer(bL21b, NCH * CE * 2);
#endif
    LocalTensor<float> lraw = bLraw.Get<float>(), cexp = bCexp.Get<float>();
    LocalTensor<float> af = bAf.Get<float>(), eye = bEye.Get<float>();
    LocalTensor<bfloat16_t> ab = bAb.Get<bfloat16_t>();
    GlobalTensor<float> L, Eye;
    GlobalTensor<bfloat16_t> A16;
#if KDA_SOLVE_WIDE_SUBB > 1
    LocalTensor<float> l21f = bL21f.Get<float>();
    LocalTensor<bfloat16_t> l21b = bL21b.Get<bfloat16_t>();
    LocalTensor<float> zf = bZf.Get<float>();
    LocalTensor<bfloat16_t> zrow = bZb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Lneg;
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
#endif
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pEye));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));

    DataCopy(eye, Eye[0], DataCopyParams(M, M / 8, 0, 0));
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t s = 0; s < SB; ++s) {
            DataCopy(lraw[i * RW + s * NCH * M],
                     L[static_cast<uint64_t>(c0) * PC * PC + (s * M + i) * PC + s * M],
                     DataCopyParams(NCH, M / 8, (PC * PC - M) / 8, 0));
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    // Negate L21 before export so Cube can absorb the sign into its operand.
    for (int32_t ch = 0; ch < NCH; ++ch) {
        DataCopy(l21f[ch * MM],
                 L[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC],
                 DataCopyParams(M, M / 8, (PC - M) / 8, 0));
    }
#endif
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    // Negate the gathered tile in batches to amortize instruction overhead;
    // split at CM to respect the count-mode limit.
    for (int32_t off = 0; off < CH; off += CM) {
        const int32_t n = (CH - off < CM) ? (CH - off) : CM;
        Muls(lraw[off], lraw[off], -1.0f, n);
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    Muls(l21f, l21f, -1.0f, NCH * CE);
    Cast(l21b, l21f, RoundMode::CAST_RINT, NCH * CE);
#endif
#if KDA_SOLVE_WIDE_SUBB > 1
    Duplicate(zf, 0.0f, MM);
    PipeBarrier<PIPE_V>();
    Cast(zrow, zf, RoundMode::CAST_RINT, MM);
#endif
    PipeBarrier<PIPE_V>();

    for (int32_t i = 0; i < M; ++i) {
        // Broadcast one identity row across instances to avoid duplicate source loads.
        Adds(af[i * RW], eye[i * M], 0.0f, M, NC, UnaryRepeatParams(1, 1, M / 8, 0));
        Brcb(cexp, lraw[i * RW], NC * M / 8, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < i; ++j) {
            MulAddDst(af[i * RW], af[j * RW], cexp[j * 8], M, NC,
                      BinaryRepeatParams(1, 1, 0, M / 8, M / 8, M));
        }
        PipeBarrier<PIPE_V>();
    }
    for (int32_t off = 0; off < CH; off += CM) {
        const int32_t n = (CH - off < CM) ? (CH - off) : CM;
        Cast(ab[off], af[off], RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
    }

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
#if KDA_SOLVE_WIDE_SUBB > 1
    // Zero both exports so the debug FP32 view agrees with the BF16 inverse
    // consumed by Cube.
    for (int32_t s = 0; s + 1 < SB; ++s) {
        for (int32_t s2 = s + 1; s2 < SB; ++s2) {
            for (int32_t ch = 0; ch < NCH; ++ch) {
                const uint64_t d = static_cast<uint64_t>(c0 + ch) * PC * PC
                                   + (s * M) * PC + s2 * M;
                DataCopy(A16[d], zrow,
                         DataCopyParams(M, M / 16, 0, (PC - M) / 16));
            }
        }
    }
#endif
    // Gather rows on export to avoid a separate transpose of the UB tile.
    for (int32_t s = 0; s < SB; ++s) {
        for (int32_t ch = 0; ch < NCH; ++ch) {
            const int32_t t = s * NCH + ch;
            const uint64_t d =
                static_cast<uint64_t>(c0 + ch) * PC * PC + (s * M) * PC + s * M;
            DataCopy(A16[d], ab[t * M],
                     DataCopyParams(M, M / 16, (NC - 1) * (M / 16), (PC - M) / 16));
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    for (int32_t ch = 0; ch < NCH; ++ch) {
        DataCopy(Lneg[static_cast<uint64_t>(c0 + ch) * CE], l21b[ch * CE],
                 DataCopyParams(1, CE / 16, 0, 0));
    }
#endif
    PipeBarrier<PIPE_ALL>();
}
