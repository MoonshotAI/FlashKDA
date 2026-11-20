#ifndef FLASH_KDA_950_PRE_GRAM_H
#define FLASH_KDA_950_PRE_GRAM_H

// Pre-Gram arithmetic, gate prefixes and optional direct communication.
#include "kernel_operator.h"
#if defined(KDA_950_ROW_VF) || defined(KDA_950_PREFIX_VF)
#include "reg_compute/kernel_reg_compute_intf.h"
#endif

namespace kda950 {
// Conservative GM baseline: native 2Dv2 loads, single live transaction.
// Recenter and BF16 rounding remain in the existing vector producer.
template<int Chunk>
__aicore__ inline void PreGram(GM_ADDR pa, GM_ADDR pk, GM_ADDR pb, GM_ADDR px,
                              GM_ADDR po, GM_ADDR pl, int count, int unroll) {
    constexpr int Depth=128, Band=Chunk>32?32:Chunk;
    static_assert(Chunk==16 || Chunk==32 || Chunk==64, "950 baseline supports M16/32/64");
    static_assert(2*Chunk*Depth*2<=64*1024, "L0A capacity exceeded");
    static_assert(Chunk*Depth*2<=64*1024, "L0B capacity exceeded");
    static_assert(2*Chunk*Chunk*4<=256*1024, "L0C capacity exceeded");
    TPipe pipe;
    TBuf<TPosition::A1> bufA,bufK;
    TBuf<TPosition::B1> bufB;
    pipe.InitBuffer(bufA,Chunk*Depth*2);
    pipe.InitBuffer(bufK,Chunk*Depth*2);
    pipe.InitBuffer(bufB,Chunk*Depth*2);
    auto a1=bufA.Get<bfloat16_t>(), k1=bufK.Get<bfloat16_t>(), b1=bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> a2(TPosition::A2,0,Chunk*Depth);
    LocalTensor<bfloat16_t> k2(TPosition::A2,Chunk*Depth*2,Chunk*Depth);
    LocalTensor<bfloat16_t> b2(TPosition::B2,0,Chunk*Depth);
    LocalTensor<float> co(TPosition::CO1,0,Chunk*Chunk);
    LocalTensor<float> cl(TPosition::CO1,Chunk*Chunk*4,Chunk*Chunk);
    GlobalTensor<bfloat16_t> a,k,b,x;
    GlobalTensor<float> o,l;
    a.SetGlobalBuffer((__gm__ bfloat16_t*)pa); k.SetGlobalBuffer((__gm__ bfloat16_t*)pk);
    b.SetGlobalBuffer((__gm__ bfloat16_t*)pb); x.SetGlobalBuffer((__gm__ bfloat16_t*)px);
    o.SetGlobalBuffer((__gm__ float*)po); l.SetGlobalBuffer((__gm__ float*)pl);
    for(int u=0;u<unroll;++u) {
        CrossCoreWaitFlag(8);
        for(int s=0;s<2;++s) {
            int c=GetBlockIdx()*2*unroll+2*u+s;
            if(c>=count)c=count-1;
            uint64_t off=uint64_t(c)*Chunk*Depth, out=uint64_t(c)*Chunk*Chunk;
            DataCopy(a1,a[off],Nd2NzParams(1,Chunk,Depth,0,Depth,Chunk,1,0));
            DataCopy(k1,k[off],Nd2NzParams(1,Chunk,Depth,0,Depth,Chunk,1,0));
            DataCopy(b1,b[off],Nd2NzParams(1,Chunk,Depth,0,Depth,Chunk,1,0));
            PipeBarrier<PIPE_ALL>();
            LoadNz<Chunk,Depth>(a2,a1); LoadNz<Chunk,Depth>(k2,k1); LoadNz<Chunk,Depth>(b2,b1);
            PipeBarrier<PIPE_ALL>();
            Mmad(co,a2,b2,MmadParams(Chunk,Chunk,Depth,0,false,true));
            Mmad(cl,k2,b2,MmadParams(Chunk,Chunk,Depth,0,false,true));
            PipeBarrier<PIPE_ALL>();
            StoreNd<Chunk,Chunk>(o[out],co,Chunk); StoreNd<Chunk,Chunk>(l[out],cl,Chunk);
            PipeBarrier<PIPE_ALL>();
            if constexpr(Chunk>32) {
                DataCopy(a1,a[off+Band*Depth],Nd2NzParams(1,Band,Depth,0,Depth,Band,1,0));
                DataCopy(k1,k[off+Band*Depth],Nd2NzParams(1,Band,Depth,0,Depth,Band,1,0));
                DataCopy(b1,x[uint64_t(c)*Band*Depth],Nd2NzParams(1,Band,Depth,0,Depth,Band,1,0));
                PipeBarrier<PIPE_ALL>();
                LoadNz<Band,Depth>(a2,a1); LoadNz<Band,Depth>(k2,k1); LoadNz<Band,Depth>(b2,b1);
                PipeBarrier<PIPE_ALL>();
                Mmad(co,a2,b2,MmadParams(Band,Band,Depth,0,false,true));
                Mmad(cl,k2,b2,MmadParams(Band,Band,Depth,0,false,true));
                PipeBarrier<PIPE_ALL>();
                StoreNd<Band,Band>(o[out+Band*Chunk],co,Chunk);
                StoreNd<Band,Band>(l[out+Band*Chunk],cl,Chunk);
                PipeBarrier<PIPE_ALL>();
            }
        }
        CrossCoreSetFlag<2,PIPE_FIX>(9);
    }
}

#if defined(KDA_950_ROW_VF)
// Match the high-level BF16 Cast: RINT, SAT, packed 32-bit register layout.
__simd_callee__ inline void StoreRounded(__ubuf__ bfloat16_t* dst,
                                        AscendC::Reg::RegTensor<float>& value,
                                        AscendC::Reg::MaskReg& mask) {
    using namespace AscendC;
    using namespace AscendC::Reg;
    static constexpr CastTrait trait = {RegLayout::ZERO, SatMode::SAT,
                                 MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> rounded;
    Cast<bfloat16_t, float, trait>(rounded, value, mask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dst, rounded, mask);
}

// Q/K have already crossed the normalization BF16 rounding boundary.
// Each Mul remains a separate FP32 operation; no FMA or reassociation.
template<int Rows>
__simd_vf__ inline void PublicRowsVF(
    __ubuf__ float* q, __ubuf__ float* k, __ubuf__ bfloat16_t* v,
    __ubuf__ float* gate, __ubuf__ float* last, __ubuf__ float* beta,
    __ubuf__ bfloat16_t* qg, __ubuf__ bfloat16_t* rk,
    __ubuf__ bfloat16_t* rv, __ubuf__ bfloat16_t* kg) {
    using namespace AscendC;
    using namespace AscendC::Reg;
    static constexpr CastTrait up = {RegLayout::ZERO, SatMode::SAT,
                              MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
    uint32_t lanes = 64;
    auto mask = UpdateMask<float>(lanes);
    RegTensor<float> qr, kr, vr, gr, lr, br, e, t;
    RegTensor<bfloat16_t> vb;
    for (uint16_t half = 0; half < 2; ++half) {
        LoadAlign(lr, last + half * 64);
        for (uint16_t row = 0; row < Rows; ++row) {
            uint32_t off = row * 128 + half * 64;
            LoadAlign(qr, q + off); LoadAlign(kr, k + off);
            LoadAlign(gr, gate + off);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(br, beta + row * 8);
            Muls(e, gr, 0.6931471805599453f, mask); Exp(e, e, mask);
            Mul(t, qr, e, mask); StoreRounded(qg + off, t, mask);
            Mul(t, kr, e, mask); Mul(t, t, br, mask); StoreRounded(rk + off, t, mask);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vb, v + off);
            Cast<float, bfloat16_t, up>(vr, vb, mask);
            Mul(t, vr, br, mask); StoreRounded(rv + off, t, mask);
            Sub(t, lr, gr, mask); Muls(t, t, 0.6931471805599453f, mask);
            Exp(t, t, mask); Mul(t, t, kr, mask); StoreRounded(kg + off, t, mask);
        }
    }
}

template<int Rows>
__simd_vf__ inline void GramRowsVF(
    __ubuf__ float* q, __ubuf__ float* k, __ubuf__ float* gate,
    __ubuf__ float* reference, __ubuf__ float* beta,
    __ubuf__ bfloat16_t* a, __ubuf__ bfloat16_t* kb,
    __ubuf__ bfloat16_t* b) {
    using namespace AscendC::Reg;
    uint32_t lanes = 64;
    auto mask = UpdateMask<float>(lanes);
    RegTensor<float> qr, kr, gr, ref, br, z, e, t;
    for (uint16_t half = 0; half < 2; ++half) {
        LoadAlign(ref, reference + half * 64);
        for (uint16_t row = 0; row < Rows; ++row) {
            uint32_t off = row * 128 + half * 64;
            LoadAlign(qr, q + off); LoadAlign(kr, k + off); LoadAlign(gr, gate + off);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(br, beta + row * 8);
            Sub(z, gr, ref, mask);
            Muls(e, z, 0.6931471805599453f, mask); Exp(e, e, mask);
            Mul(t, qr, e, mask); StoreRounded(a + off, t, mask);
            Mul(t, kr, e, mask); Mul(t, t, br, mask); StoreRounded(kb + off, t, mask);
            Muls(e, z, -0.6931471805599453f, mask); Exp(e, e, mask);
            Mul(t, kr, e, mask); StoreRounded(b + off, t, mask);
        }
    }
}
#endif

#if defined(KDA_950_PREFIX_VF)
// Match the existing radix-8 parenthesization exactly. Keep block totals
// in registers, so no VF invocation reads its own earlier UB stores.
template<int Rows>
__simd_vf__ inline void GatePrefixVF(__ubuf__ float* gate) {
    using namespace AscendC::Reg;
    uint32_t lanes = 64;
    auto mask = UpdateMask<float>(lanes);
    RegTensor<float> r0, r1, r2, r3, r4, r5, r6, r7, previous;
    for (uint16_t half = 0; half < 2; ++half) {
        for (uint16_t block = 0; block < Rows / 8; ++block) {
            auto p = gate + block * 8 * 128 + half * 64;
            LoadAlign(r0, p); LoadAlign(r1, p + 128);
            LoadAlign(r2, p + 256); LoadAlign(r3, p + 384);
            LoadAlign(r4, p + 512); LoadAlign(r5, p + 640);
            LoadAlign(r6, p + 768); LoadAlign(r7, p + 896);
            Add(r1, r1, r0, mask); Add(r2, r2, r1, mask);
            Add(r3, r3, r2, mask); Add(r4, r4, r3, mask);
            Add(r5, r5, r4, mask); Add(r6, r6, r5, mask);
            Add(r7, r7, r6, mask);
            if (block != 0) {
                Add(r0, r0, previous, mask); Add(r1, r1, previous, mask);
                Add(r2, r2, previous, mask); Add(r3, r3, previous, mask);
                Add(r4, r4, previous, mask); Add(r5, r5, previous, mask);
                Add(r6, r6, previous, mask); Add(r7, r7, previous, mask);
            }
            Copy(previous, r7, mask);
            StoreAlign(p, r0, mask); StoreAlign(p + 128, r1, mask);
            StoreAlign(p + 256, r2, mask); StoreAlign(p + 384, r3, mask);
            StoreAlign(p + 512, r4, mask); StoreAlign(p + 640, r5, mask);
            StoreAlign(p + 768, r6, mask); StoreAlign(p + 896, r7, mask);
        }
    }
}
#endif

#if defined(KDA_950_PRE_CV)
template<int Rows> struct PreGramCvLayout {
    static constexpr int Depth = 128, Band = Rows > 32 ? 32 : Rows;
    static constexpr int A = 0, K = Rows * Depth * 2, B = 2 * Rows * Depth * 2;
    static constexpr int X = 3 * Rows * Depth * 2;
    static constexpr int PerAiv = X + Band * Depth * 2;
    static constexpr int PerSlot = 2 * PerAiv;
    static constexpr int RxQ = 192 * 1024, RxL = 208 * 1024;
    static_assert(2 * PerSlot <= 512 * 1024, "two Pre-Gram L1 slots");
    static_assert(Rows * Rows * 4 <= RxL - RxQ, "Pre-Gram Q receiver");
    static_assert(RxL + Rows * Rows * 4 <= 224 * 1024, "Pre-Gram L receiver");
    __aicore__ static inline int Base(int slot, int subcore) {
        return slot * PerSlot + subcore * PerAiv;
    }
};

// The vector producer prepares u+1 while Cube consumes u. Two independent
// L1 slots are required; DONE for u is consumed before READY for u+1.
template<int Rows>
__aicore__ inline void PreGramCv(int unroll) {
    using namespace AscendC;
    using L = PreGramCvLayout<Rows>;
    constexpr int Depth = 128, Band = L::Band;
    LocalTensor<bfloat16_t> a2(TPosition::A2, 0, Rows * Depth);
    LocalTensor<bfloat16_t> k2(TPosition::A2, Rows * Depth * 2, Rows * Depth);
    LocalTensor<bfloat16_t> b2(TPosition::B2, 0, Rows * Depth);
    LocalTensor<float> co(TPosition::CO1, 0, Rows * Rows);
    LocalTensor<float> cl(TPosition::CO1, Rows * Rows * 4, Rows * Rows);
    LocalTensor<float> rq(TPosition::VECCALC, L::RxQ, Rows * Rows);
    LocalTensor<float> rl(TPosition::VECCALC, L::RxL, Rows * Rows);
    for (int u = 0; u < unroll; ++u) {
        CrossCoreWaitFlag(8);
        for (int iv = 0; iv < 2; ++iv) {
            const int base = L::Base(u & 1, iv);
            LocalTensor<bfloat16_t> a1(TPosition::A1, base + L::A, Rows * Depth);
            LocalTensor<bfloat16_t> k1(TPosition::A1, base + L::K, Rows * Depth);
            LocalTensor<bfloat16_t> b1(TPosition::B1, base + L::B, Rows * Depth);
            LocalTensor<bfloat16_t> x1(TPosition::B1, base + L::X, Band * Depth);
            PipeBarrier<PIPE_ALL>();
            LoadNz<Rows, Depth>(a2, a1); LoadNz<Rows, Depth>(k2, k1); LoadNz<Rows, Depth>(b2, b1);
            PipeBarrier<PIPE_ALL>();
            Mmad(co, a2, b2, MmadParams(Rows, Rows, Depth, 0, false, true));
            Mmad(cl, k2, b2, MmadParams(Rows, Rows, Depth, 0, false, true));
            PipeBarrier<PIPE_ALL>();
            L0cToUb<Rows, Rows>(rq, co, iv); L0cToUb<Rows, Rows>(rl, cl, iv);
            PipeBarrier<PIPE_ALL>();
            if constexpr (Rows > 32) {
                // Select lower rows from full NZ; preserve source row stride
                // while compacting them into the 32-row L0 matrices.
                LoadData2DParamsV2 slice{};
                slice.mStep = Band / 16; slice.kStep = Depth / 16;
                slice.srcStride = Rows / 16; slice.dstStride = Band / 16;
                LoadData(a2, a1[Band * 16], slice);
                LoadData(k2, k1[Band * 16], slice);
                LoadNz<Band, Depth>(b2, x1);
                PipeBarrier<PIPE_ALL>();
                Mmad(co, a2, b2, MmadParams(Band, Band, Depth, 0, false, true));
                Mmad(cl, k2, b2, MmadParams(Band, Band, Depth, 0, false, true));
                PipeBarrier<PIPE_ALL>();
                L0cToUb<Band, Band, Band, Rows>(rq[Band * Rows], co, iv);
                L0cToUb<Band, Band, Band, Rows>(rl[Band * Rows], cl, iv);
                PipeBarrier<PIPE_ALL>();
            }
        }
        CrossCoreSetFlag<2, PIPE_FIX>(9);
    }
}
#endif
}

#endif
