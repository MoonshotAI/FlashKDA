// Move dense Gram products to Cube to reduce vector work.
// Reuse Aqk32/L for the handoff to avoid extra GM
// buffers, and retain FP32 accumulation until masking and rounding.
// Adjacent event pairs protect reused staging without carrying event state
// between passes. Buffer placement is layout-sensitive on this CANN runtime;
// changes to declaration order or pipeline placement need hardware validation.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
constexpr int32_t M = KDA_CHUNK;
constexpr int32_t D = 128;
constexpr int32_t N = M * D;
// Process row-local work in 16-row passes to fit the 192 KiB UB as chunks
// grow. Keep only the gate chunk-wide because its prefix sum couples rows.
constexpr int32_t MT = M > 16 ? 16 : M;
constexpr int32_t NP = M / MT;
constexpr int32_t NG = MT * D;
constexpr int32_t KF = M / 16;
constexpr int32_t DF = D / 16;
// Recenter gates per band to keep exponentials within FP32/BF16 range.
// Cross-band Gram operands must share a reference so their factors cancel.
constexpr int32_t BS = M > 32 ? 32 : M;
constexpr int32_t MID0 = BS / 2;
constexpr bool XBAND = M > BS;
constexpr float RCP_LN2 = 1.4426950216f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float EPS = 1e-6f;
// Mode-2 waits consume both AIV publications together. Publish READY only
// after the previous DONE wait: flags are levels, so an early publication
// can satisfy the wrong step and leave a later wait blocked.
constexpr uint16_t FL_READY = 8;
constexpr uint16_t FL_DONE = 9;

// Halve rows before reduction to fit the 64-lane reduction while keeping
// independent rows in their own slots.
static __aicore__ inline void RowReduce(LocalTensor<float> rs, LocalTensor<float> tmp,
                                        const LocalTensor<float> tile) {
    Add(tmp, tile, tile[64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    WholeReduceSum(rs, tmp, 64, MT, 1, 1, 16);
}



// Reuse Cube result buffers for masking and rounding to avoid extra GM
// storage. A two-deep queue protects staging reuse while overlapping the
// next band's loads with the current band's vector work.
static __aicore__ inline void post_gram(TQue<TPosition::VECIN, 2>& qin,
                                        TQue<TPosition::VECIN, 1>& qmk,
                                        TQue<TPosition::VECOUT, 2>& qout,
                                        TQue<TPosition::VECOUT, 2>& qo16,
                                        const LocalTensor<uint8_t> mbitsAll,
                                        const GlobalTensor<float> Aqk32,
                                        const GlobalTensor<float> L,
                                        const GlobalTensor<bfloat16_t> Aqk16,
                                        const GlobalTensor<float> MaskS,
                                        const GlobalTensor<float> MaskL,
                                        int32_t c, float scale, bool buildMasks) {
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    CrossCoreWaitFlag(FL_DONE);
    // Build invariant mask bits once per block to avoid repeating conversion.
    constexpr int32_t BB = 16 * M / 8;
    // Use 16-row bands to fit Gram, mask and rounding staging in UB.
    // Prefetch one band ahead to hide MTE2 latency; queue depth two keeps
    // both bands live until the current band releases its storage.
    {
        LocalTensor<float> gpre = qin.AllocTensor<float>();
        DataCopy(gpre, Aqk32[m0], 16 * M);
        DataCopy(gpre[16 * M], L[m0], 16 * M);
        qin.EnQue(gpre);
    }
    for (int32_t mm = 0; mm < KF; ++mm) {
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;
    const uint64_t mo = static_cast<uint64_t>(mm) * 16 * M;
    constexpr int32_t NB = 16 * M;
    LocalTensor<float> gA = qin.DeQue<float>();
    if (mm + 1 < KF) {
        LocalTensor<float> gnx = qin.AllocTensor<float>();
        const uint64_t on = m0 + static_cast<uint64_t>(mm + 1) * 16 * M;
        DataCopy(gnx, Aqk32[on], NB);
        DataCopy(gnx[NB], L[on], NB);
        qin.EnQue(gnx);
    }
    LocalTensor<float> gmk, gmaskS, gmaskL;
    if (buildMasks) {
        gmk = qmk.AllocTensor<float>();
        gmaskS = gmk; gmaskL = gmk[NB];
        DataCopy(gmaskS, MaskS[mo], NB);
        DataCopy(gmaskL, MaskL[mo], NB);
        qmk.EnQue(gmk);
    }
    LocalTensor<float> ga32 = gA, gl32 = gA[NB];
    LocalTensor<float> gM;
    if (buildMasks) {
        gM = qmk.DeQue<float>();
        gmaskS = gM; gmaskL = gM[NB];
    }
    LocalTensor<float> gout = qout.AllocTensor<float>();
    LocalTensor<float> ga32o = gout, gl32o = gout[NB];
    LocalTensor<uint8_t> gmaskBits = mbitsAll[mm * BB];
    LocalTensor<uint8_t> gmaskBitsL = mbitsAll[M * M / 8 + mm * BB];
    // Select masked values to zero because multiplication would retain Inf/NaN.
    if (buildMasks) {
        Compares(gmaskBits, gmaskS, 0.5f, CMPMODE::GT, NB);
        PipeBarrier<PIPE_V>();
        Compares(gmaskBitsL, gmaskL, 0.5f, CMPMODE::GT, NB);
        PipeBarrier<PIPE_V>();
    }
    Select(ga32o, gmaskBits, ga32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    Muls(ga32o, ga32o, scale, NB);
    PipeBarrier<PIPE_V>();
    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> g16 = qo16.AllocTensor<bfloat16_t>();
    Cast(g16, ga32o, RoundMode::CAST_RINT, NB);
    qout.EnQue(gout);
    qo16.EnQue(g16);
    LocalTensor<float> gA2 = qout.DeQue<float>();
    LocalTensor<float> ga32s = gA2, gl32s = gA2[NB];
    LocalTensor<bfloat16_t> g16s = qo16.DeQue<bfloat16_t>();
    // Skip the masked FP32 store because only the rounded Aqk16 has a
    // consumer; retain raw Aqk32 for the Cube-to-vector handoff.
    DataCopy(L[o], gl32s, NB);
    DataCopy(Aqk16[o], g16s, NB);
    qout.FreeTensor(gA2);
    qo16.FreeTensor(g16s);
    qin.FreeTensor(gA);
    if (buildMasks) qmk.FreeTensor(gM);
    }
}

static __aicore__ inline void run_gram_aic(GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb,
                                           GM_ADDR pGx,
                                           GM_ADDR pAqk32, GM_ADDR pL,
                                           int32_t nchunk, int32_t unroll) {
    const int32_t base = GetBlockIdx() * 2 * unroll;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 4> qa, qb;
    pipe.InitBuffer(qa, 4, M * D * 2);
    pipe.InitBuffer(qb, 4, M * D * 2);
    TQue<QuePosition::CO1, 2> qc;
    pipe.InitBuffer(qc, 2, M * M * 4);
    TQue<QuePosition::B1, 2> qx;
    pipe.InitBuffer(qx, 2, BS * D * 2);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 4 * M * D * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * M * D * 2 + (XBAND ? 2 * BS * D * 2 : 0));
    GlobalTensor<bfloat16_t> Ga, Gk, Gb, Gx;
    GlobalTensor<float> Aqk32, L;
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
    Gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGx));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));

    for (int32_t u = 0; u < unroll; ++u) {
        // Wait once for both AIV publications so the paired chunks stay in step.
        CrossCoreWaitFlag(FL_READY);
        // Clamp tail work to keep flag counts balanced; repeated work is idempotent.
        int32_t c0 = base + 2 * u;
        int32_t c1 = c0 + 1;
        if (c0 >= nchunk) c0 = nchunk - 1;
        if (c1 >= nchunk) c1 = nchunk - 1;
        const uint64_t o0 = static_cast<uint64_t>(c0) * N;
        const uint64_t o1 = static_cast<uint64_t>(c1) * N;
        auto ta0 = qa.AllocTensor<bfloat16_t>();
        auto tk0 = qa.AllocTensor<bfloat16_t>();
        auto ta1 = qa.AllocTensor<bfloat16_t>();
        auto tk1 = qa.AllocTensor<bfloat16_t>();
        auto tb0 = qb.AllocTensor<bfloat16_t>();
        auto tb1 = qb.AllocTensor<bfloat16_t>();
        // Separate cross-band staging avoids enlarging the shared L0B slots.
        LocalTensor<bfloat16_t> tx0, tx1;
        if (XBAND) {
            tx0 = qx.AllocTensor<bfloat16_t>();
            tx1 = qx.AllocTensor<bfloat16_t>();
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(ta0[mm * DF * 256], Ga[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tk0[mm * DF * 256], Gk[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tb0[mm * DF * 256], Gb[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(ta1[mm * DF * 256], Ga[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tk1[mm * DF * 256], Gk[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tb1[mm * DF * 256], Gb[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; XBAND && mm < BS / 16; ++mm) {
            DataCopy(tx0[mm * DF * 256], Gx[static_cast<uint64_t>(c0) * BS * D + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
            DataCopy(tx1[mm * DF * 256], Gx[static_cast<uint64_t>(c1) * BS * D + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        qa.EnQue(ta0);
        qa.EnQue(tk0);
        qa.EnQue(ta1);
        qa.EnQue(tk1);
        qb.EnQue(tb0);
        qb.EnQue(tb1);
        if (XBAND) {
            qx.EnQue(tx0);
            qx.EnQue(tx1);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        ta0 = qa.DeQue<bfloat16_t>();
        tk0 = qa.DeQue<bfloat16_t>();
        ta1 = qa.DeQue<bfloat16_t>();
        tk1 = qa.DeQue<bfloat16_t>();
        tb0 = qb.DeQue<bfloat16_t>();
        tb1 = qb.DeQue<bfloat16_t>();
        if (XBAND) {
            tx0 = qx.DeQue<bfloat16_t>();
            tx1 = qx.DeQue<bfloat16_t>();
        }
        LocalTensor<bfloat16_t> la0 = a8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk0 = a8[M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> la1 = a8[2 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk1 = a8[3 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb0 = b8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb1 = b8[M * D * 2].ReinterpretCast<bfloat16_t>();
        constexpr int32_t KX = BS / 16;
        LocalTensor<bfloat16_t> lx0 =
            b8[2 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lx1 =
            b8[2 * M * D * 2 + (XBAND ? BS * D * 2 : 0)].ReinterpretCast<bfloat16_t>();
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(la0[(mm * DF + dd) * 256], ta0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lk0[(mm * DF + dd) * 256], tk0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(la1[(mm * DF + dd) * 256], ta1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lk1[(mm * DF + dd) * 256], tk1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lb0[(dd * KF + mm) * 256], tb0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lb1[(dd * KF + mm) * 256], tb1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; XBAND && dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KX; ++mm) {
                LoadData(lx0[(dd * KX + mm) * 256], tx0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; XBAND && dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KX; ++mm) {
                LoadData(lx1[(dd * KX + mm) * 256], tx1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        SetFlag<HardEvent::MTE1_M>(e1m);
        WaitFlag<HardEvent::MTE1_M>(e1m);
        // Use the full tile's C0 stride so Fixpipe reads the correct L0C rows
        // when the accumulator spans multiple n-blocks.
        constexpr int32_t FX_SRC_STRIDE = M > 16 ? M : 1;
        auto ip = FixpipeParamsV220(M, M, FX_SRC_STRIDE, M, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        // Use the sub-tile's C0 stride but retain the parent destination stride
        // so the cross-band correction lands within the existing Gram tile.
        auto ipx = FixpipeParamsV220(BS, BS, BS, M, false);
        ipx.quantPre = QuantMode_t::NoQuant;
        ipx.unitFlag = 0;
        for (int32_t s = 0; s < 2; ++s) {
            LocalTensor<float> cf0 = qc.AllocTensor<float>();
            LocalTensor<float> cf1 = qc.AllocTensor<float>();
            if (s == 0) {
                Mmad(cf0, la0, lb0, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk0, lb0, MmadParams(M, M, D, 0, false, true));
            } else {
                Mmad(cf0, la1, lb1, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk1, lb1, MmadParams(M, M, D, 0, false, true));
            }
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            const uint64_t m0 = static_cast<uint64_t>(s == 0 ? c0 : c1) * M * M;
            Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32[m0], cf0, ip);
            Fixpipe<float, float, CFG_ROW_MAJOR>(L[m0], cf1, ip);
            SetFlag<HardEvent::FIX_M>(efm);
            WaitFlag<HardEvent::FIX_M>(efm);
            qc.FreeTensor(cf0);
            qc.FreeTensor(cf1);
            if (XBAND) {
                // Both operands need band 1's reference to avoid exponent underflow
                // and preserve cancellation. Overwrite the cross-band block because
                // the full-tile product used different references for its two sides.
                LocalTensor<bfloat16_t> laX = s == 0 ? la0[2 * DF * 256] : la1[2 * DF * 256];
                LocalTensor<bfloat16_t> lkX = s == 0 ? lk0[2 * DF * 256] : lk1[2 * DF * 256];
                LocalTensor<bfloat16_t> lx = s == 0 ? lx0 : lx1;
                LocalTensor<float> cf2 = qc.AllocTensor<float>();
                LocalTensor<float> cf3 = qc.AllocTensor<float>();
                Mmad(cf2, laX, lx, MmadParams(BS, BS, D, 0, false, true));
                Mmad(cf3, lkX, lx, MmadParams(BS, BS, D, 0, false, true));
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32[m0 + BS * M], cf2, ipx);
                Fixpipe<float, float, CFG_ROW_MAJOR>(L[m0 + BS * M], cf3, ipx);
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qc.FreeTensor(cf2);
                qc.FreeTensor(cf3);
            }
        }
        qa.FreeTensor(ta0);
        qa.FreeTensor(tk0);
        qa.FreeTensor(ta1);
        qa.FreeTensor(tk1);
        qb.FreeTensor(tb0);
        qb.FreeTensor(tb1);
        if (XBAND) {
            qx.FreeTensor(tx0);
            qx.FreeTensor(tx1);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(FL_DONE);
    }
}

extern "C" __global__ __aicore__ void kda_pre_gram_mix(
    GM_ADDR pQ, GM_ADDR pK, GM_ADDR pV, GM_ADDR pG, GM_ADDR pBeta,
    GM_ADDR pAlog, GM_ADDR pBias,
    GM_ADDR pDecay, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pQg, GM_ADDR pKg,
    GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb, GM_ADDR pGx,
    GM_ADDR pAqk32, GM_ADDR pAqk16, GM_ADDR pL, GM_ADDR pMaskS, GM_ADDR pMaskL,
    GM_ADDR pLengths, GM_ADDR pSequences,
    int32_t B, int32_t T, int32_t H, float lower_bound, float scale, int32_t unroll,
    int32_t xRowBytes, int32_t gRowBytes, int32_t gateBf16) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // Sequence offsets avoid packing the full input for variable-length batches.
    const auto sequences = reinterpret_cast<__gm__ int64_t *>(pSequences);
    const int32_t totalChunks = pSequences == nullptr ? B * H * ((T + M - 1) / M)
                                                      : sequences[B * 4 + 2] * H;
    if ASCEND_IS_AIC {
        run_gram_aic(pGa, pGk, pGb, pGx, pAqk32, pL, totalChunks, unroll);
    }
    if ASCEND_IS_AIV {
    const int32_t nt = (T + M - 1) / M;
    const DataCopyPadExtParams<bfloat16_t> padNop(false, 0, 0, 0);
    const DataCopyPadExtParams<float> padNopF(false, 0, 0, 0);
    const DataCopyPadExtParams<float> padScalar(true, 0, 7, 0);
    const int32_t nchunk = totalChunks;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vq = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vk = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vg = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vv = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vs = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID evs = pipe.AllocEventID<HardEvent::V_S>();
    TEventID e3d = pipe.AllocEventID<HardEvent::MTE3_V>();
    // Drain stores before vector writes reuse their source buffers.
    TEventID e3p = pipe.AllocEventID<HardEvent::MTE3_V>();
    // Cross-band stores occur after the pass drain, so rvbo reuse needs
    // its own MTE3-to-vector dependency.
    TEventID e3x = pipe.AllocEventID<HardEvent::MTE3_V>();
    // Order the next pass's loads after the current pass's final buffer reads.
    TEventID em2 = pipe.AllocEventID<HardEvent::V_MTE2>();
    // Reserve explicit UB for queues because TPipe does not overlay them
    // with scratch buffers.
    TQue<TPosition::VECIN, 2> qgin;
    TQue<TPosition::VECIN, 1> qgmk;
    TQue<TPosition::VECOUT, 2> qgout, qgo16;
    TBuf<TPosition::VECCALC> bQf, bKf, bT0, bT2, bEf, bRed,
        bQnb, bKnb, bRkb, bRvb, bQgb, bKgb, bBias, bBeta, bBb, bAlog, bZz,
        bGef, bGateBf16, bRvo, bMfull, bBetaIndex;
    pipe.InitBuffer(bQf, NG * 4); pipe.InitBuffer(bKf, NG * 4);
    pipe.InitBuffer(bT0, N * 4); pipe.InitBuffer(bT2, NG * 4);
    pipe.InitBuffer(bEf, NG * 4); pipe.InitBuffer(bRed, 384 * 4);
    pipe.InitBuffer(bQnb, NG * 2); pipe.InitBuffer(bKnb, NG * 2); pipe.InitBuffer(bRkb, NG * 2);
    pipe.InitBuffer(bRvb, NG * 2); pipe.InitBuffer(bQgb, NG * 2); pipe.InitBuffer(bKgb, NG * 2);
    pipe.InitBuffer(bBias, D * 4); pipe.InitBuffer(bBeta, M * 4);
    pipe.InitBuffer(bBetaIndex, M * sizeof(uint32_t));
    // A single 32-byte broadcast block per row avoids a full-width beta tile.
    pipe.InitBuffer(bBb, M * 8 * 4); pipe.InitBuffer(bAlog, 8 * 4);
    pipe.InitBuffer(bZz, NG * 4);
    // Reuse consumed exponential and Q/K landing tiles to reduce UB pressure;
    // em2 prevents later loads from overwriting normalization's final reads.
    pipe.InitBuffer(bGef, NG * 4);
    pipe.InitBuffer(bGateBf16, N * 2);
    pipe.InitBuffer(bRvo, NG * 2);
    pipe.InitBuffer(qgin, 2, 2 * 16 * M * 4);
    // Serial mask-band lifetimes allow a single queue slot.
    pipe.InitBuffer(qgmk, 1, 2 * 16 * M * 4);
    pipe.InitBuffer(qgout, 2, 2 * 16 * M * 4);
    pipe.InitBuffer(qgo16, 2, 16 * M * 2);
    // Cache invariant mask bits to avoid rebuilding them for every chunk.
    pipe.InitBuffer(bMfull, 2 * M * M / 8 + 64);
    // Publish one pass at a time to limit operand staging. FL_READY on
    // MTE3 keeps Cube from reading until every band has reached GM.
    TBuf<TPosition::VECCALC> bPga0, bPgk0, bPgb0;
    pipe.InitBuffer(bPga0, NG * 2);
    pipe.InitBuffer(bPgk0, NG * 2);
    pipe.InitBuffer(bPgb0, NG * 2);
    LocalTensor<float> qf = bQf.Get<float>(), kf = bKf.Get<float>();
    LocalTensor<float> gf = bT0.Get<float>(), t2 = bT2.Get<float>();
    LocalTensor<float> ef = bEf.Get<float>(), red = bRed.Get<float>();
    LocalTensor<float> bias = bBias.Get<float>(), beta = bBeta.Get<float>();
    LocalTensor<float> bb = bBb.Get<float>(), alog = bAlog.Get<float>();
    LocalTensor<float> zz = bZz.Get<float>();
    LocalTensor<float> gef = bGef.Get<float>();
    LocalTensor<bfloat16_t> qnb2 = bQnb.Get<bfloat16_t>(), knb2 = bKnb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> gnb = bGateBf16.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> rvbo = bRvo.Get<bfloat16_t>();
    LocalTensor<uint8_t> mbitsAll = bMfull.Get<uint8_t>();
    bool masksBuilt = false;
    LocalTensor<uint32_t> betaIndex = bBetaIndex.Get<uint32_t>();
    if (pSequences != nullptr) {
        for (int32_t row = 0; row < M; ++row) betaIndex.SetValue(row, row * 32);
        PipeBarrier<PIPE_ALL>();
    }

    LocalTensor<bfloat16_t> qnb = bQnb.Get<bfloat16_t>(), knb = bKnb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> rkb = bRkb.Get<bfloat16_t>(), rvb = bRvb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> qgb = bQgb.Get<bfloat16_t>(), kgb = bKgb.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> Q, K, V, Rk, Rv, Qg, Kg, Ga, Gk, Gb, Gx;
    GlobalTensor<bfloat16_t> GBf16;
    GlobalTensor<float> G, Beta, Alog, Bias, Decay;
    Q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQ));
    K.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pK));
    V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pV));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
    Gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGx));
    GBf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pG));
    G.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pG));
    Beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBeta));
    Alog.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAlog));
    Bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBias));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));

    GlobalTensor<bfloat16_t> Aqk16;
    GlobalTensor<float> Aqk32, L, MaskS, MaskL;
    Aqk16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk16));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    MaskS.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskS));
    MaskL.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskL));
    // Interleave subcore chunks so Cube reads adjacent operand tiles.
    // Clamp out-of-range chunks to balance flag counts; repeated work
    // is idempotent.
    const int32_t ratio = static_cast<int32_t>(GetTaskRation());
    const int32_t group = ratio == 0 ? static_cast<int32_t>(GetBlockIdx())
                                     : static_cast<int32_t>(GetBlockIdx()) / ratio;
    const int32_t sub = static_cast<int32_t>(GetSubBlockIdx());
    int32_t cprev = -1;
    int32_t cachedHead = -1;
    float aexp = 0.0f;
    for (int32_t u = 0; u < unroll; ++u) {
    int32_t c = group * 2 * unroll + 2 * u + sub;
    if (c >= nchunk) c = nchunk - 1;
    int32_t sequence = c / (nt * H);
    int32_t sequenceNt = nt;
    int32_t localChunk = c - sequence * H * nt;
    int64_t tokenStart = static_cast<int64_t>(sequence) * T;
    int32_t sequenceLength = T;
    if (pSequences != nullptr) {
        // Choose the rightmost prefix to skip empty sequences with repeated offsets.
        int32_t lo = 0, hi = B;
        while (lo + 1 < hi) {
            const int32_t mid = lo + (hi - lo) / 2;
            if (sequences[mid * 4 + 2] * H <= c) lo = mid;
            else hi = mid;
        }
        sequence = lo;
        tokenStart = sequences[lo * 4];
        sequenceLength = sequences[lo * 4 + 1];
        sequenceNt = sequences[lo * 4 + 3];
        localChunk = c - sequences[lo * 4 + 2] * H;
    }
    const int32_t head = localChunk / sequenceNt;
    const int32_t ck = localChunk % sequenceNt;
    const bool headChanged = head != cachedHead;
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    const uint64_t xg = static_cast<uint64_t>(c) * BS * D;
    const uint64_t cm = static_cast<uint64_t>(c) * M;
    // Read the public strided layout directly to avoid packing a full input copy.
    const int32_t b = sequence;
    const int32_t physicalRows = sequenceLength - ck * M;
    int32_t validRows = physicalRows < M ? physicalRows : M;
    if (pLengths != nullptr) {
        const int32_t remaining = reinterpret_cast<__gm__ int32_t *>(pLengths)[b] - ck * M;
        const int32_t logicalRows = remaining < 0 ? 0 : remaining;
        if (logicalRows < validRows) validRows = logicalRows;
    }
    const uint64_t token = tokenStart + static_cast<int64_t>(ck) * M;
    const uint64_t xb = token *
                            static_cast<uint64_t>(H) * D +
                        static_cast<uint64_t>(head) * D;

    // Use byte-strided DataCopyPad because block-strided DataCopy is
    // unreliable for this GM gather on the target hardware.
    // Keep the row-coupled prefix sum chunk-wide; use small passes for
    // row-local work to limit UB staging.
    if (validRows > 0) {
        if (gateBf16) {
            DataCopyPad(gnb, GBf16[xb], DataCopyExtParams(validRows, D * 2, gRowBytes, 0, 0), padNop);
        } else {
            DataCopyPad(gf, G[xb], DataCopyExtParams(validRows, D * 4, gRowBytes, 0, 0), padNopF);
        }
    }
    SetFlag<HardEvent::MTE2_V>(e2vg);
    if (headChanged) {
        // Load only A_log's scalar to avoid reading past GM for the final head.
        DataCopyPad(alog, Alog[head], DataCopyExtParams(1, sizeof(float), 0, 0, 0),
                    padScalar);
    }
    if (pSequences == nullptr) {
        DataCopy(beta, Beta[cm], DataCopyParams(1, M / 8, 0, 0));
    } else {
        // Reuse beta broadcast storage for aligned scalar loads to avoid extra UB.
        DataCopyPad(bb, Beta[token * H + head],
                    DataCopyExtParams(validRows, sizeof(float), (H - 1) * sizeof(float), 0, 0), padScalar);
    }
    // Wait for gate/log/beta loads before vector consumers read their UB buffers.
    SetFlag<HardEvent::MTE2_V>(e2vs);
    WaitFlag<HardEvent::MTE2_V>(e2vg);
    if (gateBf16) {
        Cast(gf, gnb, RoundMode::CAST_NONE, N);
        PipeBarrier<PIPE_V>();
    }
    if (validRows < M) {
        Duplicate(gf[validRows * D], 0.0f, (M - validRows) * D);
        PipeBarrier<PIPE_V>();
    }

    WaitFlag<HardEvent::MTE2_V>(e2vs);
    if (pSequences != nullptr) {
        if (validRows < M) {
            // Align every staging row to 32 bytes so one-token tails remain valid.
            Duplicate(bb[validRows * 8], 0.0f, (M - validRows) * 8);
            PipeBarrier<PIPE_V>();
        }
        Gather(beta, bb, betaIndex, 0, M);
        PipeBarrier<PIPE_V>();
    }
    Muls(beta, beta, -1.0f, M);
    Exp(beta, beta, M);
    Adds(beta, beta, 1.0f, M);
    Duplicate(red, 1.0f, M);
    PipeBarrier<PIPE_V>();
    Div(beta, red, beta, M);
    PipeBarrier<PIPE_V>();
    Brcb(bb, beta, M / 8, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();

    if (pBias != nullptr) {
        if (headChanged) {
            DataCopy(bias, Bias[head * D], DataCopyParams(1, 16, 0, 0));
            SetFlag<HardEvent::MTE2_V>(e2v);
            WaitFlag<HardEvent::MTE2_V>(e2v);
        }
        Add(gf, gf, bias, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        Add(gf[64], gf[64], bias[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        PipeBarrier<PIPE_V>();
    }
    if (headChanged) {
        // Cache bias and its exponential across chunks sharing a head to avoid
        // repeated loads and exponentiation.
        Exp(alog, alog, 8);
        SetFlag<HardEvent::V_S>(evs);
        WaitFlag<HardEvent::V_S>(evs);
        aexp = -alog.GetValue(0);
        cachedHead = head;
    }
    // Split sigmoid work into passes because its temporary operand only
    // fits an MT-row tile; the row-coupled prefix sum must stay chunk-wide.
    Duplicate(t2, 1.0f, NG);
    for (int32_t hp = 0; hp < NP; ++hp) {
    LocalTensor<float> gfs = gf[hp * NG];
    Muls(gfs, gfs, aexp, NG);
    Exp(gfs, gfs, NG);
    Adds(gfs, gfs, 1.0f, NG);
    PipeBarrier<PIPE_V>();
    Div(gfs, t2, gfs, NG);
    Muls(gfs, gfs, lower_bound, NG);
    PipeBarrier<PIPE_V>();
    }
    // Neutralize padded log-decay after activation so padded tokens preserve
    // state exactly, independent of gate logits, bias and A_log.
    if (validRows < M) {
        Duplicate(gf[validRows * D], 0.0f, (M - validRows) * D);
        PipeBarrier<PIPE_V>();
    }
    // A radix-8 scan separates aliasing reads and writes across instructions.
    // A naive in-place log scan creates repeat-level RAW dependencies that
    // this hardware does not order. Skip block-total rows in the final add
    // because they already contain their prefix and would alias its source.
    {
        constexpr int32_t BS = 8;
        constexpr int32_t NB = M / BS;
        constexpr int32_t ROW8 = 8 * 16;
        for (int32_t j = 1; j < BS; ++j) {
            Add(gf[D * j], gf[D * j], gf[D * (j - 1)], 64, NB,
                BinaryRepeatParams(1, 1, 1, ROW8, ROW8, ROW8));
            Add(gf[D * j + 64], gf[D * j + 64], gf[D * (j - 1) + 64], 64, NB,
                BinaryRepeatParams(1, 1, 1, ROW8, ROW8, ROW8));
            PipeBarrier<PIPE_V>();
        }
        for (int32_t b = 1; b < NB; ++b) {
            Add(gf[D * (BS * b + BS - 1)], gf[D * (BS * b + BS - 1)],
                gf[D * (BS * (b - 1) + BS - 1)], 128);
            PipeBarrier<PIPE_V>();
        }
        for (int32_t b = 1; b < NB; ++b) {
            Add(gf[D * BS * b], gf[D * BS * b], gf[D * (BS * (b - 1) + BS - 1)],
                64, BS - 1, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
            Add(gf[D * BS * b + 64], gf[D * BS * b + 64],
                gf[D * (BS * (b - 1) + BS - 1) + 64], 64, BS - 1,
                BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        }
        PipeBarrier<PIPE_V>();
    }
    Muls(gf, gf, RCP_LN2, N);
    Muls(t2, gf[(M - 1) * D], LN2, D);
    Exp(t2, t2, D);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Decay[static_cast<uint64_t>(c) * D], t2, DataCopyParams(1, 16, 0, 0));
    // MTE3_V protects t2 reuse without draining all pipes. Wait once before
    // the pass loop: repeating a wait for one publication would block forever.
    SetFlag<HardEvent::MTE3_V>(e3d);
    LocalTensor<bfloat16_t> pga = bPga0.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgk = bPgk0.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgb = bPgb0.Get<bfloat16_t>();

    WaitFlag<HardEvent::MTE3_V>(e3d);
    for (int32_t hp = 0; hp < NP; ++hp) {
    const int32_t gh = hp * NG;
    const uint64_t xh = x0 + gh;
    const uint64_t xbh = xb + static_cast<uint64_t>(hp) * MT * H * D;
    LocalTensor<float> gfp = gf[gh];
    LocalTensor<float> bbp = bb[hp * MT * 8];
    const int32_t remainingRows = validRows - hp * MT;
    const int32_t rows = remainingRows < 0 ? 0 : (remainingRows < MT ? remainingRows : MT);
    if (rows > 0) {
        DataCopyPad(qnb, Q[xbh], DataCopyExtParams(rows, D * 2, xRowBytes, 0, 0), padNop);
    }
    SetFlag<HardEvent::MTE2_V>(e2vq);
    if (rows > 0) {
        DataCopyPad(knb, K[xbh], DataCopyExtParams(rows, D * 2, xRowBytes, 0, 0), padNop);
    }
    SetFlag<HardEvent::MTE2_V>(e2vk);
    if (rows > 0) {
        DataCopyPad(rvb, V[xbh], DataCopyExtParams(rows, D * 2, xRowBytes, 0, 0), padNop);
    }
    SetFlag<HardEvent::MTE2_V>(e2vv);
    WaitFlag<HardEvent::MTE2_V>(e2vq);
    if (rows < MT) {
        // Zero through half to avoid numeric conversion and out-of-range GM reads.
        Duplicate(qnb.ReinterpretCast<half>()[rows * D], half(0.0f), (MT - rows) * D);
        PipeBarrier<PIPE_V>();
    }
    Cast(qf, qnb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    Mul(t2, qf, qf, NG);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, MT);
    Rsqrt(red, red, MT);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(qf, qf, red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(qf[64], qf[64], red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(qnb2, qf, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    Cast(qf, qnb2, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    WaitFlag<HardEvent::MTE2_V>(e2vk);
    if (rows < MT) {
        Duplicate(knb.ReinterpretCast<half>()[rows * D], half(0.0f), (MT - rows) * D);
        PipeBarrier<PIPE_V>();
    }
    Cast(kf, knb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, kf, NG);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, MT);
    Rsqrt(red, red, MT);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(kf, kf, red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(kf[64], kf[64], red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(knb2, kf, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    Cast(kf, knb2, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    // Center each gate band to keep both operand exponentials within FP32 range.
    const int32_t mid = MID0 + (hp * MT / BS) * BS;
    Sub(zz, gfp, gf[mid * D], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Sub(zz[64], gfp[64], gf[mid * D + 64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Muls(ef, gfp, LN2, NG);
    Exp(ef, ef, NG);
    PipeBarrier<PIPE_V>();

    Mul(t2, qf, ef, NG);
    PipeBarrier<PIPE_V>();
    Cast(qgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    Mul(t2, kf, ef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rkb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    WaitFlag<HardEvent::MTE2_V>(e2vv);
    if (rows < MT) {
        Duplicate(rvb.ReinterpretCast<half>()[rows * D], half(0.0f), (MT - rows) * D);
        PipeBarrier<PIPE_V>();
    }
    Cast(t2, rvb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rvbo, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    // Place V_MTE2 after the final landing-buffer read so the next pass's
    // loads cannot overwrite it. The earlier MTE3 drain cannot protect a
    // later reader because its wait does not stall scalar instruction issue.
    SetFlag<HardEvent::V_MTE2>(em2); WaitFlag<HardEvent::V_MTE2>(em2);

    Sub(t2, gf[(M - 1) * D], gfp, 64, MT, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    Sub(t2[64], gf[(M - 1) * D + 64], gfp[64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    PipeBarrier<PIPE_V>();
    Muls(t2, t2, LN2, NG);
    Exp(t2, t2, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, kf, NG);
    PipeBarrier<PIPE_V>();
    Cast(kgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    // Issue stores early to overlap their transfer with Gram preparation.
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Qg[xh], qgb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Kg[xh], kgb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Rk[xh], rkb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Rv[xh], rvbo, DataCopyParams(MT, 8, 0, 0));
    // Store operands per pass to avoid whole-chunk UB staging. Reuse t2
    // and gef after their final readers to keep the same rounding points
    // with fewer temporary tiles.
    Muls(gef, zz, LN2, NG);
    Exp(gef, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, qf, gef, NG);
    PipeBarrier<PIPE_V>();
    Cast(pga, t2, RoundMode::CAST_RINT, NG);
    Mul(t2, kf, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    if (pSequences != nullptr && hp + 1 == NP) {
        // Fence beta's final bb read before the next chunk reuses bb for MTE2 loads.
        SetFlag<HardEvent::V_MTE2>(em2);
        WaitFlag<HardEvent::V_MTE2>(em2);
    }
    Cast(pgk, t2, RoundMode::CAST_RINT, NG);
    Muls(gef, zz, -LN2, NG);
    Exp(gef, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, gef, NG);
    PipeBarrier<PIPE_V>();
    Cast(pgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Ga[xh], pga, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Gk[xh], pgk, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Gb[xh], pgb, DataCopyParams(MT, 8, 0, 0));
    // Drain MTE3 reads before the next pass overwrites their vector sources.
    // The separate V_MTE2 pair protects the opposite load/write dependency.
    SetFlag<HardEvent::MTE3_V>(e3p); WaitFlag<HardEvent::MTE3_V>(e3p);

    // Republish band-0 keys with band 1's reference so cross-band factors
    // cancel without underflow. Reuse rvbo after its Rv store drains to
    // avoid another landing tile.
    if (XBAND && mid == MID0) {
        Sub(t2, gfp, gf[(MID0 + BS) * D], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        Sub(t2[64], gfp[64], gf[(MID0 + BS) * D + 64], 64, MT,
            BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        PipeBarrier<PIPE_V>();
        Muls(t2, t2, -LN2, NG);
        Exp(t2, t2, NG);
        PipeBarrier<PIPE_V>();
        Mul(t2, t2, kf, NG);
        PipeBarrier<PIPE_V>();
        Cast(rvbo, t2, RoundMode::CAST_RINT, NG);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gx[xg + gh], rvbo, DataCopyParams(MT, 8, 0, 0));
        // Fence this late store separately: the pass drain precedes it and
        // cannot protect rvbo from the next pass's cast.
        SetFlag<HardEvent::MTE3_V>(e3x);
        WaitFlag<HardEvent::MTE3_V>(e3x);
    }
    }
    if (cprev >= 0) {
        post_gram(qgin, qgmk, qgout, qgo16, mbitsAll, Aqk32, L, Aqk16,
                  MaskS, MaskL, cprev, scale, !masksBuilt);
        masksBuilt = true;
    }
    // Wait for DONE before publishing again because flags are levels,
    // not counters; overlapping publications can lose a handshake.
    CrossCoreSetFlag<2, PIPE_MTE3>(FL_READY);
    cprev = c;
    // The pass-level V_MTE2 and MTE3_V pairs also protect chunk boundaries,
    // including single-pass chunks, so an additional full drain is unnecessary.
    }
    // Finish the final Gram handoff before releasing pipeline resources.
    post_gram(qgin, qgmk, qgout, qgo16, mbitsAll, Aqk32, L, Aqk16,
              MaskS, MaskL, cprev, scale, !masksBuilt);
    masksBuilt = true;
    PipeBarrier<PIPE_ALL>();
    }
}
