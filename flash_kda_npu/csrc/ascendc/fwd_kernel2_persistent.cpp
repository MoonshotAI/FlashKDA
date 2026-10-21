// Keeping recurrence on device removes per-chunk launches and FP32 state
// round trips. Resident FP32 state also avoids cumulative BF16 rounding.
// Independent loads run ahead of cross-core waits to hide memory latency;
// barriers still protect staging buffers read by the previous iteration.
//
// FP32 D1/D2/D3 handoffs preserve cancellation accuracy for weak decay and
// large initial states. Per-head flags let chunks reuse one GM slot safely:
// FL_R protects D1/D2/D3 reuse, and the following FL_C1 protects Vt reuse.
//
// Publishing Vt once in packed fractal order avoids a second GM copy.
// Strided loads derive the B layout from that same copy; 16-row bands satisfy
// the L0 layout, while K == FR permits a contiguous-load shortcut.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
#ifndef KDA_MAXH
#define KDA_MAXH 4
#endif
constexpr int32_t M = KDA_CHUNK;
constexpr int32_t FR = 16;
constexpr int32_t NB = M / FR;
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t K = M;
constexpr int32_t N = 64;
constexpr int32_t N_D4 = 128;
constexpr int32_t TILE = M * BV;
constexpr int32_t S_TILE = BV * D;
constexpr int32_t NG = 2 * BV / M;
constexpr int32_t MAXH = KDA_MAXH;
// Reusing L0A across stages reduces storage; size it for the larger live
// operand set so either stage can use the same allocation.
constexpr int32_t L0A_ELEMS =
    (2 * M * D > NG * M * K + M * K) ? 2 * M * D : NG * M * K + M * K;
// Keep flag IDs within the usable range to avoid cross-core channel conflicts.
constexpr uint16_t FL_C1 = 0;
constexpr uint16_t FL_V = 1;
constexpr uint16_t FL_C2 = 2;
constexpr uint16_t FL_R = 3;

extern "C" __global__ __aicore__ void kda_k2_persistent_loop(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4,
    GM_ADDR pOut, GM_ADDR pVnewT,
    GM_ADDR pH0, GM_ADDR pFinal, GM_ADDR pS16, GM_ADDR pSequences, GM_ADDR pHeadMap,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, float scale, int32_t NH,
    int32_t finalBF16, int32_t T) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int32_t nv = NV;
    const auto sequences = reinterpret_cast<__gm__ int64_t *>(pSequences);
    // Both engines must visit the same head order so their flag handshakes match.
    const auto headMap = reinterpret_cast<__gm__ int64_t *>(pHeadMap);

    if ASCEND_IS_AIC {
        const int32_t blk = static_cast<int32_t>(GetBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        if (pHeadMap != nullptr) {
            for (int32_t slot = 0; slot < MAXH; ++slot) {
                const int32_t h = headMap[blk * 4 + slot];
                if (h >= 0) heads[nh++] = h;
            }
        } else {
            for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        }
        if (nh == 0) return;
        // Cache the immutable schedule to avoid scalar GM reads in every chunk stage.
        int32_t chunkCounts[MAXH], chunkBases[MAXH], lengths[MAXH];
        int64_t tokenStarts[MAXH];
        int32_t blockChunks = 0;
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t bh = heads[s], seq = bh / NH;
            chunkCounts[s] = pSequences == nullptr ? NT : sequences[seq * 4 + 3];
            chunkBases[s] = pSequences == nullptr ? bh * NT
                : sequences[seq * 4 + 2] * NH + (bh % NH) * chunkCounts[s];
            lengths[s] = pSequences == nullptr ? T : sequences[seq * 4 + 1];
            tokenStarts[s] = pSequences == nullptr ? static_cast<int64_t>(seq) * T : sequences[seq * 4];
            if (chunkCounts[s] > blockChunks) blockChunks = chunkCounts[s];
        }

        GlobalTensor<bfloat16_t> W, Qg, Aqk, Vt, Kt, S16;
        GlobalTensor<float> D4;
        GlobalTensor<float> D1, D2, D3;
        W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
        Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
        Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

        TPipe pipe;
        TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
        TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
        TEventID em1 = pipe.AllocEventID<HardEvent::M_MTE1>();
        TQue<QuePosition::B1, 1> qw, qg, qs, qa, qv, qx, qk;
        pipe.InitBuffer(qw, 1, M * D * 2);
        pipe.InitBuffer(qg, 1, M * D * 2);
        pipe.InitBuffer(qs, 1, nv * BV * D * 2);
        pipe.InitBuffer(qa, 1, M * K * 2);
        pipe.InitBuffer(qv, 1, NG * M * K * 2);
        pipe.InitBuffer(qx, 1, nv * BV * K * 2);
        pipe.InitBuffer(qk, 1, D * K * 2);
        TQue<QuePosition::CO1, 1> qc;
        pipe.InitBuffer(qc, 1, (NG * M * N_D4 + nv * M * N) * 4);
        LocalTensor<float> cf = qc.AllocTensor<float>();
        LocalTensor<uint8_t> a8(TPosition::A2, 0, L0A_ELEMS * 2);
        // Reuse L0B across stages to avoid allocating separate operand buffers.
        LocalTensor<uint8_t> b8(TPosition::B2, 0, nv * BV * D * 2);
        LocalTensor<bfloat16_t> l0a = a8.ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b = b8.ReinterpretCast<bfloat16_t>();

        for (int32_t chunk = 0; chunk < blockChunks; ++chunk) {
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t chunks = chunkCounts[s];
                if (chunk >= chunks) continue;
                const int32_t chunkBase = chunkBases[s];
                const uint64_t a0 = static_cast<uint64_t>(chunkBase + chunk) * M * D;
                auto lw = qw.AllocTensor<bfloat16_t>();
                auto lg = qg.AllocTensor<bfloat16_t>();
                auto ls = qs.AllocTensor<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lw[b * FR * D], W[a0 + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                    DataCopy(lg[b * FR * D], Qg[a0 + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_R);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t s0 = static_cast<uint64_t>(bh * nv + iv) * BV * D;
                    DataCopy(ls[iv * BV * D], S16[s0], Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qw.EnQue(lw);
                qg.EnQue(lg);
                qs.EnQue(ls);
                lw = qw.DeQue<bfloat16_t>();
                lg = qg.DeQue<bfloat16_t>();
                ls = qs.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[b * FR * D], lw[b * FR * D],
                             LoadData2dParams(0, D / FR, 1, 0, 0, false, 0));
                    LoadData(l0a[M * D + b * FR * D], lg[b * FR * D],
                             LoadData2dParams(0, D / FR, 1, 0, 0, false, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[iv * BV * D], ls[iv * BV * D],
                             LoadData2dParams(0, BV * D / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[iv * M * N], l0a, l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                    Mmad(cf[(nv + iv) * M * N], l0a[M * D], l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t o0 = static_cast<uint64_t>(bh * nv + iv) * TILE;
                    auto ip1 = FixpipeParamsV220(N, M, M, N, false);
                    ip1.quantPre = QuantMode_t::NoQuant;
                    ip1.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0], cf[iv * M * N], ip1);
                    auto ip2 = FixpipeParamsV220(N, M, M, N, false);
                    ip2.quantPre = QuantMode_t::NoQuant;
                    ip2.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0], cf[(nv + iv) * M * N], ip2);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qw.FreeTensor(lw);
                qg.FreeTensor(lg);
                qs.FreeTensor(ls);
                // M_MTE1 prevents the next LoadData from overwriting operands still read
                // by Mmad without draining every pipe. FIX_M protects only L0C, and a
                // PIPE_M barrier cannot order MTE1 writes.
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C1);
            }
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t chunks = chunkCounts[s];
                if (chunk >= chunks) continue;
                const int32_t chunkBase = chunkBases[s];
                const int32_t c = chunkBase + chunk;
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lv = qv.AllocTensor<bfloat16_t>();
                auto lx = qx.AllocTensor<bfloat16_t>();
                auto lk = qk.AllocTensor<bfloat16_t>();
                if (K == FR) {
                    // With one C0 block per row, ND already matches fractal order;
                    // a plain burst avoids layout conversion.
                    DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], M * K);
                } else {
                    for (int32_t b = 0; b < NB; ++b) {
                        DataCopy(la[b * FR * K],
                                 Aqk[static_cast<uint64_t>(c) * M * K + b * FR * K],
                                 Nd2NzParams(1, FR, K, 0, K, FR, 1, 0));
                    }
                }
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lk[b * FR * D], Kt[static_cast<uint64_t>(c) * M * D + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_V);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const int32_t task = bh * nv + iv;
                    const uint64_t t0 = static_cast<uint64_t>(task) * BV * K;
                    if (K == FR) {
                        // With K == FR, both operands can read Vt directly because each
                        // row contains one C0 block.
                        DataCopy(lv[iv * BV * K], Vt[t0], BV * K);
                        DataCopy(lx[iv * BV * K], Vt[t0], BV * K);
                    } else {
                        // Explicit block strides reuse Vt's packed A layout for B without
                        // another GM copy. Iterate K / FR bands with BV / FR source stride;
                        // confusing these dimensions overruns qx when K differs from BV.
                        DataCopy(lv[iv * BV * K], Vt[t0], BV * K);
                        for (int32_t c = 0; c < K / FR; ++c) {
                            DataCopy(lx[iv * BV * K + c * (BV / FR) * FR * FR],
                                     Vt[t0 + c * FR * FR],
                                     DataCopyParams(BV / FR, FR * FR / 16,
                                                    (K / FR) * FR * FR / 16 - FR * FR / 16, 0));
                        }
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qa.EnQue(la);
                qv.EnQue(lv);
                qx.EnQue(lx);
                qk.EnQue(lk);
                la = qa.DeQue<bfloat16_t>();
                lv = qv.DeQue<bfloat16_t>();
                lx = qx.DeQue<bfloat16_t>();
                lk = qk.DeQue<bfloat16_t>();
                // Load both value tiles: the d4 operand has NG * M rows, requiring
                // NG * NB fractal bands regardless of the chunk size.
                for (int32_t b = 0; b < NG * NB; ++b) {
                    LoadData(l0a[b * FR * K], lv[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                }
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[NG * M * K + b * FR * K], la[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                    LoadDataWithTranspose(l0b[b * (D / FR) * 256], lk[b * FR * D],
                                          LoadData2dTransposeParams(0, D / FR, 1, 0, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[D * K + iv * BV * K], lx[iv * BV * K],
                             LoadData2dParams(0, BV * K / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                // One Mmad keeps d4 contiguous in L0C so a single row-major Fixpipe
                // can export the whole tile.
                Mmad(cf, l0a, l0b, MmadParams(NG * M, N_D4, K, 0, false, true));
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[NG * M * N_D4 + iv * M * N], l0a[NG * M * K],
                         l0b[D * K + iv * BV * K], MmadParams(M, N, K, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                {
                    auto ip = FixpipeParamsV220(N_D4, NG * M, NG * M, N_D4, false);
                    ip.quantPre = QuantMode_t::NoQuant;
                    ip.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D4[static_cast<uint64_t>(bh) * D * D], cf[0], ip);
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    auto ip = FixpipeParamsV220(N, M, M, N, false);
                    ip.unitFlag = 0;
                    ip.quantPre = QuantMode_t::NoQuant;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D3[static_cast<uint64_t>(bh * nv + iv) * TILE],
                        cf[NG * M * N_D4 + iv * M * N], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qa.FreeTensor(la);
                qv.FreeTensor(lv);
                qx.FreeTensor(lx);
                qk.FreeTensor(lk);
                // Order operand reuse after Mmad reads without draining unrelated pipes.
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C2);
            }
        }
        qc.FreeTensor(cf);
        return;
    }

    if ASCEND_IS_AIV {
        const int32_t ratio = static_cast<int32_t>(GetTaskRation());
        const int32_t raw = static_cast<int32_t>(GetBlockIdx());
        const int32_t blk = ratio == 0 ? raw : raw / ratio;
        // Matching flag sequences keep the two value subcores synchronized.
        const int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        if (pHeadMap != nullptr) {
            for (int32_t slot = 0; slot < MAXH; ++slot) {
                const int32_t h = headMap[blk * 4 + slot];
                if (h >= 0) heads[nh++] = h;
            }
        } else {
            for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        }
        if (nh == 0) return;
        // Cache the immutable schedule to avoid scalar GM reads in every chunk stage.
        int32_t chunkCounts[MAXH], chunkBases[MAXH], lengths[MAXH];
        int64_t tokenStarts[MAXH];
        int32_t blockChunks = 0;
        for (int32_t s = 0; s < nh; ++s) {
            const int32_t bh = heads[s], seq = bh / NH;
            chunkCounts[s] = pSequences == nullptr ? NT : sequences[seq * 4 + 3];
            chunkBases[s] = pSequences == nullptr ? bh * NT
                : sequences[seq * 4 + 2] * NH + (bh % NH) * chunkCounts[s];
            lengths[s] = pSequences == nullptr ? T : sequences[seq * 4 + 1];
            tokenStarts[s] = pSequences == nullptr ? static_cast<int64_t>(seq) * T : sequences[seq * 4];
            if (chunkCounts[s] > blockChunks) blockChunks = chunkCounts[s];
        }

        GlobalTensor<bfloat16_t> Vt, S16, Out, Final16;
        GlobalTensor<float> U, D4, Decay, H0, Final32;
        GlobalTensor<float> D1, D2, D3;
        U.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pU));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
        Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
        Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
        H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
        Final32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pFinal));
        Final16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pFinal));

        TPipe pipe;
        TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        TEventID evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
        TEventID e3v = pipe.AllocEventID<HardEvent::MTE3_V>();
        // Alias staging across stages 2 and 4 to leave UB space for resident
        // FP32 states. Their entry barriers prevent overlapping buffer use;
        // V_MTE2 after Sub separately protects d1f before sc/vt overwrite it.
        // B/D must also fit state slices when a chunk tile is smaller.
        // At M=64, 56.5 KiB staging plus four 32 KiB states fits the 192 KiB UB.
        constexpr int32_t HALF_BYTES = (TILE > S_TILE / 2) ? TILE * 2 : S_TILE;
        TBuf<TPosition::VECCALC> uA, uB, uC, uD, uE, udec, us;
        pipe.InitBuffer(uA, TILE * sizeof(float));
        pipe.InitBuffer(uB, HALF_BYTES);
        pipe.InitBuffer(uC, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uD, HALF_BYTES);
        pipe.InitBuffer(uE, TILE * sizeof(float));
        pipe.InitBuffer(udec, D * sizeof(float));
        pipe.InitBuffer(us, MAXH * S_TILE * sizeof(float));

        LocalTensor<float> vf = uA.Get<float>();
        LocalTensor<float> of = uA.Get<float>();
        LocalTensor<bfloat16_t> sc = uE.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vt = sc[TILE];
        LocalTensor<float> d1f = uE.Get<float>();
        LocalTensor<float> d2f = uE.Get<float>();
        LocalTensor<float> d3f = uA.Get<float>();
        LocalTensor<bfloat16_t> s16 = uB.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> ob = uC.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vb = uD.Get<bfloat16_t>();
        LocalTensor<float> d4 = uD.Get<float>();
        LocalTensor<float> dec = udec.Get<float>();
        LocalTensor<float> st = us.Get<float>();

        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            LocalTensor<float> state = st[s * S_TILE];
            if (pH0 == nullptr) {
                Duplicate(state, 0.0f, S_TILE);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(state, H0[static_cast<uint64_t>(task) * S_TILE],
                         DataCopyParams(BV, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
            }
            // Publishing state in halves keeps BF16 staging small enough for four
            // resident FP32 states. Drain each store before reusing its source.
            for (int32_t hf = 0; hf < 2; ++hf) {
                // A cross-pipe barrier is needed: PIPE_MTE3 alone cannot order
                // V-pipe writes after MTE3 reads.
                if (hf == 1) PipeBarrier<PIPE_ALL>();
                LocalTensor<float> sh = state[hf * (S_TILE / 2)];
                Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 2);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                const uint64_t offset = static_cast<uint64_t>(task) * S_TILE + hf * (BV / 2) * D;
                DataCopy(S16[offset], s16, DataCopyParams(BV / 2, D / 16, 0, 0));
                if (pSequences != nullptr && chunkCounts[s] == 0
                        && pFinal != nullptr && finalBF16) {
                    DataCopy(Final16[offset], s16, DataCopyParams(BV / 2, D / 16, 0, 0));
                }
            }

            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t s = 0; s < nh; ++s) {
            if (pSequences != nullptr && chunkCounts[s] == 0) continue;
            CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
        }

        for (int32_t chunk = 0; chunk < blockChunks; ++chunk) {
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t chunks = chunkCounts[s];
                if (chunk >= chunks) continue;
                const int32_t chunkBase = chunkBases[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = chunkBase + chunk;
                // Drain previous reads before reloading aliased UB buffers; persistent
                // iterations reuse staging that separate launches would isolate.
                PipeBarrier<PIPE_ALL>();
                const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
                const uint64_t out0 = static_cast<uint64_t>(task) * TILE;
                // Read only this subcore's value half to avoid duplicate U transfers.
                DataCopy(vf, U[u0 + iv * BV], DataCopyParams(M, BV / 8, (D - BV) / 8, 0));
                CrossCoreWaitFlag(FL_C1);
                DataCopy(d1f, D1[out0], DataCopyParams(M, BV / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                PipeBarrier<PIPE_V>();
                Sub(vf, vf, d1f, BV, M,
                    BinaryRepeatParams(1, 1, 1, BV / 8, BV / 8, BV / 8));
                Cast(vb, vf, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(evm2);
                WaitFlag<HardEvent::V_MTE2>(evm2);
                // Transpose into packed fractals so the Cube can consume Vt without
                // an additional GM layout conversion.
                for (int32_t j0 = 0; j0 < BV / FR; ++j0) {
                    for (int32_t m0 = 0; m0 < NB; ++m0) {
                        DataCopy(sc[(j0 * NB + m0) * FR * FR], vb[m0 * FR * BV + j0 * FR],
                                 DataCopyParams(FR, 1, BV / FR - 1, 0));
                    }
                }
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                for (int32_t bl = 0; bl < (BV / FR) * NB; ++bl) {
                    AscendC::Transpose(vt[bl * FR * FR], sc[bl * FR * FR]);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                // Publish the packed tile in one transfer; stage 3 derives the B
                // layout with load strides to avoid storing a second copy.
                DataCopy(Vt[out0], vt, BV * M);
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_V);
            }
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t chunks = chunkCounts[s];
                if (chunk >= chunks) continue;
                const int32_t chunkBase = chunkBases[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = chunkBase + chunk;
                // Drain previous vector reads and output/state stores before
                // reusing the same staging buffers.
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> state = st[s * S_TILE];
                const uint64_t t0 = static_cast<uint64_t>(task) * TILE;
                const uint64_t d4base = static_cast<uint64_t>(bh) * D * D +
                                        static_cast<uint64_t>(iv) * BV * D;
                DataCopy(d2f, D2[t0], DataCopyParams(M, BV / 8, 0, 0));
                DataCopy(dec, Decay[static_cast<uint64_t>(c) * D], DataCopyParams(1, D / 8, 0, 0));
                CrossCoreWaitFlag(FL_C2);
                DataCopy(d3f, D3[t0], DataCopyParams(M, BV / 8, 0, 0));
                DataCopy(d4, D4[d4base], DataCopyParams(BV / 4, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                // Round only after FP32 addition to preserve output cancellation accuracy.
                Muls(d2f, d2f, scale, TILE);
                Add(of, d3f, d2f, TILE);
                Cast(ob, of, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                // Quarter-sized d4 staging shares the v_new buffer and leaves UB
                // space for resident states. Sixteen rows preserve whole-row repeat
                // strides while allowing loads to overlap vector work.
                const bool nextChunk = chunk + 1 < chunks;
                const bool publish = nextChunk || (pFinal != nullptr && finalBF16);
                for (int32_t hf = 0; hf < 4; ++hf) {
                    if (hf > 0) {
                        WaitFlag<HardEvent::MTE2_V>(e2v);
                        // MTE3_V prevents Cast from overwriting s16 before its store
                        // has read it; PIPE_MTE3 alone cannot order the V pipe.
                        if (publish) WaitFlag<HardEvent::MTE3_V>(e3v);
                    }
                    LocalTensor<float> sh = state[hf * (S_TILE / 4)];
                    Mul(sh, sh, dec, 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Mul(sh[64], sh[64], dec[64], 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Add(sh, sh, d4, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    if (publish) Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(ev3);
                    WaitFlag<HardEvent::V_MTE3>(ev3);
                    if (hf == 0) {
                        // Store directly in the public layout to avoid an output permutation.
                        const int32_t hh = bh - (bh / NH) * NH;
                        const uint64_t token = tokenStarts[s] + static_cast<uint64_t>(chunk) * M;
                        const uint64_t obase = token * NH * D +
                                               static_cast<uint64_t>(hh) * D + iv * BV;
                        const int32_t remaining = lengths[s] - chunk * M;
                        const int32_t rows = remaining < M ? remaining : M;
                        DataCopy(Out[obase], ob,
                                 DataCopyParams(rows, BV / 16, 0,
                                                static_cast<uint16_t>(NH * D / 16 - BV / 16)));
                    }
                    if (publish) {
                        const uint64_t offset = static_cast<uint64_t>(task) * S_TILE + hf * (BV / 4) * D;
                        if (nextChunk) {
                            DataCopy(S16[offset], s16, DataCopyParams(BV / 4, D / 16, 0, 0));
                        } else {
                            DataCopy(Final16[offset], s16, DataCopyParams(BV / 4, D / 16, 0, 0));
                        }
                    }
                    if (hf < 3) {
                        // Release s16 only after the store has consumed it.
                        if (publish) SetFlag<HardEvent::MTE3_V>(e3v);
                        SetFlag<HardEvent::V_MTE2>(evm2);
                        WaitFlag<HardEvent::V_MTE2>(evm2);
                        DataCopy(d4, D4[d4base + (hf + 1) * (BV / 4) * D],
                                 DataCopyParams(BV / 4, D / 8, 0, 0));
                        SetFlag<HardEvent::MTE2_V>(e2v);
                    }
                }
                if (nextChunk) CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
            }
        }
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        for (int32_t s = 0; pFinal != nullptr && !finalBF16 && s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            DataCopy(Final32[static_cast<uint64_t>(task) * S_TILE], st[s * S_TILE],
                     DataCopyParams(BV, D / 8, 0, 0));
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
}
