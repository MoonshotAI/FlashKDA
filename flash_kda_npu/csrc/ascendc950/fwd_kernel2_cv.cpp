// Ascend 950 resident-head pipelined direct CV recurrence. State remains FP32
// in each AIV; only rounded operands cross UB->L1 and FP32 products return
// through L0C->UB. The unused GM handoff arguments preserve the launcher ABI.
#include "kernel_operator.h"
using namespace AscendC;
namespace kda950cv {
constexpr int M = KDA_CHUNK, D = 128, BV = 64, FR = 16, NB = M / FR;
constexpr int TILE = M * BV, STATE = BV * D, MAXH = KDA_MAXH;
constexpr int RX_STRIDE = 64 * BV;
static_assert(M == 16 || M == 32 || M == 64, "CV1 supports M16/32/64");
static_assert(MAXH <= 4, "CV1 head map and UB budget require MAXH <= 4");
// One physical UB contract shared by AIC Fixpipe and both AIV consumers.
constexpr int D13 = 128 * 1024, WORK = 192 * 1024, TX = 208 * 1024;
constexpr int D4 = 224 * 1024, OUT = TX, DEC = TX + TILE * 2;
static_assert(MAXH * RX_STRIDE * 4 <= WORK - D13, "CV3 per-head receive slots");
static_assert(DEC + D * 4 <= TX + STATE * 2, "CV2 output/decay alias budget");
static_assert(D4 + STATE / 2 * 4 <= 256 * 1024, "CV2 UB capacity");
// L1 operands have independent state and residual slots. AIV i only writes
// its own slice; AIC does not reuse a slot until the transaction is done.
constexpr int LS = 0, LV = 128 * 1024, LX = 192 * 1024;
constexpr int LW = 256 * 1024, LQ = 272 * 1024, LA = 288 * 1024, LK = 296 * 1024;
static_assert(LK + M * D * 2 <= 512 * 1024, "CV1 L1 capacity");
constexpr int A_ELEMS = 2 * M * D > D * M + M * M ? 2 * M * D : D * M + M * M;
static_assert(A_ELEMS * 2 <= 64 * 1024, "CV1 L0A capacity");
static_assert((D * M + 2 * BV * M) * 2 <= 64 * 1024, "CV1 L0B capacity");
constexpr int HOLD_D2 = D * D + 2 * TILE;
static_assert((HOLD_D2 + MAXH * 2 * TILE) * 4 <= 256 * 1024, "CV2 L0C capacity");
constexpr uint16_t STATE_READY = 0, RESID_READY = 4, PRODUCTS_READY = 8;
constexpr uint16_t UPDATE_READY = 12, UPDATE_FREE = 13;
struct Schedule {
    int heads[MAXH], counts[MAXH], bases[MAXH], lengths[MAXH], nh = 0, chunks = 0;
    int64_t starts[MAXH];
    __aicore__ inline Schedule(int blk, int blocks, int bh, int nt, int nh_total, int tokens,
                               GM_ADDR seq_ptr, GM_ADDR map_ptr) {
        const auto seq = reinterpret_cast<__gm__ int64_t*>(seq_ptr);
        const auto map = reinterpret_cast<__gm__ int64_t*>(map_ptr);
        if (map != nullptr) {
            for (int s = 0; s < MAXH; ++s) {
                int h = map[blk * 4 + s];
                if (h >= 0) heads[nh++] = h;
            }
        } else {
            for (int h = blk; h < bh && nh < MAXH; h += blocks) heads[nh++] = h;
        }
        for (int s = 0; s < nh; ++s) {
            int h = heads[s], si = h / nh_total;
            counts[s] = seq == nullptr ? nt : seq[si * 4 + 3];
            bases[s] = seq == nullptr ? h * nt : seq[si * 4 + 2] * nh_total
                                                   + h % nh_total * counts[s];
            lengths[s] = seq == nullptr ? tokens : seq[si * 4 + 1];
            starts[s] = seq == nullptr ? int64_t(si) * tokens : seq[si * 4];
            if (counts[s] > chunks) chunks = counts[s];
        }
    }
};
}
extern "C" __global__ __aicore__ void kda_k2_persistent_loop(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4, GM_ADDR pOut, GM_ADDR pVnewT,
    GM_ADDR pH0, GM_ADDR pFinal, GM_ADDR pS16, GM_ADDR pSequences, GM_ADDR pHeadMap,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, float scale, int32_t NH,
    int32_t finalBF16, int32_t T) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using namespace kda950cv;
    TPipe pipe;
    if ASCEND_IS_AIC {
        auto e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        auto e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        auto emf = pipe.AllocEventID<HardEvent::M_FIX>();
        Schedule plan(GetBlockIdx(), NBLK, BH, NT, NH, T, pSequences, pHeadMap);
        if (!plan.nh) return;
        GlobalTensor<bfloat16_t> w, q, aq, kg;
        w.SetGlobalBuffer((__gm__ bfloat16_t*)pW); q.SetGlobalBuffer((__gm__ bfloat16_t*)pQg);
        aq.SetGlobalBuffer((__gm__ bfloat16_t*)pAqk); kg.SetGlobalBuffer((__gm__ bfloat16_t*)pKgT);
        LocalTensor<bfloat16_t> ls_all(TPosition::B1, LS, MAXH * 2 * STATE);
        LocalTensor<bfloat16_t> lv_all(TPosition::A1, LV, MAXH * D * M);
        LocalTensor<bfloat16_t> lx_all(TPosition::B1, LX, MAXH * D * M);
        LocalTensor<bfloat16_t> lw(TPosition::A1, LW, M * D);
        LocalTensor<bfloat16_t> lq(TPosition::A1, LQ, M * D);
        LocalTensor<bfloat16_t> la(TPosition::A1, LA, M * M);
        LocalTensor<bfloat16_t> lk(TPosition::B1, LK, M * D);
        LocalTensor<bfloat16_t> aa(TPosition::A2, 0, A_ELEMS);
        LocalTensor<bfloat16_t> bb(TPosition::B2, 0, 2 * STATE);
        LocalTensor<float> cc(TPosition::CO1, 0, HOLD_D2 + MAXH * 2 * TILE);
        LocalTensor<float> rx13_all(TPosition::VECCALC, D13, 4 * RX_STRIDE);
        LocalTensor<float> rx2(TPosition::VECCALC, WORK, TILE);
        LocalTensor<float> rx4(TPosition::VECCALC, D4, STATE / 2);
        for (int chunk = 0; chunk < plan.chunks; ++chunk) {
            for (int s = 0; s < plan.nh; ++s) {
                if (chunk >= plan.counts[s]) continue;
                int c = plan.bases[s] + chunk;
                auto ls = ls_all[s * 2 * STATE];
                auto rx13 = rx13_all[s * RX_STRIDE];
                uint64_t off = uint64_t(c) * M * D;
                PipeBarrier<PIPE_ALL>();
                DataCopy(lw, w[off], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
                DataCopy(lq, q[off], Nd2NzParams(1, M, D, 0, D, M, 1, 0));
                CrossCoreWaitFlag(STATE_READY + s);
                SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
                kda950::LoadAFromNz<M, D>(aa, lw);
                kda950::LoadAFromNz<M, D>(aa[M * D], lq);
                for (int iv = 0; iv < 2; ++iv)
                    LoadData(bb[iv * STATE], ls[iv * STATE], LoadData2dParams(0, STATE / 256, 1, 0, 0, false, 0));
                SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int iv = 0; iv < 2; ++iv) {
                    Mmad(cc[iv * TILE], aa, bb[iv * STATE], MmadParams(M, BV, D, 0, false, true));
                    Mmad(cc[HOLD_D2 + (s * 2 + iv) * TILE], aa[M * D], bb[iv * STATE], MmadParams(M, BV, D, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
                for (int iv = 0; iv < 2; ++iv) {
                    kda950::L0cToUb<M, BV>(rx13, cc[iv * TILE], iv);
                }
                CrossCoreSetFlag<2, PIPE_FIX>(PRODUCTS_READY + s);
            }
            for (int s = 0; s < plan.nh; ++s) {
                if (chunk >= plan.counts[s]) continue;
                int c = plan.bases[s] + chunk;
                uint64_t off = uint64_t(c) * M * D;
                auto lv = lv_all[s * D * M];
                auto lx = lx_all[s * D * M];
                auto rx13 = rx13_all[s * RX_STRIDE];
                // All D1 slots are consumed before this phase. A different
                // head's free D1 slot receives the upper half of D4. The
                // shared UPDATE credit protects it until all Vector reads end.
                auto rx4_upper = rx13_all[((s + 1) % 4) * RX_STRIDE];
                PipeBarrier<PIPE_ALL>();
                if constexpr (M == FR) DataCopy(la, aq[uint64_t(c) * M * M], M * M);
                else DataCopy(la, aq[uint64_t(c) * M * M], Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                for (int b = 0; b < NB; ++b)
                    DataCopy(lk[b * FR * D], kg[off + b * FR * D], Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                CrossCoreWaitFlag(RESID_READY + s);
                SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
                kda950::LoadAFromNz<D, M>(aa, lv);
                kda950::LoadAFromNz<M, M>(aa[D * M], la);
                for (int b = 0; b < NB; ++b)
                    LoadDataWithTranspose(bb[b * (D / FR) * 256], lk[b * FR * D],
                                          LoadData2dTransposeParams(0, D / FR, 1, 0, 0));
                for (int iv = 0; iv < 2; ++iv)
                    LoadData(bb[D * M + iv * BV * M], lx[iv * BV * M],
                             LoadData2dParams(0, BV * M / 256, 1, 0, 0, false, 0));
                SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
                Mmad(cc, aa, bb, MmadParams(D, D, M, 0, false, true));
                for (int iv = 0; iv < 2; ++iv)
                    Mmad(cc[D * D + iv * TILE], aa[D * M], bb[D * M + iv * BV * M],
                         MmadParams(M, BV, M, 0, false, true));
                SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
                CrossCoreWaitFlag(UPDATE_FREE);
                for (int iv = 0; iv < 2; ++iv) {
                    kda950::L0cToUb<M, BV>(rx2, cc[HOLD_D2 + (s * 2 + iv) * TILE], iv);
                    kda950::L0cToUb<BV / 2, D, D>(rx4, cc[iv * BV * 16], iv);
                    kda950::L0cToUb<BV / 2, D, D>(rx4_upper, cc[(iv * BV + BV / 2) * 16], iv);
                    kda950::L0cToUb<M, BV>(rx13, cc[D * D + iv * TILE], iv);
                }
                CrossCoreSetFlag<2, PIPE_FIX>(UPDATE_READY);
            }
            CrossCoreWaitFlag(UPDATE_FREE);
        }
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIV {
        auto e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        auto ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        auto e3v = pipe.AllocEventID<HardEvent::MTE3_V>();
        auto evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
        int ratio = GetTaskRation(), raw = GetBlockIdx();
        int blk = ratio == 0 ? raw : raw / ratio, iv = GetSubBlockIdx();
        Schedule plan(blk, NBLK, BH, NT, NH, T, pSequences, pHeadMap);
        if (!plan.nh) return;
        GlobalTensor<float> u, decay, initial, final32;
        GlobalTensor<bfloat16_t> output, final16;
        u.SetGlobalBuffer((__gm__ float*)pU); decay.SetGlobalBuffer((__gm__ float*)pDecay);
        initial.SetGlobalBuffer((__gm__ float*)pH0); final32.SetGlobalBuffer((__gm__ float*)pFinal);
        final16.SetGlobalBuffer((__gm__ bfloat16_t*)pFinal); output.SetGlobalBuffer((__gm__ bfloat16_t*)pOut);
        LocalTensor<float> states(TPosition::VECCALC, 0, MAXH * STATE);
        LocalTensor<bfloat16_t> tx(TPosition::VECCALC, TX, STATE);
        LocalTensor<float> work(TPosition::VECCALC, WORK, TILE);
        LocalTensor<float> rx13_all(TPosition::VECCALC, D13, 4 * RX_STRIDE);
        LocalTensor<float> rx4(TPosition::VECCALC, D4, STATE / 2);
        LocalTensor<float> dec(TPosition::VECCALC, DEC, D);
        LocalTensor<bfloat16_t> ob(TPosition::VECCALC, OUT, TILE);
        for (int s = 0; s < plan.nh; ++s) {
            if (pH0 == nullptr) Duplicate(states[s * STATE], 0.0f, STATE);
            else DataCopy(states[s * STATE], initial[uint64_t(plan.heads[s] * 2 + iv) * STATE], STATE);
            PipeBarrier<PIPE_ALL>();
            if (plan.counts[s] == 0) continue;
            auto packed = rx13_all[s * RX_STRIDE].ReinterpretCast<bfloat16_t>();
#ifdef KDA_950_STATE_PACK_VF
            kda950::UpdateStatePackVF<false>((__ubuf__ float*)states[s * STATE].GetPhyAddr(),
                nullptr, nullptr, nullptr, (__ubuf__ bfloat16_t*)packed.GetPhyAddr());
            PipeBarrier<PIPE_V>();
#else
            Cast(tx, states[s * STATE], RoundMode::CAST_RINT, STATE);
            PipeBarrier<PIPE_V>();
            kda950::PackNzUb<BV, D>(packed, tx);
#endif
            SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
            LocalTensor<bfloat16_t> state_slot(TPosition::B1, LS + (s * 2 + iv) * STATE * 2, STATE);
            DataCopy(state_slot, packed, STATE);
            CrossCoreSetFlag<2, PIPE_MTE3>(STATE_READY + s);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
        for (int chunk = 0; chunk < plan.chunks; ++chunk) {
            for (int s = 0; s < plan.nh; ++s) {
                if (chunk >= plan.counts[s]) continue;
                int bh = plan.heads[s], c = plan.bases[s] + chunk;
                auto rx13 = rx13_all[s * RX_STRIDE];
                LocalTensor<bfloat16_t> resid_a(TPosition::A1, LV + (s * D * M + iv * BV * FR) * 2, BV * M);
                LocalTensor<bfloat16_t> resid_b(TPosition::B1, LX + (s * D * M + iv * BV * M) * 2, BV * M);
                DataCopy(work, u[uint64_t(c) * M * D + iv * BV], DataCopyParams(M, BV / 8, (D - BV) / 8, 0));
                CrossCoreWaitFlag(PRODUCTS_READY + s);
                SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
#ifdef KDA_950_RESIDUAL_VF
                kda950::ResidualNzVF<M>((__ubuf__ float*)work.GetPhyAddr(),
                    (__ubuf__ float*)rx13.GetPhyAddr(), (__ubuf__ bfloat16_t*)tx[TILE].GetPhyAddr());
                PipeBarrier<PIPE_V>();
#else
                Sub(work, work, rx13, BV, M, BinaryRepeatParams(1, 1, 1, BV / 8, BV / 8, BV / 8));
                PipeBarrier<PIPE_V>();
                auto vb = rx13.ReinterpretCast<bfloat16_t>();
                Cast(vb, work, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                for (int j = 0; j < BV / FR; ++j) for (int b = 0; b < NB; ++b)
                    DataCopy(tx[(j * NB + b) * 256], vb[b * FR * BV + j * FR],
                             DataCopyParams(FR, 1, BV / FR - 1, 0));
                PipeBarrier<PIPE_V>();
                // Transpose each tile and publish one full NZ matrix. AIV0
                // and AIV1 fill disjoint row slices of the same 128-row A.
                for (int j = 0; j < BV / FR; ++j) for (int b = 0; b < NB; ++b)
                    AscendC::Transpose(tx[TILE + (b * (BV / FR) + j) * 256],
                                      tx[(j * NB + b) * 256]);
#endif
                SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                DataCopy(resid_a, tx[TILE], DataCopyParams(NB, BV * FR / 16, 0, BV * FR / 16));
                DataCopy(resid_b, tx[TILE], BV * M);
                CrossCoreSetFlag<2, PIPE_MTE3>(RESID_READY + s);
                SetFlag<HardEvent::MTE3_V>(e3v); WaitFlag<HardEvent::MTE3_V>(e3v);
                SetFlag<HardEvent::V_MTE2>(evm2); WaitFlag<HardEvent::V_MTE2>(evm2);
            }
            // WORK is also the D2 receive slot: publish its initial credit
            // only after every head's residual has stopped using WORK.
            CrossCoreSetFlag<2, PIPE_V>(UPDATE_FREE);
            for (int s = 0; s < plan.nh; ++s) {
                if (chunk >= plan.counts[s]) continue;
                int bh = plan.heads[s], c = plan.bases[s] + chunk;
                auto state = states[s * STATE];
                auto rx13 = rx13_all[s * RX_STRIDE];
                auto rx4_upper = rx13_all[((s + 1) % 4) * RX_STRIDE];
                // The independent decay DMA can overlap Cube completion.
                // DEC aliases TX, and the last head may still be using DEC
                // in its VF when no next-state publication drains Vector.
                SetFlag<HardEvent::V_MTE2>(evm2); WaitFlag<HardEvent::V_MTE2>(evm2);
                DataCopy(dec, decay[uint64_t(c) * D], D);
                CrossCoreWaitFlag(UPDATE_READY);
                SetFlag<HardEvent::MTE2_V>(e2v); WaitFlag<HardEvent::MTE2_V>(e2v);
                Muls(work, work, scale, TILE);
                Add(rx13, rx13, work, TILE);
                Cast(ob, rx13, RoundMode::CAST_RINT, TILE);
                SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                int hh = bh % NH, remaining = plan.lengths[s] - chunk * M;
                int valid = remaining < M ? remaining : M;
                uint64_t token = plan.starts[s] + uint64_t(chunk) * M;
                DataCopy(output[token * NH * D + uint64_t(hh) * D + iv * BV], ob,
                         DataCopyParams(valid, BV / 16, 0, NH * D / 16 - BV / 16));
#ifdef KDA_950_STATE_PACK_VF
                if (chunk + 1 < plan.counts[s]) {
                    auto packed = rx13.ReinterpretCast<bfloat16_t>();
                    kda950::UpdateStatePackVF<true>((__ubuf__ float*)state.GetPhyAddr(),
                        (__ubuf__ float*)dec.GetPhyAddr(), (__ubuf__ float*)rx4.GetPhyAddr(),
                        (__ubuf__ float*)rx4_upper.GetPhyAddr(),
                        (__ubuf__ bfloat16_t*)packed.GetPhyAddr());
                } else
#endif
                {
                kda950::UpdateStateVF((__ubuf__ float*)state.GetPhyAddr(),
                    (__ubuf__ float*)dec.GetPhyAddr(), (__ubuf__ float*)rx4.GetPhyAddr(),
                    (__ubuf__ float*)rx4_upper.GetPhyAddr());
                }
                // Return the aliased receive-slot credit only after the VF
                // and output DMA drain. A PIPE_V notification alone exposed
                // intermittent output corruption under repeated full-grid runs.
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_V>(UPDATE_FREE);
                // TX aliases output; wait only for its output DMA before reuse.
                SetFlag<HardEvent::MTE3_V>(e3v); WaitFlag<HardEvent::MTE3_V>(e3v);
                if (chunk + 1 < plan.counts[s]) {
                    auto packed = rx13.ReinterpretCast<bfloat16_t>();
#ifndef KDA_950_STATE_PACK_VF
                    Cast(tx, state, RoundMode::CAST_RINT, STATE);
                    PipeBarrier<PIPE_V>();
                    kda950::PackNzUb<BV, D>(packed, tx);
#endif
                    SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                    LocalTensor<bfloat16_t> state_slot(TPosition::B1, LS + (s * 2 + iv) * STATE * 2, STATE);
                    DataCopy(state_slot, packed, STATE);
                    CrossCoreSetFlag<2, PIPE_MTE3>(STATE_READY + s);
                    SetFlag<HardEvent::MTE3_V>(e3v); WaitFlag<HardEvent::MTE3_V>(e3v);
                }
            }
        }
        for (int s = 0; pFinal != nullptr && s < plan.nh; ++s) {
            uint64_t off = uint64_t(plan.heads[s] * 2 + iv) * STATE;
            if (finalBF16) {
                Cast(tx, states[s * STATE], RoundMode::CAST_RINT, STATE);
                PipeBarrier<PIPE_ALL>();
                DataCopy(final16[off], tx, STATE);
                PipeBarrier<PIPE_ALL>();
            } else {
                SetFlag<HardEvent::V_MTE3>(ev3); WaitFlag<HardEvent::V_MTE3>(ev3);
                DataCopy(final32[off], states[s * STATE], STATE);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
}
