// Offloading the coupling product X22 @ Lneg21 @ X11 to Cube lets the
// vector solve use smaller diagonal blocks, reducing substitution work.
// Keeping the intermediate in L1 avoids a GM round trip; batching loads
// and final stores amortizes transfer setup across chunks.
#include "kernel_operator.h"
using namespace AscendC;
namespace kda_assembly {

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;
constexpr int32_t M = PC / 2;
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;
constexpr int32_t BANDE = KF * 256;
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;
constexpr int32_t SLOTS = PASSES * NC;

static __aicore__ inline void assemble_chunk(
    int32_t pass, int32_t slot,
    LocalTensor<bfloat16_t> la, LocalTensor<bfloat16_t> lb,
    LocalTensor<bfloat16_t> lp, LocalTensor<uint8_t>& a8,
    LocalTensor<uint8_t>& b8, LocalTensor<float>& cfall,
    int32_t ch, TEventID e1m, TEventID emf) {
    LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    if (pass == 1) {
        LoadData(a, la, LoadData2dParams(0, KF * KF, 1, 0, 0, false, 0));
    } else {
        for (int32_t dd = 0; dd < KF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
    }
    if (pass == 0) {
        LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
    } else {
        // Reorder NZ source blocks to match L0B's fractal layout without
        // materializing another intermediate.
        const int32_t src = ch * MM;
        LoadDataWithTranspose(b, lp[src], LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[2 * 256], lp[src + 256],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[1 * 256], lp[src + 512],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[3 * 256], lp[src + 768],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
    }
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);
    LocalTensor<float> cf = cfall[slot * MM];
    Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);
    // Keep the intermediate in L1 so the second pass avoids a GM round trip.
    if (pass == 0) {
        auto ipnz = FixpipeParamsV220(M, M, M, M, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lp[ch * MM], cf, ipnz);
    }
}

static __aicore__ inline void run(
    GM_ADDR pA16, GM_ADDR pLneg, int32_t C) {
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufA;
    pipe.InitBuffer(bufA, NC * LBSZ);
    TBuf<TPosition::B1> bufP;
    pipe.InitBuffer(bufP, NC * LBSZ);
    TBuf<TPosition::B1> bufX;
    pipe.InitBuffer(bufX, NC * 2 * MM * 2);
    // Separate slots prevent operand/result reuse while earlier work is in flight.
    // Size storage from SLOTS and MM because fused and standalone batch sizes differ.
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Lneg;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lxAll = bufX.Get<bfloat16_t>();

    // Gather directly from A16 to avoid a separate diagonal-block copy.
    DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
             Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
    for (int32_t ch = 0; ch < nch; ++ch) {
        for (int32_t diag = 0; diag < 2; ++diag) {
            DataCopy(lxAll[ch * 2 * MM + diag * MM],
                     A16[static_cast<uint64_t>(c0 + ch) * PC * PC + diag * (M * PC + M)],
                     Nd2NzParams(KF, 16, M, 16 * PC, PC, 16, 1, BANDE));
        }
    }
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);
    for (int32_t pass = 0; pass < PASSES; ++pass) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la =
                (pass == 0) ? laAll[ch * MM] : lxAll[ch * 2 * MM + MM];
            LocalTensor<bfloat16_t> lb =
                (pass == 0) ? lxAll[ch * 2 * MM] : lpAll;
            assemble_chunk(pass, pass * NC + ch, la, lb,
                           lpAll, a8, b8, cfall, ch, e1m, emf);
        }
        if (pass == 1) {
            // Batch stores to amortize Fixpipe setup. Each chunk has already
            // waited for M_FIX, so all source slots are ready for the transfer.
            auto ipb = FixpipeParamsV220(
                M, M, M, PC, false, QuantMode_t::F322BF16, 0,
                static_cast<uint16_t>(nch),
                static_cast<uint16_t>(MM / 256),
                static_cast<uint16_t>(PC * PC), 0);
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                A16[static_cast<uint64_t>(c0) * PC * PC + M * PC],
                cfall[NC * MM], ipb);
        }
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
}

#ifndef KDA_SOLVE_FUSED
extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pLneg, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    kda_assembly::run(pA16, pLneg, C);
}
#endif
