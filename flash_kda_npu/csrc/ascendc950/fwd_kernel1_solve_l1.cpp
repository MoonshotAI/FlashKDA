// 950-only assembly/WU: preserve both BF16 coupling boundaries in L1.
// A16's assembled GM view remains available, but W/U consumes the L1 tile.
#include "kernel_operator.h"
using namespace AscendC;
namespace kda950 {
template<bool Transpose>
__aicore__ inline void LoadInverseSub(LocalTensor<bfloat16_t> dst,
                                     LocalTensor<bfloat16_t> src) {
    LoadData2DParamsV2 p{};
    p.mStep=2; p.kStep=2; p.srcStride=4; p.dstStride=2;
    p.ifTranspose=Transpose;
    LoadData(dst,src,p);
}
}
extern "C" __global__ __aicore__ void kda_solve_fused(
    GM_ADDR pX, GM_ADDR pNeg, GM_ADDR pRk, GM_ADDR pRv,
    GM_ADDR pW, GM_ADDR pU, int32_t count) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    static_assert(KDA_CHUNK==64, "L1 assembly/WU requires M64");
    constexpr int M=64, S=32, D=128, Batch=KDA_WU_NCHUNK;
    static_assert(Batch>=1 && Batch*M*D*2<=64*1024, "single-pass L0B capacity");
    static_assert(Batch*M*D*4<=256*1024, "single-pass L0C capacity");
    const int first=GetBlockIdx()*Batch;
    if(first>=count) return;
    const int nch=count-first<Batch?count-first:Batch;
    TPipe pipe;
    auto e21=pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    auto e1m=pipe.AllocEventID<HardEvent::MTE1_M>();
    auto emf=pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::A1> bx,bn;
    TBuf<TPosition::B1> bp,br;
    pipe.InitBuffer(bx,Batch*M*M*2); pipe.InitBuffer(bn,Batch*S*S*2);
    pipe.InitBuffer(bp,Batch*S*S*2); pipe.InitBuffer(br,Batch*M*D*2);
    auto x=bx.Get<bfloat16_t>(), neg=bn.Get<bfloat16_t>();
    auto product=bp.Get<bfloat16_t>(), rhs=br.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> aa(TPosition::A2,0,Batch*M*M), bb(TPosition::B2,0,Batch*M*D);
    LocalTensor<float> cc(TPosition::CO1,0,Batch*M*D);
    GlobalTensor<bfloat16_t> gx,gn,gk,gv,gw; GlobalTensor<float> gu;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)pX); gn.SetGlobalBuffer((__gm__ bfloat16_t*)pNeg);
    gk.SetGlobalBuffer((__gm__ bfloat16_t*)pRk); gv.SetGlobalBuffer((__gm__ bfloat16_t*)pRv);
    gw.SetGlobalBuffer((__gm__ bfloat16_t*)pW); gu.SetGlobalBuffer((__gm__ float*)pU);
    for(int ch=0;ch<nch;++ch) {
        DataCopy(x[ch*M*M],gx[uint64_t(first+ch)*M*M],Nd2NzParams(1,M,M,0,M,M,1,0));
        DataCopy(neg[ch*S*S],gn[uint64_t(first+ch)*S*S],Nd2NzParams(1,S,S,0,S,S,1,0));
    }
    SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
    for(int pass=0;pass<2;++pass) {
        for(int ch=0;ch<nch;++ch) {
            auto a=aa[ch*M*M], b=bb[ch*M*D]; auto c=cc[ch*M*D]; auto tile=x[ch*M*M];
            if(pass==0) {
                kda950::LoadNz<S,S>(a,neg[ch*S*S]);
                kda950::LoadInverseSub<true>(b,tile);
            } else {
                kda950::LoadInverseSub<false>(a,tile[2*M*16+S*16]);
                kda950::LoadTransposeNz<S,S>(b,product[ch*S*S]);
            }
            SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
            Mmad(c,a,b,MmadParams(S,S,S,0,false,true));
            SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
            if(pass==0) kda950::L0cToL1Nz<S,S>(product[ch*S*S],c);
            else {
                kda950::L0cToL1Nz<S,S,M>(tile[S*16],c);
                kda950::StoreNd<S,S>(gx[uint64_t(first+ch)*M*M+S*M],c,M);
            }
        }
        // Phase boundary drains FIX->L1 before operand slots are reused.
        PipeBarrier<PIPE_ALL>();
    }
    for(int ch=0;ch<nch;++ch) kda950::LoadNz<M,M>(aa[ch*M*M],x[ch*M*M]);
    for(int pass=0;pass<2;++pass) {
        auto input=pass==0?gk:gv;
        for(int ch=0;ch<nch;++ch)
            DataCopy(rhs[ch*M*D],input[uint64_t(first+ch)*M*D],Nd2NzParams(1,M,D,0,D,M,1,0));
        SetFlag<HardEvent::MTE2_MTE1>(e21); WaitFlag<HardEvent::MTE2_MTE1>(e21);
        for(int ch=0;ch<nch;++ch) {
            auto a=aa[ch*M*M], b=bb[ch*M*D]; auto c=cc[ch*M*D];
            uint64_t ri=uint64_t(first+ch)*M*D;
            kda950::LoadTransposeNz<M,D>(b,rhs[ch*M*D]);
            SetFlag<HardEvent::MTE1_M>(e1m); WaitFlag<HardEvent::MTE1_M>(e1m);
            Mmad(c,a,b,MmadParams(M,D,M,0,false,true));
            SetFlag<HardEvent::M_FIX>(emf); WaitFlag<HardEvent::M_FIX>(emf);
            if(pass==0) kda950::StoreNd<M,D>(gw[ri],c,D);
            else kda950::StoreNd<M,D>(gu[ri],c,D);
        }
        // One L0B/C slot per chunk; reuse between W and U only after drain.
        PipeBarrier<PIPE_ALL>();
    }
}
