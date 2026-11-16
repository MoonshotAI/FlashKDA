#ifndef FLASH_KDA_950_SOLVE_H
#define FLASH_KDA_950_SOLVE_H

// Inverse assembly, vector row solves and optional L1 transfers.
#include "kernel_operator.h"
#if defined(KDA_950_SOLVE_VF) && defined(KDA_950_WIDE_SOLVE)
#include "reg_compute/kernel_reg_compute_intf.h"
#endif

namespace kda950 {
using namespace AscendC;
template<int Rows, int Cols>
__aicore__ inline void LoadTransposeNz(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src) {
    LoadData2DParamsV2 p{};
    p.mStep=Rows/16; p.kStep=Cols/16;
    p.srcStride=Rows/16; p.dstStride=Cols/16; p.ifTranspose=true;
    LoadData(dst,src,p);
}
template<int Chunk,int Batch>
__aicore__ inline void Assemble(GM_ADDR px,GM_ADDR pn,int count) {
    constexpr int Sub=Chunk/2, Tile=Sub*Sub;
    int first=GetBlockIdx()*Batch;
    TPipe pipe;
    TBuf<TPosition::A1> ba;
    TBuf<TPosition::B1> bb;
    pipe.InitBuffer(ba,Tile*2); pipe.InitBuffer(bb,Tile*2);
    auto a1=ba.Get<bfloat16_t>(), b1=bb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> a2(TPosition::A2,0,Tile),b2(TPosition::B2,0,Tile);
    LocalTensor<float> c(TPosition::CO1,0,Tile);
    GlobalTensor<bfloat16_t> x,neg;
    x.SetGlobalBuffer((__gm__ bfloat16_t*)px); neg.SetGlobalBuffer((__gm__ bfloat16_t*)pn);
    for(int ch=first;ch<first+Batch && ch<count;++ch) {
        uint64_t base=uint64_t(ch)*Chunk*Chunk;
        DataCopy(a1,neg[uint64_t(ch)*Tile],Nd2NzParams(1,Sub,Sub,0,Sub,Sub,1,0));
        DataCopy(b1,x[base],Nd2NzParams(1,Sub,Sub,0,Chunk,Sub,1,0));
        PipeBarrier<PIPE_ALL>();
        LoadNz<Sub,Sub>(a2,a1); LoadTransposeNz<Sub,Sub>(b2,b1);
        PipeBarrier<PIPE_ALL>();
        Mmad(c,a2,b2,MmadParams(Sub,Sub,Sub,0,false,true));
        PipeBarrier<PIPE_ALL>();
        // Use the owned X21 GM tile as the BF16 intermediate in the baseline.
        // L0C -> L1 NZ reuse is enabled only after its own routing microtest.
        StoreNd<Sub,Sub>(x[base+Sub*Chunk],c,Chunk);
        PipeBarrier<PIPE_ALL>();
        DataCopy(a1,x[base+Sub*Chunk+Sub],Nd2NzParams(1,Sub,Sub,0,Chunk,Sub,1,0));
        DataCopy(b1,x[base+Sub*Chunk],Nd2NzParams(1,Sub,Sub,0,Chunk,Sub,1,0));
        PipeBarrier<PIPE_ALL>();
        LoadNz<Sub,Sub>(a2,a1); LoadTransposeNz<Sub,Sub>(b2,b1);
        PipeBarrier<PIPE_ALL>();
        Mmad(c,a2,b2,MmadParams(Sub,Sub,Sub,0,false,true));
        PipeBarrier<PIPE_ALL>();
        StoreNd<Sub,Sub>(x[base+Sub*Chunk],c,Chunk);
        PipeBarrier<PIPE_ALL>();
    }
}

#if defined(KDA_950_SOLVE_VF) && defined(KDA_950_WIDE_SOLVE)
// Keep one row in registers across its predecessor loop. Previous rows are
// published by separate VF invocations with PIPE_V completion between rows.
template<int Rows, int Batch>
__simd_vf__ inline void SolveRowVF(__ubuf__ float* af, __ubuf__ float* lraw,
                                  __ubuf__ float* eye, int row_index) {
    using namespace AscendC::Reg;
    constexpr int Stride = Batch * Rows;
    uint32_t lanes = Rows;
    auto mask = UpdateMask<float>(lanes);
    RegTensor<float> row, previous, coefficient;
    for (uint16_t ch = 0; ch < Batch; ++ch) {
        LoadAlign(row, eye + row_index * Rows);
        for (uint16_t j = 0; j < row_index; ++j) {
            LoadAlign(previous, af + j * Stride + ch * Rows);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                coefficient, lraw + row_index * Stride + ch * Rows + j);
            MulAddDst(row, previous, coefficient, mask);
        }
        StoreAlign(af + row_index * Stride + ch * Rows, row, mask);
    }
}
#endif

#if defined(KDA_950_SOLVE_L1)
// dav-3510 L0C->L1 NZ dstStride is the column-block pitch in elements.
// DestRows can exceed Rows when replacing a block of a larger inverse tile.
template<int Rows, int Cols, int DestRows=Rows>
__aicore__ inline void L0cToL1Nz(AscendC::LocalTensor<bfloat16_t> dst,
                                AscendC::LocalTensor<float> src) {
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> p;
    p.mSize=Rows; p.nSize=Cols; p.srcStride=Rows; p.dstStride=DestRows*16;
    p.quantPre=QuantMode_t::F322BF16;
    AscendC::Fixpipe<bfloat16_t,float,AscendC::CFG_NZ>(dst,src,p);
}
#endif
}

#endif
