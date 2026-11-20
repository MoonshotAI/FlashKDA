#ifndef FLASH_KDA_950_COMMON_H
#define FLASH_KDA_950_COMMON_H

// Shared matrix layouts and Cube/Vector data transfers.
#include "kernel_operator.h"

namespace kda950 {
// Legacy KDA stages L1 as consecutive 16-row NZ bands. dav-3510 L0A
// orders the fractal grid by K, then M. Preserve the GM/L1 contracts
// and permute the blocks on the MTE1 load.
template<int Rows, int Cols>
__aicore__ inline void LoadAFromBands(
    AscendC::LocalTensor<bfloat16_t> dst,
    AscendC::LocalTensor<bfloat16_t> src) {
    for (int k=0; k<Cols/16; ++k) {
        for (int m=0; m<Rows/16; ++m) {
            AscendC::LoadData(dst[(k*(Rows/16)+m)*256],
                             src[(m*(Cols/16)+k)*256],
                             AscendC::LoadData2dParams(0,1,1,0,0,false,0));
        }
    }
}
template<int Rows, int Cols>
__aicore__ inline void LoadAFromNz(
    AscendC::LocalTensor<bfloat16_t> dst,
    AscendC::LocalTensor<bfloat16_t> src) {
    AscendC::LoadData2DParamsV2 ld{};
    ld.mStep=Rows/16; ld.kStep=Cols/16;
    ld.srcStride=Rows/16; ld.dstStride=Rows/16;
    AscendC::LoadData(dst,src,ld);
}

using namespace AscendC;
template<int Rows, int Depth>
__aicore__ inline void LoadNz(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src) {
    LoadData2DParamsV2 p{};
    p.mStep=Rows/16; p.kStep=Depth/16;
    p.srcStride=Rows/16; p.dstStride=Rows/16;
    LoadData(dst,src,p);
}
template<int Rows, int Cols, typename T>
__aicore__ inline void StoreNd(GlobalTensor<T> dst, LocalTensor<float> src, int stride) {
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> p;
    p.mSize=Rows; p.nSize=Cols; p.srcStride=Rows; p.dstStride=stride;
    p.quantPre=sizeof(T)==2 ? QuantMode_t::F322BF16 : QuantMode_t::NoQuant;
    Fixpipe(dst,src,p);
}

#if defined(KDA_950_DIRECT_CV) || defined(KDA_950_PRE_CV)
template<int Rows, int Cols>
__aicore__ inline void PackNzUb(AscendC::LocalTensor<bfloat16_t> dst,
                               AscendC::LocalTensor<bfloat16_t> src) {
    for (int c = 0; c < Cols / 16; ++c)
        AscendC::DataCopy(dst[c * Rows * 16], src[c * 16],
            AscendC::DataCopyParams(Rows, 1, Cols / 16 - 1, 0));
}
// Explicit UB ND -> Cube L1 NZ using the hardware MTE3 route. Unlike the
// convenience ND2NZ overload, this needs no implicit UB stack workspace.
template<int Rows, int Cols, int L1Rows = Rows>
__aicore__ inline void UbToL1Nz(AscendC::LocalTensor<bfloat16_t> dst,
                               AscendC::LocalTensor<bfloat16_t> src) {
    static_assert(Rows % 16 == 0 && Cols % 16 == 0, "CV tiles must be C0 aligned");
    for (int c = 0; c < Cols / 16; ++c) {
        AscendC::DataCopy(dst[c * L1Rows * 16], src[c * 16],
            AscendC::DataCopyParams(Rows, 1, Cols / 16 - 1, 0));
    }
}
constexpr AscendC::FixpipeConfig kRowMajorUb = {AscendC::CO2Layout::ROW_MAJOR, true};
template<int Rows, int Cols, int SourceRows = Rows, int DestinationStride = Cols>
__aicore__ inline void L0cToUb(AscendC::LocalTensor<float> dst,
                              AscendC::LocalTensor<float> src, int subcore) {
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> p;
    p.mSize = Rows; p.nSize = Cols;
    p.srcStride = SourceRows; p.dstStride = DestinationStride;
    p.quantPre = QuantMode_t::NoQuant;
    p.dualDstCtl = 0; p.subBlockId = subcore != 0;
    AscendC::Fixpipe<float, float, kRowMajorUb>(dst, src, p);
}
#endif
}

#endif
