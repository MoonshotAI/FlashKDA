// Fusing assembly and W/U saves one host launch per solve slice.
// Scoped TPipe lifetimes drain work before reusing L0 and events, keeping
// storage at the larger phase's requirement instead of their sum.
// Retain the GM A16 boundary to preserve BF16 rounding and layout.
static_assert(kda_assembly::NC == kda_wu::NC, "fused chunk ownership must match");
extern "C" __global__ __aicore__ void kda_solve_fused(
    GM_ADDR pA16, GM_ADDR pLneg, GM_ADDR pRk, GM_ADDR pRv,
    GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    kda_assembly::run(pA16, pLneg, C);
    kda_wu::run(pA16, pRk, pRv, pW, pU, C);
}
