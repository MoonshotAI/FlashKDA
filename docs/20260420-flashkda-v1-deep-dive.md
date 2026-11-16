# FlashKDA v1: A Deep Dive

*2026-04-20*

This report summarizes the key design decisions behind FlashKDA v1, our fused
kernel implementation of Kimi Delta Attention. We walk through chunk-size
selection, kernel fusion strategy, numerical-precision trade-offs, and the
low-level optimizations that let the kernel reach its current performance
level.

## 1. Chunk Size Selection

Unlike Flash Linear Attention, which uses `CHUNK = 64`, FlashKDA v1
uses `CHUNK = 16`. This decision is motivated by three considerations:

- **Numerical range fits within bf16.** With the gate `lower_bound` set to
  `-5`, `CHUNK = 16` keeps the range of `exp(cumsum(g))` within the
  representable precision of bf16. This eliminates the need for the elaborate
  intra-chunk rescaling tricks that larger chunk sizes require.
- **Cheap matrix inversion.** Inverting a `16 × 16` matrix is dramatically
  cheaper than inverting a `64 × 64` one, and the former can be computed
  directly by forward substitution without further decomposition.
- **SM80-only MMA path.** All `CHUNK = 16` math maps cleanly onto SM80 MMA
  instructions. This keeps the kernel simple and makes it portable across a
  wide range of modern NVIDIA GPUs without relying on architecture-specific
  features.

## 2. Kernel Fusion Strategy

We partition the full computation into two kernels along their natural
parallelism axes:

- **K1 (token-parallel, grid = `N × H × num_chunks`):** `g` activation → L2
  normalization → decay application → `L` / `Mqk` construction → matrix
  inversion.
- **K2 (head-parallel only, grid = `N × H`):** chunk-by-chunk delta-rule
  recurrence → output projection → running state accumulation.

Early prototypes of FlashKDA used a single fused kernel. In that design, the
token-parallel work in the K1 stage was bottlenecked by the much lower
parallelism of the recurrence in K2, leaving a large fraction of the SMs
idle. Splitting the pipeline into two kernels yielded at least a **15%**
end-to-end speedup and made each stage independently tunable.

## 3. Numerical Precision

**Forward-substitution matrix inversion.** The `16 × 16` inverse is computed
exactly: the seed `L` stays in fp32 end to end (the `L` GEMM stores its raw
fp32 accumulator and the tril/beta masking is done in fp32), the two diagonal
`8 × 8` blocks are inverted by fp32 forward substitution, and the
off-diagonal block is merged with two bf16-HMMA GEMMs (`dc = P @ M`,
`o = (−dc) @ P`, both fp32-accumulate).

### Accuracy vs. `fla_chunk_kda`

![Accuracy comparison between `fla_chunk_kda` and FlashKDA across various input cases.](assets/compare_with_fla.png)

## 4. Other Optimizations

- **Base-2 exponent.** In the `g_act` stage we rebase the exponent to 2 and
  use `ex2.approx.ftz.f32`. This removes the change-of-base FMA entirely and
  benefits from the higher throughput of `ex2` compared to `exp`.
- **K1 occupancy.** Through aggressive shared-memory reuse (unions over
  lifetimes that do not overlap) and `__launch_bounds__(256, 8)`, we trade a
  small amount of register spilling for a substantial increase in thread
  blocks per SM. The net effect is a significant improvement in K1
  throughput.
- **K2 register-file transposes.** By fusing the stages of K2 tightly and
  using the `MOVM_T` instruction to transpose operands directly inside the
  register file, we eliminate every intermediate shared-memory round trip
  between stages. This also shrinks the shared-memory buffer requirement of
  K2.
