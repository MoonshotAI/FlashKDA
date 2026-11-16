# KDA Optimizations on Ascend 950

The 950 backend uses the `flash_kda_npu.fwd` API and the Pre-Gram → Solve/WU →
persistent K2 pipeline. It adapts matrix layouts, Vector register arithmetic and
Cube/Vector data exchange for `dav-3510`. With the default chunk=64, row-wise
triangular-solve VF, Pre-Gram prefix/row VF and K2 direct C/V are enabled.
See the [950 benchmark](../BENCHMARK_950.md) for latency and FLA comparisons.

## 1. Computation and Optimization Targets

| Stage | Main computation | 950 optimization |
| --- | --- | --- |
| Pre-Gram | Q/K normalization, gate/beta activation, gate prefixes and Gram matrices | Keep prefix sums and row-wise arithmetic in registers; use 950 NZ loading layouts on Cube |
| Solve/WU | Triangular inversion, block coupling and W/U generation | Retain the active solve row in registers; split chunk=64 into two 32×32 diagonal blocks and fuse assembly with W/U |
| Persistent K2 | Sequential state updates and output generation | Retain FP32 state in AIV UB; send BF16 operands directly to AIC L1 and return FP32 products directly to AIV UB |

Pre-Gram and Solve parallelize across chunks and heads. K2 must advance state in
sequence order. Pre-Gram therefore targets Vector instructions and UB traffic,
Solve targets dependent row operations, and K2 targets GM round trips during
Cube/Vector exchange.

## 2. Matrix Layouts and Data Paths for dav-3510

[common.h](../csrc/ascendc950/common.h) implements loading, conversion and output.
The 950 L0A fractal grid is ordered by K, then M, while the shared L1 staging
layout consists of consecutive 16-row NZ bands. `LoadAFromBands` rearranges these
blocks during MTE1 loading while preserving the upstream GM/L1 storage contract.
Complete NZ matrices use `LoadData2DParamsV2`, with transpose and stride settings
chosen for each operand.

```text
Shared GM handoff: Vector UB → GM → Cube L1 → L0A/L0B → L0C → GM → Vector UB
950 K2 direct C/V: Vector UB ─────→ Cube L1 → L0A/L0B → L0C ─────→ Vector UB
```

`UbToL1Nz` uses explicit MTE3 copies to transfer ND operands into L1 NZ in
16-column blocks. `L0cToUb` uses `FixpipeParamsArch3510` to write into the two
AIV receive regions. Cube operands retain their BF16 rounding boundaries;
products return to Vector in FP32. Pre-Gram uses GM handoffs by default, with a
separate switch for direct C/V.

## 3. Pre-Gram: Reducing UB Traffic in Vector Arithmetic

The implementation is in [pre_gram.h](../csrc/ascendc950/pre_gram.h), called by the
shared [Pre-Gram kernel](../csrc/ascendc/fwd_kernel1_pre_gram.cpp).

### Register-resident gate prefixes

`GatePrefixVF` divides the 128 dimensions into two 64-lane halves and processes
eight rows at a time. It preserves the FP32 addition order within each group and
keeps the preceding group's total in registers. This removes reads of prefix
values just written to UB within the same VF invocation. Scaling after the
prefix sum retains its arithmetic boundary.

### Fused row-wise exponentials, products and BF16 conversion

`PublicRowsVF` generates Qg, Rk, Rv and Kg; `GramRowsVF` generates Gram operands.
Each invocation loads a row of gate values, Q/K and beta, performs exponentials,
products and BF16 conversion in registers, then writes the operands directly.

Q/K squaring, row reduction, Rsqrt and normalization rounding remain unchanged.
VF receives Q/K after BF16 normalization rounding and conversion back to FP32.
Subsequent products execute separately, with RINT conversion. For chunk=64, the
correction across the two 32-row bands retains its own arithmetic chain. These
boundaries determine the values used by recurrence.

### Optional Pre-Gram direct C/V

[fwd_kernel1_pre_gram_cv.cpp](../csrc/ascendc950/fwd_kernel1_pre_gram_cv.cpp) uses
two L1 slots so Vector can prepare the next item while Cube reads the preceding
one. READY/DONE handshakes protect operands and results before slot reuse.
FP32 L and BF16 Aqk still go to GM for the separate Solve stage.

This path reduces GM workspace for Gram operands and products, but Vector
operand preparation can limit the stage. Pre-Gram direct C/V is disabled by
default. In both paths, a `V_MTE2` event protects the last Vector read of the
FP32 gate buffer before MTE2 overwrites it.

## 4. Solve/WU: Row-wise VF and Two-level Block Inversion

`SolveRowVF` in [solve.h](../csrc/ascendc950/solve.h) solves one row at a time:

```text
row = identity[i, :]
for j in 0 .. i-1:
    row = MulAddDst(row, previous_solved_row[j], negative_L[i, j])
store(row)
PIPE_V completion
```

VF retains the active row in registers and writes it to UB once the row is
complete. Preceding rows are read from UB in the same accumulation order. A
`PIPE_V` completion point publishes each row before the next row consumes it.
Only the active row, preceding row and coefficient occupy registers, keeping
the register arrays within the VF stack limit.

For chunk=64, Vector solves two 32×32 diagonal blocks and Cube computes the
off-diagonal coupling. The intermediate results retain two BF16 rounding points:

```text
P16 = round_bf16((-L21) @ X11)
X21 = round_bf16(X22 @ P16)
```

By default, `KDA_SOLVE_FUSE=1` combines inverse assembly and W/U generation in the
[fused kernel](../csrc/ascendc/fwd_kernel1_solve_fused.cpp).
[ascendc.py](../ascendc.py) divides tasks into slices and uses stream/event
dependencies to overlap Vector solve with Cube assembly on different slices.
L0A/L0B and UB capacity constrain batch size.

The optional [L1 solve kernel](../csrc/ascendc950/fwd_kernel1_solve_l1.cpp) retains
assembly intermediates through L0C→L1 NZ before continuing W/U, reducing GM
traffic while preserving both BF16 rounding points. It is controlled by
`KDA_950_SOLVE_L1` and disabled by default.

## 5. Persistent K2: Resident State and Direct C/V Exchange

[fwd_kernel2_cv.cpp](../csrc/ascendc950/fwd_kernel2_cv.cpp) advances multiple chunks
within one mixed kernel call. Each group contains one AIC and two AIVs. Each AIV
owns 64 value rows; together they cover all 128 dimensions. FP32 state remains in
UB across chunks, avoiding full-state GM writeback between updates.

A group holds up to four heads, with the host and kernel sharing the
`KDA_PERSIST_LOOP_MAXH` limit. Heads provide independent work to interleave while
each head preserves its recurrence order.

For each chunk:

1. AIV publishes a BF16 state copy to L1; AIC computes state-dependent products.
2. AIC returns FP32 products through L0C→UB. AIV computes the residual and sends
   its BF16 copy to L1.
3. AIC computes output products and state increments, returning them to AIV.
4. AIV combines the output and updates its resident FP32 state.

The direct path omits GM handoff buffers for D1–D4; the launcher ABI retains
their argument slots. `UpdateStateVF` in [state.h](../csrc/ascendc950/state.h)
reuses decay in registers and updates state with separate FP32 Mul and Add
operations.

### Slot ownership and buffer reuse

`STATE_READY`, `PRODUCTS_READY` and `RESID_READY` order state/residual exchange.
`UPDATE_READY` and `UPDATE_FREE` protect the reused product receive region.
Returning a credit requires completion of Vector reads and output DMA.

The kernel checks UB, L1, L0A/L0B and L0C layouts against their capacities:
256 KiB UB, 512 KiB L1, 64 KiB L0A/L0B and 256 KiB L0C. Head count, tile sizes
and aliased storage regions must fit together.

## 6. Shared Packed Path and Runtime Dispatch

[ascendc.py](../ascendc.py) builds sequence starts, lengths, chunk ranges and head
mappings for packed inputs. Kernels read the original tensors through this
metadata. Equal-length, aligned sequences can use a batch view; unequal lengths
and tails use the explicit metadata path. Each sequence has an independent state.

[device_config.py](../device_config.py) queries the active context/device for the
SoC and core counts. The 950 path uses `--npu-arch=dav-3510` and checks the
1 Cube : 2 Vector pairing. Source selection, compilation caches and workspaces
share the same device identity. A3 and 950 share the source package, API and
[source installation command](../README.md#installation). Installation builds
the host launcher against the active Python, CANN and PyTorch environment.
The 950 VF/CV switches apply only to the 950 implementation.

## 7. Defaults and Optional Paths

These switches are read at module import. Start a separate Python process for
each configuration.

| Switch | chunk=64 default | chunk=16/32 default | Purpose |
| --- | ---: | ---: | --- |
| `KDA_950_SOLVE_VF` | 1 | 1 | Row-wise triangular-solve VF |
| `KDA_950_DIRECT_CV` | 1 | 0 | Direct Cube/Vector exchange in K2 |
| `KDA_950_PREFIX_VF` | 1 | 0 | Register-resident gate prefixes |
| `KDA_950_ROW_VF` | 1 | 0 | Fused row-wise Pre-Gram arithmetic |
| `KDA_SOLVE_FUSE` | 1 | N/A | Fused inverse assembly and W/U |
| `KDA_950_PRE_CV` | 0 | 0 | Optional direct Pre-Gram exchange |
| `KDA_950_SOLVE_L1` | 0 | N/A | Optional L1 assembly and W/U |
| `KDA_950_SOLVE_OFFSETS` | 0 | N/A | Offset parameters for SOLVE_L1 |
| `KDA_950_RESIDUAL_VF` | 0 | 0 | Fused residual-to-NZ conversion; requires DIRECT_CV |
| `KDA_950_STATE_PACK_VF` | 0 | 0 | Fused state update and BF16 NZ publication; requires DIRECT_CV |

`KDA_CHUNK` supports 16, 32 and 64, with 64 as the default. Smaller chunks
reduce work within each block for short sequences but increase recurrence
iterations for long sequences. Each chunk size also has its own default
optimization combination.
