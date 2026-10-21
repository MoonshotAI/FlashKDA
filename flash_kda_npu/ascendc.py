"""AscendC (RTC-compiled) KDA v1 forward."""
from __future__ import annotations

import heapq
import math
import os
import struct
import time
from functools import lru_cache, wraps
from pathlib import Path

import torch
import torch_npu

from flash_kda_npu_C import launch_argsarray_engine, rtc_compile, runtime_key

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

# Fix geometry at import so host allocations and RTC kernels use matching sizes.
CHUNK = int(os.environ.get("KDA_CHUNK", "64"))
SUPPORTED_CHUNKS = frozenset({16, 32, 64})
# Limit batched sub-blocks to keep wide-solve staging within the 192 KiB UB.
SOLVE_WIDE_NCHUNK = (int(os.environ.get("KDA_SOLVE_WIDE_NCHUNK", "0"))
                     or (32 if CHUNK <= 16 else (16 if CHUNK <= 32 else 12)))
# Split large triangles to reduce vector work and offload coupling to Cube.
SOLVE_WIDE_SUBB = 2 if CHUNK >= 64 else 1
SOLVE_WIDE_NCH = SOLVE_WIDE_NCHUNK // SOLVE_WIDE_SUBB
ASM_NCHUNK = int(os.environ.get("KDA_ASM_NCHUNK", "0")) or 4
SOLVE_OVERLAP = int(os.environ.get("KDA_SOLVE_OVERLAP", "24"))
SOLVE_SLICE_TARGET = int(os.environ.get("KDA_SOLVE_SLICE_TARGET", "20"))
# Cap resident heads so FP32 states and aliased staging fit in the 192 KiB UB.
# Keep the cap consistent with the compiled kernel to prevent buffer overruns.
PERSIST_MAXH = max(1, min(4 if CHUNK <= 64 else 2,
                          int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0"))
                          or (4 if CHUNK <= 64 else 2)))
# Bound Cube batching by the 128 KiB L0C capacity.
WU_NCHUNK = int(os.environ.get("KDA_WU_NCHUNK", "0")) or (4 if CHUNK <= 32 else 2)
SOLVE_FUSE = SOLVE_WIDE_SUBB > 1 and os.environ.get("KDA_SOLVE_FUSE", "1") == "1"
D = 128
BV = 64
NV = 2
_COMPILED: set[tuple[int, int]] = set()
_LAST_PROFILE: dict[str, object] = {}
# Cache invariant masks to avoid rebuilding them for each forward call.
_GRAM_MASKS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Cache identity tiles to avoid repeated allocation and initialization.
_EYE_TILES: dict[tuple, torch.Tensor] = {}
_LAUNCH_COUNTS: dict[str, int] = {}
_LAUNCH_BLOCKS: dict[str, int] = {}


def _tri_eye(device: torch.device, n: int | None = None) -> torch.Tensor:
    """Identity tile for the K1 solve: A_inv starts from it.

    The wide kernel reads it as one dense n x n block, so the two-level path
    wants the M x M sub-block identity, not the CHUNK x CHUNK one.
    """
    n = CHUNK if n is None else n
    key = (device, n)
    eye = _EYE_TILES.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=torch.float32).contiguous()
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls, so publish the tile before any kernel reads it.
        torch.npu.synchronize()
        _EYE_TILES[key] = eye
    return eye


# Cache device properties to avoid querying hardware on every launch.
_CORE_COUNTS: dict = {}


def _core_count(device, prop: str, env: str) -> int:
    entry = _CORE_COUNTS.setdefault(str(device), {})
    if prop not in entry:
        n = int(os.environ.get(env, "0") or 0)
        if not n:
            try:
                idx = device.index
                if idx is None:
                    idx = torch_npu.npu.current_device()
                props = torch_npu.npu.get_device_properties(idx)
                n = int(getattr(props, prop, 0) or 0)
            except Exception:
                n = 0
        entry[prop] = n
    return entry[prop]


def _aic_core_count(device) -> int:
    return _core_count(device, "cube_core_num", "KDA_AIC_CORES") or 20


def _aiv_core_count(device) -> int:
    n = _core_count(device, "vector_core_num", "KDA_AIV_CORES")
    return n if n else 2 * _aic_core_count(device)


def _solve_padded_chunks(c: int) -> int:
    """Round chunk/head work items to whole wide/assembly/Cube launch groups."""
    unit = math.lcm(SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK)
    return (c + unit - 1) // unit * unit


def _solve_slice_chunks(c_solve: int, unit: int, nch: int, aiv_cores: int,
                        target: int) -> int:
    """Two-level solve slice size: a whole number of AIV waves.

    One wave is `aiv_cores * nch` chunks, lcm'd with the wiring unit so a
    slice boundary is still a whole (wide / assemble / Cube) group.  The size
    is the multiple closest to `c_solve / target`; a smaller remainder slice
    carries the tail.
    """
    wave = math.lcm(unit, aiv_cores * nch)
    k = max(1, int(round(c_solve / float(target) / wave)))
    return k * wave


def _pre_gram_unroll(c: int, cores: int) -> int:
    """Balance chunk work and block startup across whole Cube waves.

    c counts (sequence chunk, head) work items, not tokens. Each mixed block
    processes two items per unroll step; cores counts physical Cube cores.
    The search limit and one-step startup charge are empirical heuristics.
    """
    if c < 256:
        return 1
    best_u, best_cost = 1, None
    for u in range(88, 0, -1):
        grid = -(-c // (2 * u))
        waves = -(-grid // cores)
        steps = waves * u
        # A wave also pays block initialization and pipeline fill/drain.
        # Approximate that fixed cost by one chunk step: minimizing steps
        # alone can replace a few long-lived blocks with thousands of tiny
        # blocks for a negligible arithmetic saving. Keep fewer steps on
        # equal cost so startup savings do not add needless padded work.
        cost = (steps + waves, steps)
        if best_cost is None or cost < best_cost:
            best_cost, best_u = cost, u
    return best_u


@lru_cache(maxsize=128)
def _persistent_varlen_blocks(counts: tuple[int, ...], heads: int, cores: int,
                              max_heads: int, default: int) -> int:
    """Estimate the existing stride map, including its single-head tails.

    A head alone loses the depth-one protocol's engine overlap. Charge its
    unmatched tail twice, then simulate blocks retiring on the available
    Cubes. Fixed/equal-length batches retain the established wave picker.
    Only compare a bounded set of head counts and nearby whole-wave grids.
    counts contains chunks per sequence; heads is heads per sequence; nb is
    the proposed block count. Costs are relative chunk steps, not milliseconds.
    """
    if len(set(counts)) <= 1:
        return default
    bh = len(counts) * heads
    lower = (bh + max_heads - 1) // max_heads
    candidates = {default}
    for mh in range(2, max_heads + 1):
        nb = (bh + mh - 1) // mh
        candidates.update((nb, nb // cores * cores, (nb + cores - 1) // cores * cores))

    def cost(nb):
        slots = [0] * cores
        # The stride map changes only where a sequence boundary crosses a
        # block index. Evaluate each constant run once rather than sorting
        # the same head lengths separately for every block.
        boundaries = sorted({0, nb, *(seq * heads % nb
                                    for seq in range(1, len(counts) + 1))})
        for start, end in zip(boundaries, boundaries[1:]):
            work = sorted((counts[i // heads] for i in range(start, bh, nb)),
                          reverse=True)
            tail = work[0] - (work[1] if len(work) > 1 else 0)
            weight = sum(work) + tail
            remaining = end - start
            if weight == 0:
                continue
            # Once all cores are within one job of each other, a complete
            # wave adds the same duration to every core.
            while remaining and max(slots) - slots[0] > weight:
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
                remaining -= 1
            waves, remaining = divmod(remaining, cores)
            if waves:
                slots = [ready + waves * weight for ready in slots]
            for _ in range(remaining):
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
        return max(slots)

    return min((nb for nb in candidates if lower <= nb <= bh),
               key=lambda nb: (cost(nb), nb != default, nb))


@lru_cache(maxsize=128)
def _persistent_fixed_blocks(bh: int, cores: int) -> int:
    """Compare stride grids using per-block retirement, preserving tied choices.

    bh is batch times heads. Equal recurrence lengths cancel from comparisons,
    so each head contributes one work unit, plus a single-head overlap penalty.
    """
    if bh <= cores or PERSIST_MAXH == 1:
        return bh
    best = None
    candidates = set()
    for mh in range(2, PERSIST_MAXH + 1):
        nb = (bh + mh - 1) // mh
        steps = -(-nb // cores) * mh
        if best is None or steps <= best[0]:
            best = (steps, nb)
        candidates.update((nb, nb // cores * cores, -(-nb // cores) * cores))
    default = best[1]

    def cost(nb):
        # Stride mapping puts the heavier blocks first. A lone head loses
        # AIC/AIV overlap, just as in the variable-length estimate.
        nh, heavy = divmod(bh, nb)
        slots = [0] * cores
        for count, heads in ((heavy, nh + 1), (nb - heavy, nh)):
            weight = heads + (heads == 1)
            while count and max(slots) - slots[0] > weight:
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
                count -= 1
            waves, tail = divmod(count, cores)
            if waves:
                slots = [ready + waves * weight for ready in slots]
            for _ in range(tail):
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
        return max(slots)

    lower = (bh + PERSIST_MAXH - 1) // PERSIST_MAXH
    return min((nb for nb in candidates if lower <= nb <= bh),
               key=lambda nb: (cost(nb), nb != default, nb))


@lru_cache(maxsize=128)
def _persistent_grouped_plan(counts: tuple[int, ...], heads: int, cores: int,
                             cap: int, default: int, override: int = 0,
                             force: bool = False):
    """Group similar recurrence lengths and launch long blocks first.

    Estimate the existing depth-one protocol's single-head tails and block
    startup on a greedy Cube schedule. Require a useful margin before paying
    for an explicit map. The returned rows append to the sequence metadata,
    so the schedule needs no additional allocation or device upload.
    """
    if len(set(counts)) <= 1 and not force:
        return default, ()
    bh = len(counts) * heads
    sequence_order = sorted(range(len(counts)), key=lambda seq: -counts[seq])
    order = [h for seq in sequence_order for h in range(seq * heads, (seq + 1) * heads)]
    ordered_work = [counts[h // heads] for h in order]
    prefix = [0]
    for work in ordered_work:
        prefix.append(prefix[-1] + work)

    def grouped_score(nb):
        # Prefix sums avoid rebuilding/sorting every candidate's head groups.
        slots = [0] * cores
        width, extra = divmod(bh, nb)
        start = 0
        for b in range(nb):
            size = width + (b < extra)
            end = start + size
            tail = ordered_work[start] - (ordered_work[start + 1] if size > 1 else 0)
            weight = 1 + prefix[end] - prefix[start] + tail
            ready = heapq.heappop(slots)
            heapq.heappush(slots, ready + weight)
            start = end
        return max(slots)

    slots = [0] * cores
    for b in range(default):
        work = sorted((counts[h // heads] for h in range(b, bh, default)), reverse=True)
        weight = 1 + sum(work) + work[0] - (work[1] if len(work) > 1 else 0)
        ready = heapq.heappop(slots)
        heapq.heappush(slots, ready + weight)
    baseline = max(slots)
    candidates = ({override} if override else
                  {default, *((bh + nh - 1) // nh for nh in range(2, cap + 1))})
    scores = {nb: grouped_score(nb) for nb in candidates}
    nb = min(candidates, key=lambda n: (scores[n], n))
    gain = baseline - scores[nb]
    # Empirical guard: require both 16 work units and a 5% modeled saving.
    # This accounts for explicit-map overhead; it is not a timing guarantee.
    if not force and gain < max(16, baseline / 20):
        return default, ()
    width, extra = divmod(bh, nb)
    start = 0
    rows = []
    for b in range(nb):
        size = width + (b < extra)
        rows.append(tuple(order[start:start + size] + [-1] * (4 - size)))
        start += size
    return nb, tuple(rows)


def _pack_ptrs(xs):
    return [struct.pack("<Q", 0 if x is None else int(x.data_ptr())) for x in xs]


def _pack_int32(x: int) -> bytes:
    return struct.pack("<i", int(x))


def _pack_float32(x: float) -> bytes:
    return struct.pack("<f", float(x))


def _launch(name: str, blocks: int, args: list[bytes], stream):
    _LAUNCH_COUNTS[name] = _LAUNCH_COUNTS.get(name, 0) + 1
    _LAUNCH_BLOCKS[name] = _LAUNCH_BLOCKS.get(name, 0) + int(blocks)
    launch_argsarray_engine(name, int(blocks), stream, args, 0)


# Reuse per-device streams to overlap vector and Cube work
# without stream creation overhead.
_SOLVE_STREAMS: dict = {}


def _solve_streams(device: torch.device):
    key = (device, runtime_key())
    pair = _SOLVE_STREAMS.get(key)
    if pair is None:
        pair = (torch_npu.npu.Stream(device=device),
                torch_npu.npu.Stream(device=device))
        _SOLVE_STREAMS[key] = pair
    return pair


def _launch_solve_two_level(c_solve, c, nch, asm_nchunk, wu_nchunk, overlap, L, eye,
                            a16, lneg, rk, rv, W, U, stream) -> None:
    """Two-level solve: wide substitution (AIV), then assembly and W/U (AIC).

    Slices contain whole launch groups. Multiple slices overlap on separate
    streams with per-slice events; overlap <= 0 uses the caller's stream.
    c_solve includes padding, while c counts real chunk/head work items.
    rk/rv contain only real items; W/U have padded storage but K2 reads c items.
    """
    chunks_per_group = math.lcm(nch, asm_nchunk, wu_nchunk)
    group_count = c_solve // chunks_per_group
    # Default slices span whole AIV waves. Cap their count at
    # max(1, group_count // 8) to amortize block startup.
    slice_chunks_override = int(os.environ.get("KDA_SOLVE_SLICE_CHUNKS", "0"))
    if overlap <= 0:
        group_ranges = [(0, group_count)]
    elif slice_chunks_override < 0:
        slice_count = min(overlap, max(1, group_count // 8))
        group_ranges = [(i * group_count // slice_count, (i + 1) * group_count // slice_count)
                        for i in range(slice_count)]
    else:
        if slice_chunks_override > 0:
            groups_per_slice = max(1, min(group_count, slice_chunks_override // chunks_per_group))
        else:
            groups_per_slice = max(1, min(group_count, _solve_slice_chunks(
                c_solve, chunks_per_group, nch, _aiv_core_count(a16.device),
                SOLVE_SLICE_TARGET) // chunks_per_group))
        slice_count = -(-group_count // groups_per_slice)
        group_ranges = [(i * groups_per_slice, min((i + 1) * groups_per_slice, group_count))
                        for i in range(slice_count)]
        if slice_count > max(1, group_count // 8):
            slice_count = max(1, group_count // 8)
            group_ranges = [(i * group_count // slice_count, (i + 1) * group_count // slice_count)
                            for i in range(slice_count)]
    use_overlap = len(group_ranges) >= 2
    caller_stream = torch_npu.npu.current_stream()
    if use_overlap:
        vector_stream, cube_stream = _solve_streams(a16.device)
        # Wait for the caller's Gram before vector work, and join Cube results
        # before recurrence so separate streams preserve data dependencies.
        vector_stream.wait_stream(caller_stream)
    for group_start, group_end in group_ranges:
        chunk_start, slice_chunks = (
            group_start * chunks_per_group, (group_end - group_start) * chunks_per_group)
        wide_args = _pack_ptrs([
            L[chunk_start:], eye, a16[chunk_start:], lneg[chunk_start:],
        ]) + [_pack_int32(slice_chunks)]
        real_slice_chunks = min(slice_chunks, c - chunk_start)
        if SOLVE_FUSE:
            assembly_kernel = "kda_solve_fused"
            assembly_args = _pack_ptrs([
                a16[chunk_start:], lneg[chunk_start:], rk[chunk_start:],
                rv[chunk_start:], W[chunk_start:], U[chunk_start:],
            ]) + [_pack_int32(real_slice_chunks)]
        else:
            assembly_kernel = "kda_solve_assemble"
            assembly_args = _pack_ptrs([
                a16[chunk_start:], lneg[chunk_start:],
            ]) + [_pack_int32(slice_chunks)]
            cube_args = _pack_ptrs([
                a16[chunk_start:], rk[chunk_start:], rv[chunk_start:],
                W[chunk_start:], U[chunk_start:],
            ]) + [_pack_int32(slice_chunks)]
        assembly_blocks = ((real_slice_chunks + wu_nchunk - 1) // wu_nchunk if SOLVE_FUSE
                           else (slice_chunks + asm_nchunk - 1) // asm_nchunk)
        if use_overlap:
            _launch("kda_solve_wu_wide", slice_chunks // nch, wide_args, vector_stream.npu_stream)
            wide_done = torch_npu.npu.Event()
            wide_done.record(vector_stream)
            cube_stream.wait_event(wide_done)
            _launch(assembly_kernel, assembly_blocks,
                    assembly_args, cube_stream.npu_stream)
            if real_slice_chunks > 0 and not SOLVE_FUSE:
                _launch("kda_solve_wu_cube_kernel",
                        (real_slice_chunks + wu_nchunk - 1) // wu_nchunk,
                        cube_args, cube_stream.npu_stream)
        else:
            _launch("kda_solve_wu_wide", slice_chunks // nch, wide_args, stream)
            _launch(assembly_kernel, assembly_blocks,
                    assembly_args, stream)
            if real_slice_chunks > 0 and not SOLVE_FUSE:
                _launch("kda_solve_wu_cube_kernel",
                        (real_slice_chunks + wu_nchunk - 1) // wu_nchunk, cube_args, stream)
    if use_overlap:
        caller_stream.wait_stream(cube_stream)


def _defines() -> str:
    """Override kernel defaults so RTC geometry matches host buffer layouts."""
    return ("#define KDA_CHUNK %d\n#define KDA_SOLVE_WIDE_NCHUNK %d\n"
            "#define KDA_SOLVE_WIDE_SUBB %d\n#define KDA_ASM_NCHUNK %d\n"
            "#define KDA_WU_NCHUNK %d\n#define KDA_MAXH %d\n"
            % (CHUNK, SOLVE_WIDE_NCHUNK, SOLVE_WIDE_SUBB, ASM_NCHUNK,
               WU_NCHUNK, PERSIST_MAXH))


def compile_config() -> dict[str, int]:
    """Report effective compile geometry and solve defaults alongside timings."""
    return {
        "KDA_CHUNK": CHUNK,
        "KDA_MAXH": PERSIST_MAXH,
        "KDA_SOLVE_WIDE_NCHUNK": SOLVE_WIDE_NCHUNK,
        "KDA_SOLVE_WIDE_SUBB": SOLVE_WIDE_SUBB,
        "KDA_ASM_NCHUNK": WU_NCHUNK if SOLVE_FUSE else ASM_NCHUNK,
        "KDA_WU_NCHUNK": WU_NCHUNK,
        "KDA_SOLVE_OVERLAP": SOLVE_OVERLAP,
        "KDA_SOLVE_FUSE": int(SOLVE_FUSE),
        "nv": NV,
    }


def _rtc(rel: str, name: str, sizes: list[int]) -> None:
    """Compile a repository-relative source with host geometry and no UTF-8 BOM."""
    prefix = _defines()
    if name == "kda_solve_fused":
        prefix += ("#define KDA_SOLVE_FUSED 1\n#undef KDA_ASM_NCHUNK\n"
                   "#define KDA_ASM_NCHUNK KDA_WU_NCHUNK\n")
        sources = ("fwd_kernel1_solve_assemble.cpp", "fwd_kernel1_solve_cube.cpp",
                   "fwd_kernel1_solve_fused.cpp")
        source = "\n".join((PACKAGE_ROOT / "csrc/ascendc" / path).read_text(
            encoding="utf-8-sig") for path in sources)
    else:
        source = (ROOT / rel).read_text(encoding="utf-8-sig")
    rtc_compile(prefix + source, name, sizes)


_SOURCES: list[tuple[str, str, list[int]]] = [
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_pre_gram.cpp", "kda_pre_gram_mix", [8] * 23 + [4] * 9),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_wide.cpp", "kda_solve_wu_wide", [8] * 4 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_assemble.cpp", "kda_solve_assemble", [8] * 2 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_cube.cpp", "kda_solve_wu_cube_kernel", [8] * 5 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel2_persistent.cpp", "kda_k2_persistent_loop", [8] * 17 + [4] * 8),
]
if SOLVE_FUSE:
    _SOURCES = [entry for entry in _SOURCES if entry[1] not in (
        "kda_solve_assemble", "kda_solve_wu_cube_kernel")]
    _SOURCES.append(("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_fused.cpp",
                     "kda_solve_fused", [8] * 6 + [4]))


def _compile_all() -> None:
    key = runtime_key()
    if key in _COMPILED:
        return
    for rel, name, sizes in _SOURCES:
        _rtc(rel, name, sizes)
    _COMPILED.add(key)


def _tri_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = _GRAM_MASKS.get(device)
    if masks is None:
        idx = torch.arange(CHUNK, device=device)
        masks = ((idx[None, :] <= idx[:, None]).to(torch.float32).contiguous(),
                 (idx[None, :] < idx[:, None]).to(torch.float32).contiguous())
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls made by the launcher, so make sure the mask
        # tensors are on device before any kernel can read them.
        torch.npu.synchronize()
        _GRAM_MASKS[device] = masks
    return masks


def _check_inputs(q, k, v, g, beta, A_log, bias, initial_state, state_sequences=None):
    if q.device.type != "npu":
        raise ValueError("AscendC v1 requires NPU tensors")
    if any(x is not None and x.device != q.device
           for x in (k, v, g, beta, A_log, bias, initial_state)):
        raise ValueError("all inputs must be on q's NPU device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q/k/v must be BF16")
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape) or tuple(q.shape) != tuple(v.shape):
        raise ValueError("q/k/v must have identical [B,T,H,128] shape")
    b, t, h, d = q.shape
    if b < 1 or h < 1 or t < 1 or d != D:
        raise ValueError("support is B,H,T > 0, D=128")
    if g.shape != q.shape or g.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("g must be BF16 or FP32 [B,T,H,128]")
    if beta.shape != (b, t, h) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B,T,H]")
    if A_log.shape != (h,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if bias is not None and (bias.shape != (h, D) or bias.dtype != torch.float32):
        raise ValueError("bias must be FP32 [H,128]")
    sequences = b if state_sequences is None else state_sequences
    if initial_state is not None and (initial_state.shape != (sequences, h, D, D) or initial_state.dtype != torch.float32):
        raise ValueError("initial_state must be FP32 [N,H,128,128]")
    return b, t, h


def get_last_profile() -> dict[str, object]:
    profile = dict(_LAST_PROFILE)
    profile["launch_counts"] = dict(_LAUNCH_COUNTS)
    profile["launch_blocks"] = dict(_LAUNCH_BLOCKS)
    profile["launch_total"] = sum(_LAUNCH_COUNTS.values())
    # Record compile geometry because RTC kernels do not expose their defines;
    # performance measurements are only comparable with that context.
    profile["compile"] = compile_config()
    return profile


def _merge_profiles(profiles, real_tokens, padded_tokens):
    """Publish all native groups belonging to one packed public call."""
    global _LAST_PROFILE, _LAUNCH_COUNTS, _LAUNCH_BLOCKS
    _LAST_PROFILE = dict(sequence_groups=len(profiles), real_tokens=real_tokens,
                         padded_tokens=padded_tokens, input_packing=bool(profiles))
    _LAUNCH_COUNTS, _LAUNCH_BLOCKS = {}, {}
    for profile in profiles:
        for key, value in profile.items():
            if key.endswith("_ms"):
                _LAST_PROFILE[key] = _LAST_PROFILE.get(key, 0.0) + value
        for key, value in profile["launch_counts"].items():
            _LAUNCH_COUNTS[key] = _LAUNCH_COUNTS.get(key, 0) + value
        for key, value in profile["launch_blocks"].items():
            _LAUNCH_BLOCKS[key] = _LAUNCH_BLOCKS.get(key, 0) + value


def _on_input_device(fn):
    @wraps(fn)
    def run(q, *args, **kwargs):
        if q.device.type == "npu" and q.device.index != torch_npu.npu.current_device():
            with torch_npu.npu.device(q.device):
                return fn(q, *args, **kwargs)
        return fn(q, *args, **kwargs)
    return run


@_on_input_device
def kda_fwd_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    A_log: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    valid_lengths: torch.Tensor | None = None,
    sequence_offsets: tuple[int, ...] | None = None,
):
    """Run native AscendC KDA with the configured chunk size.

    ``q/k/v`` are BF16 ``[B, T, H, 128]``, ``g`` is BF16/FP32 ``[B, T, H, 128]``,
    ``beta`` and ``A_log``/``bias`` are FP32; returns ``(out, final_state)``.
    Any positive fixed sequence length is supported, including partial chunks.
    ``valid_lengths`` is internal packing metadata: counts in [0, T], with
    zero Q/K/V beyond each count. ``sequence_offsets`` selects direct packed
    input/output with compact per-sequence chunks; boundaries have already been
    copied to the host for public validation, never Q/K/V/g themselves.
    """
    if CHUNK not in SUPPORTED_CHUNKS and os.environ.get(
            "KDA_ALLOW_UNSUPPORTED_CHUNK", "0") != "1":
        raise ValueError(
            f"KDA_CHUNK={CHUNK} is an unsupported build; supported sizes are "
            "16, 32 and 64. Set KDA_ALLOW_UNSUPPORTED_CHUNK=1 for debugging only; "
            "other sizes may exceed device memory limits or produce incorrect results.")
    global _LAST_PROFILE
    global _LAUNCH_COUNTS, _LAUNCH_BLOCKS
    _LAUNCH_COUNTS = {}
    _LAUNCH_BLOCKS = {}
    _LAST_PROFILE = {}
    profile = os.environ.get("KDA_PROFILE", "0") == "1"
    prof: dict[str, float] = {}

    def mark(name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = time.perf_counter()

    def finish(name: str, start_name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = (time.perf_counter() - prof[start_name]) * 1e3

    state_sequences = None
    if sequence_offsets is not None:
        if (len(sequence_offsets) < 2 or sequence_offsets[0] != 0
                or sequence_offsets[-1] != q.shape[1]
                or any(a > b for a, b in zip(sequence_offsets, sequence_offsets[1:]))
                or q.shape[0] != 1 or valid_lengths is not None):
            raise ValueError("invalid packed sequence offsets")
        state_sequences = len(sequence_offsets) - 1
    batch, steps, heads = _check_inputs(
        q, k, v, g, beta, A_log, bias, initial_state, state_sequences)
    sequences = batch if state_sequences is None else state_sequences
    if valid_lengths is not None and (
            valid_lengths.shape != (batch,) or valid_lengths.dtype != torch.int32
            or valid_lengths.device != q.device or not valid_lengths.is_contiguous()):
        raise ValueError("valid_lengths must be contiguous INT32 [B] on q's NPU device")
    if out is not None and (
            out.shape != q.shape or out.dtype != torch.bfloat16
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16 with q/k/v's shape")
    if final_state is not None:
        if not output_final_state:
            raise ValueError("final_state requires output_final_state=True")
        if (final_state.shape != (sequences, heads, D, D)
                or final_state.dtype not in (torch.float32, torch.bfloat16)
                or final_state.device != q.device or not final_state.is_contiguous()):
            raise ValueError(
                "final_state must be contiguous FP32 or BF16 [B,H,128,128]")
    if scale is None:
        scale = D ** -0.5
    max_chunks = (steps + CHUNK - 1) // CHUNK
    sequence_meta = None
    head_map = None
    total_chunks = batch * max_chunks
    sequence_chunks = None
    if sequence_offsets is not None:
        # Upload sequence offsets and optional head mapping together to avoid
        # separate metadata transfers for grouped scheduling.
        records = []
        total_chunks = 0
        max_chunks = 0
        for start, end in zip(sequence_offsets, sequence_offsets[1:]):
            count = (end - start + CHUNK - 1) // CHUNK
            records.append((start, end - start, total_chunks, count))
            total_chunks += count
            max_chunks = max(max_chunks, count)
        records.append((steps, 0, total_chunks, 0))
        sequence_chunks = tuple(record[3] for record in records[:-1])

    sequence_heads = sequences * heads
    aic_cores = _aic_core_count(q.device)
    persistent_blocks = _persistent_fixed_blocks(sequence_heads, aic_cores)
    if sequence_chunks is not None:
        persistent_blocks = _persistent_varlen_blocks(
            sequence_chunks, heads, aic_cores, PERSIST_MAXH, persistent_blocks)
    requested_blocks = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
    override = requested_blocks if 0 < requested_blocks <= sequence_heads else 0
    if override:
        override = max(override, (sequence_heads + PERSIST_MAXH - 1) // PERSIST_MAXH)
        persistent_blocks = override
    # Respect the compiled head cap to prevent per-block state buffer overruns.
    persistent_blocks = max(persistent_blocks, (sequence_heads + PERSIST_MAXH - 1) // PERSIST_MAXH)

    if sequence_offsets is not None:
        mapping_rows = ()
        mode = os.environ.get("KDA_PERSIST_HEAD_MAP", "auto")
        if mode not in ("auto", "stride", "grouped"):
            raise ValueError("KDA_PERSIST_HEAD_MAP must be auto, stride or grouped")
        if mode != "stride":
            persistent_blocks, mapping_rows = _persistent_grouped_plan(
                sequence_chunks, heads, aic_cores, PERSIST_MAXH,
                persistent_blocks, override, mode == "grouped")
        sequence_meta = torch.tensor(records + list(mapping_rows), dtype=torch.int64, device=q.device)
        if mapping_rows:
            head_map = sequence_meta[sequences + 1:]
    chunk_heads = total_chunks * heads
    state_tiles = sequence_heads * NV
    q, k, v, g, beta, A_log = [x.contiguous() for x in (q, k, v, g, beta, A_log)]
    if bias is not None:
        bias = bias.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    # Read strided inputs directly to avoid full-size packing copies; the
    # kernel uses byte-strided DataCopyPad for the supported GM gather.
    qk_row_bytes = heads * D * 2 - D * 2
    g_row_bytes = (heads - 1) * D * g.element_size()
    if sequence_meta is not None:
        beta_pack = beta
    elif steps % CHUNK == 0:
        beta_pack = (beta.view(batch, max_chunks, CHUNK, heads)
                     .permute(0, 3, 1, 2).contiguous().view(chunk_heads, CHUNK))
    else:
        # Beta already needs a head-major copy. Pad only this small scalar
        # buffer; pre-Gram reads Q/K/V/g directly, padding their tails in UB.
        beta_pack = torch.empty((batch, heads, max_chunks * CHUNK), dtype=beta.dtype, device=q.device)
        beta_pack[:, :, :steps].copy_(beta.permute(0, 2, 1))
        beta_pack[:, :, steps:].zero_()
        beta_pack = beta_pack.view(chunk_heads, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    decay = torch.empty((chunk_heads, D), dtype=torch.float32, device=q.device)
    rk = torch.empty((chunk_heads, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    rv = torch.empty_like(rk)
    qg = torch.empty_like(rk)
    kg = torch.empty_like(rk)
    aqk32 = torch.empty((chunk_heads, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    aqk16 = torch.empty((chunk_heads, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    # Pad L with zeros so extra solve tiles become identity without a tail kernel.
    solve_chunk_heads = _solve_padded_chunks(chunk_heads)
    L = torch.empty((solve_chunk_heads, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    if solve_chunk_heads != chunk_heads:
        L[chunk_heads:].zero_()
    mask_s, mask_l = _tri_masks(q.device)
    pre_unroll = (int(os.environ.get("KDA_PRE_UNROLL", "0"))
                  or _pre_gram_unroll(chunk_heads, _aic_core_count(q.device)))
    mark("pre_gram_start")
    gram_ops = torch.empty((3, chunk_heads, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    # Recenter cross-band keys so both Gram operands share a reference
    # without exceeding the exponential range.
    gram_x = torch.empty((chunk_heads, max(1, CHUNK // 2), D), dtype=torch.bfloat16,
                         device=q.device)
    pre_args = _pack_ptrs([
        q, k, v, g, beta_pack, A_log, bias,
        decay, rk, rv, qg, kg,
        gram_ops[0], gram_ops[1], gram_ops[2], gram_x,
        aqk32, aqk16, L, mask_s, mask_l,
        valid_lengths, sequence_meta,
    ])
    pre_args += [
        _pack_int32(sequences), _pack_int32(steps), _pack_int32(heads),
        _pack_float32(lower_bound), _pack_float32(scale), _pack_int32(pre_unroll),
        _pack_int32(qk_row_bytes), _pack_int32(g_row_bytes),
        _pack_int32(g.dtype == torch.bfloat16),
    ]
    _launch("kda_pre_gram_mix", (chunk_heads + 2 * pre_unroll - 1) // (2 * pre_unroll),
            pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a16 = torch.empty((solve_chunk_heads, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty((solve_chunk_heads, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    # U is consumed by vector subtraction, so retain its FP32 Cube sum.
    U = torch.empty(W.shape, dtype=torch.float32, device=q.device)
    mark("solve_start")
    if SOLVE_WIDE_SUBB > 1:
        sub = CHUNK // SOLVE_WIDE_SUBB
        bf16 = torch.bfloat16
        lneg = torch.empty((solve_chunk_heads, sub * sub), dtype=bf16, device=q.device)
        _launch_solve_two_level(
            solve_chunk_heads, chunk_heads, SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK,
            SOLVE_OVERLAP, L, _tri_eye(q.device, CHUNK // SOLVE_WIDE_SUBB),
            a16, lneg, rk, rv, W, U, stream)
    else:
        # Retain the unused coupling argument so single-level builds share the same ABI.
        solve_args = _pack_ptrs([L, _tri_eye(q.device), a16, None])
        solve_args += [_pack_int32(chunk_heads)]
        _launch("kda_solve_wu_wide", solve_chunk_heads // SOLVE_WIDE_NCHUNK, solve_args, stream)
        # The Cube needs the bf16 A_inv the substitution just wrote, so the two
        # launches stay ordered on the stream.
        _launch("kda_solve_wu_cube_kernel", (chunk_heads + WU_NCHUNK - 1) // WU_NCHUNK,
                _pack_ptrs([a16, rk, rv, W, U]) + [_pack_int32(chunk_heads)], stream)
    finish("solve_ms", "solve_start")

    # Keep FP32 state resident to avoid per-chunk state transfers and rounding.
    state_output = None
    if output_final_state:
        state_output = (final_state.view(state_tiles, BV, D) if final_state is not None
                        else torch.empty((state_tiles, BV, D), dtype=torch.float32, device=q.device))
    s16 = torch.empty((state_tiles, BV, D), dtype=torch.bfloat16, device=q.device)
    # FP32 handoffs preserve cancellation accuracy. Flags guard slot reuse,
    # so each head needs one chunk's storage regardless of sequence length.
    d1 = torch.empty((state_tiles, CHUNK, BV), dtype=torch.float32, device=q.device)
    d2 = torch.empty_like(d1)
    d3 = torch.empty_like(d1)
    d4f = torch.empty((sequence_heads, D, D), dtype=torch.float32, device=q.device)
    # The loop writes the public [B, T, H, D] layout directly with a
    # strided store, so no host-side permute is needed.
    out_public = out if out is not None else torch.empty(
        (batch, steps, heads, D), dtype=torch.bfloat16, device=q.device)
    vnew_t = torch.empty((state_tiles, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
    h0 = None if initial_state is None else initial_state.view(sequence_heads, D, D)
    # Transpose kg fractals during L0B loading to avoid a separate layout copy.
    mark("k2_start")
    _launch("kda_k2_persistent_loop", persistent_blocks,
            _pack_ptrs([U, W, qg, aqk16, kg, decay, d1, d2, d3, d4f,
                        out_public, vnew_t, h0, state_output, s16, sequence_meta, head_map]) +
            [_pack_int32(sequence_heads), _pack_int32(max_chunks), _pack_int32(NV),
             _pack_int32(persistent_blocks), _pack_float32(scale), _pack_int32(heads),
             _pack_int32(state_output is not None and state_output.dtype == torch.bfloat16),
             _pack_int32(steps)], stream)
    finish("k2_ms", "k2_start")
    final_state = None if not output_final_state else (
        final_state if final_state is not None else
        state_output.view(sequence_heads, NV, BV, D).reshape(sequences, heads, D, D))
    if profile:
        prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
        _LAST_PROFILE = prof
    _LAST_PROFILE.update(sequence_groups=1, real_tokens=batch * steps,
                         padded_tokens=total_chunks * CHUNK, input_packing=False,
                         sequence_count=sequences, packed_direct=sequence_meta is not None,
                         pre_unroll=pre_unroll, persistent_blocks=persistent_blocks,
                         persistent_head_map="grouped" if head_map is not None else "stride")
    return out_public, final_state


# Preserve the original entry point so existing callers continue to work.
kda_bt16_fwd_ascendc = kda_fwd_ascendc
