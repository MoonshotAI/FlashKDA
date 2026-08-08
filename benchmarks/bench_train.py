import os

# Pin the FLA baseline to the Triton path: with working dispatch, no_grad
# chunk_kda calls would otherwise route to the flash_kda inference backend.
os.environ.setdefault("FLA_FLASH_KDA", "0")
os.environ.setdefault("FLA_FLASH_KDA_TRAIN", "0")

import torch
import torch.nn.functional as F
import math

from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda import chunk_kda
from flash_kda.train import chunk_kda_train_bwd, chunk_kda_train_fwd, prepare_chunk_indices


def bench_fn(fn, warmup, iters, repeats):
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()

    all_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        all_ms.extend([s.elapsed_time(e) for s, e in zip(starts, ends)])

    xs = sorted(float(x) for x in all_ms)
    n = len(xs)
    mean = sum(xs) / n if n else float("nan")
    mn = xs[0] if n else float("nan")
    mx = xs[-1] if n else float("nan")
    return mean, mn, mx


def run_case(seq_lens, H, D, warmup, iters, repeats):
    device = torch.device("cuda")
    LOWER_BOUND = -5.0
    scale_float = 1.0 / math.sqrt(D)

    varlen = len(seq_lens) > 1
    T_total = sum(seq_lens)
    N = len(seq_lens)

    if varlen:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long, device=device,
        )
        print(f"varlen shape=[{T_total},{H},{D}] seq_lens={seq_lens} warmup={warmup} iters={iters} repeats={repeats}")
        extra = {"cu_seqlens": cu_seqlens}
    else:
        print(f"shape=[{T_total},{H},{D}] warmup={warmup} iters={iters} repeats={repeats}")
        extra = {}

    chunk_indices = None
    if varlen:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        extra_train = {"cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices}
    else:
        extra_train = {}

    q = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    # fla chunk_kda expects a flat dt_bias of shape [H * D] (bwd returns it flat).
    dt_bias = torch.rand(H * D, dtype=torch.float32, device=device)

    initial_state = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    # upstream chunk_kda hasn't implemented use_beta_sigmoid_in_kernel;
    # both paths take post-sigmoid beta explicitly.
    beta_sig = beta.sigmoid().contiguous()

    do = torch.randn_like(v)
    dht = torch.randn(N, H, D, D, dtype=torch.float32, device=device)

    # l2norm_fwd matches the cost of fla's use_qk_l2norm_in_kernel and mirrors
    # the FLA dispatch wrapper, which applies l2norm before the CUDA kernels.
    def flash_fwd(qn, kn):
        return chunk_kda_train_fwd(
            q=qn, k=kn, v=v, g=g, beta=beta_sig, scale=scale_float,
            initial_state=initial_state, output_final_state=True,
            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND, **extra_train,
        )

    def flash_step():
        qn = l2norm_fwd(q)[0]
        kn = l2norm_fwd(k)[0]
        return flash_fwd(qn, kn)

    # --- flash_kda train: fwd ---
    mean, mn, mx = bench_fn(flash_step, warmup, iters, repeats)
    print(f"  flash_kda_train fwd    : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- flash_kda train: fwd+bwd ---
    def flash_fwdbwd():
        qn = l2norm_fwd(q)[0]
        kn = l2norm_fwd(k)[0]
        o, final_state, g_cumsum, Aqk, Akk = flash_fwd(qn, kn)
        chunk_kda_train_bwd(
            q=qn, k=kn, v=v, beta=beta_sig, Aqk=Aqk, Akk=Akk, scale=scale_float,
            initial_state=initial_state, do=do, dht=dht,
            g=g_cumsum, g_org=g,
            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND, **extra_train,
        )

    mean, mn, mx = bench_fn(flash_fwdbwd, warmup, iters, repeats)
    print(f"  flash_kda_train fwdbwd : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- fla chunk_kda: fwd ---
    def run_chunk_kda_fwd():
        with torch.no_grad():
            chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta_sig,
                scale=scale_float,
                initial_state=initial_state,
                output_final_state=True,
                use_gate_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                A_log=A_log, dt_bias=dt_bias,
                lower_bound=LOWER_BOUND,
                **extra,
            )

    mean, mn, mx = bench_fn(run_chunk_kda_fwd, warmup, iters, repeats)
    print(f"  fla_chunk_kda fwd      : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- fla chunk_kda: fwd+bwd ---
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    gg = g.clone().requires_grad_(True)
    bg = beta_sig.clone().requires_grad_(True)
    h0g = initial_state.clone().requires_grad_(True)
    A_log_g = A_log.clone().requires_grad_(True)
    dt_bias_g = dt_bias.clone().requires_grad_(True)

    def run_chunk_kda_fwdbwd():
        o, ht = chunk_kda(
            q=qg, k=kg, v=vg, g=gg, beta=bg,
            scale=scale_float,
            initial_state=h0g,
            output_final_state=True,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log_g, dt_bias=dt_bias_g,
            lower_bound=LOWER_BOUND,
            **extra,
        )
        ((o * do).sum() + (ht * dht).sum()).backward()

    mean, mn, mx = bench_fn(run_chunk_kda_fwdbwd, warmup, iters, repeats)
    print(f"  fla_chunk_kda fwdbwd   : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")


FIXED_CASES = [
    [8192],
]

VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--D", type=int, default=128)
    args = p.parse_args()

    cases = []
    if args.mode in ("fixed", "all"):
        cases.extend(FIXED_CASES)
    if args.mode in ("varlen", "all"):
        cases.extend(VARLEN_CASES)

    for seq_lens in cases:
        run_case(seq_lens, args.H, args.D, args.warmup, args.iters, args.repeats)


if __name__ == "__main__":
    main()
