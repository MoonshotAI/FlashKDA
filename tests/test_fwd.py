import torch
import torch.nn.functional as F
import flash_kda
import math

from torch_ref import torch_ref


# ============================================================
# Test helpers
# ============================================================

def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def print_error_stats(label, actual, expected):
    diff = (actual.float() - expected.float()).abs()
    ref_abs = expected.float().abs()
    avg_rtol = diff.mean() / (ref_abs.mean() + 1e-8)
    max_rtol = diff.max() / (ref_abs.max() + 1e-8)
    print(f"  {label} | avg_rtol: {avg_rtol}, max_rtol: {max_rtol}")
    print(f"  {label} | avg_atol: {diff.mean()}, max_atol: {diff.max()}")


def collect_windowed_errors(gold, pred, window):
    T_len = gold.shape[1]
    errors = []
    for i in range(0, T_len, window):
        end = min(i + window, T_len)
        s = slice(i, end)
        errors.append((i, end,
                        (gold[:, s] - pred[:, s]).abs().max().item(),
                        (gold[:, s] - pred[:, s]).abs().mean().item(),
                        get_err_ratio(gold[:, s], pred[:, s])))
    return errors


# ============================================================
# FLA comparison helpers
# ============================================================

def make_test_cases(H, D, T_shape, dtype, device):
    """Build (name, g, dt_bias) tuples for g/bias sweep."""
    g_specs = [
        ("g=-8",       lambda s: torch.full(s, -8.0, dtype=dtype, device=device)),
        ("g=-4",       lambda s: torch.full(s, -4.0, dtype=dtype, device=device)),
        ("g=0",        lambda s: torch.full(s,  0.0, dtype=dtype, device=device)),
        ("g=8",        lambda s: torch.full(s,  8.0, dtype=dtype, device=device)),
        ("g=U(-8,8)",  lambda s: torch.zeros(s, dtype=dtype, device=device).uniform_(-8, 8)),
        ("g=N(0,8)",   lambda s: torch.randn(s, dtype=dtype, device=device) * 8),
    ]
    bias_specs = [
        ("bias={-4,4}",  torch.where(torch.rand(H, D, device=device) < 0.5, torch.tensor(-4.0), torch.tensor(4.0)).to(torch.float32)),
        ("bias={-8,8}",  torch.where(torch.rand(H, D, device=device) < 0.5, torch.tensor(-8.0), torch.tensor(8.0)).to(torch.float32)),
        ("bias=U(-8,8)", torch.zeros(H, D, dtype=torch.float32, device=device).uniform_(-8, 8)),
        ("bias=N(0,8)",  torch.randn(H, D, dtype=torch.float32, device=device) * 8),
    ]

    cases = []
    for g_name, g_fn in g_specs:
        cases.append((f"{g_name}_bias=0", g_fn(T_shape), torch.zeros(H, D, dtype=torch.float32, device=device)))
    g_zero = torch.full(T_shape, 0.0, dtype=dtype, device=device)
    for b_name, b_val in bias_specs:
        cases.append((f"g=0_{b_name}", g_zero.clone(), b_val.clone()))
    return cases


def run_fla_gold_reference(q, k, v, g, beta, h0, A_log, dt_bias, scale, lower_bound, cu_seqlens=None):
    """Run fused_recurrent_kda in fp64 as gold reference, and chunk_kda in bf16.
    beta: [B, T, H] bf16 logits (pre-sigmoid).
    """
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    H = A_log.shape[0]
    g_fp64 = g.clone().to(torch.float64) + dt_bias.to(torch.float64).unsqueeze(0).unsqueeze(0)
    A_log_fp64 = A_log.to(torch.float64)
    g_activated_fp64 = lower_bound * torch.sigmoid(torch.exp(A_log_fp64.view(1, 1, H, 1)) * g_fp64)

    # fused_recurrent_kda expects post-sigmoid beta
    beta_activated_fp64 = torch.sigmoid(beta.clone().to(torch.float64))

    fla_kwargs = dict(cu_seqlens=cu_seqlens) if cu_seqlens is not None else {}

    tri, tri_ht = fused_recurrent_kda(
        q=q.clone().to(torch.float64),
        k=k.clone().to(torch.float64),
        v=v.clone().to(torch.float64),
        g=g_activated_fp64,
        beta=beta_activated_fp64,
        A_log=None, dt_bias=None,
        scale=scale,
        initial_state=h0.clone().to(torch.float64),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        lower_bound=None,
        transpose_state_layout=True,
        **fla_kwargs,
    )
    tri = tri.to(torch.float32)
    tri_ht = tri_ht.to(torch.float32)

    # upstream hasn't implemented use_beta_sigmoid_in_kernel; pass post-sigmoid beta explicitly.
    beta_activated = torch.sigmoid(beta.clone().float()).to(torch.bfloat16)
    chunk_o, chunk_ht = chunk_kda(
        q=q.clone().to(torch.bfloat16),
        k=k.clone().to(torch.bfloat16),
        v=v.clone(),
        g=g.clone(),
        beta=beta_activated,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone(),
        lower_bound=lower_bound,
        transpose_state_layout=True,
        **fla_kwargs,
    )

    return tri, tri_ht, chunk_o, chunk_ht


def run_flash_kda_batched(q, k, v, g, beta, h0, A_log, dt_bias, scale, lower_bound, cu_seqlens=None):
    """Run flash_kda.fwd with [B, T, H, D] inputs from FLA tensor layout."""
    h0_bf16 = h0.to(torch.bfloat16).clone()
    final_state_fk = torch.zeros_like(h0_bf16)
    out_fk = torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device)
    fwd_kwargs = dict(cu_seqlens=cu_seqlens) if cu_seqlens is not None else {}
    flash_kda.fwd(q.to(torch.bfloat16), k.to(torch.bfloat16), v.clone(), g.to(torch.bfloat16),
                  beta.to(torch.bfloat16).clone(), scale, out_fk,
                  A_log=A_log.clone(), dt_bias=dt_bias.clone(), lower_bound=lower_bound,
                  initial_state=h0_bf16, final_state=final_state_fk, **fwd_kwargs)
    torch.cuda.synchronize()

    return out_fk, final_state_fk.float()


def plot_error_comparison(results, save_path):
    import matplotlib.pyplot as plt

    def moving_avg(data, w):
        return [sum(data[max(0, i-w):i+1]) / min(i+1, w) for i in range(len(data))]

    n_cases = len(results)
    fig, axes = plt.subplots(5, n_cases, figsize=(4 * n_cases, 20))

    for col, r in enumerate(results):
        positions = [e[0] for e in r["errors_flash"]]
        flash_max = [e[2] for e in r["errors_flash"]]
        flash_mean = [e[3] for e in r["errors_flash"]]
        chunk_max = [e[2] for e in r["errors_chunk"]]
        chunk_mean = [e[3] for e in r["errors_chunk"]]
        flash_err_ratio = [e[4] for e in r["errors_flash"]]
        chunk_err_ratio = [e[4] for e in r["errors_chunk"]]

        ax = axes[0, col]
        ax.plot(positions, flash_max, 'b-', alpha=0.7, label='flash_kda')
        ax.plot(positions, chunk_max, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='Max Error')
        ax.set_title(f'{r["name"]}\nMax Error')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(positions, moving_avg(flash_max, 20), 'b-', lw=2, label='flash_kda (MA-20)')
        ax.plot(positions, moving_avg(chunk_max, 20), 'r-', lw=2, label='chunk_kda (MA-20)')
        ax.set(xlabel='Token Position', ylabel='Max Error (MA)')
        ax.set_title('Max Error MA-20')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        ax.plot(positions, flash_mean, 'b-', alpha=0.7, label='flash_kda')
        ax.plot(positions, chunk_mean, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='Mean Error')
        ax.set_title('Mean Error')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        ax.plot(positions, flash_err_ratio, 'b-', alpha=0.7, label='flash_kda')
        ax.plot(positions, chunk_err_ratio, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='RMSE Ratio')
        ax.set_title('RMSE Ratio')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[4, col]
        ax.hist(flash_max, bins=30, alpha=0.5, label='flash_kda', color='blue')
        ax.hist(chunk_max, bins=30, alpha=0.5, label='chunk_kda', color='red')
        ax.set(xlabel='Max Error', ylabel='Frequency')
        ax.set_title('Error Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)

    state_lines = []
    for r in results:
        n = r['name']
        c_max = (r['tri_ht'] - r['chunk_ht']).abs().max().item()
        c_mean = (r['tri_ht'] - r['chunk_ht']).abs().mean().item()
        c_ratio = get_err_ratio(r['tri_ht'], r['chunk_ht'])
        f_max = (r['tri_ht'] - r['final_state']).abs().max().item()
        f_mean = (r['tri_ht'] - r['final_state']).abs().mean().item()
        f_ratio = get_err_ratio(r['tri_ht'], r['final_state'])
        state_lines.append(
            f"{n:>25s}  chunk_kda ht: max={c_max:.6f} mean={c_mean:.2e} ratio={c_ratio:.2e}"
            f"  |  flash_kda ht: max={f_max:.6f} mean={f_mean:.2e} ratio={f_ratio:.2e}"
        )
    plt.tight_layout()
    fig.text(0.5, -0.01, "State Diff:\n" + "\n".join(state_lines),
             ha='center', va='top', fontsize=15, family='monospace')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


# ============================================================
# Tests
# ============================================================

def test_fwd():
    """Test: cutlass kernel vs torch ref, require exact match."""
    B, T, H, D = 1, 8192, 96, 128
    LOWER_BOUND = -5.0

    print(f"Testing shape: [{B}, {T}, {H}, {D}]")
    torch.manual_seed(0)

    q = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device='cuda'), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device='cuda'), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((B, T, H, D), dtype=torch.bfloat16, device='cuda')
    g = torch.randn((B, T, H, D), dtype=torch.bfloat16, device='cuda')
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device='cuda')

    A_log = torch.rand(H, dtype=torch.float32, device='cuda')
    dt_bias = torch.rand(H, D, dtype=torch.float32, device='cuda')

    initial_state = torch.arange(H * D * D, dtype=torch.float32, device='cuda').reshape(1, H, D, D).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # cutlass kernel
    final_state_kernel = torch.zeros_like(initial_state)
    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=initial_state.clone(), final_state=final_state_kernel)
    torch.cuda.synchronize()

    # torch ref
    final_state_ref = torch.zeros_like(initial_state)
    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=initial_state.clone(), final_state=final_state_ref)

    print(f"{torch.max(out_kernel)} {torch.max(out_ref)}")
    print_error_stats("output", out_kernel, out_ref)

    assert torch.equal(out_kernel, out_ref), "output mismatch between kernel and torch ref"
    assert torch.equal(final_state_kernel, final_state_ref), "final_state mismatch between kernel and torch ref"
    print("Success: kernel == torch ref (exact match)")


def test_fwd_varlen():
    """Test: varlen cutlass kernel vs torch ref, require exact match."""
    H, D = 96, 128
    LOWER_BOUND = -5.0
    seq_lens = [1300, 547, 2048, 963, 271, 3063]
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device='cuda',
    )

    print(f"\ntest_fwd_varlen: seq_lens={seq_lens}, T_total={T_total}, N={N}, H={H}, D={D}")
    torch.manual_seed(0)

    q = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device='cuda'), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device='cuda'), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device='cuda')
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device='cuda')
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device='cuda')

    A_log = torch.rand(H, dtype=torch.float32, device='cuda')
    dt_bias = torch.rand(H, D, dtype=torch.float32, device='cuda')

    initial_state = torch.arange(N * H * D * D, dtype=torch.float32, device='cuda').reshape(N, H, D, D).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # cutlass kernel
    final_state_kernel = torch.zeros_like(initial_state)
    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=initial_state.clone(), final_state=final_state_kernel, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    # torch ref
    final_state_ref = torch.zeros_like(initial_state)
    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=initial_state.clone(), final_state=final_state_ref, cu_seqlens=cu_seqlens)

    print(f"{torch.max(out_kernel)} {torch.max(out_ref)}")
    print_error_stats("output", out_kernel, out_ref)

    assert torch.equal(out_kernel, out_ref), "output mismatch between kernel and torch ref"
    assert torch.equal(final_state_kernel, final_state_ref), "final_state mismatch between kernel and torch ref"
    print("Success: varlen kernel == torch ref (exact match)")


@torch.inference_mode()
def test_fwd_vs_fla():
    from fla.utils import assert_close, device

    B, T, H, D = 1, 8192, 1, 128
    dtype = torch.bfloat16
    scale = 1 / math.sqrt(D)
    WINDOW = 8
    LOWER_BOUND = -5.0

    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    A_log = torch.full((H,), 0.0, dtype=torch.float32, device=device)
    cases = make_test_cases(H, D, (B, T, H, D), dtype, device)
    results = []

    for case_name, g, dt_bias in cases:
        print(f"\n{'='*80}")
        print(f"Case: {case_name}")

        tri, tri_ht, chunk_o, chunk_ht = run_fla_gold_reference(
            q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND)
        out_fk, final_state_fk = run_flash_kda_batched(
            q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND)

        print(f"  chunk_kda  | Output err_ratio: {get_err_ratio(tri, chunk_o):.6e}, State err_ratio: {get_err_ratio(tri_ht, chunk_ht):.6e}")
        print(f"  flash_kda  | Output err_ratio: {get_err_ratio(tri, out_fk):.6e}, State err_ratio: {get_err_ratio(tri_ht, final_state_fk):.6e}")

        results.append(dict(
            name=case_name, tri=tri, tri_ht=tri_ht,
            chunk_o=chunk_o, chunk_ht=chunk_ht,
            out=out_fk, final_state=final_state_fk,
            errors_flash=collect_windowed_errors(tri, out_fk, WINDOW),
            errors_chunk=collect_windowed_errors(tri, chunk_o, WINDOW),
        ))

    plot_error_comparison(results, 'plot.png')

    print(f"\n{'='*80}")
    for r in results:
        assert_close(f"{r['name']} o", r["tri"], r["out"], 0.005)
        assert_close(f"{r['name']} ht", r["tri_ht"], r["final_state"], 0.005, warning=True)
        assert_close(f"{r['name']} chunk_kda ht", r["tri_ht"], r["chunk_ht"], 0.005, warning=True)
    print("Assert results: Success")


@torch.inference_mode()
def test_fwd_varlen_vs_fla():
    from fla.utils import assert_close, device

    H, D = 1, 128
    dtype = torch.bfloat16
    scale = 1 / math.sqrt(D)
    WINDOW = 8
    LOWER_BOUND = -5.0

    seq_lens = [1300, 547, 2048, 963, 271, 3063]
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device=device,
    )

    print(f"seq_lens: {seq_lens}, T_total: {T_total}, N: {N}")

    torch.manual_seed(42)
    q = torch.rand(1, T_total, H, D, dtype=dtype, device=device)
    k = torch.rand(1, T_total, H, D, dtype=dtype, device=device)
    v = torch.rand(1, T_total, H, D, dtype=dtype, device=device)
    beta = torch.randn(1, T_total, H, dtype=dtype, device=device)
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)

    A_log = torch.full((H,), 0.0, dtype=torch.float32, device=device)
    cases = make_test_cases(H, D, (1, T_total, H, D), dtype, device)
    results = []

    for case_name, g, dt_bias in cases:
        print(f"\n{'='*80}")
        print(f"Case: {case_name}")

        tri, tri_ht, chunk_o, chunk_ht = run_fla_gold_reference(
            q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND, cu_seqlens=cu_seqlens)
        out_fk, final_state_fk = run_flash_kda_batched(
            q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND, cu_seqlens=cu_seqlens)

        print(f"  chunk_kda   | Output err_ratio: {get_err_ratio(tri, chunk_o):.6e}, State err_ratio: {get_err_ratio(tri_ht, chunk_ht):.6e}")
        print(f"  flash_kda   | Output err_ratio: {get_err_ratio(tri, out_fk):.6e}, State err_ratio: {get_err_ratio(tri_ht, final_state_fk):.6e}")

        results.append(dict(
            name=case_name, tri=tri, tri_ht=tri_ht,
            chunk_o=chunk_o, chunk_ht=chunk_ht,
            out=out_fk, final_state=final_state_fk,
            errors_flash=collect_windowed_errors(tri, out_fk, WINDOW),
            errors_chunk=collect_windowed_errors(tri, chunk_o, WINDOW),
        ))

    plot_error_comparison(results, 'plot_varlen.png')

    print(f"\n{'='*80}")
    for r in results:
        assert_close(f"{r['name']} o", r["tri"], r["out"], 0.006)
        assert_close(f"{r['name']} ht", r["tri_ht"], r["final_state"], 0.006, warning=True)
        assert_close(f"{r['name']} chunk_kda ht", r["tri_ht"], r["chunk_ht"], 0.005, warning=True)
    print("Assert results: Success")


if __name__ == "__main__":
    test_fwd()
    test_fwd_varlen()
    test_fwd_vs_fla()
    test_fwd_varlen_vs_fla()
