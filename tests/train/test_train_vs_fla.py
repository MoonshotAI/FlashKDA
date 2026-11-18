"""Precision comparison: flash_kda training kernels vs FLA chunk_kda (Triton),
both against an fp64 naive-recurrent gold, forward output and all gradients.

Mirrors tests/test_fwd.py::test_fwd_vs_fla (same g/bias sweep, windowed error
curves, 5-row plot layout) and writes docs/assets/compare_train_with_fla.png.
"""
import os

# Keep the FLA baseline on the Triton path (see benchmarks/bench_train.py).
os.environ.setdefault("FLA_FLASH_KDA", "0")
os.environ.setdefault("FLA_FLASH_KDA_TRAIN", "0")

import math
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAVE_PATH = REPO_ROOT / "docs" / "assets" / "compare_train_with_fla.png"


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


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


def make_test_cases(H, D, T_shape, dtype, device):
    """Build (name, g, dt_bias) tuples for the g/bias sweep (same as test_fwd.py)."""
    g_specs = [
        ("g=-8",      lambda s: torch.full(s, -8.0, dtype=dtype, device=device)),
        ("g=-4",      lambda s: torch.full(s, -4.0, dtype=dtype, device=device)),
        ("g=0",       lambda s: torch.full(s,  0.0, dtype=dtype, device=device)),
        ("g=8",       lambda s: torch.full(s,  8.0, dtype=dtype, device=device)),
        ("g=U(-8,8)", lambda s: torch.zeros(s, dtype=dtype, device=device).uniform_(-8, 8)),
        ("g=N(0,8)",  lambda s: torch.randn(s, dtype=dtype, device=device) * 8),
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


def activate_g(g, dt_bias, A_log, lower_bound):
    """lower_bound * sigmoid(exp(A_log) * (g + dt_bias)), kept in the autograd graph."""
    H = A_log.shape[0]
    g = g + dt_bias.unsqueeze(0).unsqueeze(0)
    return lower_bound * torch.sigmoid(torch.exp(A_log).view(1, 1, H, 1) * g)


def gold_recurrence(q, k, v, g_act, beta, h0, scale):
    """Differentiable fp64 naive KDA recurrence (same math as fla.ops.kda.naive)."""
    B, T, H, D = q.shape
    S = h0.clone()
    o = torch.zeros(B, T, H, D, dtype=torch.float64, device=q.device)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g_act[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i * scale, S)
    return o, S


def run_gold(q, k, v, g, beta_sig, h0, A_log, dt_bias, scale, lower_bound, do, dht):
    leaves = [x.to(torch.float64).detach().requires_grad_(True) for x in (q, k, v, g, beta_sig, h0)]
    q64, k64, v64, g64, b64, h064 = leaves
    g_act = activate_g(g64, dt_bias.to(torch.float64), A_log.to(torch.float64), lower_bound)
    o, ht = gold_recurrence(q64, k64, v64, g_act, b64, h064, scale)
    ((o * do.to(torch.float64)).sum() + (ht * dht.to(torch.float64)).sum()).backward()
    grads = {n: x.grad.clone() for n, x in zip(("dq", "dk", "dv", "dg", "db", "dh0"), leaves)}
    return o.detach(), ht.detach(), grads


def run_flash_train(q, k, v, g, beta_sig, h0, A_log, dt_bias, scale, lower_bound, do, dht):
    from flash_kda.train import chunk_kda_train_bwd, chunk_kda_train_fwd

    o, ht, g_cumsum, Aqk, Akk = chunk_kda_train_fwd(
        q=q, k=k, v=v, g=g, beta=beta_sig, scale=scale,
        initial_state=h0, output_final_state=True,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    )
    dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_train_bwd(
        q=q, k=k, v=v, beta=beta_sig, Aqk=Aqk, Akk=Akk, scale=scale,
        initial_state=h0, do=do, dht=dht, g=g_cumsum, g_org=g,
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    )
    grads = {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "db": db, "dh0": dh0}
    return o.detach(), ht.detach(), {n: x.float() for n, x in grads.items()}


def run_fla_chunk_train(q, k, v, g, beta_sig, h0, A_log, dt_bias, scale, lower_bound, do, dht):
    from fla.ops.kda import chunk_kda

    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta_sig, h0)]
    qg, kg, vg, gg, bg, h0g = leaves
    o, ht = chunk_kda(
        q=qg, k=kg, v=vg, g=gg, beta=bg, scale=scale,
        initial_state=h0g, output_final_state=True,
        use_gate_in_kernel=True, use_qk_l2norm_in_kernel=False,
        A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    )
    ((o * do).sum() + (ht.float() * dht).sum()).backward()
    grads = {n: x.grad.clone() for n, x in zip(("dq", "dk", "dv", "dg", "db", "dh0"), leaves)}
    return o.detach(), ht.detach(), {n: x.float() for n, x in grads.items()}


def plot_error_comparison(results, save_path):
    import matplotlib.pyplot as plt

    def moving_avg(data, w):
        return [sum(data[max(0, i - w):i + 1]) / min(i + 1, w) for i in range(len(data))]

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
        ax.plot(positions, flash_max, 'b-', alpha=0.7, label='flash_kda_train')
        ax.plot(positions, chunk_max, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='Max Error')
        ax.set_title(f'{r["name"]}\nMax Error')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(positions, moving_avg(flash_max, 20), 'b-', lw=2, label='flash_kda_train (MA-20)')
        ax.plot(positions, moving_avg(chunk_max, 20), 'r-', lw=2, label='chunk_kda (MA-20)')
        ax.set(xlabel='Token Position', ylabel='Max Error (MA)')
        ax.set_title('Max Error MA-20')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        ax.plot(positions, flash_mean, 'b-', alpha=0.7, label='flash_kda_train')
        ax.plot(positions, chunk_mean, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='Mean Error')
        ax.set_title('Mean Error')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[3, col]
        ax.plot(positions, flash_err_ratio, 'b-', alpha=0.7, label='flash_kda_train')
        ax.plot(positions, chunk_err_ratio, 'r-', alpha=0.7, label='chunk_kda')
        ax.set(xlabel='Token Position', ylabel='RMSE Ratio')
        ax.set_title('RMSE Ratio')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[4, col]
        ax.hist(flash_max, bins=30, alpha=0.5, label='flash_kda_train', color='blue')
        ax.hist(chunk_max, bins=30, alpha=0.5, label='chunk_kda', color='red')
        ax.set(xlabel='Max Error', ylabel='Frequency')
        ax.set_title('Error Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)

    lines = []
    for r in results:
        n = r['name']
        parts = [f"{n:>18s}"]
        for label, gold_val, f_val, c_val in r["summary"]:
            parts.append(
                f"{label}: chunk={get_err_ratio(gold_val, c_val):.2e} "
                f"flash={get_err_ratio(gold_val, f_val):.2e}"
            )
        lines.append("  ".join(parts))
    plt.tight_layout()
    fig.text(0.5, -0.01, "RMSE ratio vs fp64 gold:\n" + "\n".join(lines),
             ha='center', va='top', fontsize=15, family='monospace')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    from fla.utils import assert_close

    B, T, H, D = 1, 2048, 1, 128
    dtype = torch.bfloat16
    device = torch.device("cuda")
    scale = 1 / math.sqrt(D)
    WINDOW = 8
    LOWER_BOUND = -5.0

    torch.manual_seed(42)
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid().contiguous()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    A_log = torch.full((H,), 0.0, dtype=torch.float32, device=device)
    do = torch.randn(B, T, H, D, dtype=dtype, device=device)
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    cases = make_test_cases(H, D, (B, T, H, D), dtype, device)
    results = []

    for case_name, g, dt_bias in cases:
        print(f"\n{'=' * 80}")
        print(f"Case: {case_name}")

        gold_o, gold_ht, gold_g = run_gold(q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND, do, dht)
        f_o, f_ht, f_g = run_flash_train(q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND, do, dht)
        c_o, c_ht, c_g = run_fla_chunk_train(q, k, v, g, beta, h0, A_log, dt_bias, scale, LOWER_BOUND, do, dht)

        print(f"  chunk_kda        | o err_ratio: {get_err_ratio(gold_o, c_o):.6e}, ht err_ratio: {get_err_ratio(gold_ht, c_ht):.6e}")
        print(f"  flash_kda_train  | o err_ratio: {get_err_ratio(gold_o, f_o):.6e}, ht err_ratio: {get_err_ratio(gold_ht, f_ht):.6e}")

        summary = [("ht", gold_ht, f_ht, c_ht)]
        for name in ("dq", "dk", "dv", "dg", "db", "dh0"):
            summary.append((name, gold_g[name], f_g[name], c_g[name]))

        results.append(dict(
            name=case_name,
            errors_flash=collect_windowed_errors(gold_o.float(), f_o.float(), WINDOW),
            errors_chunk=collect_windowed_errors(gold_o.float(), c_o.float(), WINDOW),
            summary=summary,
            gold_o=gold_o, f_o=f_o, c_o=c_o,
        ))

    plot_error_comparison(results, SAVE_PATH)

    print(f"\n{'=' * 80}")
    for r in results:
        assert_close(f"{r['name']} flash o", r["gold_o"].float(), r["f_o"].float(), 0.005)
        assert_close(f"{r['name']} chunk o", r["gold_o"].float(), r["c_o"].float(), 0.005)
    print("Assert results: Success")


if __name__ == "__main__":
    main()
