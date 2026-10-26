#!/usr/bin/env python3
"""
Run ``bench_fwd.py`` twice (default ``H`` and ``--H 64``), parse stdout, and
write a benchmark markdown report.

Reports mean latency for flash_kda (fp32 state), FLA chunk_kda and
chunk_gated_delta_rule, plus speedups. Generated date is UTC (YYYY-MM-DD).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_FWD = Path(__file__).resolve().parent / "bench_fwd.py"
DEFAULT_OUT = REPO_ROOT / "BENCHMARK_A3.md"
DEFAULT_DEVICE_LABEL = "Ascend / A3"

FLA_CHUNK_KDA_OPTIONS_MD = (
    "- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, "
    "`use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, "
    "`lower_bound=-5`, `transpose_state_layout=True`"
)
FLA_CHUNK_GDN_OPTIONS_MD = (
    "- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate "
    "`g=logsigmoid(randn)` of shape `(1, T, H)`, preactivated "
    "`beta=sigmoid(beta_logits)`, `use_qk_l2norm_in_kernel=True`, "
    "`transpose_state_layout=True`"
)

RE_HEADER_FIXED = re.compile(
    r"^shape=\[(\d+),(\d+),(\d+)\] warmup=(\d+) iters=(\d+) repeats=(\d+)\s*$"
)
RE_HEADER_VARLEN = re.compile(
    r"^varlen shape=\[(\d+),(\d+),(\d+)\] seq_lens=(\[[^\]]+\]) "
    r"warmup=(\d+) iters=(\d+) repeats=(\d+)\s*$"
)
RE_RESULT = re.compile(
    r"^\s+(.+?)\s*:\s*mean=([\d.]+) ms, min=([\d.]+) ms, max=([\d.]+) ms\s*$"
)

VARIANT_KEYS = {
    "bf16 state": "bf16_mean_ms",
    "no state": "nostate_mean_ms",
    "fp32 state": "fp32_mean_ms",
}


def run_bench(extra_argv: list[str]) -> str:
    cmd = [sys.executable, str(BENCH_FWD), *extra_argv]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        sys.stderr.write(proc.stdout or "")
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout


def parse_stdout(text: str) -> list[dict]:
    cases: list[dict] = []
    current: dict | None = None
    backend = "unknown"

    def new_case_base(kind: str, *, T: int, H: int, D: int,
                      warmup: int, iters: int, repeats: int,
                      seq_lens: str | None = None) -> dict:
        c: dict = {
            "kind": kind, "T": T, "H": H, "D": D,
            "warmup": warmup, "iters": iters, "repeats": repeats,
            "bf16_mean_ms": None, "nostate_mean_ms": None, "fp32_mean_ms": None,
        }
        if seq_lens is not None:
            c["seq_lens"] = seq_lens
        return c

    for line in text.splitlines():
        if line.strip().startswith("backend="):
            backend = line.strip().split()[0].split("=", 1)[1]
            continue
        m = RE_HEADER_VARLEN.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, seq_lens, w, it, rep = m.groups()
            current = new_case_base(
                "varlen", T=int(t), H=int(h), D=int(d), warmup=int(w),
                iters=int(it), repeats=int(rep), seq_lens=seq_lens)
            continue
        m = RE_HEADER_FIXED.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, w, it, rep = m.groups()
            current = new_case_base(
                "fixed", T=int(t), H=int(h), D=int(d), warmup=int(w),
                iters=int(it), repeats=int(rep))
            continue
        m = RE_RESULT.match(line)
        if m and current is not None:
            name, mean, _mn, _mx = m.groups()
            name = name.strip()
            for key, field in VARIANT_KEYS.items():
                if key in name:
                    current[field] = float(mean)
                    current[field.replace("_mean_ms", "_backend")] = backend
            if name == "chunk_kda":
                current["chunk_mean_ms"] = float(mean)
            elif name == "chunk_gated_delta_rule":
                current["gdn_mean_ms"] = float(mean)

    if current is not None:
        cases.append(current)
    return cases


def _fmt_seq_lens(seq_lens_str: str) -> str:
    try:
        xs = ast.literal_eval(seq_lens_str)
    except (ValueError, SyntaxError):
        return seq_lens_str
    if not isinstance(xs, list) or not xs:
        return seq_lens_str
    if not all(isinstance(x, int) for x in xs):
        return seq_lens_str
    first = xs[0]
    if len(xs) >= 2 and all(x == first for x in xs):
        return f"{first} x {len(xs)}"
    return seq_lens_str


def _case_detail(c: dict) -> str:
    if c["kind"] == "fixed":
        return "Fixed"
    seq = _fmt_seq_lens(c["seq_lens"])
    if seq.startswith("["):
        return f"Varlen, `seq_lens`={seq}"
    return f"Varlen, `seq_lens`=`{seq}`"


def _fmt_ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def _fmt_speedup(flash: float | None, baseline: float | None) -> str:
    if flash is None or flash <= 0 or baseline is None:
        return "—"
    return f"{baseline / flash:.2f}×"


def _argv_with_h(argv: list[str], h: int) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--H" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--H="):
            i += 1
            continue
        out.append(a)
        i += 1
    out.extend(["--H", str(h)])
    return out


def _render_table_block(cases: list[dict]) -> list[str]:
    lines: list[str] = [
        "| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | "
        "Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |",
        "|------|----------------------:|----------------------:|--------:|----------------------:|--------:|",
    ]
    for c in cases:
        cell = _case_detail(c).replace("|", "\\|")
        flash = c.get("fp32_mean_ms")
        chunk, gdn = c.get("chunk_mean_ms"), c.get("gdn_mean_ms")

        lines.append(
            f"| {cell} | {_fmt_ms(flash)} | {_fmt_ms(chunk)} | {_fmt_speedup(flash, chunk)} |"
            f" {_fmt_ms(gdn)} | {_fmt_speedup(flash, gdn)} |"
        )
    lines.append("")
    return lines


def render_markdown(sections: list[list[dict]], generated_at: str,
                    generator_cmd: str, device_label: str) -> str:
    title = "# KDA forward benchmark"
    if device_label:
        title += f" ({device_label})"

    lines: list[str] = [title, "", f"- Generated: {generated_at}", ""]
    if not sections:
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"- Command: `{generator_cmd}`")
    lines.append("")

    first = next((c for c in sections if c), None)
    if first:
        c0 = first[0]
        lines.append(
            f"- Benchmark settings: `warmup={c0['warmup']}`, `iters={c0['iters']}`, "
            f"`repeats={c0['repeats']}`"
        )
        lines.append("")
        lines.append("- `flash_kda` uses FP32 initial and final state; latency uses device events.")
        lines.append(FLA_CHUNK_KDA_OPTIONS_MD)
        lines.append(FLA_CHUNK_GDN_OPTIONS_MD)
        lines.append("- GDN input activation is outside timing. CUDA's benchmark uses raw normal gate/beta; these can overflow on long sequences, so GDN timings use different input values.")
        lines.append("")

    for cases in sections:
        if not cases:
            continue
        c0 = cases[0]
        lines.append(f"### `T={c0['T']}`, `H={c0['H']}`, `D={c0['D']}`")
        lines.append("")
        lines.extend(_render_table_block(cases))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run flash_kda_npu/benchmarks/bench_fwd.py and write a benchmark markdown report."
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                   help=f"Output markdown path (default: {DEFAULT_OUT})")
    p.add_argument("--device-label", default=DEFAULT_DEVICE_LABEL,
                   help=f"Device/platform label for the report title (default: {DEFAULT_DEVICE_LABEL!r})")
    args, bench_extra = p.parse_known_args()

    def _fmt_generator_cmd(extra: list[str]) -> str:
        cmd = "python flash_kda_npu/benchmarks/generate_benchmark_md.py"
        if args.output != DEFAULT_OUT:
            cmd += f" -o {args.output}"
        if args.device_label != DEFAULT_DEVICE_LABEL:
            cmd += f" --device-label {args.device_label}"
        tail = " ".join(extra)
        return f"{cmd} {tail}".strip() if tail else cmd

    argv_default = list(bench_extra)
    argv_h64 = _argv_with_h(bench_extra, 64)

    stdout_a = run_bench(argv_default)
    stdout_b = run_bench(argv_h64)
    cases_a = parse_stdout(stdout_a)
    cases_b = parse_stdout(stdout_b)

    sections: list[list[dict]] = [cases_a, cases_b]
    if not cases_a or not cases_b:
        sys.stderr.write(
            "Warning: missing benchmark rows for one or both runs.\n"
        )

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    md = render_markdown(sections, generated,
                         _fmt_generator_cmd(bench_extra), args.device_label)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
