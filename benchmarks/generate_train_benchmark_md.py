#!/usr/bin/env python3
"""
Run ``bench_train.py`` twice (default ``H`` and ``--H 64``), parse stdout, and
write a training benchmark markdown report.

Reports mean latency for ``flash_kda_train`` (CUDA training kernels) and
``fla_chunk_kda`` (FLA Triton), fwd and fwd+bwd, plus speedup
``fla_mean / flash_mean``. Generated date is UTC, day precision only
(YYYY-MM-DD).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_TRAIN = Path(__file__).resolve().parent / "bench_train.py"
DEFAULT_OUT = REPO_ROOT / "BENCHMARK_TRAIN_RTX5090.md"
DEFAULT_DEVICE_LABEL = "Blackwell / RTX 5090"

FLA_CHUNK_KDA_OPTIONS_MD = (
    "- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, "
    "`use_qk_l2norm_in_kernel=True`, post-sigmoid `beta`, "
    "`lower_bound=-5`, fp32 `initial_state`"
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


def run_bench(extra_argv: list[str]) -> str:
    cmd = [sys.executable, str(BENCH_TRAIN), *extra_argv]
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

    def new_case(kind, *, T, H, D, warmup, iters, repeats, seq_lens=None) -> dict:
        c = {
            "kind": kind, "T": T, "H": H, "D": D,
            "warmup": warmup, "iters": iters, "repeats": repeats,
            "flash_fwd_ms": None, "fla_fwd_ms": None,
            "flash_fwdbwd_ms": None, "fla_fwdbwd_ms": None,
        }
        if seq_lens is not None:
            c["seq_lens"] = seq_lens
        return c

    for line in text.splitlines():
        m = RE_HEADER_VARLEN.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, seq_lens, w, it, rep = m.groups()
            current = new_case("varlen", T=int(t), H=int(h), D=int(d),
                               warmup=int(w), iters=int(it), repeats=int(rep), seq_lens=seq_lens)
            continue

        m = RE_HEADER_FIXED.match(line)
        if m:
            if current is not None:
                cases.append(current)
            t, h, d, w, it, rep = m.groups()
            current = new_case("fixed", T=int(t), H=int(h), D=int(d),
                               warmup=int(w), iters=int(it), repeats=int(rep))
            continue

        m = RE_RESULT.match(line)
        if m and current is not None:
            name, mean, _mn, _mx = m.groups()
            name = name.strip()
            if name == "flash_kda_train fwd":
                current["flash_fwd_ms"] = float(mean)
            elif name == "fla_chunk_kda fwd":
                current["fla_fwd_ms"] = float(mean)
            elif name == "flash_kda_train fwdbwd":
                current["flash_fwdbwd_ms"] = float(mean)
            elif name == "fla_chunk_kda fwdbwd":
                current["fla_fwdbwd_ms"] = float(mean)

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


def _fmt_ms(x: float) -> str:
    return f"{x:.4f}"


def _fmt_speedup(flash: float, fla: float) -> str:
    if flash <= 0:
        return "—"
    return f"{fla / flash:.2f}×"


def _complete_cases(raw: list[dict]) -> list[dict]:
    return [
        c for c in raw
        if all(c[k] is not None for k in (
            "flash_fwd_ms", "fla_fwd_ms", "flash_fwdbwd_ms", "fla_fwdbwd_ms"))
    ]


def _render_table_block(cases: list[dict]) -> list[str]:
    lines = [
        "| Case | `flash_kda_train` fwd (ms) | `fla_chunk_kda` fwd (ms) | "
        "fwd speedup | `flash_kda_train` fwd+bwd (ms) | `fla_chunk_kda` fwd+bwd (ms) | "
        "fwd+bwd speedup |",
        "|------|------------------:|------------------:|--------:|"
        "------------------:|------------------:|--------:|",
    ]
    for c in cases:
        cell = _case_detail(c).replace("|", "\\|")
        lines.append(
            f"| {cell} | {_fmt_ms(c['flash_fwd_ms'])} | {_fmt_ms(c['fla_fwd_ms'])} |"
            f" {_fmt_speedup(c['flash_fwd_ms'], c['fla_fwd_ms'])} |"
            f" {_fmt_ms(c['flash_fwdbwd_ms'])} | {_fmt_ms(c['fla_fwdbwd_ms'])} |"
            f" {_fmt_speedup(c['flash_fwdbwd_ms'], c['fla_fwdbwd_ms'])} |"
        )
    lines.append("")
    return lines


def render_markdown(sections, generated_at, generator_cmd, device_label) -> str:
    title = "# KDA training benchmark (fwd / fwd+bwd)"
    if device_label:
        title += f" ({device_label})"

    lines = [title, "", f"- Generated: {generated_at}", ""]

    if not sections:
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    lines.append(f"- Command: `{generator_cmd}`")
    lines.append("")

    first_cases = next((c for c in sections if c), None)
    c0 = first_cases[0] if first_cases else None
    if c0 is not None:
        lines.append(
            f"- Benchmark settings: `warmup={c0['warmup']}`, `iters={c0['iters']}`, "
            f"`repeats={c0['repeats']}`"
        )
        lines.append("")
        lines.append(FLA_CHUNK_KDA_OPTIONS_MD)
        lines.append(
            "- `flash_kda_train` configuration: `flash_kda.train.chunk_kda_train_fwd`/"
            "`chunk_kda_train_bwd`, `use_gate_in_kernel=True`, post-sigmoid `beta`, "
            "`lower_bound=-5`, fp32 `initial_state`, `chunk_size=64`; "
            "q/k l2-normalized inside the timed region (matches `use_qk_l2norm_in_kernel`)"
        )
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
        description="Run bench_train.py and write a benchmark markdown report."
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                   help=f"Output markdown path (default: {DEFAULT_OUT})")
    p.add_argument("--device-label", default=DEFAULT_DEVICE_LABEL,
                   help=f"Device/platform label for the report title (default: {DEFAULT_DEVICE_LABEL!r})")
    args, bench_extra = p.parse_known_args()

    def _fmt_generator_cmd(extra: list[str]) -> str:
        cmd = "python benchmarks/generate_train_benchmark_md.py"
        if args.output != DEFAULT_OUT:
            cmd += f" -o {args.output}"
        if args.device_label != DEFAULT_DEVICE_LABEL:
            cmd += f" --device-label {args.device_label}"
        tail = " ".join(extra)
        return f"{cmd} {tail}".strip() if tail else cmd

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

    stdout_a = run_bench(list(bench_extra))
    stdout_b = run_bench(_argv_with_h(bench_extra, 64))
    cases_a = _complete_cases(parse_stdout(stdout_a))
    cases_b = _complete_cases(parse_stdout(stdout_b))

    sections = [cases_a, cases_b]

    if not cases_a or not cases_b:
        sys.stderr.write(
            "Warning: missing complete benchmark rows for one or both runs.\n"
        )

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    md = render_markdown(sections, generated, _fmt_generator_cmd(bench_extra), args.device_label)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
