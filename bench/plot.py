#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
# SPDX-License-Identifier: MIT
"""Draw README and history SVGs from published summaries: uv run bench/plot.py.

Style, palette and SVG text follow the GLM-5.3-Flash and Flash-Next recipes.
Historical qualifications stay dated; only same-chain controls support a causal
comparison. Missing stream metrics and memory categories are never inferred.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
OUT = ROOT / "docs/img"
CURRENT = RESULTS / "2026-11-15-current-r7d-qualification/summary.json"
PREVIOUS = RESULTS / "2026-11-15-current-r7c-qualification/summary.json"
EARLIER = RESULTS / "2026-11-14-current-r7b-qualification/summary.json"
DECODE, AGG, PREFILL, AGG_ONE = "#0969da", "#cf222e", "#8250df", "#f0a8ad"
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.edgecolor": "#d8dee4",
    "axes.labelcolor": "#57606a",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "deepseek-recipe",
})


def read(path):
    rec = json.loads(path.read_text())
    assert {c["concurrency"] for c in rec["decode"]["cells"]} == {1, 4, 8}
    assert not rec["prefill"]["cells"] or {(c["prompt_tokens"], c["concurrency"]) for c in rec["prefill"]["cells"]} == {
        (n, c) for n in (2048, 8192) for c in (1, 4, 8)
    }
    return rec


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Title": caption, "Date": None})
    # Matplotlib emits trailing spaces inside SVG path data; normalize for review.
    path = OUT / name
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def style(ax):
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)


def annotate(ax, xs, ys, color, fmt="{:.1f}", dy=7):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=color)


def decode(current, previous):
    cells = {c["concurrency"]: c for c in current["decode"]["cells"]}
    xs = sorted(cells)
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ys = [cells[c]["mean_tok_s"] for c in xs]
    ax.plot(xs, ys, marker="o", color=AGG, linewidth=2, label="current-r7d · 2026-11-15 · one ON boot")
    annotate(ax, xs, ys, AGG)
    refs = [cells[c]["reference_tok_s"] for c in xs]
    ax.plot(xs, refs, marker="o", color=DECODE, linestyle="--", linewidth=1.8,
            label="current-r7c · 2026-11-15 · stored same-day reference")
    annotate(ax, xs, refs, DECODE, dy=-16)
    ax.set(title="Decode aggregate against concurrency, dated qualifications", xlabel="concurrent requests",
           ylabel="decode tokens per second, sum of measured stream rates", ylim=(0, max(ys) * 1.4))
    ax.set_xticks(xs)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.suptitle("Frozen chat · 400 tokens/request · temperature 0, seed 15 · TP4, four Gen2 x16, 175 W/GPU\n"
                 "c1/c4 ordered, c8 burst · r7d ordered c8: 419.7 tok/s (not deciding) · stored cross-boot reference", fontsize=10)
    print("decode:", {c: {"r7d": cells[c]["mean_tok_s"], "protocol": cells[c]["protocol"],
                          "r7c_reference": cells[c]["reference_tok_s"]} for c in xs})
    save(fig, "decode-concurrency.svg", "Current-r7d decode and same-day current-r7c reference, c1/c4 ordered, c8 burst")


def prefill(current):
    cells = {(c["prompt_tokens"], c["concurrency"]): c for c in current["prefill"]["cells"]}
    xs = [1, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, field, title, ylabel, fmt in (
        (axes[0], "ttft_s", "Time to first token", "seconds, median of scored batches", "{:.2f}"),
        (axes[1], "input_tok_s", "Aggregate input rate", "input tokens per second, batch aggregate", "{:,.0f}"),
    ):
        for n, color in ((2048, DECODE), (8192, PREFILL)):
            ys = [cells[n, c][field] for c in xs]
            ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=f"{n:,} nominal tokens/request")
            annotate(ax, xs, ys, color, fmt=fmt, dy=7 if n == 8192 else -16)
        ax.set(title=title, xlabel="concurrent requests", ylabel=ylabel, ylim=(0, max(c[field] for c in cells.values()) * 1.3))
        ax.set_xticks(xs)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style(ax)
    fig.suptitle("Cold prefill · current-r7c · 2026-11-15 · one ON boot, median of three scored batches\n"
                 "Temperature 0 · thinking off · one output token · zero prompt-cache hits · host cache warm-steady", fontsize=10)
    print("prefill:", current["prefill"]["cells"])
    save(fig, "prefill.svg", "Current-r7c cold prefill TTFT and aggregate input rate at 2K and 8K, c1/c4/c8")


def history(current, previous, earlier):
    records = [earlier, previous, current]
    # docs/HISTORY.md is the promotion index; only entries with numeric summaries appear.
    index = (ROOT / "docs/HISTORY.md").read_text()
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    for c, color in ((1, DECODE), (4, AGG), (8, PREFILL)):
        ys = []
        for rec in records:
            assert f'| {rec["configuration"]} |' in index
            ys.append(next(cell["mean_tok_s"] for cell in rec["decode"]["cells"] if cell["concurrency"] == c))
        ax.plot(range(len(records)), ys, marker="o", color=color, linewidth=2, label=f"c{c} aggregate")
        annotate(ax, range(len(records)), ys, color)
    ax.set_xticks(range(len(records)), [f'{r["configuration"]}\n{r["date_utc"]}' for r in records])
    ax.set(xlim=(-.25, len(records) - .75), ylim=(0, 530), ylabel="decode tokens per second, sum of measured stream rates",
           title="Daily definitions over promotions (docs/HISTORY.md)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.suptitle("Frozen chat prompts · 400 output tokens/request · temperature 0, seed 15 · four Gen2 x16, 175 W/GPU\n"
                 "r7b: two-boot mean; r7c/r7d: one boot · c8 burst; 11-15 c1/c4 ordered · cross-boot drift limits attribution.", fontsize=10)
    print("history:", [(r["configuration"], r["date_utc"], {c["concurrency"]: c["mean_tok_s"] for c in r["decode"]["cells"]}) for r in records])
    save(fig, "history.svg", "Daily promotion history with dated c1/c4/c8 suite decode aggregates")


def headline(summary):
    """Only allowlisted headline summaries; incomplete cells remain gaps."""
    assert summary["schema"] == "headline-summary/1"
    cells = summary["cells"]
    xs = [1, 4, 8]
    title = f'{summary["configuration"]} · {summary["date_utc"]} UTC · base name · TP4, four Gen2 x16, 175 W/GPU'
    if not summary["publication_ready"]:
        title += " · INCOMPLETE DRAFT"
    def values(keys, field):
        return [cells.get(k, {}).get(field, float("nan")) for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    keys = [f"specbench-subset-c{c}" for c in xs]
    for ax, field, label, color in zip(axes[:2], ("output_tok_s", "per_stream_tok_s"),
            ("Wall output tok/s", "Per-stream tok/s (1000 / TPOT p50)"), (AGG, DECODE)):
        ys = values(keys, field)
        ax.plot(xs, ys, marker="o", color=color); annotate(ax, xs, ys, color)
        ax.set(title=label, xlabel="concurrent agents", ylim=(0, None)); ax.set_xticks(xs); style(ax)
    for field, label, color in (("ttft_p50_ms", "p50", DECODE), ("ttft_p99_ms", "p99", PREFILL)):
        axes[2].plot(xs, values(keys, field), marker="o", color=color, label=label)
    axes[2].set(title="TTFT, ms", xlabel="concurrent agents", ylim=(0, None))
    axes[2].set_xticks(xs); axes[2].legend(frameon=False); style(axes[2])
    fig.suptitle(title + "\nSpec-Bench · stratified 120 prompts · 256 forced tokens · T=0 · client wall includes prefill", fontsize=10)
    save(fig, "headline-agent.svg", title + "; Spec-Bench wall output, per-stream rate and TTFT")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for p, a, color in ((1024, 2048, DECODE), (32768, 2048, AGG), (65536, 8192, PREFILL)):
        keys = [f"code-shared-t{p}-a{a}-c{c}" for c in xs]
        for ax, field in zip(axes, ("aggregate_tok_s", "per_stream_tok_s", "ttft_p50_ms")):
            ax.plot(xs, values(keys, field), marker="o", color=color, label=f"{p//1024}K + {a//1024}K")
    for ax, label in zip(axes, ("Common-window aggregate tok/s", "Mean stream decode tok/s", "Append TTFT p50, ms")):
        ax.set(title=label, xlabel="concurrent agents", ylim=(0, None)); ax.set_xticks(xs)
        ax.legend(frameon=False, fontsize=8); style(ax)
    fig.suptitle(title + "\nCached code · 400 forced tokens · T=0, seed15 · shared context · one batch/cell", fontsize=10)
    save(fig, "headline-code.svg", title + "; cached-code peak and long-context decode and append TTFT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline", type=Path, help="sanitized headline summary; otherwise draw dated suite history")
    parser.add_argument("--output-dir", type=Path, help="stage generated figures in a local directory")
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir
    if args.headline:
        headline(json.loads(args.headline.read_text()))
        raise SystemExit(0)
    current, previous, earlier = read(CURRENT), read(PREVIOUS), read(EARLIER)
    decode(current, previous)
    # Retain the explicitly dated r7c historical figure; no absolute r7d prefill.
    prefill(previous)
    history(current, previous, earlier)
    print("memory: omitted; no current-r7d category measurements in the retained summaries")
    print("wrote decode-concurrency.svg, prefill.svg, history.svg")
