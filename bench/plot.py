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


if __name__ == "__main__":
    current, previous, earlier = read(CURRENT), read(PREVIOUS), read(EARLIER)
    decode(current, previous)
    # Retain the explicitly dated r7c historical figure; no absolute r7d prefill.
    prefill(previous)
    history(current, previous, earlier)
    print("memory: omitted; no current-r7d category measurements in the retained summaries")
    print("wrote decode-concurrency.svg, prefill.svg, history.svg")
