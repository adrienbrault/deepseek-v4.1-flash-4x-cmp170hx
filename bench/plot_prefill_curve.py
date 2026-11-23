#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
# SPDX-License-Identifier: MIT
"""Draw docs/img/prefill-curve.svg from a cold-prefill summary.

uv run bench/plot_prefill_curve.py bench/results/2026-11-23-current-r18b-prefill/summary.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "prefill-curve.svg"


def main(path):
    summary = json.loads(Path(path).read_text())
    cells = summary["cells"]
    xs = [c["median_prompt_tokens"] for c in cells]
    rate = [c["median_prefill_tok_s"] for c in cells]
    ttft = [c["median_ttft_s"] for c in cells]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in (left, right):
        ax.set_xscale("log")
        ticks = [x for x in xs if not 150000 < x < 220000]  # 200K sits too close to 240K
        ax.set_xticks(ticks, [f"{round(x / 1000)}K" for x in ticks])
        ax.minorticks_off()
        ax.grid(axis="y", color="#eaeef2")
        ax.set_axisbelow(True)
        ax.set_xlabel("prompt tokens")
    left.plot(xs, rate, marker="o", color="#2f6fb5")
    left.set_ylim(0, max(rate) * 1.2)
    left.set_ylabel("prefill tok/s")
    left.set_title("Cold prefill rate", fontsize=10)
    right.plot(xs, ttft, marker="o", color="#c4622d")
    right.set_ylabel("seconds")
    right.set_title("Time to first token", fontsize=10)
    for ax, ys, fmt in ((left, rate, "{:,.0f}"), (right, ttft, "{:.1f}")):
        for i, (x, y) in enumerate(zip(xs, ys)):
            near = i == len(xs) - 2
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(-8, -14) if near else (0, 7),
                        ha="right" if near else "center", fontsize=8)
    fig.suptitle(f"Cold prefill · {summary['definition']} · {summary['date_utc']} UTC · one request at a time, "
                 "median of 3, one output token, T=0", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, format="svg", bbox_inches="tight", metadata={"Title": "Cold prefill rate and time to first token", "Date": None})
    OUT.write_text("\n".join(line.rstrip() for line in OUT.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
