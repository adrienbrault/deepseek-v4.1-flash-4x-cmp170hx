#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate static qualification charts from the retained numeric summary."""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'bench/results/2026-11-14-current-r7b-qualification/summary.json'


def chart(path, title, labels, values, unit, ymax):
    width, height = 800, 320
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(title)}</title>',
             '<desc id="desc">Current-r7b, 2026-11-14 UTC, two qualification boots. Values are measured cell summaries, with no interpolation.</desc>',
             '<rect width="800" height="320" fill="#101820"/>',
             '<g fill="#edf2f7" font-family="sans-serif">',
             f'<text x="28" y="35" font-size="21">{escape(title)}</text>',
             '<text x="28" y="59" font-size="13">2026-11-14 UTC · current-r7b · loaded suite v2 · 175 W/GPU</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 84 + i * 55
        items += [f'<text x="28" y="{y+20}" font-size="15">{escape(label)}</text>',
                  f'<rect x="145" y="{y}" width="{value/ymax*510:.2f}" height="29" rx="3" fill="#64c4c6"/>',
                  f'<text x="{155+value/ymax*510:.2f}" y="{y+21}" font-size="15">{value:,.1f} {unit}</text>']
    items += ['<text x="28" y="291" font-size="12">Source: bench/results/2026-11-14-current-r7b-qualification</text>', '</g></svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(items) + '\n')


def main():
    data = json.loads(SOURCE.read_text())
    cells = sorted(data['decode']['cells'], key=lambda x: x['concurrency'])
    chart(ROOT / 'docs/img/qualification-decode.svg', 'Decode aggregate: sum of measured stream rates',
          ['c' + str(x['concurrency']) for x in cells], [x['mean_tok_s'] for x in cells], 'tok/s', 500)


if __name__ == '__main__':
    main()
