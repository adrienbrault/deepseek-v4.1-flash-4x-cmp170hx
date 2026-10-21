# SPDX-License-Identifier: MIT
import json, statistics, sys

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    if out in args:
        args.remove(out)
    path, label = (args[0], args[1] if len(args) > 1 else '')
    recs = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in recs if r.get('ok') and r.get('ttft_s') is not None and (r['wall_s'] > r['ttft_s'])]
    if not ok:
        print(f'{label} agent: NO DATA ({len(recs)} records, 0 ok)')
        return 1
    pre = [(r['t_submit'], r['t_submit'] + r['ttft_s']) for r in ok]
    t0 = min((r['t_submit'] for r in ok))
    t1 = max((r['t_submit'] + r['wall_s'] for r in ok))
    gen = sum((r['gen_tokens'] for r in ok))
    clean, mixed = ([], [])
    for i, r in enumerate(ok):
        a, b = (r['t_submit'] + r['ttft_s'], r['t_submit'] + r['wall_s'])
        rate = r['gen_tokens'] / (b - a)
        overlap = any((j != i and s < b and (e > a) for j, (s, e) in enumerate(pre)))
        (mixed if overlap else clean).append(rate)
    busy = sum((r['wall_s'] - r['ttft_s'] for r in ok))
    ttft = [r['ttft_s'] for r in ok]
    res = {'label': label, 'requests': len(ok), 'failed': len(recs) - len(ok), 'span_s': round(t1 - t0, 1), 'gen_tokens': gen, 'aggregate_tok_s': round(gen / (t1 - t0), 1), 'decode_median': round(statistics.median(clean + mixed), 1), 'decode_clean_median': round(statistics.median(clean), 1) if clean else None, 'decode_mixed_median': round(statistics.median(mixed), 1) if mixed else None, 'mixed_share': round(len(mixed) / len(ok), 3), 'ttft_median_s': round(statistics.median(ttft), 3), 'ttft_p90_s': round(pct(ttft, 0.9), 3), 'ttft_max_s': round(max(ttft), 3), 'streams_mean': round(busy / (t1 - t0), 2), 'prompt_tokens_median': statistics.median([r['prompt_tokens'] for r in ok if r.get('prompt_tokens')] or [0])}
    print(f"{label} agent: {res['requests']} req ({res['failed']} failed) aggregate {res['aggregate_tok_s']} tok/s, decode median {res['decode_median']} (clean {res['decode_clean_median']}, beside a prefill {res['decode_mixed_median']}, {100 * res['mixed_share']:.0f} % of requests), TTFT median {res['ttft_median_s']} s p90 {res['ttft_p90_s']} s max {res['ttft_max_s']} s, streams {res['streams_mean']}, prompt median {res['prompt_tokens_median']}")
    if out:
        json.dump(res, open(out, 'w'), indent=1)
    return 0
if __name__ == '__main__':
    sys.exit(main())
