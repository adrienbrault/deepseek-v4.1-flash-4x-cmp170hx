# Agent workload instruments

These clients measure serial coding requests and concurrent growing sessions through an OpenAI-compatible local API. They use Python standard-library modules and run without GPU access.

The serial client reads code files from a source directory, uses high reasoning effort and server-default sampling, and requests up to 1,500 output tokens per question. The long input limit is 270,000 characters; the short limit is 9,000 characters. Actual token counts are reported by the server. `DEPTH_PROBE_SEED` sets the request seed, and `DEPTH_PROBE_DOC` selects a frozen document for a matched comparison.

```bash
mkdir -p results
DEPTH_PROBE_SEED=42 python3 bench/depth-probe.py 8001 deepseek-v4.1-flash-1017 ./src 90000 4 results/long.jsonl
DEPTH_PROBE_SEED=42 python3 bench/depth-probe.py 8001 deepseek-v4.1-flash-1017 ./src 3000 3 results/short.jsonl
```

The agent replay sends staggered growing sessions with sampled response lengths, synthetic tool output and pauses between turns. Use the same seed, session ladder and timing settings for each arm. Run a separate warmup before the scored replay.

```bash
python3 bench/agent_replay.py --url http://127.0.0.1:8001/v1 --model deepseek-v4.1-flash-1017 --out results/warm.jsonl --prod-ladder --sessions 10 --think 10 --respawn --seed 7 --timeout 300 --max-seconds 45
python3 bench/agent_replay.py --url http://127.0.0.1:8001/v1 --model deepseek-v4.1-flash-1017 --out results/agent.jsonl --prod-ladder --sessions 10 --think 10 --respawn --seed 1 --timeout 300 --max-seconds 240 --drain --drain-grace 90
python3 bench/agent_score.py results/agent.jsonl control --json results/agent-summary.json
```

The scorer reports completed tokens divided by the full request span, per-request decode rates, time to first token and observed overlap with other requests’ prefill. Its aggregate includes prefill and the drain period; it is distinct from a sum of concurrent stream decode rates. Synthetic sessions describe arrival shape and cache reuse; they do not evaluate completed coding tasks.

`quality-probe.py 8001` checks six known answers and repetition. `serving-check.py results/serving.json` checks exact-answer prefix reuse, reported cache hits and a generated red image on port 8001. Both fail on unmet checks. Each measurement directory records its model, image identity, runtime settings, sampler and UTC date.

`cpu-clock-screen.py` compares the same loaded model at normal CPU frequency limits and a temporary 1.6 GHz maximum. It alternates three normal and three limited rounds at concurrency one and four, using 400-token raw completions, temperature zero, seed 15 and ignored EOS. It saves each policy’s original maximum before changing it, restores every policy on exit, and records frequency samples and request receipts. Run this on an otherwise unoccupied Linux host with `acpi-cpufreq` and permission to write its frequency policies. A lower-clock comparison measures clock sensitivity; it does not predict the gain from a different CPU architecture.

```bash
sudo python3 bench/cpu-clock-screen.py --snapshot results/cpu-original.json --out results/cpu-clock.json
sudo python3 bench/cpu-clock-screen.py --restore results/cpu-original.json
```

`decode-probe.py` measures fixed-concurrency chat decode with exact server token usage, temperature zero and ignored EOS. `DECODE_PROBE_SEED` seeds prompt generation and the server request; `DECODE_PROBE_WARMUP=1` excludes one batch before the three measured batches. `DECODE_PROBE_OUT` stores numeric request receipts without generated text. The reasoning setting is the server default. Each fixed batch reports individual stream rates and their directly measured sum; this differs from the growing replay’s tokens per full wall-clock span.

```bash
DECODE_PROBE_SEED=15 DECODE_PROBE_WARMUP=1 DECODE_PROBE_OUT=results/c4.jsonl python3 bench/decode-probe.py 8001 400 3 4 prose
python3 bench/cpu-clock-screen.py --check-only
```


## Headline figures

The [2026-11-24 cached-code and Spec-Bench summary](results/2026-11-24-current-r18b-headline/) uses the published x16 ledger values. The detailed x16 report is unavailable locally. Regenerate its SVGs with `python3 bench/plot.py --headline bench/results/2026-11-24-current-r18b-headline/summary.json` from the repository root with Matplotlib installed. `fill_headline.py` is the r7f/r7g exporter for a full `headline-report/2` and the historical README marker layout.
