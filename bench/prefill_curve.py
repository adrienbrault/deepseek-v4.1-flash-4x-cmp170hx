#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cold-prefill curve on the serving daily (opera, 127.0.0.1:8001).

One request at a time. Each prompt is fresh random-word prose with its own cache
salt, so nothing hits the prefix cache. One forced output token, temperature 0.
Prefill rate = server-counted prompt tokens / client time to first token.
A request is VOID when another request was running on the server when it started
or ended, or when the server reports cached prompt tokens.
Usage: probe.py OUT.json [--targets 2000,8000,...] [--reps 3]
"""
import argparse, json, random, re, sys, time, urllib.request

URL = "http://127.0.0.1:8001"
MODEL = "deepseek-v4.1-flash-1017"
WORDS = ("river stone cloud lantern harbor meadow copper violet signal orchard canyon ember quartz willow "
         "falcon thistle granite marble cobalt saffron tundra glacier prairie juniper basalt cedar onyx "
         "pewter sorrel maple heron osprey bramble fennel indigo larch quarry spindle tarn vellum").split()


def post(path, body, timeout):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def running():
    text = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
    m = re.search(r'^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)', text, re.M)
    return float(m.group(1)) if m else -1.0


def ntokens(text):
    with post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": True}, 120) as r:
        return json.load(r)["count"]


def prose(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words)) + "."


def build(rng, target):
    # Calibrate words per token on a sample, then adjust once on the full prompt.
    words = max(16, int(target / 1.3))
    for _ in range(3):
        text = prose(rng, words)
        n = ntokens(text)
        if abs(n - target) <= target * 0.01:
            return text, n
        words = max(16, int(words * target / n))
    return text, n


def one(text, salt):
    body = dict(model=MODEL, prompt=text, max_tokens=1, temperature=0, ignore_eos=True, cache_salt=salt,
                stream=True, stream_options=dict(include_usage=True))
    start = time.monotonic(); first = None; usage = None
    with post("/v1/completions", body, 1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if first is None and chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                first = time.monotonic() - start
            if chunk.get("usage"):
                usage = chunk["usage"]
    return first, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--targets", default="2000,8000,30000,60000,120000,200000,240000")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random()
    rows = []
    for target in map(int, a.targets.split(",")):
        for rep in range(a.reps):
            text, n = build(rng, target)
            salt = f"prefill-curve-{time.time_ns()}-{rng.getrandbits(64):x}"
            before = running()
            ttft, usage = one(text, salt)
            after = running()
            cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            prompt = (usage or {}).get("prompt_tokens", n)
            void = before != 0 or after != 0 or cached != 0 or ttft is None
            row = dict(target=target, rep=rep, prompt_tokens=prompt, cached_tokens=cached, ttft_s=ttft,
                       prefill_tok_s=prompt / ttft if ttft else None, running_before=before,
                       running_after=after, void=void)
            rows.append(row)
            print(json.dumps(row), flush=True)
            json.dump(dict(schema="dsv41-prefill-curve/1", model=MODEL, rows=rows), open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
