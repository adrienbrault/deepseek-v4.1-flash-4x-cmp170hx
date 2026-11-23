#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warm the r18 DeepSeek prefill paths on a healthy loopback listener."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from threading import Barrier
from time import monotonic
from uuid import uuid4

BASE_MODEL = "deepseek-v4.1-flash-1017"
ABLITERATED_MODEL = BASE_MODEL + "-abliterated"
# See WARMUP.md for the dispatch boundaries and expected request cost.
PREFILL_SIZES = (8, 16, 33, 64, 128, 1024, 2048 + 129)
MIXED_SIZE = 1024
REQUEST_TIMEOUT = 30
PASS_BUDGET = 90


def request_body(model, prompt_tokens, max_tokens=1):
    # Send raw IDs to avoid text BPE merges or a chat template changing the count.
    return {
        "model": model,
        "prompt": [42] * prompt_tokens,
        "add_special_tokens": False,
        "cache_salt": uuid4().hex,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }


def warm_request(port, model, size, max_tokens, phase, deadline, start_gate=None):
    body = json.dumps(request_body(model, size, max_tokens))
    if start_gate is not None:
        # Queue the HTTP requests together so the scheduler can mix them.
        start_gate.wait()
    start = monotonic()
    label = f"{phase} {model} prompt_tokens={size}"
    connection = None
    try:
        remaining = deadline - start
        if remaining <= 0:
            raise TimeoutError("warm-up budget exhausted")
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=min(REQUEST_TIMEOUT, remaining)
        )
        connection.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        with connection.getresponse() as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            result = json.load(response)
        usage = result.get("usage") if isinstance(result, dict) else None
        if not isinstance(usage, dict) or usage.get("prompt_tokens") != size:
            raise ValueError("server prompt count differs from requested count")
        if usage.get("completion_tokens") != max_tokens:
            raise ValueError("server did not generate the requested token count")
    except (OSError, http.client.HTTPException, ValueError) as exc:
        print(f"r18-warmup WARNING {label} seconds={monotonic() - start:.3f} {exc}", flush=True)
        return False
    finally:
        if connection is not None:
            connection.close()
    print(f"r18-warmup OK {label} seconds={monotonic() - start:.3f}", flush=True)
    return True


def warmup(port):
    start = monotonic()
    deadline = start + PASS_BUDGET
    models = (BASE_MODEL, ABLITERATED_MODEL)
    outcomes = [
        warm_request(port, model, size, 1, "single", deadline)
        for size in PREFILL_SIZES
        for model in models
    ]
    # Allow a few decode steps so there is also a chance to overlap decode with
    # prefill if the two requests land in different scheduler iterations.
    with ThreadPoolExecutor(max_workers=2) as executor:
        gate = Barrier(2, timeout=REQUEST_TIMEOUT)
        pending = [
            executor.submit(warm_request, port, model, MIXED_SIZE, 16, "mixed", deadline, gate)
            for model in models
        ]
        outcomes.extend(future.result() for future in pending)
    succeeded = sum(outcomes)
    print(
        f"r18-warmup finished seconds={monotonic() - start:.3f} "
        f"ok={succeeded} failed={len(outcomes) - succeeded}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001, help="r18 port on 127.0.0.1")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if os.environ.get("WARMUP", "1") == "0":
        print("r18-warmup skipped (WARMUP=0)", flush=True)
        return
    warmup(args.port)


if __name__ == "__main__":
    main()
