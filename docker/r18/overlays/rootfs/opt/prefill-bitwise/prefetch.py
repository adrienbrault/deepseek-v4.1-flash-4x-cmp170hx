# SPDX-License-Identifier: MIT
"""0003: bounded future-chunk page-cache warming; never supplies model bytes.

Hash formula follows GNmac's dsv41_engram_prefetch and the incumbent Triton
kernel, including signed int64 wrap and padding. No torch/model imports.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from common import enabled


def i64(n):
    return (n + (1 << 63)) % (1 << 64) - (1 << 63)


def hashes(prompt, start, end, token_map, multipliers, primes, offsets, pad):
    """[positions][layers][columns]; raw text tokens only, no dead/image tokens."""
    if not 0 <= start <= end <= len(prompt):
        raise ValueError('invalid prompt interval')
    if any(v < 0 or v >= len(token_map) for v in prompt):
        raise ValueError('unknown/multimodal token; skip this request')
    out = []
    for pos in range(start, end):
        row = []
        for layer, mult in enumerate(multipliers):
            rolling, values = 0, []
            for shift, factor in enumerate(mult):
                value = pad if pos < shift else token_map[prompt[pos - shift]]
                rolling = i64(rolling ^ i64(value * factor))
                if shift:
                    for head, prime in enumerate(primes[layer][shift - 1]):
                        if prime <= 0:
                            raise ValueError('hash prime must be positive')
                        col = (shift - 1) * len(primes[layer][shift - 1]) + head
                        values.append(rolling % prime + offsets[layer][col])
            row.append(values)
        out.append(row)
    return out


class RowReader:
    """Own read-only fd, bounded pread; arbitrary bytes never reach the model.

    regions: (weight_offset, weight_bytes_per_row), (scale_offset, scale_bytes_per_row).
    Shard-global row IDs outside this rank's vocab range are ignored, like row_store_range.
    """
    def __init__(self, path, rows, regions, lo=0, hi=None):
        self.fd = os.open(path, os.O_RDONLY)
        self.rows, self.lo, self.hi = rows, lo, rows if hi is None else hi
        self.regions = tuple(regions)
        size = os.fstat(self.fd).st_size
        if not 0 <= self.lo <= self.hi <= rows or any(
                off < 0 or width <= 0 or off + rows * width > size for off, width in self.regions):
            self.close()
            raise ValueError('row region outside shard')

    def warm(self, ids):
        count = 0
        for row in sorted(set(ids)):
            if not 0 <= row < self.rows:
                raise ValueError('row ID outside table')
            if not self.lo <= row < self.hi:
                continue
            for off, width in self.regions:
                data = os.pread(self.fd, width, off + row * width)
                if len(data) != width:
                    raise OSError('short shard read')
                count += len(data)
        return count

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class FutureChunks:
    """One running job, zero queued jobs; explicit off drains before A timing.

    A dropped hint is retried at a later step; request IDs never retain a stale
    prompt across reuse. Callback receives immutable prompt and absolute interval.
    """
    def __init__(self, read, chunk=2048, ahead=1):
        if not 1 <= chunk <= 2048 or not 1 <= ahead <= 2:
            raise ValueError('bounded chunk/ahead required')
        self.read, self.chunk, self.ahead = read, chunk, ahead
        self.on = enabled(os.environ, 11)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix='prefill-engram')
        self.future = None
        self.requests = {}
        self.closed = False
        self.stats = dict(submitted=0, dropped=0, errors=0, bytes=0)
        self.lock = threading.Lock()

    def _collect(self, wait=False):
        if self.future is not None and (wait or self.future.done()):
            try:
                self.stats['bytes'] += self.future.result()
            except Exception:
                self.stats['errors'] += 1
            self.future = None

    def select(self, on):
        with self.lock:
            if self.closed:
                raise RuntimeError('prefetcher closed')
            # Drain at drained client boundaries, never during a measured forward.
            if not on:
                self._collect(wait=True)
            self.on = bool(on)

    def step(self, new, computed, scheduled, finished=()):
        """new maps req_id->full raw-token prompt; computed is pre-step position."""
        with self.lock:
            if self.closed:
                raise RuntimeError('prefetcher closed')
            self._collect()
            for rid in finished:
                self.requests.pop(rid, None)
            for rid, prompt in new.items():
                if prompt:  # replacement also handles request-slot reuse
                    self.requests[rid] = (tuple(prompt), 0)
                else:
                    self.requests.pop(rid, None)
            for rid, n in scheduled.items():
                if rid not in self.requests or rid not in computed:
                    continue
                prompt, done = self.requests[rid]
                nxt = computed[rid] + n
                if n <= 0 or nxt < 0:
                    continue
                if nxt >= len(prompt):
                    self.requests.pop(rid, None)
                    continue
                start, end = max(nxt, done), min(len(prompt), nxt + self.ahead * self.chunk)
                if not self.on or start >= end:
                    continue
                if self.future is not None:
                    self.stats['dropped'] += 1
                    continue
                self.future = self.pool.submit(self.read, prompt, start, end)
                self.requests[rid] = (prompt, end)
                self.stats['submitted'] += 1

    def close(self):
        with self.lock:
            if self.closed:
                return
            self._collect(wait=True)
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.requests.clear()
            self.closed = True


class EngramPlan:
    """Bind an already-loaded incumbent runner; no import-time side effects.

    Construct before scored work. Caller invokes after_execute after the normal
    execute_model returns, and select(False) at a drained OFF boundary. Tables
    are opened separately; the native row cache/handles are never touched.
    """
    def __init__(self, runner, model_dir, chunk=2048, ahead=1):
        from vllm.models.deepseek_v41.ssd.engram_ssd import find_engram_shard
        model = runner.get_model()
        model = model.unwrap() if hasattr(model, 'unwrap') else model
        modules = list(model.modules())
        hs = [m for m in modules if type(m).__name__ == 'NgramHashState']
        es = [m for m in modules if type(m).__name__ == 'Engram'
              and getattr(m.embed_tokens, 'ssd_table', None) is not None]
        if len(hs) != 1 or not es:
            raise RuntimeError('expected one hash state and existing NVMe Engram tables')
        h = hs[0]
        self.token_map = h.token_map.cpu().tolist()
        self.mult = h.multipliers.cpu().tolist()
        self.primes = h.primes.cpu().tolist()
        self.offsets = h.offsets.cpu().tolist()
        self.pad = int(h.pad_id)
        self.tables = []
        try:
            for e in es:
                emb = e.embed_tokens
                path, rows, woff, soff = find_engram_shard(model_dir, h.layout.layer_ids[e.layer_hash_index])
                reader = RowReader(path, rows, ((woff, 256), (soff, 8)),
                                   emb.vocab_start_idx, emb.vocab_end_idx)
                self.tables.append((e.layer_hash_index, emb.head_start, emb.part_n_hash_cols, reader))
            self.worker = FutureChunks(self.read, chunk, ahead)
        except BaseException:
            for _, _, _, r in self.tables:
                r.close()
            raise

    def read(self, prompt, start, end):
        ids = hashes(prompt, start, end, self.token_map, self.mult, self.primes, self.offsets, self.pad)
        return sum(reader.warm([v for row in ids for v in row[layer][h0:h0 + nh]])
                   for layer, h0, nh, reader in self.tables)

    def after_execute(self, so):
        # Never hash multimodal/dead/replaced token sequences as plain text.
        new = {r.req_id: (r.prompt_token_ids if not getattr(r, 'mm_features', None) else ())
               for r in so.scheduled_new_reqs}
        computed = {r.req_id: r.num_computed_tokens for r in so.scheduled_new_reqs}
        cached = so.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        self.worker.step(new, computed, so.num_scheduled_tokens, so.finished_req_ids)

    def select(self, on):
        self.worker.select(on)

    def close(self):
        self.worker.close()
        for _, _, _, reader in self.tables:
            reader.close()
