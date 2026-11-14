// Engram row store: exact FP8 row + UE8M0 scale retrieval straight from the
// original safetensors shard on the NVMe, served through the kernel page cache.
//
// Design adapted from 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 adapter/row_store.cpp
// (MIT, Copyright (c) 2026 0xSero); rewritten here with a thread pool so that a
// prefill chunk (thousands of tokens x 24 rows) does not serialize on one pread.
// No CUDA calls in the callback: it is launched via cudaLaunchHostFunc and is
// therefore legal inside CUDA graph capture/replay.
//
// Row layout in the checkpoint (per Engram table):
//   weight: [rows, 256] F8_E4M3   at byte offset woff (contiguous 256 B rows)
//   scale : [rows,   8] F8_E8M0   at byte offset soff (contiguous   8 B rows)
#include <atomic>
#include <array>
#include <memory>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <sys/stat.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kBatchRows = 64;

struct Pool {
  std::vector<std::thread> threads;
  std::mutex m;
  std::condition_variable cv, done_cv;
  std::function<void(int)> job;   // job(index)
  uint64_t njobs = 0, next = 0, finished = 0;
  bool stop = false;
  explicit Pool(int n) {
    for (int i = 0; i < n; ++i) threads.emplace_back([this] { loop(); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> g(m); stop = true; }
    cv.notify_all();
    for (auto &t : threads) t.join();
  }
  void loop() {
    for (;;) {
      uint64_t idx;
      {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return stop || next < njobs; });
        if (stop) return;
        idx = next++;
      }
      job(idx);
      {
        std::lock_guard<std::mutex> g(m);
        if (++finished == njobs) done_cv.notify_all();
      }
    }
  }
  void run(uint64_t n, std::function<void(int)> f) {
    {
      std::lock_guard<std::mutex> g(m);
      job = std::move(f); njobs = n; next = 0; finished = 0;
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lk(m);
    done_cv.wait(lk, [&] { return finished == njobs; });
  }
};


// Bounded direct-mapped row cache, local to this immutable Store. Stripe locks
// protect tag + payload as a unit; misses read outside locks. Concurrent misses
// for the same row can duplicate I/O, but can never expose an unfinished row.
struct RowCache {
  struct Entry {
    int64_t id = -1;
    uint8_t w[256], sc[8];
  };
  static constexpr size_t stripes = 64;
  std::array<std::mutex, stripes> locks;
  std::vector<Entry> entries;
  std::atomic<uint64_t> hits{0}, misses{0};
  explicit RowCache(uint64_t bytes) {
    size_t n = 1;
    while ((n * 2) <= bytes / sizeof(Entry)) n *= 2;
    entries.resize(n);
  }
  size_t slot(int64_t id) const {
    uint64_t x = uint64_t(id);
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x & (entries.size() - 1);
  }
  bool get(int64_t id, uint8_t *w, uint8_t *sc) {
    size_t i = slot(id);
    std::lock_guard<std::mutex> guard(locks[i % stripes]);
    const Entry &e = entries[i];
    if (e.id != id) { misses.fetch_add(1, std::memory_order_relaxed); return false; }
    std::memcpy(w, e.w, 256); std::memcpy(sc, e.sc, 8);
    hits.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  void put(int64_t id, const uint8_t *w, const uint8_t *sc) {
    size_t i = slot(id);
    std::lock_guard<std::mutex> guard(locks[i % stripes]);
    Entry &e = entries[i];
    std::memcpy(e.w, w, 256); std::memcpy(e.sc, sc, 8); e.id = id;
  }
};

struct Store {
  std::unique_ptr<RowCache> cache;
  int fd = -1;
  uint64_t rows = 0, woff = 0, soff = 0, row_lo = 0, row_hi = 0;
  Pool *pool = nullptr;
  bool batch_reads = false;
  std::atomic<bool> nowait_supported{true};
  std::atomic<uint64_t> lookups{0}, batches{0};
};

struct Work {
  Store *store;
  const int64_t *ids;   // [count] host (pinned) row ids, -1 = skip
  uint8_t *weights;     // [count, 256] host (pinned)
  uint8_t *scales;      // [count, 8]   host (pinned)
  uint64_t count;
};

void fail(const char *reason) {
  std::fprintf(stderr, "engram row_store: %s (errno=%d %s)\n", reason, errno, std::strerror(errno));
  std::abort();  // never let a generation continue with missing/stale rows
}

void pread_all(int fd, void *out, size_t len, uint64_t off) {
  uint8_t *p = static_cast<uint8_t *>(out);
  while (len) {
    ssize_t got = pread(fd, p, len, off);
    if (got < 0) { if (errno == EINTR) continue; fail("pread failed"); }
    if (got == 0) fail("short read (EOF)");
    p += got; len -= got; off += got;
  }
}

void fetch_one(Store *s, int64_t id, uint8_t *w, uint8_t *sc) {
  if (id < 0 || uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
    std::memset(w, 0, 256); std::memset(sc, 0, 8); return;
  }
  if (uint64_t(id) >= s->rows) fail("row id out of bounds");
  pread_all(s->fd, w, 256, s->woff + uint64_t(id) * 256);
  pread_all(s->fd, sc, 8, s->soff + uint64_t(id) * 8);
}

// Adapted from GNmac-lab's MIT-licensed patch_row_store_overlap.py.
// Copyright (c) 2026 GNmac-lab; see NOTICE.row_store_batch.md.
// Only a complete cache hit is consumed. A partial probe is overwritten by the
// normal exact blocking read, never exposed as a completed/stale row.
bool read_cached(Store *s, void *out, size_t len, uint64_t off) {
  if (!s->nowait_supported.load(std::memory_order_relaxed)) return false;
  struct iovec v{out, len};
  for (;;) {
    const ssize_t got = preadv2(s->fd, &v, 1, off_t(off), RWF_NOWAIT);
    if (got == ssize_t(len)) return true;
    if (got >= 0 || errno == EAGAIN) return false;
    if (errno == EINTR) continue;
    if (errno == EOPNOTSUPP || errno == ENOSYS || errno == EINVAL) {
      s->nowait_supported.store(false, std::memory_order_relaxed);
      return false;
    }
    fail("preadv2 failed");
  }
}

struct Miss { uint8_t *dst; size_t len; uint64_t off; };

void fetch_range_uncached(Store *s, const int64_t *ids, uint8_t *weights,
                 uint8_t *scales, uint64_t lo, uint64_t hi) {
  if (!s->batch_reads) {
    for (uint64_t i = lo; i < hi; ++i)
      fetch_one(s, ids[i], weights + i * 256, scales + i * 8);
    return;
  }
  // At most two descriptors per row, independent of total prefill size.
  Miss misses[2 * kBatchRows];
  size_t count = 0;
  for (uint64_t i = lo; i < hi; ++i) {
    const int64_t id = ids[i];
    uint8_t *w = weights + i * 256, *sc = scales + i * 8;
    if (id < 0 || uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(w, 0, 256); std::memset(sc, 0, 8); continue;
    }
    if (uint64_t(id) >= s->rows) fail("row id out of bounds");
    const uint64_t woff = s->woff + uint64_t(id) * 256;
    const uint64_t soff = s->soff + uint64_t(id) * 8;
    if (!read_cached(s, w, 256, woff)) misses[count++] = {w, 256, woff};
    if (!read_cached(s, sc, 8, soff)) misses[count++] = {sc, 8, soff};
  }
  // Advisory hints may fail or complete synchronously. Correctness depends
  // only on the exact reads below; this is not guaranteed async disk I/O.
  for (size_t k = 0; k < count; ++k)
    posix_fadvise(s->fd, off_t(misses[k].off), off_t(misses[k].len), POSIX_FADV_WILLNEED);
  for (size_t k = 0; k < count; ++k)
    pread_all(s->fd, misses[k].dst, misses[k].len, misses[k].off);
}


void fetch_range(Store *s, const int64_t *ids, uint8_t *w, uint8_t *sc,
                 uint64_t lo, uint64_t hi) {
  if (!s->cache) { fetch_range_uncached(s, ids, w, sc, lo, hi); return; }
  // Preserve the baseline's up-to-64-row batched miss handling. Cache hits
  // split miss runs; each run is filled completely before its tags publish.
  uint64_t begin = lo;
  auto finish = [&](uint64_t end) {
    if (begin == end) return;
    fetch_range_uncached(s, ids, w, sc, begin, end);
    for (uint64_t k = begin; k < end; ++k) {
      int64_t id = ids[k];
      if (id >= 0 && uint64_t(id) >= s->row_lo && uint64_t(id) < s->row_hi)
        s->cache->put(id, w + k * 256, sc + k * 8);
    }
  };
  for (uint64_t k = lo; k < hi; ++k) {
    int64_t id = ids[k];
    // Invalid/foreign IDs retain exact baseline zero-fill behavior.
    if (id >= 0 && uint64_t(id) >= s->row_lo && uint64_t(id) < s->row_hi &&
        s->cache->get(id, w + k * 256, sc + k * 8)) {
      finish(k); begin = k + 1;
    }
  }
  finish(hi);
}

}  // namespace

extern "C" Store *row_store_open(const char *path, uint64_t rows, uint64_t woff,
                                 uint64_t soff, int nthreads) {
  auto *s = new Store;
  s->fd = open(path, O_RDONLY | O_CLOEXEC);
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat st;
  if (fstat(s->fd, &st) || woff + rows * 256 > uint64_t(st.st_size) ||
      soff + rows * 8 > uint64_t(st.st_size)) { close(s->fd); delete s; return nullptr; }
  s->rows = rows; s->woff = woff; s->soff = soff; s->row_lo = 0; s->row_hi = rows;
  const char *batch = std::getenv("DSV41_ENGRAM_BATCH_READS");
  s->batch_reads = batch && std::strcmp(batch, "1") == 0;
  if (s->batch_reads) posix_fadvise(s->fd, 0, 0, POSIX_FADV_RANDOM);
  if (nthreads < 1) nthreads = 1;
  const char *setting = std::getenv("DSV41_EG_ROW_CACHE_MB");
  if (setting && std::strcmp(setting, "0") != 0) {
    char *end = nullptr;
    unsigned long mb = std::strtoul(setting, &end, 10);
    if (!*setting || *end || mb < 1 || mb > 512) fail("row cache MB must be 0..512");
    s->cache.reset(new RowCache(uint64_t(mb) << 20));
    std::fprintf(stderr, "EG_ROW_CACHE entries=%zu bytes=%zu requested_mb=%lu\n",
                 s->cache->entries.size(), s->cache->entries.size() * sizeof(RowCache::Entry), mb);
  }
  s->pool = new Pool(nthreads);
  return s;
}

extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  s->row_lo = lo; s->row_hi = hi;
}

// Host callback (cudaLaunchHostFunc signature: void(*)(void*)).
extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  const uint64_t n = work->count;
  s->lookups.fetch_add(n, std::memory_order_relaxed);
  s->batches.fetch_add(1, std::memory_order_relaxed);
  if (n == 0) return;
  const uint64_t chunk = kBatchRows;
  const uint64_t njobs = (n + chunk - 1) / chunk;
  if (njobs == 1) {
    fetch_range(s, work->ids, work->weights, work->scales, 0, n);
    return;
  }
  s->pool->run(njobs, [&](int j) {
    const uint64_t lo = uint64_t(j) * chunk, hi = std::min<uint64_t>(n, lo + chunk);
    fetch_range(s, work->ids, work->weights, work->scales, lo, hi);
  });
}

// Synchronous variant for tests / warmup (same semantics).
extern "C" void row_store_lookup_sync(Store *s, const int64_t *ids, uint8_t *weights,
                                      uint8_t *scales, uint64_t count) {
  Work w{s, ids, weights, scales, count};
  row_store_lookup(&w);
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->lookups.load(); out[1] = s->batches.load();
}

extern "C" void row_store_close(Store *s) {
  if (!s) return;
  delete s->pool;
  close(s->fd);
  delete s;
}


// New optional ABI; the existing stats struct and Work layout stay unchanged.
extern "C" void row_store_cache_stats(Store *s, uint64_t *out) {
  out[0] = s->cache ? s->cache->hits.load() : 0;
  out[1] = s->cache ? s->cache->misses.load() : 0;
  out[2] = s->cache ? s->cache->entries.size() * sizeof(RowCache::Entry) : 0;
}
extern "C" int row_store_gap_revision() { return 1; }
