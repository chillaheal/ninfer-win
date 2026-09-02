// Persistent worker pool for the CPU vision encoder's parallel kernels (2026-09-02).
//
// Replaces the per-call std::thread spawn+join both pfor variants used. The 2026-09-02
// profile measured ~1.7 ms/layer of spawn+join under attention alone (120 ms wall vs ~73 ms
// of per-job phase time over 16 jobs); the same overhead sits under every GEMM/elementwise
// pfor call (~138 per encode). The workers are created once (min(16, hw) = the old per-call
// cap — 16 = one per physical core on the 9950X3D; SMT pairs add no fp32-FMA throughput) and
// steal blocks dynamically. The barrier preserves the old join contract: the dispatcher
// returns only after every body invocation has completed, so bodies may reference the
// caller's frame.
//
// Semantics unchanged vs the old spawn+join: the blocks are independent (per-block bodies
// write disjoint output); the bodies are [&]-style closures with no mutable closure state,
// so concurrent invocation of the one shared closure object is safe (the old version copied
// the closure per thread — that copy only ever protected mutable closure members, of which
// none exist); thread_local state inside the bodies now PERSISTS across calls (the old
// thread-local died with the thread) — every body re-stages its buffers per call and never
// assumes fresh per-call state.
//
// NOT reentrant: a body must never call pfor/pfor_light (the workers would hold each other
// at the barrier) — all current bodies are pure per-block work.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
// windows.h defines the min/max MACROS (windef.h) without NOMINMAX — the project CMake sets
// it globally (CMakeLists.txt:39), but standalone benches (fc2_sustained.bat) don't, so the
// guard goes here: std::min/max would otherwise break in any TU that includes this header
// before the project-wide define (C2059 at vision_cpu_simd.cpp:803, 2026-09-02).
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // SetThreadAffinityMask (NINFER_VISION_PFOR_AFFINITY override)
#endif

namespace ninfer::targets::qwen3_6::detail {

struct PforPool {
    static constexpr std::size_t kCap = 16;  // the old per-call thread cap

    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv_work;  // dispatch -> workers
    std::condition_variable cv_done;  // last block done -> the waiting dispatcher
    std::function<void(std::size_t)> body;  // guarded by mtx; valid dispatch..barrier
    std::size_t count = 0;
    std::size_t next  = 0;
    std::atomic<std::size_t> finished{0};
    unsigned gen      = 0;
    bool stopping     = false;

    PforPool() {
        const unsigned hw  = std::max(1u, std::thread::hardware_concurrency());
        const unsigned nth = std::min<unsigned>(kCap, hw);
        for (unsigned i = 0; i < nth; ++i) {
            workers.emplace_back([this] {
                unsigned seen = 0;
                for (;;) {
                    std::size_t blk = 0;
                    bool took      = false;
                    std::size_t n  = 0;  // generation's block count, captured under the lock
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        if (next < count) {
                            // Steal immediately: never park while the generation still has
                            // work. (The first version parked after one block on the
                            // predicate "gen != seen" alone — with count > worker count the
                            // first wave took one block each and every worker re-parked in
                            // the SAME generation, so blocks 17..count were never taken and
                            // the dispatcher waited on cv_done forever.)
                            blk  = next++;
                            n    = count;
                            took = true;
                            seen = gen;
                        } else {
                            // Park only when this generation is exhausted. Wake on the next
                            // dispatch (gen change); the `next < count` clause re-checks on
                            // spurious wakeups. `seen` is synced in BOTH branches so a
                            // parked worker never holds a stale gen and busy-loops.
                            cv_work.wait(lk, [this, &seen] {
                                return stopping || gen != seen || next < count;
                            });
                            if (stopping) { return; }
                            seen = gen;
                            if (next < count) { blk = next++; n = count; took = true; }
                        }
                    }
                    if (took) {
                        body(blk);
                        // `n` (not `count`) for the last-block test: the next generation's
                        // dispatch cannot have reset `count` before this generation's barrier
                        // (single-dispatcher invariant: run() blocks on cv_done, so no second
                        // run() can start until every block of this one has finished).
                        if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
                            cv_done.notify_all();
                        }
                    }
                }
            });
        }
#ifdef _WIN32
        // NINFER_VISION_PFOR_AFFINITY (2026-09-02, #19 STEP 10b): pin the pool workers to a
        // set of logical processors — "first16" (0-15), "last16" (hw-16..hw-1), or a
        // comma-separated list of inclusive ranges ("0-15", "16-31", "0-7,16-23");
        // unset/"all" = the OS default (workers spread across both CCDs). Experiment knob for
        // the 9950X3D V-Cache CCD: the 96 MB V-Cache lives on ONE CCD (8 cores = 16 SMT
        // threads = exactly kCap); pinning the whole pool there keeps the ViT working set
        // (~50-60 MB: X + W + C + activations) local to that CCD's cache instead of spilling
        // to DRAM / crossing Infinity Fabric. Parsed ONCE here (workers are created once);
        // a failed SetThreadAffinityMask leaves the worker unpinned (degenerate case only).
        const DWORD mask = parse_affinity_mask();
        if (mask != 0) {
            for (std::thread& w : workers) {
                const HANDLE h = static_cast<HANDLE>(w.native_handle());
                SetThreadAffinityMask(h, mask);
            }
        }
#endif
    }

#ifdef _WIN32
    // "all"/unset = 0 (no mask). "first16"/"last16" and inclusive ranges of logical
    // processor indices. Indices >= 64 are ignored (the DWORD mask covers one group; this
    // machine has 32 logical processors).
    static DWORD parse_affinity_mask() {
        const char* e = std::getenv("NINFER_VISION_PFOR_AFFINITY");
        if (e == nullptr || e[0] == '\0' || std::strcmp(e, "all") == 0) { return 0; }
        if (std::strcmp(e, "first16") == 0) { return (1u << 16) - 1u; }
        if (std::strcmp(e, "last16") == 0) {
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            const unsigned lo = hw > 16 ? hw - 16 : 0;
            DWORD mask = 0;
            for (unsigned i = lo; i < hw; ++i) { mask |= 1u << i; }
            return mask;
        }
        DWORD mask = 0;
        const char* p = e;
        while (*p) {
            const char* dash = std::strpbrk(p, "-");
            if (dash == nullptr) { break; }
            const long a = std::atol(p);
            const long b = std::atol(dash + 1);
            for (long i = a; i <= b && i < 64; ++i) {
                if (i >= 0) { mask |= 1u << i; }
            }
            p = dash + 1;
            while (*p && *p != ',') { ++p; }
            if (*p == ',') { ++p; }
        }
        return mask;
    }
#endif

    ~PforPool() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stopping = true;
        }
        cv_work.notify_all();
        for (std::thread& t : workers) { t.join(); }
    }

    void run(std::size_t n, std::function<void(std::size_t)> b) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            body   = std::move(b);
            count  = n;
            next   = 0;
            finished.store(0, std::memory_order_relaxed);
            ++gen;
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this, n] { return finished.load(std::memory_order_acquire) >= n; });
    }

    static PforPool& instance() {
        static PforPool pool;  // Meyers: constructed on first pfor, workers joined at exit
        return pool;
    }

    // Worker count. The vector is only mutated in the constructor, so this needs no lock.
    std::size_t size() const { return workers.size(); }
};

// Shared dispatcher: blocks are stolen dynamically across the persistent workers; returns
// after all of them have completed (the old join contract).
//
// Coarsening: the dispatch unit is a RUN of `per = ceil(count/nth)` units, so the block count
// stays at ~nth. The elementwise kernels block at 16-lane vector granularity (gelu: 59180
// blocks of 16 floats); stealing one of those per global-mutex round-trip cost ~4 us/block
// against ~100 ns of block work — measured 15.9 ms/call idle (pfor_bench, 2026-09-02) and
// 30-38 ms/call in-encode (A/B step10/step10b) vs ~1-2 ms with the old contiguous-range
// spawn+join. Coarse blocks reproduce that old distribution (each block is a contiguous run,
// as the old per-thread ranges were); the unit bodies are invoked exactly once each and are
// independent, so the results are unchanged.
template <typename Body>
void pfor_run(std::size_t count, Body&& body) {
    if (count == 0) { return; }
    PforPool& pool = PforPool::instance();
    const std::size_t nth = pool.size();
    const std::size_t per = (count + nth - 1) / nth;
    const std::size_t blocks = (count + per - 1) / per;
    pool.run(blocks, [&](std::size_t b) {
        const std::size_t lo = b * per;
        const std::size_t hi = lo + per < count ? lo + per : count;
        for (std::size_t i = lo; i < hi; ++i) { body(i); }
    });
}

// Parallel-for for the AVX-512 kernels (the blocks are heavy, ~1 ms+ each): parallelize as
// soon as there is more than one unit.
template <typename Body>
void pfor(std::size_t count, Body&& body) {
    if (count <= 1) {
        for (std::size_t i = 0; i < count; ++i) { body(i); }
        return;
    }
    pfor_run(count, std::forward<Body>(body));
}

// Parallel-for for the scalar fallback kernels (lighter 16-element chunks / rows): below 64
// units the dispatch round-trip exceeds the work (the same threshold as the old spawn+join).
template <typename Body>
void pfor_light(std::size_t count, Body&& body) {
    if (count < 64) {
        for (std::size_t i = 0; i < count; ++i) { body(i); }
        return;
    }
    pfor_run(count, std::forward<Body>(body));
}

}  // namespace ninfer::targets::qwen3_6::detail
