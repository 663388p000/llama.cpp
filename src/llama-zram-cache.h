#pragma once

// ============================================================================
// llama-zram-cache.h
//
// Dynamic, frequently-refreshed cache subsystem backed by Linux zram
// (compressed RAM block device).
//
// Goals (as requested):
//   1. Detect total system RAM and the presence/creatability of zram devices.
//   2. Route "dynamic" caches -- i.e. buffers that are rewritten very often,
//      such as KV-cache pages, compressed-attention state, indexer keys,
//      etc. -- through zram-backed storage instead of plain, uncompressed
//      heap/mmap memory. ALL such caches are required to live in zram.
//   3. Apply UMA (Unified Memory Architecture) aware allocation policy: on
//      platforms where the CPU and the compute accelerator (integrated
//      GPU/NPU) share one physical RAM pool, avoid the extra host<->device
//      staging copies that a discrete-GPU (dedicated VRAM) design would
//      need. NOTE: "UMA" here refers to *shared system RAM*, not VRAM on a
//      discrete GPU -- those are different memory pools and are out of
//      scope for this subsystem.
//   4. Optionally use O_DIRECT ("DMA-style") I/O against the zram block
//      device to bypass the page cache for large transfers, when the
//      kernel and the underlying block device support it. True hardware
//      DMA is a driver-level concept not reachable from userspace; O_DIRECT
//      is the closest userspace-visible approximation and is what's
//      implemented here.
//
// This subsystem is Linux-only, since zram is a Linux kernel driver. On
// non-Linux platforms every entry point still works but transparently
// degrades to a plain anonymous-memory cache (LLAMA_CACHE_BACKEND_RAM_PLAIN).
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// System memory topology detection
// ----------------------------------------------------------------------------

struct llama_mem_info {
    uint64_t ram_total_bytes  = 0; // /proc/meminfo MemTotal
    uint64_t ram_avail_bytes  = 0; // /proc/meminfo MemAvailable
    uint64_t swap_total_bytes = 0; // /proc/meminfo SwapTotal

    uint64_t zram_total_bytes = 0; // sum of disksize of all active zram devices found
    bool     zram_supported   = false; // zram-control / zram module usable on this system

    // true if the CPU and the compute accelerator (GPU/NPU) address the same
    // physical RAM pool (Unified Memory Architecture) -- e.g. most ARM SBCs,
    // laptop/desktop iGPUs, some SoCs. This is a heuristic (see .cpp for the
    // detection method) and is NOT about a discrete GPU's local VRAM.
    bool     is_uma = false;

    std::string uma_reason; // human readable detection note, useful for logging
};

// Detect total/available RAM, swap, and zram support, and estimate whether the
// platform is UMA. Cheap enough to call once at startup and cache the result;
// safe to call repeatedly (e.g. to refresh ram_avail_bytes).
llama_mem_info llama_mem_info_detect();

// ----------------------------------------------------------------------------
// zram device management
// ----------------------------------------------------------------------------

struct llama_zram_device {
    std::string dev_path;       // e.g. "/dev/zram3"
    std::string sys_path;       // e.g. "/sys/block/zram3"
    uint64_t    disksize = 0;   // logical (uncompressed) size, bytes
    std::string comp_algorithm; // e.g. "zstd", "lz4", "lzo-rle"
    bool        is_new = false; // true if this process created the device
};

// Process-wide manager for zram devices used as cache backing storage.
class llama_zram_manager {
public:
    static llama_zram_manager & instance();

    // Whether the zram kernel driver / sysfs control interface is usable.
    bool available() const { return m_available; }

    // Ensure a zram device with logical size >= min_size_bytes exists and is
    // activated (creating one via /sys/class/zram-control/hot_add if
    // necessary), and set its compression algorithm. Returns nullptr on
    // failure (caller should fall back to plain RAM).
    //
    // NOTE: min_size_bytes is the *logical* (uncompressed) size exposed by
    // the block device; the kernel compresses pages transparently, so actual
    // resident RAM usage is typically a fraction of this, workload dependent
    // (roughly 30-60% of logical size for zstd on typical KV-cache-shaped
    // data, better for sparse/idle regions since untouched pages cost zero).
    const llama_zram_device * get_or_create_device(uint64_t min_size_bytes,
                                                     const char * comp_algorithm = "zstd");

    // Grow an existing device's disksize if it was created by us and the new
    // size is larger. zram only supports resizing while the device has no
    // data written yet in most kernel versions, so this is best-effort; on
    // failure the caller should create a new (bigger) device instead.
    bool try_grow_device(const llama_zram_device * dev, uint64_t new_min_size_bytes);

    // Reset/remove a device this manager created. No-op for devices this
    // process did not create (e.g. pre-existing system zram swap devices).
    void release_device(const llama_zram_device * dev);

private:
    llama_zram_manager();
    ~llama_zram_manager();

    bool detect_support();
    static bool write_sysfs(const std::string & path, const std::string & value);
    static bool read_sysfs(const std::string & path, std::string & out);

    bool         m_available = false;
    std::mutex   m_mutex;
    std::vector<std::unique_ptr<llama_zram_device>> m_devices;
};

// ----------------------------------------------------------------------------
// Dynamic (hot / frequently-refreshed) cache backed by zram
// ----------------------------------------------------------------------------

enum class llama_cache_backend {
    LLAMA_CACHE_BACKEND_RAM_PLAIN,   // plain anonymous RAM -- fallback only, no zram found
    LLAMA_CACHE_BACKEND_RAM_UMA,     // anonymous RAM, tuned for UMA sharing -- fallback only
    LLAMA_CACHE_BACKEND_ZRAM,        // compressed RAM block device, buffered pread/pwrite
    LLAMA_CACHE_BACKEND_ZRAM_DIRECT, // same, but O_DIRECT ("DMA-style") path for large I/O
};

struct llama_dynamic_cache_config {
    // Logical (uncompressed) capacity requested for this cache instance.
    uint64_t capacity_bytes = 0;

    // All dynamic caches are required to live in zram; this is enforced by
    // default (true). Set to false only for debugging/benchmarking against a
    // plain-RAM baseline.
    bool force_zram = true;

    // Attempt O_DIRECT (bypass page cache) for reads/writes at or above this
    // size, when offset/length/buffer alignment allows it. 0 disables it.
    size_t direct_io_min_bytes = 1u << 20; // 1 MiB

    const char * comp_algorithm = "zstd";

    std::string debug_name = "llama_dynamic_cache";
};

// A generic "hot data" cache: a flat logical address space of `capacity()`
// bytes, backed entirely by a zram block device. Intended for data that is
// rewritten very frequently (KV-cache pages, compressed-attention state,
// lightning-indexer keys, ...), where:
//   - the *logical* footprint can exceed what would be comfortable as plain,
//     uncompressed RAM,
//   - the *physical* RAM footprint is reduced via in-kernel compression,
//   - cold/idle regions cost far less resident memory than hot ones, since
//     zram compresses per page and untouched pages are never allocated.
//
// I/O goes directly against the zram block device node via pread/pwrite (or
// their O_DIRECT counterparts) -- there is no filesystem involved, matching
// zram's "RAM disk" semantics.
class llama_dynamic_zram_cache {
public:
    explicit llama_dynamic_zram_cache(const llama_dynamic_cache_config & cfg);
    ~llama_dynamic_zram_cache();

    llama_dynamic_zram_cache(const llama_dynamic_zram_cache &) = delete;
    llama_dynamic_zram_cache & operator=(const llama_dynamic_zram_cache &) = delete;

    llama_cache_backend backend() const { return m_backend; }
    bool                is_zram_backed() const {
        return m_backend == llama_cache_backend::LLAMA_CACHE_BACKEND_ZRAM ||
               m_backend == llama_cache_backend::LLAMA_CACHE_BACKEND_ZRAM_DIRECT;
    }

    // Write `len` bytes at logical `offset`. Returns false on I/O error or
    // out-of-range access.
    bool write(uint64_t offset, const void * data, size_t len);

    // Read `len` bytes at logical `offset`. Returns false on I/O error or
    // out-of-range access. As with any block device, reading a region that
    // was never written returns whatever zero/garbage the backend yields --
    // callers that need "was this written" semantics should track their own
    // validity bitmap (KV-cache cell metadata already does this).
    bool read(uint64_t offset, void * data, size_t len) const;

    // Best-effort hint that [offset, offset+len) will not be read again soon.
    // Currently informational only (no discard/TRIM is issued to zram in
    // this implementation); kept as an API seam for future work.
    void invalidate(uint64_t offset, size_t len);

    uint64_t capacity() const { return m_cfg.capacity_bytes; }

    struct stats_t {
        uint64_t bytes_written      = 0;
        uint64_t bytes_read         = 0;
        uint64_t n_direct_io_ops    = 0;
        uint64_t n_buffered_io_ops  = 0;
    };
    stats_t stats() const;

private:
    bool open_zram_backend();
    bool open_ram_backend(bool uma_tuned);

    bool pio(uint64_t offset, void * buf, size_t len, bool is_write);
    bool pio_direct(uint64_t offset, void * buf, size_t len, bool is_write);

    llama_dynamic_cache_config m_cfg;
    llama_cache_backend        m_backend = llama_cache_backend::LLAMA_CACHE_BACKEND_RAM_PLAIN;

    // zram path
    const llama_zram_device * m_zdev         = nullptr;
    int                       m_fd_buffered  = -1; // page-cache backed pread/pwrite
    int                       m_fd_direct    = -1; // O_DIRECT fd, aligned ops only
    size_t                    m_direct_align = 4096;

    // plain-RAM fallback path (only used if zram is unavailable)
    std::unique_ptr<uint8_t[]> m_ram_buf;
    void *                     m_ram_alloc_base = nullptr; // for aligned free()

    mutable std::atomic<uint64_t> m_bytes_written{0};
    mutable std::atomic<uint64_t> m_bytes_read{0};
    mutable std::atomic<uint64_t> m_n_direct_io_ops{0};
    mutable std::atomic<uint64_t> m_n_buffered_io_ops{0};
};

// Convenience factory: build a dynamic cache sized to hold `n_bytes` of hot
// data, automatically wired to zram (falls back to RAM only if zram is truly
// unavailable on this system, e.g. non-Linux or missing kernel module and no
// permission to load it).
std::unique_ptr<llama_dynamic_zram_cache> llama_make_dynamic_cache(
        uint64_t n_bytes,
        const char * debug_name = "llama_dynamic_cache");
