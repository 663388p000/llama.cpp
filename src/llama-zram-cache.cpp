#include "llama-zram-cache.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>

#if defined(__linux__)
#  define LLAMA_ZRAM_LINUX 1
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  include <dirent.h>
#  include <errno.h>
#else
#  define LLAMA_ZRAM_LINUX 0
#endif

// ============================================================================
// Small helpers
// ============================================================================

namespace {

bool llama_zram_debug_enabled() {
    static const bool dbg = [] {
        const char * env = getenv("LLAMA_ZRAM_DEBUG");
        return env && atoi(env) > 0;
    }();
    return dbg;
}

#define LLAMA_ZRAM_LOG(...)                          \
    do {                                              \
        if (llama_zram_debug_enabled()) {              \
            fprintf(stderr, "llama-zram: " __VA_ARGS__); \
        }                                              \
    } while (0)

#if LLAMA_ZRAM_LINUX
bool read_file_string(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    // trim trailing whitespace/newlines
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return true;
}

bool write_file_string(const std::string & path, const std::string & value) {
    std::ofstream f(path);
    if (!f.is_open()) {
        LLAMA_ZRAM_LOG("failed to open '%s' for write: %s\n", path.c_str(), strerror(errno));
        return false;
    }
    f << value;
    if (!f.good()) {
        LLAMA_ZRAM_LOG("failed to write '%s' to '%s'\n", value.c_str(), path.c_str());
        return false;
    }
    return true;
}

bool path_exists(const std::string & path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}
#endif // LLAMA_ZRAM_LINUX

} // namespace

// ============================================================================
// llama_mem_info_detect()
// ============================================================================

llama_mem_info llama_mem_info_detect() {
    llama_mem_info info;

#if LLAMA_ZRAM_LINUX
    // ---- RAM / swap totals from /proc/meminfo ----
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (f && std::getline(f, line)) {
            uint64_t kb = 0;
            if (sscanf(line.c_str(), "MemTotal: %lu kB", &kb) == 1) {
                info.ram_total_bytes = kb * 1024ull;
            } else if (sscanf(line.c_str(), "MemAvailable: %lu kB", &kb) == 1) {
                info.ram_avail_bytes = kb * 1024ull;
            } else if (sscanf(line.c_str(), "SwapTotal: %lu kB", &kb) == 1) {
                info.swap_total_bytes = kb * 1024ull;
            }
        }
    }

    // ---- zram support / already-active zram devices ----
    info.zram_supported = path_exists("/sys/class/zram-control/hot_add") ||
                           path_exists("/sys/block/zram0");

    {
        DIR * d = opendir("/sys/block");
        if (d) {
            struct dirent * ent;
            while ((ent = readdir(d)) != nullptr) {
                const std::string name = ent->d_name;
                if (name.rfind("zram", 0) != 0) {
                    continue;
                }
                std::string disksize_str;
                if (read_file_string("/sys/block/" + name + "/disksize", disksize_str)) {
                    uint64_t sz = strtoull(disksize_str.c_str(), nullptr, 10);
                    info.zram_total_bytes += sz;
                }
            }
            closedir(d);
        }
    }

    // ---- UMA heuristic ----
    // A platform is treated as UMA if none of the GPU devices exposed under
    // /sys/class/drm/card*/device report a dedicated VRAM pool. Discrete GPUs
    // (amdgpu, nvidia via nouveau, some Intel Arc configs) expose
    // "mem_info_vram_total"; integrated GPUs generally do not, because they
    // draw directly from system RAM.
    {
        bool found_any_gpu   = false;
        bool found_dgpu_vram = false;

        DIR * d = opendir("/sys/class/drm");
        if (d) {
            struct dirent * ent;
            while ((ent = readdir(d)) != nullptr) {
                const std::string name = ent->d_name;
                // only look at top-level cardN entries, not cardN-<connector>
                if (name.rfind("card", 0) != 0) {
                    continue;
                }
                if (name.find('-') != std::string::npos) {
                    continue;
                }
                const std::string dev_dir = "/sys/class/drm/" + name + "/device";
                if (!path_exists(dev_dir)) {
                    continue;
                }
                found_any_gpu = true;
                if (path_exists(dev_dir + "/mem_info_vram_total")) {
                    found_dgpu_vram = true;
                }
            }
            closedir(d);
        }

        if (found_dgpu_vram) {
            info.is_uma = false;
            info.uma_reason = "at least one GPU exposes a dedicated VRAM pool (mem_info_vram_total)";
        } else if (found_any_gpu) {
            info.is_uma = true;
            info.uma_reason = "GPU device(s) found, none expose dedicated VRAM -> assumed to share system RAM";
        } else {
            // no GPU device nodes found at all (headless / CPU-only / some ARM SoCs
            // expose their GPU under a different subsystem) -- default to UMA=true
            // since there is no evidence of a separate discrete memory pool, and
            // system RAM is the only pool that exists in that case anyway.
            info.is_uma = true;
            info.uma_reason = "no /sys/class/drm GPU nodes found -> system RAM is the only pool";
        }
    }
#else
    // Non-Linux: we cannot read /proc/meminfo or /sys; leave everything zeroed
    // except a conservative UMA=false default so callers don't assume sharing
    // semantics we can't verify.
    info.zram_supported = false;
    info.is_uma          = false;
    info.uma_reason      = "non-Linux platform: RAM/zram/UMA detection not implemented";
#endif

    LLAMA_ZRAM_LOG("mem_info: total=%.1f MiB avail=%.1f MiB swap=%.1f MiB zram_supported=%d "
                    "zram_total=%.1f MiB is_uma=%d (%s)\n",
            info.ram_total_bytes  / 1024.0 / 1024.0,
            info.ram_avail_bytes  / 1024.0 / 1024.0,
            info.swap_total_bytes / 1024.0 / 1024.0,
            (int) info.zram_supported,
            info.zram_total_bytes / 1024.0 / 1024.0,
            (int) info.is_uma,
            info.uma_reason.c_str());

    return info;
}

// ============================================================================
// llama_zram_manager
// ============================================================================

llama_zram_manager & llama_zram_manager::instance() {
    static llama_zram_manager mgr;
    return mgr;
}

llama_zram_manager::llama_zram_manager() {
    m_available = detect_support();
}

llama_zram_manager::~llama_zram_manager() {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto & dev : m_devices) {
        if (dev && dev->is_new) {
            release_device(dev.get());
        }
    }
}

bool llama_zram_manager::detect_support() {
#if LLAMA_ZRAM_LINUX
    if (path_exists("/sys/class/zram-control/hot_add")) {
        return true;
    }

    // module not loaded yet -- try to load it. This requires root/CAP_SYS_MODULE;
    // failure here is expected (and fine) when running unprivileged, in which
    // case get_or_create_device() will still try to reuse a pre-existing device.
    int rc = system("modprobe zram > /dev/null 2>&1");
    (void) rc;

    if (path_exists("/sys/class/zram-control/hot_add")) {
        return true;
    }

    // even without hot_add, a statically-sized zram0 may already exist and be
    // reusable (common on distros that pre-configure zram-backed swap).
    return path_exists("/sys/block/zram0");
#else
    return false;
#endif
}

bool llama_zram_manager::write_sysfs(const std::string & path, const std::string & value) {
#if LLAMA_ZRAM_LINUX
    return write_file_string(path, value);
#else
    (void) path; (void) value;
    return false;
#endif
}

bool llama_zram_manager::read_sysfs(const std::string & path, std::string & out) {
#if LLAMA_ZRAM_LINUX
    return read_file_string(path, out);
#else
    (void) path; (void) out;
    return false;
#endif
}

const llama_zram_device * llama_zram_manager::get_or_create_device(
        uint64_t min_size_bytes, const char * comp_algorithm) {
#if LLAMA_ZRAM_LINUX
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_available) {
        LLAMA_ZRAM_LOG("get_or_create_device: zram not available on this system\n");
        return nullptr;
    }

    // 1) reuse a device we already created that has enough room and no cache
    //    attached to it yet (we only ever attach one cache per device here for
    //    simplicity/isolation between caches).
    for (const auto & dev : m_devices) {
        if (dev->disksize >= min_size_bytes) {
            return dev.get();
        }
    }

    // 2) create a fresh device via hot_add
    std::string id_str;
    if (!path_exists("/sys/class/zram-control/hot_add") ||
        !read_sysfs("/sys/class/zram-control/hot_add", id_str)) {
        LLAMA_ZRAM_LOG("get_or_create_device: hot_add unavailable/failed\n");
        return nullptr;
    }

    const int id = atoi(id_str.c_str());
    if (id < 0) {
        LLAMA_ZRAM_LOG("get_or_create_device: hot_add returned invalid id '%s'\n", id_str.c_str());
        return nullptr;
    }

    auto dev = std::make_unique<llama_zram_device>();
    dev->sys_path = "/sys/block/zram" + std::to_string(id);
    dev->dev_path = "/dev/zram" + std::to_string(id);
    dev->comp_algorithm = comp_algorithm ? comp_algorithm : "zstd";
    dev->is_new = true;

    // best-effort: set compression algorithm before sizing (zram requires this
    // order on some kernel versions; ignore failure and keep the default algo)
    write_sysfs(dev->sys_path + "/comp_algorithm", dev->comp_algorithm);

    // round up to a 4K page multiple, zram requires page-aligned disksize
    uint64_t size = ((min_size_bytes + 4095ull) / 4096ull) * 4096ull;
    if (size == 0) {
        size = 4096;
    }

    if (!write_sysfs(dev->sys_path + "/disksize", std::to_string(size))) {
        LLAMA_ZRAM_LOG("get_or_create_device: failed to set disksize on zram id=%d\n", id);
        // try to clean up the half-created device
        write_sysfs("/sys/class/zram-control/hot_remove", std::to_string(id));
        return nullptr;
    }

    dev->disksize = size;

    LLAMA_ZRAM_LOG("created zram device %s, disksize=%.1f MiB, comp=%s\n",
            dev->dev_path.c_str(), size / 1024.0 / 1024.0, dev->comp_algorithm.c_str());

    m_devices.push_back(std::move(dev));
    return m_devices.back().get();
#else
    (void) min_size_bytes; (void) comp_algorithm;
    return nullptr;
#endif
}

bool llama_zram_manager::try_grow_device(const llama_zram_device * dev, uint64_t new_min_size_bytes) {
#if LLAMA_ZRAM_LINUX
    if (!dev || !dev->is_new) {
        return false;
    }
    if (new_min_size_bytes <= dev->disksize) {
        return true;
    }

    // zram only allows changing disksize while the device is not in active use
    // (no open handles / not mounted / no data). Attempting this on a live
    // cache will typically fail with EBUSY -- callers should treat a `false`
    // return as "create a second device instead".
    uint64_t size = ((new_min_size_bytes + 4095ull) / 4096ull) * 4096ull;
    if (!write_sysfs(dev->sys_path + "/disksize", std::to_string(size))) {
        LLAMA_ZRAM_LOG("try_grow_device: resize of %s to %.1f MiB failed (device busy?)\n",
                dev->dev_path.c_str(), size / 1024.0 / 1024.0);
        return false;
    }

    const_cast<llama_zram_device *>(dev)->disksize = size;
    return true;
#else
    (void) dev; (void) new_min_size_bytes;
    return false;
#endif
}

void llama_zram_manager::release_device(const llama_zram_device * dev) {
#if LLAMA_ZRAM_LINUX
    if (!dev || !dev->is_new) {
        return;
    }

    // extract the numeric id from "/dev/zramN"
    const std::string & path = dev->dev_path;
    const size_t pos = path.find_last_not_of("0123456789");
    if (pos == std::string::npos || pos + 1 >= path.size()) {
        return;
    }
    const std::string id_str = path.substr(pos + 1);

    write_sysfs(dev->sys_path + "/reset", "1");
    write_sysfs("/sys/class/zram-control/hot_remove", id_str);

    LLAMA_ZRAM_LOG("released zram device %s\n", dev->dev_path.c_str());
#else
    (void) dev;
#endif
}

// ============================================================================
// llama_dynamic_zram_cache
// ============================================================================

llama_dynamic_zram_cache::llama_dynamic_zram_cache(const llama_dynamic_cache_config & cfg) : m_cfg(cfg) {
    bool ok = false;

    if (m_cfg.force_zram || llama_zram_manager::instance().available()) {
        ok = open_zram_backend();
    }

    if (!ok) {
        if (m_cfg.force_zram) {
            LLAMA_ZRAM_LOG("'%s': zram backend unavailable despite force_zram=true, "
                            "falling back to RAM (results will use MORE uncompressed RAM "
                            "than requested -- fix zram/module permissions to avoid this)\n",
                    m_cfg.debug_name.c_str());
        }

        const llama_mem_info mi = llama_mem_info_detect();
        ok = open_ram_backend(mi.is_uma);
    }

    if (!ok) {
        LLAMA_ZRAM_LOG("'%s': failed to allocate ANY backend for %.1f MiB cache!\n",
                m_cfg.debug_name.c_str(), m_cfg.capacity_bytes / 1024.0 / 1024.0);
    }
}

llama_dynamic_zram_cache::~llama_dynamic_zram_cache() {
#if LLAMA_ZRAM_LINUX
    if (m_fd_direct >= 0) {
        close(m_fd_direct);
    }
    if (m_fd_buffered >= 0) {
        close(m_fd_buffered);
    }
#endif
    if (m_ram_alloc_base) {
        free(m_ram_alloc_base);
    }
}

bool llama_dynamic_zram_cache::open_zram_backend() {
#if LLAMA_ZRAM_LINUX
    m_zdev = llama_zram_manager::instance().get_or_create_device(m_cfg.capacity_bytes, m_cfg.comp_algorithm);
    if (!m_zdev) {
        return false;
    }

    m_fd_buffered = ::open(m_zdev->dev_path.c_str(), O_RDWR | O_CLOEXEC);
    if (m_fd_buffered < 0) {
        LLAMA_ZRAM_LOG("'%s': failed to open %s: %s\n",
                m_cfg.debug_name.c_str(), m_zdev->dev_path.c_str(), strerror(errno));
        return false;
    }

    m_backend = llama_cache_backend::LLAMA_CACHE_BACKEND_ZRAM;

    if (m_cfg.direct_io_min_bytes > 0) {
        int fd_o_direct = ::open(m_zdev->dev_path.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC);
        if (fd_o_direct >= 0) {
            m_fd_direct = fd_o_direct;
            m_direct_align = 4096; // zram is page-granular; 4K covers virtually all systems
            m_backend = llama_cache_backend::LLAMA_CACHE_BACKEND_ZRAM_DIRECT;
            LLAMA_ZRAM_LOG("'%s': O_DIRECT path enabled on %s (align=%zu)\n",
                    m_cfg.debug_name.c_str(), m_zdev->dev_path.c_str(), m_direct_align);
        } else {
            LLAMA_ZRAM_LOG("'%s': O_DIRECT not available on %s (%s); using buffered I/O only\n",
                    m_cfg.debug_name.c_str(), m_zdev->dev_path.c_str(), strerror(errno));
        }
    }

    LLAMA_ZRAM_LOG("'%s': backed by %s, capacity=%.1f MiB, backend=%s\n",
            m_cfg.debug_name.c_str(), m_zdev->dev_path.c_str(),
            m_cfg.capacity_bytes / 1024.0 / 1024.0,
            is_zram_backed() ?
                (m_backend == llama_cache_backend::LLAMA_CACHE_BACKEND_ZRAM_DIRECT ? "zram+direct" : "zram") :
                "?");

    return true;
#else
    return false;
#endif
}

bool llama_dynamic_zram_cache::open_ram_backend(bool uma_tuned) {
    if (m_cfg.capacity_bytes == 0) {
        m_backend = uma_tuned ?
            llama_cache_backend::LLAMA_CACHE_BACKEND_RAM_UMA :
            llama_cache_backend::LLAMA_CACHE_BACKEND_RAM_PLAIN;
        return true;
    }

    // Allocate page-aligned memory even in the fallback path: this keeps the
    // read()/write() implementation uniform with the O_DIRECT path (aligned
    // buffers are a prerequisite there) and, on UMA systems, page-aligned
    // allocations are also what the driver stack (e.g. dma-buf/ion-style
    // allocators, or GBM on Mesa) expects if this memory is later imported
    // for zero-copy access by the integrated GPU -- avoiding a bounce copy
    // that a misaligned malloc() would otherwise force.
    void * mem = nullptr;
#if LLAMA_ZRAM_LINUX
    if (posix_memalign(&mem, 4096, m_cfg.capacity_bytes) != 0) {
        mem = nullptr;
    }
#else
    mem = malloc(m_cfg.capacity_bytes);
#endif
    if (!mem) {
        LLAMA_ZRAM_LOG("'%s': RAM fallback allocation of %.1f MiB failed\n",
                m_cfg.debug_name.c_str(), m_cfg.capacity_bytes / 1024.0 / 1024.0);
        return false;
    }

    memset(mem, 0, m_cfg.capacity_bytes);
    m_ram_alloc_base = mem;

    m_backend = uma_tuned ?
        llama_cache_backend::LLAMA_CACHE_BACKEND_RAM_UMA :
        llama_cache_backend::LLAMA_CACHE_BACKEND_RAM_PLAIN;

    LLAMA_ZRAM_LOG("'%s': RAM fallback backend (%s), capacity=%.1f MiB (uncompressed!)\n",
            m_cfg.debug_name.c_str(), uma_tuned ? "UMA-tuned" : "plain",
            m_cfg.capacity_bytes / 1024.0 / 1024.0);

    return true;
}

bool llama_dynamic_zram_cache::pio_direct(uint64_t offset, void * buf, size_t len, bool is_write) {
#if LLAMA_ZRAM_LINUX
    if (m_fd_direct < 0) {
        return false;
    }

    const size_t align = m_direct_align;

    const uint64_t aligned_off  = offset & ~(uint64_t) (align - 1);
    const uint64_t off_delta    = offset - aligned_off;
    const size_t   aligned_len  = ((off_delta + len + align - 1) / align) * align;

    void * bounce = nullptr;
    if (posix_memalign(&bounce, align, aligned_len) != 0) {
        return false;
    }

    bool ok = true;

    if (is_write) {
        // read-modify-write when the request isn't already page-aligned, so we
        // don't clobber neighboring bytes that share the first/last page.
        if (off_delta != 0 || aligned_len != len) {
            ssize_t r = ::pread(m_fd_direct, bounce, aligned_len, (off_t) aligned_off);
            if (r < 0) {
                ok = false;
            }
        }
        if (ok) {
            memcpy((uint8_t *) bounce + off_delta, buf, len);
            ssize_t w = ::pwrite(m_fd_direct, bounce, aligned_len, (off_t) aligned_off);
            ok = (w == (ssize_t) aligned_len);
        }
    } else {
        ssize_t r = ::pread(m_fd_direct, bounce, aligned_len, (off_t) aligned_off);
        if (r < 0 || (size_t) r < off_delta + len) {
            ok = false;
        } else {
            memcpy(buf, (uint8_t *) bounce + off_delta, len);
        }
    }

    free(bounce);

    if (ok) {
        m_n_direct_io_ops.fetch_add(1, std::memory_order_relaxed);
    }

    return ok;
#else
    (void) offset; (void) buf; (void) len; (void) is_write;
    return false;
#endif
}

bool llama_dynamic_zram_cache::pio(uint64_t offset, void * buf, size_t len, bool is_write) {
    if (offset + len > m_cfg.capacity_bytes) {
        LLAMA_ZRAM_LOG("'%s': out-of-range %s at offset=%lu len=%zu capacity=%lu\n",
                m_cfg.debug_name.c_str(), is_write ? "write" : "read",
                (unsigned long) offset, len, (unsigned long) m_cfg.capacity_bytes);
        return false;
    }

    if (is_zram_backed()) {
#if LLAMA_ZRAM_LINUX
        const bool aligned = (offset % m_direct_align == 0) && (len % m_direct_align == 0) &&
                              (((uintptr_t) buf) % m_direct_align == 0);

        if (m_fd_direct >= 0 && len >= m_cfg.direct_io_min_bytes) {
            // aligned buffer/offset/length -> go straight to O_DIRECT without a
            // bounce buffer; otherwise pio_direct() itself uses one internally.
            if (aligned) {
                ssize_t rc = is_write ?
                    ::pwrite(m_fd_direct, buf, len, (off_t) offset) :
                    ::pread (m_fd_direct, buf, len, (off_t) offset);
                if (rc == (ssize_t) len) {
                    m_n_direct_io_ops.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                LLAMA_ZRAM_LOG("'%s': aligned O_DIRECT %s failed (%s), falling back to buffered\n",
                        m_cfg.debug_name.c_str(), is_write ? "write" : "read", strerror(errno));
            } else if (pio_direct(offset, buf, len, is_write)) {
                return true;
            }
            // fall through to buffered path on any direct-I/O failure
        }

        ssize_t rc = is_write ?
            ::pwrite(m_fd_buffered, buf, len, (off_t) offset) :
            ::pread (m_fd_buffered, buf, len, (off_t) offset);

        if (rc != (ssize_t) len) {
            LLAMA_ZRAM_LOG("'%s': buffered %s failed at offset=%lu len=%zu: %s\n",
                    m_cfg.debug_name.c_str(), is_write ? "write" : "read",
                    (unsigned long) offset, len, strerror(errno));
            return false;
        }

        m_n_buffered_io_ops.fetch_add(1, std::memory_order_relaxed);
        return true;
#else
        return false;
#endif
    }

    // plain-RAM backend
    if (!m_ram_alloc_base) {
        return len == 0;
    }
    if (is_write) {
        memcpy((uint8_t *) m_ram_alloc_base + offset, buf, len);
    } else {
        memcpy(buf, (uint8_t *) m_ram_alloc_base + offset, len);
    }
    m_n_buffered_io_ops.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool llama_dynamic_zram_cache::write(uint64_t offset, const void * data, size_t len) {
    if (len == 0) {
        return true;
    }
    const bool ok = pio(offset, const_cast<void *>(data), len, /*is_write=*/true);
    if (ok) {
        m_bytes_written.fetch_add(len, std::memory_order_relaxed);
    }
    return ok;
}

bool llama_dynamic_zram_cache::read(uint64_t offset, void * data, size_t len) const {
    if (len == 0) {
        return true;
    }
    auto * self = const_cast<llama_dynamic_zram_cache *>(this);
    const bool ok = self->pio(offset, data, len, /*is_write=*/false);
    if (ok) {
        m_bytes_read.fetch_add(len, std::memory_order_relaxed);
    }
    return ok;
}

void llama_dynamic_zram_cache::invalidate(uint64_t offset, size_t len) {
    LLAMA_ZRAM_LOG("'%s': invalidate hint offset=%lu len=%zu (no-op: no discard/TRIM wired up)\n",
            m_cfg.debug_name.c_str(), (unsigned long) offset, len);
}

llama_dynamic_zram_cache::stats_t llama_dynamic_zram_cache::stats() const {
    stats_t s;
    s.bytes_written     = m_bytes_written.load(std::memory_order_relaxed);
    s.bytes_read        = m_bytes_read.load(std::memory_order_relaxed);
    s.n_direct_io_ops   = m_n_direct_io_ops.load(std::memory_order_relaxed);
    s.n_buffered_io_ops = m_n_buffered_io_ops.load(std::memory_order_relaxed);
    return s;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<llama_dynamic_zram_cache> llama_make_dynamic_cache(uint64_t n_bytes, const char * debug_name) {
    llama_dynamic_cache_config cfg;
    cfg.capacity_bytes = n_bytes;
    cfg.force_zram      = true; // per requirement: all dynamic caches must live in zram
    cfg.debug_name       = debug_name ? debug_name : "llama_dynamic_cache";
    return std::make_unique<llama_dynamic_zram_cache>(cfg);
}
