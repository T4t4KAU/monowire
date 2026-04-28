#include "monowire-mmap.h"

#include "monowire-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <stdexcept>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// TODO: consider moving to monowire-impl.h if needed in more places
#if defined(_WIN32)
static std::string monowire_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size
        = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                         NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// monowire_file

struct monowire_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
            error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char *fname, const char *mode, [[maybe_unused]] const bool use_direct_io = false) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE *file) : owns_fp(false) {
        fp = file;
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void *ptr, size_t len) {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64 * 1024 * 1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char *>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void *ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64 * 1024 * 1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const *>(ptr) + bytes_written, chunk_size,
                                    &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const { write_raw(&val, sizeof(val)); }

    bool has_direct_io() const { return true; }

    ~impl() {
        if (fp && owns_fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char *fname, const char *mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {
#ifdef __linux__
        // Try unbuffered I/O for read only
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_fd()) {
                return;
            }
            MONOWIRE_LOG_WARN("Failed to open file '%s' with error: %s. Falling back "
                              "to buffered I/O",
                              fname, strerror(errno));
        }
#endif
        init_fp(mode);
    }

#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats {};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
#endif

    void init_fp(const char *mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE *file) : fname("(file*)"), owns_fp(false) {
        fp = file;
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (fd == -1) {
            long ret = std::ftell(fp);
            if (ret == -1) {
                throw std::runtime_error(format("ftell error: %s", strerror(errno)));
            }

            return (size_t)ret;
        }

        off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos == -1) {
            throw std::runtime_error(format("lseek error: %s", strerror(errno)));
        }
        return (size_t)pos;
    }

    void seek(size_t offset, int whence) const {
        off_t ret = 0;
        if (fd == -1) {
            ret = std::fseek(fp, (long)offset, whence);
        } else {
            ret = lseek(fd, offset, whence);
        }
        if (ret == -1) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw_unsafe(void *ptr, size_t len) {
        if (len == 0) {
            return;
        }
        errno = 0;
        if (fd == -1) {
            const size_t curr_off = tell();
            const size_t to_read = std::min(len, size - curr_off);

            std::size_t ret = std::fread(ptr, to_read, 1, fp);
            if (ferror(fp)) {
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (to_read > 0 && ret != 1) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
        } else {
            size_t bytes_read = 0;
            while (bytes_read < len) {
                const size_t to_read = len - bytes_read;
                ssize_t ret = ::read(fd, reinterpret_cast<char *>(ptr) + bytes_read, to_read);

                if (ret == -1) {
                    if (errno == EINTR) {
                        continue; // Interrupted by signal, retry
                    }
                    // Fallback to std::fread in case the DMA controller cannot access the
                    // buffer
                    if (errno == EFAULT || errno == EINVAL) {
                        MONOWIRE_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
                        auto curr_off = tell();
                        close(fd);
                        fd = -1;
                        alignment = 1;
                        init_fp("rb");
                        seek(curr_off, SEEK_SET);
                        read_raw_unsafe(ptr, len);
                        return;
                    }
                    throw std::runtime_error(format("read error: %s", strerror(errno)));
                }
                if (ret == 0) {
                    // EOF: allow if this read was only pulling alignment padding past
                    // file end
                    off_t pos = lseek(fd, 0, SEEK_CUR);
                    if (pos != -1 && (size_t)pos == size) {
                        std::memset(reinterpret_cast<char *>(ptr) + bytes_read, 0, len - bytes_read);
                        return;
                    }
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += (size_t)ret;
            }
        }
    }

    void read_aligned_chunk(void *dest, size_t size) {
        size_t offset = tell();
        off_t aligned_offset = offset & ~(alignment - 1);
        off_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + size + alignment - 1) & ~(alignment - 1);

        void *raw_buffer = nullptr;
        int ret = posix_memalign(&raw_buffer, alignment, bytes_to_read);
        if (ret != 0) {
            throw std::runtime_error(format("posix_memalign failed with error %d", ret));
        }

        struct aligned_buffer_deleter {
            void operator()(void *p) const { free(p); }
        };
        std::unique_ptr<void, aligned_buffer_deleter> buffer(raw_buffer);

        seek(aligned_offset, SEEK_SET);
        read_raw_unsafe(buffer.get(), bytes_to_read);

        uintptr_t actual_data = reinterpret_cast<uintptr_t>(buffer.get()) + offset_from_alignment;
        memcpy(dest, reinterpret_cast<void *>(actual_data), size);
    }

    void read_raw(void *ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    uint32_t read_u32() {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void *ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const { write_raw(&val, sizeof(val)); }

    bool has_direct_io() const { return fd != -1 && alignment > 1; }

    ~impl() {
        if (fd != -1) {
            close(fd);
        } else if (owns_fp) {
            std::fclose(fp);
        }
    }
    int fd = -1;
    std::string fname;
#endif

    size_t read_alignment() const { return alignment; }

    size_t alignment = 1;

    FILE *fp{};
    size_t size{};
    bool owns_fp = true;
};

monowire_file::monowire_file(const char *fname, const char *mode, const bool use_direct_io)
    : pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

monowire_file::monowire_file(FILE *file) : pimpl(std::make_unique<impl>(file)) {}

monowire_file::~monowire_file() = default;

size_t monowire_file::tell() const { return pimpl->tell(); }
size_t monowire_file::size() const { return pimpl->size; }

size_t monowire_file::read_alignment() const { return pimpl->read_alignment(); }
bool monowire_file::has_direct_io() const { return pimpl->has_direct_io(); }

int monowire_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void monowire_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void monowire_file::read_raw(void *ptr, size_t len) { pimpl->read_raw(ptr, len); }
#ifdef _WIN32
void monowire_file::read_raw_unsafe(void *ptr, size_t len) { pimpl->read_raw(ptr, len); }
#else
void monowire_file::read_raw_unsafe(void *ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }
#endif

uint32_t monowire_file::read_u32() { return pimpl->read_u32(); }

void monowire_file::write_raw(const void *ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void monowire_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// monowire_mmap

struct monowire_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
    int fd = -1;

    impl(struct monowire_file *file, size_t prefetch, bool numa) {
        size = file->size();
        fd = dup(file->file_id());
        if (fd == -1) {
            throw std::runtime_error(format("dup failed: %s", strerror(errno)));
        }
        int flags = MAP_SHARED;
        if (numa) {
            prefetch = 0;
        }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            MONOWIRE_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n", strerror(errno));
        }
        if (prefetch) {
            flags |= MAP_POPULATE;
        }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                MONOWIRE_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n", strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                MONOWIRE_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

    static void align_range(size_t *first, size_t *last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void advise_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

#ifdef __linux__
        // Runtime streaming knows the exact file offsets for upcoming layer
        // weights. fadvise gives the kernel a file-level readahead hint before
        // the mmap pages are touched by compute threads.
        const int fadvise_ret = fd >= 0 ? posix_fadvise(fd, first, len, POSIX_FADV_WILLNEED) : 0;
        if (fadvise_ret != 0) {
            MONOWIRE_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_WILLNEED) failed: %s\n", strerror(fadvise_ret));
        }
#ifdef SYS_readahead
        // Make the runtime prefetch worker actively submit page-cache reads.
        // This is still mmap-backed, but it is closer to FlexInfer's explicit
        // asynchronous I/O pipeline than relying on future page faults.
        if (fd >= 0 && syscall(SYS_readahead, fd, (off_t)first, len) == -1) {
            MONOWIRE_LOG_WARN("warning: readahead(..) failed: %s\n", strerror(errno));
        }
#endif
#endif
        if (posix_madvise((uint8_t *)addr + first, len, POSIX_MADV_WILLNEED)) {
            MONOWIRE_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n", strerror(errno));
        }
    }

    void read_fragment(size_t first, size_t last) {
        advise_fragment(first, last);

        if (fd < 0) {
            return;
        }

        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        last = std::min(last, size);
        if (last <= first) {
            return;
        }
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        // Pull pages into the filesystem cache from the background streaming
        // worker. This is a bridge toward FlexInfer's explicit I/O pipeline:
        // compute still uses mmap-backed tensors, but page faults should find
        // most upcoming weights already resident.
        static constexpr size_t read_chunk = 16 * 1024 * 1024;
        thread_local std::vector<uint8_t> buffer;
        const size_t buffer_size = std::min(read_chunk, len);
        if (buffer.size() < buffer_size) {
            buffer.resize(buffer_size);
        }

        size_t offset = first;
        while (offset < last) {
            const size_t to_read = std::min(buffer.size(), last - offset);
            size_t bytes_read = 0;
            while (bytes_read < to_read) {
                const ssize_t ret
                    = pread(fd, buffer.data() + bytes_read, to_read - bytes_read, (off_t)(offset + bytes_read));
                if (ret == -1) {
                    if (errno == EINTR) {
                        continue;
                    }
                    MONOWIRE_LOG_WARN("warning: pread(..) streaming prefetch failed: %s\n", strerror(errno));
                    return;
                }
                if (ret == 0) {
                    return;
                }
                bytes_read += (size_t)ret;
            }
            offset += to_read;
        }
    }

    bool read_to(size_t offset, void *dst, size_t bytes) {
        if (bytes == 0) {
            return true;
        }
        if (dst == nullptr || offset >= size || bytes > size - offset || fd < 0) {
            MONOWIRE_LOG_WARN(
                "warning: streaming direct buffer read rejected: dst=%p offset=%zu bytes=%zu map_size=%zu fd=%d\n", dst,
                offset, bytes, size, fd);
            return false;
        }

        size_t bytes_read = 0;
        while (bytes_read < bytes) {
            const ssize_t ret
                = pread(fd, (uint8_t *)dst + bytes_read, bytes - bytes_read, (off_t)(offset + bytes_read));
            if (ret == -1) {
                if (errno == EINTR) {
                    continue;
                }
                MONOWIRE_LOG_WARN("warning: pread(..) streaming direct buffer failed: %s\n", strerror(errno));
                return false;
            }
            if (ret == 0) {
                return false;
            }
            bytes_read += (size_t)ret;
        }

#ifdef __linux__
        // The direct-buffer path owns a private copy after this read. Drop the
        // just-read file-cache pages so they do not compete with the streaming
        // window in tight memory cgroups.
        const int fadvise_ret = posix_fadvise(fd, offset, bytes, POSIX_FADV_DONTNEED);
        if (fadvise_ret != 0) {
            MONOWIRE_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_DONTNEED) failed: %s\n", strerror(fadvise_ret));
        }
#endif

        return true;
    }

    void evict_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

#ifdef MADV_DONTNEED
        if (madvise((uint8_t *)addr + first, len, MADV_DONTNEED)) {
            MONOWIRE_LOG_WARN("warning: madvise(.., MADV_DONTNEED) failed: %s\n", strerror(errno));
        }
#endif
#ifdef __linux__
        // Pair mmap eviction with a file-cache hint so low-memory cgroups do
        // not keep already-consumed dynamic weights ahead of future layers.
        const int fadvise_ret = fd >= 0 ? posix_fadvise(fd, first, len, POSIX_FADV_DONTNEED) : 0;
        if (fadvise_ret != 0) {
            MONOWIRE_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_DONTNEED) failed: %s\n", strerror(fadvise_ret));
        }
#endif
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void *next_page_start = (uint8_t *)addr + first;

        if (munmap(next_page_start, len)) {
            MONOWIRE_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto &frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto &frag : mapped_fragments) {
            if (munmap((char *)addr + frag.first, frag.second - frag.first)) {
                MONOWIRE_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
        if (fd != -1) {
            close(fd);
        }
    }
#elif defined(_WIN32)
    HANDLE hMapping = nullptr;

    impl(struct monowire_file *file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE)_get_osfhandle(file->file_id());

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", monowire_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();

        if (addr == NULL) {
            CloseHandle(hMapping);
            throw std::runtime_error(format("MapViewOfFile failed: %s", monowire_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL(WINAPI * pPrefetchVirtualMemory)
            (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory
                = (decltype(pPrefetchVirtualMemory))(void *)GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T)std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    MONOWIRE_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                                      monowire_format_win_err(GetLastError()).c_str());
                }
            }
#else
            MONOWIRE_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void advise_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    void read_fragment(size_t first, size_t last) { advise_fragment(first, last); }

    bool read_to(size_t offset, void *dst, size_t bytes) {
        if (bytes == 0) {
            return true;
        }
        if (dst == nullptr || offset >= size || bytes > size - offset) {
            return false;
        }
        memcpy(dst, (uint8_t *)addr + offset, bytes);
        return true;
    }

    void evict_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (hMapping) {
            if (addr) {
                if (!UnmapViewOfFile(addr)) {
                    MONOWIRE_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                                      monowire_format_win_err(GetLastError()).c_str());
                }
            }
            if (!CloseHandle(hMapping)) {
                MONOWIRE_LOG_WARN("warning: CloseHandle failed: %s\n", monowire_format_win_err(GetLastError()).c_str());
            }
        }
    }
#else
    impl(struct monowire_file *file, size_t prefetch, bool numa) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void advise_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    void read_fragment(size_t first, size_t last) { advise_fragment(first, last); }

    bool read_to(size_t offset, void *dst, size_t bytes) {
        GGML_UNUSED(offset);
        GGML_UNUSED(dst);
        GGML_UNUSED(bytes);
        return false;
    }

    void evict_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void *addr;
    size_t size;
};

monowire_mmap::monowire_mmap(struct monowire_file *file, size_t prefetch, bool numa)
    : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
monowire_mmap::~monowire_mmap() = default;

size_t monowire_mmap::size() const { return pimpl->size; }
void *monowire_mmap::addr() const { return pimpl->addr; }

void monowire_mmap::advise_fragment(size_t first, size_t last) { pimpl->advise_fragment(first, last); }
void monowire_mmap::read_fragment(size_t first, size_t last) { pimpl->read_fragment(first, last); }
bool monowire_mmap::read_to(size_t offset, void *dst, size_t size) { return pimpl->read_to(offset, dst, size); }
void monowire_mmap::evict_fragment(size_t first, size_t last) { pimpl->evict_fragment(first, last); }
void monowire_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool monowire_mmap::SUPPORTED = true;
#else
const bool monowire_mmap::SUPPORTED = false;
#endif

// monowire_mlock

struct monowire_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() { return (size_t)sysconf(_SC_PAGESIZE); }

    bool raw_lock(const void *addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                                                                               \
    "Try increasing the sysctl values 'vm.user_wire_limit' and "                                                       \
    "'vm.global_user_wire_limit' and/or "                                                                              \
    "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing "                                                \
    "RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char *errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
        // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
        // Skip resource limit checks on these platforms
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && ((uint64_t)lock_limit.rlim_max > (uint64_t)lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        MONOWIRE_LOG_WARN("warning: failed to mlock %zu-byte buffer (after "
                          "previously locking %zu bytes): %s\n%s",
                          size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void *addr, size_t size) {
        if (munlock(addr, size)) {
            MONOWIRE_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t)si.dwPageSize;
    }

    bool raw_lock(void *ptr, size_t len) const {
        for (int tries = 1;; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                MONOWIRE_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer "
                                  "(after previously locking %zu bytes): %s\n",
                                  len, size, monowire_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                MONOWIRE_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                                  monowire_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                MONOWIRE_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                                  monowire_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void *ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            MONOWIRE_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                              monowire_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() { return (size_t)65536; }

    bool raw_lock(const void *addr, size_t len) const {
        MONOWIRE_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void *addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void *ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *)addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void lock_range(size_t offset, size_t len) {
        GGML_ASSERT(addr);
        if (failed_already || len == 0) {
            return;
        }

        const size_t granularity = lock_granularity();
        size_t first = offset & ~(granularity - 1);
        size_t last = (offset + len + granularity - 1) & ~(granularity - 1);

        if (last <= first) {
            return;
        }

        if (raw_lock((uint8_t *)addr + first, last - first)) {
            size += last - first;
        } else {
            failed_already = true;
        }
    }

    void *addr;
    size_t size;

    bool failed_already;
};

monowire_mlock::monowire_mlock() : pimpl(std::make_unique<impl>()) {}
monowire_mlock::~monowire_mlock() = default;

void monowire_mlock::init(void *ptr) { pimpl->init(ptr); }
void monowire_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }
void monowire_mlock::lock_range(size_t offset, size_t size) { pimpl->lock_range(offset, size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool monowire_mlock::SUPPORTED = true;
#else
const bool monowire_mlock::SUPPORTED = false;
#endif

size_t monowire_path_max() { return PATH_MAX; }
