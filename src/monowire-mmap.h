#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

struct monowire_file;
struct monowire_mmap;
struct monowire_mlock;

using monowire_files = std::vector<std::unique_ptr<monowire_file>>;
using monowire_mmaps = std::vector<std::unique_ptr<monowire_mmap>>;
using monowire_mlocks = std::vector<std::unique_ptr<monowire_mlock>>;

struct monowire_file {
    monowire_file(const char *fname, const char *mode, bool use_direct_io = false);
    monowire_file(FILE *file);
    ~monowire_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void *ptr, size_t len);
    void read_raw_unsafe(void *ptr, size_t len);
    void read_aligned_chunk(void *dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void *ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct monowire_mmap {
    monowire_mmap(const monowire_mmap &) = delete;
    monowire_mmap(struct monowire_file *file, size_t prefetch = (size_t)-1, bool numa = false);
    ~monowire_mmap();

    size_t size() const;
    void *addr() const;

    void advise_fragment(size_t first, size_t last);
    void read_fragment(size_t first, size_t last);
    bool read_to(size_t offset, void *dst, size_t size);
    void evict_fragment(size_t first, size_t last);
    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct monowire_mlock {
    monowire_mlock();
    ~monowire_mlock();

    void init(void *ptr);
    void grow_to(size_t target_size);
    void lock_range(size_t offset, size_t size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t monowire_path_max();
