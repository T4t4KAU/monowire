#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class monowire_io_write_i {
public:
    monowire_io_write_i() = default;
    virtual ~monowire_io_write_i() = default;

    virtual void write(const void *src, size_t size) = 0;
    virtual void write_tensor(const ggml_tensor *tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string &str);
};

class monowire_io_read_i {
public:
    monowire_io_read_i() = default;
    virtual ~monowire_io_read_i() = default;

    virtual const uint8_t *read(size_t size) = 0;
    virtual void read_to(void *dst, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string &str);
};
