#pragma once

#include "monowire-kv-cache.h"

#include <vector>

//
// monowire_kv_cache_iswa
//

// utilizes two instances of monowire_kv_cache
//   the first instance is for the non-SWA layers of the model and the second
//   instance is for the SWA layers

class monowire_kv_cache_iswa : public monowire_memory_i {
public:
    monowire_kv_cache_iswa(const monowire_model &model, ggml_type type_k, ggml_type type_v, bool v_trans, bool offload,
                           bool swa_full, bool unified, uint32_t kv_size, uint32_t n_seq_max, uint32_t n_ubatch,
                           uint32_t n_pad, const layer_filter_cb &filter, const layer_reuse_cb &reuse);

    ~monowire_kv_cache_iswa() = default;

    //
    // monowire_memory_i
    //

    monowire_memory_context_ptr init_batch(monowire_batch_allocr &balloc, uint32_t n_ubatch, bool embd_all) override;

    monowire_memory_context_ptr init_full() override;

    monowire_memory_context_ptr init_update(monowire_context *lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1) override;
    void seq_cp(monowire_seq_id seq_id_src, monowire_seq_id seq_id_dst, monowire_pos p0, monowire_pos p1) override;
    void seq_keep(monowire_seq_id seq_id) override;
    void seq_add(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1, monowire_pos shift) override;
    void seq_div(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1, int d) override;

    monowire_pos seq_pos_min(monowire_seq_id seq_id) const override;
    monowire_pos seq_pos_max(monowire_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(monowire_io_write_i &io, monowire_seq_id seq_id = -1,
                     monowire_state_seq_flags flags = 0) const override;
    void state_read(monowire_io_read_i &io, monowire_seq_id seq_id = -1, monowire_state_seq_flags flags = 0) override;

    //
    // monowire_kv_cache_iswa specific API
    //

    monowire_kv_cache *get_base() const;
    monowire_kv_cache *get_swa() const;

private:
    const monowire_hparams &hparams;

    const bool unified;

    std::unique_ptr<monowire_kv_cache> kv_base;
    std::unique_ptr<monowire_kv_cache> kv_swa;
};

class monowire_kv_cache_iswa_context : public monowire_memory_context_i {
public:
    using slot_info_vec_t = monowire_kv_cache::slot_info_vec_t;

    // used for errors
    monowire_kv_cache_iswa_context(monowire_memory_status status);

    // used to create a full-cache context
    monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv);

    // used to create an update context
    monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv, monowire_context *lctx, bool optimize);

    // used to create a batch processing context from a batch
    monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv, slot_info_vec_t sinfos_base, slot_info_vec_t sinfos_swa,
                                   std::vector<monowire_ubatch> ubatches);

    virtual ~monowire_kv_cache_iswa_context();

    //
    // monowire_memory_context_i
    //

    bool next() override;
    bool apply() override;

    monowire_memory_status get_status() const override;
    const monowire_ubatch &get_ubatch() const override;

    //
    // monowire_kv_cache_iswa_context specific API
    //

    const monowire_kv_cache_context *get_base() const;
    const monowire_kv_cache_context *get_swa() const;

private:
    // monowire_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<monowire_ubatch> ubatches;

    const monowire_memory_context_ptr ctx_base;
    const monowire_memory_context_ptr ctx_swa;

    const monowire_memory_status status;
};
