#pragma once

#include "monowire-batch.h"
#include "monowire-graph.h"
#include "monowire-kv-cache.h"
#include "monowire-memory-recurrent.h"
#include "monowire-memory.h"

#include <memory>
#include <vector>

//
// monowire_memory_hybrid
//

// utilizes instances of monowire_memory_recurrent and monowire_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class monowire_memory_hybrid : public monowire_memory_i {
public:
    monowire_memory_hybrid(const monowire_model &model,
                           /* attn */
                           ggml_type type_k, ggml_type type_v, bool v_trans, uint32_t kv_size, uint32_t n_pad,
                           uint32_t n_swa, monowire_swa_type swa_type,
                           /* recurrent */
                           ggml_type type_r, ggml_type type_s, uint32_t rs_size,
                           /* common */
                           uint32_t n_seq_max, bool offload, bool unified,
                           /* layer filters */
                           const layer_filter_cb &filter_attn = nullptr, const layer_filter_cb &filter_recr = nullptr);

    ~monowire_memory_hybrid() = default;

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
    // monowire_memory_hybrid specific API
    //

    monowire_kv_cache *get_mem_attn() const;
    monowire_memory_recurrent *get_mem_recr() const;

private:
    const monowire_hparams &hparams;

    const std::unique_ptr<monowire_kv_cache> mem_attn;
    const std::unique_ptr<monowire_memory_recurrent> mem_recr;
};

class monowire_memory_hybrid_context : public monowire_memory_context_i {
public:
    using slot_info_vec_t = monowire_kv_cache::slot_info_vec_t;

    // init failure
    explicit monowire_memory_hybrid_context(monowire_memory_status status);

    // init full
    explicit monowire_memory_hybrid_context(monowire_memory_hybrid *mem);

    // init update
    explicit monowire_memory_hybrid_context(monowire_memory_hybrid *mem, monowire_context *lctx, bool optimize);

    // init success
    monowire_memory_hybrid_context(monowire_memory_hybrid *mem, slot_info_vec_t sinfos_attn,
                                   std::vector<monowire_ubatch> ubatches);

    ~monowire_memory_hybrid_context() = default;

    bool next() override;
    bool apply() override;

    monowire_memory_status get_status() const override;
    const monowire_ubatch &get_ubatch() const override;

    //
    // monowire_memory_hybrid_context
    //

    const monowire_kv_cache_context *get_attn() const;
    const monowire_memory_recurrent_context *get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<monowire_ubatch> ubatches;

    const monowire_memory_context_ptr ctx_attn;
    const monowire_memory_context_ptr ctx_recr;

    const monowire_memory_status status;
};
