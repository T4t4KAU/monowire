#include "monowire-kv-cache-iswa.h"

#include "monowire-batch.h"
#include "monowire-impl.h"
#include "monowire-model.h"

#include <algorithm>
#include <cassert>

//
// monowire_kv_cache_iswa
//

monowire_kv_cache_iswa::monowire_kv_cache_iswa(const monowire_model &model, ggml_type type_k, ggml_type type_v,
                                               bool v_trans, bool offload, bool swa_full, bool unified,
                                               uint32_t kv_size, uint32_t n_seq_max, uint32_t n_ubatch, uint32_t n_pad,
                                               const layer_filter_cb &filter, const layer_reuse_cb &reuse)
    : hparams(model.hparams), unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       upstream-reference
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa * (unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to
    // the base cache size
    if (swa_full) {
        MONOWIRE_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n", __func__, "upstream-reference");

        size_swa = size_base;
    }

    MONOWIRE_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    kv_base = std::make_unique<monowire_kv_cache>(model, type_k, type_v, v_trans, offload, unified, size_base,
                                                  n_seq_max, n_pad, 0, MONOWIRE_SWA_TYPE_NONE, filter_base, reuse);

    MONOWIRE_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<monowire_kv_cache>(model, type_k, type_v, v_trans, offload, unified, size_swa, n_seq_max,
                                                 n_pad, hparams.n_swa, hparams.swa_type, filter_swa, reuse);
}

void monowire_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa->clear(data);
}

bool monowire_kv_cache_iswa::seq_rm(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa->seq_rm(seq_id, p0, p1);

    return res;
}

void monowire_kv_cache_iswa::seq_cp(monowire_seq_id seq_id_src, monowire_seq_id seq_id_dst, monowire_pos p0,
                                    monowire_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void monowire_kv_cache_iswa::seq_keep(monowire_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa->seq_keep(seq_id);
}

void monowire_kv_cache_iswa::seq_add(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1, monowire_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa->seq_add(seq_id, p0, p1, shift);
}

void monowire_kv_cache_iswa::seq_div(monowire_seq_id seq_id, monowire_pos p0, monowire_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa->seq_div(seq_id, p0, p1, d);
}

monowire_pos monowire_kv_cache_iswa::seq_pos_min(monowire_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA
    // cache
    return kv_swa->seq_pos_min(seq_id);
}

monowire_pos monowire_kv_cache_iswa::seq_pos_max(monowire_seq_id seq_id) const { return kv_swa->seq_pos_max(seq_id); }

std::map<ggml_backend_buffer_type_t, size_t> monowire_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto &buft_size : kv_swa->memory_breakdown()) { mb[buft_size.first] += buft_size.second; }
    return mb;
}

monowire_memory_context_ptr monowire_kv_cache_iswa::init_batch(monowire_batch_allocr &balloc, uint32_t n_ubatch,
                                                               bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<monowire_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<monowire_kv_cache_iswa_context>(this, std::move(sinfos_base), std::move(sinfos_swa),
                                                                std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<monowire_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<monowire_kv_cache_iswa_context>(this, std::move(sinfos_base), std::move(sinfos_swa),
                                                                std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be
    //       more flexible

    return std::make_unique<monowire_kv_cache_iswa_context>(MONOWIRE_MEMORY_STATUS_FAILED_PREPARE);
}

monowire_memory_context_ptr monowire_kv_cache_iswa::init_full() {
    return std::make_unique<monowire_kv_cache_iswa_context>(this);
}

monowire_memory_context_ptr monowire_kv_cache_iswa::init_update(monowire_context *lctx, bool optimize) {
    return std::make_unique<monowire_kv_cache_iswa_context>(this, lctx, optimize);
}

bool monowire_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() && kv_swa->get_can_shift() && kv_base->get_size() == kv_swa->get_size();
}

void monowire_kv_cache_iswa::state_write(monowire_io_write_i &io, monowire_seq_id seq_id,
                                         monowire_state_seq_flags flags) const {
    if ((flags & MONOWIRE_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void monowire_kv_cache_iswa::state_read(monowire_io_read_i &io, monowire_seq_id seq_id,
                                        monowire_state_seq_flags flags) {
    if ((flags & MONOWIRE_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

monowire_kv_cache *monowire_kv_cache_iswa::get_base() const { return kv_base.get(); }

monowire_kv_cache *monowire_kv_cache_iswa::get_swa() const { return kv_swa.get(); }

//
// monowire_kv_cache_iswa_context
//

monowire_kv_cache_iswa_context::monowire_kv_cache_iswa_context(monowire_memory_status status) : status(status) {}

monowire_kv_cache_iswa_context::monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv)
    : ctx_base(kv->get_base()->init_full()), ctx_swa(kv->get_swa()->init_full()),
      status(monowire_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {}

monowire_kv_cache_iswa_context::monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv, monowire_context *lctx,
                                                               bool optimize)
    : ctx_base(kv->get_base()->init_update(lctx, optimize)), ctx_swa(kv->get_swa()->init_update(lctx, optimize)),
      status(monowire_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {}

monowire_kv_cache_iswa_context::monowire_kv_cache_iswa_context(monowire_kv_cache_iswa *kv, slot_info_vec_t sinfos_base,
                                                               slot_info_vec_t sinfos_swa,
                                                               std::vector<monowire_ubatch> ubatches)
    : ubatches(std::move(ubatches)),
      // note: here we copy the ubatches. not sure if this is ideal
      ctx_base(new monowire_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
      ctx_swa(new monowire_kv_cache_context(kv->get_swa(), std::move(sinfos_swa), this->ubatches)),
      status(monowire_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {}

monowire_kv_cache_iswa_context::~monowire_kv_cache_iswa_context() = default;

bool monowire_kv_cache_iswa_context::next() {
    assert(status == MONOWIRE_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool monowire_kv_cache_iswa_context::apply() {
    assert(!monowire_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa->apply();

    return res;
}

monowire_memory_status monowire_kv_cache_iswa_context::get_status() const { return status; }

const monowire_ubatch &monowire_kv_cache_iswa_context::get_ubatch() const {
    assert(status == MONOWIRE_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const monowire_kv_cache_context *monowire_kv_cache_iswa_context::get_base() const {
    assert(status == MONOWIRE_MEMORY_STATUS_SUCCESS);

    return static_cast<const monowire_kv_cache_context *>(ctx_base.get());
}

const monowire_kv_cache_context *monowire_kv_cache_iswa_context::get_swa() const {
    assert(status == MONOWIRE_MEMORY_STATUS_SUCCESS);

    return static_cast<const monowire_kv_cache_context *>(ctx_swa.get());
}
