#pragma once

#include "gguf.h"
#include "monowire-arch.h"
#include "monowire.h"

#include <vector>

// FIXME temporary function for better error messages
bool monowire_model_saver_supports_arch(llm_arch arch);

struct monowire_model_saver {
    struct gguf_context *gguf_ctx = nullptr;
    const bool gguf_ctx_owned;
    const struct monowire_model *model;
    const struct LLM_KV llm_kv;

    monowire_model_saver(const struct monowire_model *model);
    monowire_model_saver(enum llm_arch arch, struct gguf_context *gguf_ctx);
    ~monowire_model_saver();

    void add_kv(enum llm_kv key, uint32_t value);
    void add_kv(enum llm_kv key, int32_t value);
    void add_kv(enum llm_kv key, float value);
    void add_kv(enum llm_kv key, bool value);
    void add_kv(enum llm_kv key, const char *value);

    [[noreturn]] void add_kv(enum llm_kv key,
                             char value); // needed to make the template below compile

    template <typename Container> void add_kv(enum llm_kv key, const Container &value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> &value);

    void add_tensor(const struct ggml_tensor *tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string &path_model);
    void save(FILE *file);
};
