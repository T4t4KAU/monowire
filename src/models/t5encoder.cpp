#include "models.h"

llm_build_t5encoder::llm_build_t5encoder(const monowire_model &model, const llm_graph_params &params)
    : llm_build_t5<true>(model, params) {}
