#ifndef MONOWIRE_H
#define MONOWIRE_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef MONOWIRE_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef MONOWIRE_BUILD
#            define MONOWIRE_API __declspec(dllexport)
#        else
#            define MONOWIRE_API __declspec(dllimport)
#        endif
#    else
#        define MONOWIRE_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MONOWIRE_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define MONOWIRE_DEFAULT_SEED 0xFFFFFFFF

#define MONOWIRE_TOKEN_NULL -1

#define MONOWIRE_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define MONOWIRE_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define MONOWIRE_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define MONOWIRE_SESSION_MAGIC   MONOWIRE_FILE_MAGIC_GGSN
#define MONOWIRE_SESSION_VERSION 9

#define MONOWIRE_STATE_SEQ_MAGIC   MONOWIRE_FILE_MAGIC_GGSQ
#define MONOWIRE_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct monowire_vocab;
    struct monowire_model;
    struct monowire_context;
    struct monowire_sampler;

    typedef struct monowire_memory_i * monowire_memory_t;

    typedef int32_t monowire_pos;
    typedef int32_t monowire_token;
    typedef int32_t monowire_seq_id;

    enum monowire_vocab_type {
        MONOWIRE_VOCAB_TYPE_NONE   = 0, // For models without vocab
        MONOWIRE_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        MONOWIRE_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        MONOWIRE_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        MONOWIRE_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        MONOWIRE_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        MONOWIRE_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum monowire_rope_type {
        MONOWIRE_ROPE_TYPE_NONE   = -1,
        MONOWIRE_ROPE_TYPE_NORM   = 0,
        MONOWIRE_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        MONOWIRE_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        MONOWIRE_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        MONOWIRE_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum monowire_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        MONOWIRE_TOKEN_TYPE_UNDEFINED    = 0,
        MONOWIRE_TOKEN_TYPE_NORMAL       = 1,
        MONOWIRE_TOKEN_TYPE_UNKNOWN      = 2,
        MONOWIRE_TOKEN_TYPE_CONTROL      = 3,
        MONOWIRE_TOKEN_TYPE_USER_DEFINED = 4,
        MONOWIRE_TOKEN_TYPE_UNUSED       = 5,
        MONOWIRE_TOKEN_TYPE_BYTE         = 6,
    };

    enum monowire_token_attr {
        MONOWIRE_TOKEN_ATTR_UNDEFINED    = 0,
        MONOWIRE_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        MONOWIRE_TOKEN_ATTR_UNUSED       = 1 << 1,
        MONOWIRE_TOKEN_ATTR_NORMAL       = 1 << 2,
        MONOWIRE_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        MONOWIRE_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        MONOWIRE_TOKEN_ATTR_BYTE         = 1 << 5,
        MONOWIRE_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        MONOWIRE_TOKEN_ATTR_LSTRIP       = 1 << 7,
        MONOWIRE_TOKEN_ATTR_RSTRIP       = 1 << 8,
        MONOWIRE_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum monowire_ftype {
        MONOWIRE_FTYPE_ALL_F32              = 0,
        MONOWIRE_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // MONOWIRE_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // MONOWIRE_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // MONOWIRE_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        MONOWIRE_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //MONOWIRE_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //MONOWIRE_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //MONOWIRE_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        MONOWIRE_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        MONOWIRE_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors

        MONOWIRE_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum monowire_rope_scaling_type {
        MONOWIRE_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        MONOWIRE_ROPE_SCALING_TYPE_NONE        = 0,
        MONOWIRE_ROPE_SCALING_TYPE_LINEAR      = 1,
        MONOWIRE_ROPE_SCALING_TYPE_YARN        = 2,
        MONOWIRE_ROPE_SCALING_TYPE_LONGROPE    = 3,
        MONOWIRE_ROPE_SCALING_TYPE_MAX_VALUE   = MONOWIRE_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum monowire_pooling_type {
        MONOWIRE_POOLING_TYPE_UNSPECIFIED = -1,
        MONOWIRE_POOLING_TYPE_NONE = 0,
        MONOWIRE_POOLING_TYPE_MEAN = 1,
        MONOWIRE_POOLING_TYPE_CLS  = 2,
        MONOWIRE_POOLING_TYPE_LAST = 3,
        MONOWIRE_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum monowire_attention_type {
        MONOWIRE_ATTENTION_TYPE_UNSPECIFIED = -1,
        MONOWIRE_ATTENTION_TYPE_CAUSAL      = 0,
        MONOWIRE_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum monowire_flash_attn_type {
        MONOWIRE_FLASH_ATTN_TYPE_AUTO     = -1,
        MONOWIRE_FLASH_ATTN_TYPE_DISABLED = 0,
        MONOWIRE_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    MONOWIRE_API const char * monowire_flash_attn_type_name(enum monowire_flash_attn_type flash_attn_type);

    enum monowire_split_mode {
        MONOWIRE_SPLIT_MODE_NONE   = 0, // single GPU
        MONOWIRE_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
        MONOWIRE_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
        MONOWIRE_SPLIT_MODE_TENSOR = 3,
    };

    // TODO: simplify (upstream-reference)
    typedef struct monowire_token_data {
        monowire_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } monowire_token_data;

    typedef struct monowire_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        monowire_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } monowire_token_data_array;

    typedef bool (*monowire_progress_callback)(float progress, void * user_data);

    // Input data for monowire_encode/monowire_decode
    // A monowire_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by monowire_encode/monowire_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct monowire_batch {
        int32_t n_tokens;

        monowire_token  *  token;
        float        *  embd;
        monowire_pos    *  pos;
        int32_t      *  n_seq_id;
        monowire_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } monowire_batch;

    enum monowire_model_kv_override_type {
        MONOWIRE_KV_OVERRIDE_TYPE_INT,
        MONOWIRE_KV_OVERRIDE_TYPE_FLOAT,
        MONOWIRE_KV_OVERRIDE_TYPE_BOOL,
        MONOWIRE_KV_OVERRIDE_TYPE_STR,
    };

    enum monowire_model_meta_key {
        MONOWIRE_MODEL_META_KEY_SAMPLING_SEQUENCE,
        MONOWIRE_MODEL_META_KEY_SAMPLING_TOP_K,
        MONOWIRE_MODEL_META_KEY_SAMPLING_TOP_P,
        MONOWIRE_MODEL_META_KEY_SAMPLING_MIN_P,
        MONOWIRE_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        MONOWIRE_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        MONOWIRE_MODEL_META_KEY_SAMPLING_TEMP,
        MONOWIRE_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        MONOWIRE_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        MONOWIRE_MODEL_META_KEY_SAMPLING_MIROSTAT,
        MONOWIRE_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        MONOWIRE_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct monowire_model_kv_override {
        enum monowire_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct monowire_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct monowire_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct monowire_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum monowire_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is MONOWIRE_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: monowire_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        monowire_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct monowire_model_kv_override * kv_overrides;

        // maximum amount of mmap-backed model weights to preserve in memory using memory-aware streaming balancing
        uint64_t streaming_budget_bytes;

        // number of decoding layers to keep in the memory streaming runtime prefetch window
        uint32_t streaming_window;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_direct_io;   // use direct io, takes precedence over use_mmap when supported
        bool use_mlock;       // force system to keep model in RAM
        bool streaming_prefetch; // proactively prefetch preserved mmap ranges
        bool streaming_evict; // evict dynamic streaming ranges after the layer is consumed
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
    };

    struct monowire_sampler_seq_config {
        monowire_seq_id           seq_id;
        struct monowire_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       upstream-reference
    struct monowire_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to monowire_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum monowire_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum monowire_rope_scaling_type`
        enum monowire_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum monowire_attention_type    attention_type;    // attention type to use for embeddings
        enum monowire_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: upstream-reference
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of monowire_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (upstream-reference)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: upstream-reference
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: upstream-reference

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use monowire_sampler_chain_init)
        struct monowire_sampler_seq_config * samplers;
        size_t                            n_samplers;
    };

    struct monowire_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct monowire_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // model quantization parameters
    typedef struct monowire_model_quantize_params {
        int32_t nthread;                                            // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum monowire_ftype ftype;                                     // quantize to this monowire_ftype
        enum ggml_type output_tensor_type;                          // output tensor type
        enum ggml_type token_embedding_type;                        // token embeddings tensor type
        bool allow_requantize;                                      // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;                                // quantize output.weight
        bool only_copy;                                             // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                                                  // quantize all tensors to the default type
        bool keep_split;                                            // quantize to the same number of shards
        bool dry_run;                                               // calculate and show the final quantization size without performing quantization
        const struct monowire_model_imatrix_data * imatrix;            // pointer to importance matrix data
        const struct monowire_model_kv_override * kv_overrides;        // pointer to kv overrides
        const struct monowire_model_tensor_override * tt_overrides;    // pointer to tensor overrides
        const int32_t * prune_layers;                               // pointer to layer indices to prune
    } monowire_model_quantize_params;

    typedef struct monowire_logit_bias {
        monowire_token token;
        float bias;
    } monowire_logit_bias;

    typedef struct monowire_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } monowire_sampler_chain_params;

    // used in chat template
    typedef struct monowire_chat_message {
        const char * role;
        const char * content;
    } monowire_chat_message;

    // lora adapter
    struct monowire_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (upstream-reference)
    MONOWIRE_API struct monowire_model_params          monowire_model_default_params(void);
    MONOWIRE_API struct monowire_context_params        monowire_context_default_params(void);
    MONOWIRE_API struct monowire_sampler_chain_params  monowire_sampler_chain_default_params(void);
    MONOWIRE_API struct monowire_model_quantize_params monowire_model_quantize_default_params(void);

    // Initialize the Monowire + GGML backend.
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    MONOWIRE_API void monowire_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    MONOWIRE_API void monowire_backend_free(void);

    //optional:
    MONOWIRE_API void monowire_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    MONOWIRE_API void monowire_attach_threadpool(
            struct monowire_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    MONOWIRE_API void monowire_detach_threadpool(struct monowire_context * ctx);

    typedef void (*monowire_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    MONOWIRE_API struct monowire_model * monowire_model_init_from_user(
                    struct gguf_context * metadata,
          monowire_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct monowire_model_params   params);

    DEPRECATED(MONOWIRE_API struct monowire_model * monowire_load_model_from_file(
                             const char * path_model,
              struct monowire_model_params   params),
            "use monowire_model_load_from_file instead");

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use monowire_model_load_from_splits
    MONOWIRE_API struct monowire_model * monowire_model_load_from_file(
                             const char * path_model,
              struct monowire_model_params   params);

    // Load a model from an open FILE pointer
    MONOWIRE_API struct monowire_model * monowire_model_load_from_file_ptr(
                                   FILE * file,
              struct monowire_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    MONOWIRE_API struct monowire_model * monowire_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct monowire_model_params    params);

    MONOWIRE_API void monowire_model_save_to_file(
            const struct monowire_model * model,
                        const char * path_model);

    DEPRECATED(MONOWIRE_API void monowire_free_model(struct monowire_model * model),
            "use monowire_model_free instead");

    MONOWIRE_API void monowire_model_free(struct monowire_model * model);

    MONOWIRE_API struct monowire_context * monowire_init_from_model(
                     struct monowire_model * model,
            struct monowire_context_params   params);

    DEPRECATED(MONOWIRE_API struct monowire_context * monowire_new_context_with_model(
                     struct monowire_model * model,
            struct monowire_context_params   params),
            "use monowire_init_from_model instead");

    // Frees all allocated memory
    MONOWIRE_API void monowire_free(struct monowire_context * ctx);

    MONOWIRE_API int64_t monowire_time_us(void);

    MONOWIRE_API size_t monowire_max_devices(void);
    MONOWIRE_API size_t monowire_max_parallel_sequences(void);
    MONOWIRE_API size_t monowire_max_tensor_buft_overrides(void);

    MONOWIRE_API bool monowire_supports_mmap       (void);
    MONOWIRE_API bool monowire_supports_mlock      (void);
    MONOWIRE_API bool monowire_supports_gpu_offload(void);
    MONOWIRE_API bool monowire_supports_rpc        (void);

    // NOTE: After creating a monowire_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via monowire_context_params may differ from the actual values used by the context
    //       ref: upstream-reference
    MONOWIRE_API uint32_t monowire_n_ctx      (const struct monowire_context * ctx);
    MONOWIRE_API uint32_t monowire_n_ctx_seq  (const struct monowire_context * ctx);
    MONOWIRE_API uint32_t monowire_n_batch    (const struct monowire_context * ctx);
    MONOWIRE_API uint32_t monowire_n_ubatch   (const struct monowire_context * ctx);
    MONOWIRE_API uint32_t monowire_n_seq_max  (const struct monowire_context * ctx);

    DEPRECATED(MONOWIRE_API int32_t monowire_n_ctx_train(const struct monowire_model * model), "use monowire_model_n_ctx_train instead");
    DEPRECATED(MONOWIRE_API int32_t monowire_n_embd     (const struct monowire_model * model), "use monowire_model_n_embd instead");
    DEPRECATED(MONOWIRE_API int32_t monowire_n_layer    (const struct monowire_model * model), "use monowire_model_n_layer instead");
    DEPRECATED(MONOWIRE_API int32_t monowire_n_head     (const struct monowire_model * model), "use monowire_model_n_head instead");

    DEPRECATED(MONOWIRE_API int32_t monowire_n_vocab    (const struct monowire_vocab * vocab), "use monowire_vocab_n_tokens instead");

    MONOWIRE_API const struct monowire_model * monowire_get_model   (const struct monowire_context * ctx);
    MONOWIRE_API           monowire_memory_t   monowire_get_memory  (const struct monowire_context * ctx);
    MONOWIRE_API  enum monowire_pooling_type   monowire_pooling_type(const struct monowire_context * ctx); // TODO: rename to monowire_get_pooling_type

    MONOWIRE_API const struct monowire_vocab * monowire_model_get_vocab(const struct monowire_model * model);
    MONOWIRE_API enum monowire_rope_type       monowire_model_rope_type(const struct monowire_model * model);

    MONOWIRE_API int32_t monowire_model_n_ctx_train(const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_embd     (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_embd_inp (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_embd_out (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_layer    (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_head     (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_head_kv  (const struct monowire_model * model);
    MONOWIRE_API int32_t monowire_model_n_swa      (const struct monowire_model * model);

    // Get the model's RoPE frequency scaling factor
    MONOWIRE_API float monowire_model_rope_freq_scale_train(const struct monowire_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    MONOWIRE_API uint32_t monowire_model_n_cls_out(const struct monowire_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    MONOWIRE_API const char * monowire_model_cls_label(const struct monowire_model * model, uint32_t i);

    MONOWIRE_API enum monowire_vocab_type monowire_vocab_type(const struct monowire_vocab * vocab);

    MONOWIRE_API int32_t monowire_vocab_n_tokens(const struct monowire_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    MONOWIRE_API int32_t monowire_model_meta_val_str(const struct monowire_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    MONOWIRE_API int32_t monowire_model_meta_count(const struct monowire_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    MONOWIRE_API const char * monowire_model_meta_key_str(enum monowire_model_meta_key key);

    // Get metadata key name by index
    MONOWIRE_API int32_t monowire_model_meta_key_by_index(const struct monowire_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    MONOWIRE_API int32_t monowire_model_meta_val_str_by_index(const struct monowire_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    MONOWIRE_API int32_t monowire_model_desc(const struct monowire_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    MONOWIRE_API uint64_t monowire_model_size(const struct monowire_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    MONOWIRE_API const char * monowire_model_chat_template(const struct monowire_model * model, const char * name);

    // Returns the total number of parameters in the model
    MONOWIRE_API uint64_t monowire_model_n_params(const struct monowire_model * model);

    // Returns true if the model contains an encoder that requires monowire_encode() call
    MONOWIRE_API bool monowire_model_has_encoder(const struct monowire_model * model);

    // Returns true if the model contains a decoder that requires monowire_decode() call
    MONOWIRE_API bool monowire_model_has_decoder(const struct monowire_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    MONOWIRE_API monowire_token monowire_model_decoder_start_token(const struct monowire_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    MONOWIRE_API bool monowire_model_is_recurrent(const struct monowire_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    MONOWIRE_API bool monowire_model_is_hybrid(const struct monowire_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    MONOWIRE_API bool monowire_model_is_diffusion(const struct monowire_model * model);

    // Returns 0 on success
    MONOWIRE_API uint32_t monowire_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const monowire_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    MONOWIRE_API struct monowire_adapter_lora * monowire_adapter_lora_init(
            struct monowire_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    MONOWIRE_API int32_t monowire_adapter_meta_val_str(const struct monowire_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    MONOWIRE_API int32_t monowire_adapter_meta_count(const struct monowire_adapter_lora * adapter);

    // Get metadata key name by index
    MONOWIRE_API int32_t monowire_adapter_meta_key_by_index(const struct monowire_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    MONOWIRE_API int32_t monowire_adapter_meta_val_str_by_index(const struct monowire_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
    MONOWIRE_API void monowire_adapter_lora_free(struct monowire_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    MONOWIRE_API uint64_t            monowire_adapter_get_alora_n_invocation_tokens(const struct monowire_adapter_lora * adapter);
    MONOWIRE_API const monowire_token * monowire_adapter_get_alora_invocation_tokens  (const struct monowire_adapter_lora * adapter);

    // The following functions operate on a monowire_context, hence the naming: monowire_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    MONOWIRE_API int32_t monowire_set_adapters_lora(
            struct monowire_context * ctx,
            struct monowire_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a monowire_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See monowire_control_vector_load in common to load a control vector.
    MONOWIRE_API int32_t monowire_set_adapter_cvec(
            struct monowire_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    MONOWIRE_API void monowire_memory_clear(
            monowire_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    MONOWIRE_API bool monowire_memory_seq_rm(
            monowire_memory_t mem,
              monowire_seq_id seq_id,
                 monowire_pos p0,
                 monowire_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    MONOWIRE_API void monowire_memory_seq_cp(
            monowire_memory_t mem,
              monowire_seq_id seq_id_src,
              monowire_seq_id seq_id_dst,
                 monowire_pos p0,
                 monowire_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    MONOWIRE_API void monowire_memory_seq_keep(
            monowire_memory_t mem,
              monowire_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    MONOWIRE_API void monowire_memory_seq_add(
            monowire_memory_t mem,
              monowire_seq_id seq_id,
                 monowire_pos p0,
                 monowire_pos p1,
                 monowire_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    MONOWIRE_API void monowire_memory_seq_div(
            monowire_memory_t mem,
              monowire_seq_id seq_id,
                 monowire_pos p0,
                 monowire_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    MONOWIRE_API monowire_pos monowire_memory_seq_pos_min(
            monowire_memory_t mem,
              monowire_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    MONOWIRE_API monowire_pos monowire_memory_seq_pos_max(
            monowire_memory_t mem,
              monowire_seq_id seq_id);

    // Check if the memory supports shifting
    MONOWIRE_API bool monowire_memory_can_shift(monowire_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    MONOWIRE_API size_t monowire_state_get_size(struct monowire_context * ctx);
    MONOWIRE_API DEPRECATED(size_t monowire_get_state_size(struct monowire_context * ctx),
        "use monowire_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    MONOWIRE_API size_t monowire_state_get_data(
            struct monowire_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    MONOWIRE_API DEPRECATED(size_t monowire_copy_state_data(
            struct monowire_context * ctx,
                         uint8_t * dst),
        "use monowire_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    MONOWIRE_API size_t monowire_state_set_data(
            struct monowire_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    MONOWIRE_API DEPRECATED(size_t monowire_set_state_data(
            struct monowire_context * ctx,
                   const uint8_t * src),
        "use monowire_state_set_data instead");

    // Save/load session file
    MONOWIRE_API bool monowire_state_load_file(
            struct monowire_context * ctx,
                      const char * path_session,
                     monowire_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    MONOWIRE_API DEPRECATED(bool monowire_load_session_file(
            struct monowire_context * ctx,
                      const char * path_session,
                     monowire_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use monowire_state_load_file instead");

    MONOWIRE_API bool monowire_state_save_file(
            struct monowire_context * ctx,
                      const char * path_session,
               const monowire_token * tokens,
                          size_t   n_token_count);
    MONOWIRE_API DEPRECATED(bool monowire_save_session_file(
            struct monowire_context * ctx,
                      const char * path_session,
               const monowire_token * tokens,
                          size_t   n_token_count),
        "use monowire_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    MONOWIRE_API size_t monowire_state_seq_get_size(
            struct monowire_context * ctx,
                    monowire_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    MONOWIRE_API size_t monowire_state_seq_get_data(
            struct monowire_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    monowire_seq_id   seq_id);

    // Copy the sequence data (originally copied with `monowire_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    MONOWIRE_API size_t monowire_state_seq_set_data(
            struct monowire_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    monowire_seq_id   dest_seq_id);

    MONOWIRE_API size_t monowire_state_seq_save_file(
            struct monowire_context * ctx,
                      const char * filepath,
                    monowire_seq_id   seq_id,
               const monowire_token * tokens,
                          size_t   n_token_count);

    MONOWIRE_API size_t monowire_state_seq_load_file(
            struct monowire_context * ctx,
                      const char * filepath,
                    monowire_seq_id   dest_seq_id,
                     monowire_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

// for backwards-compat
#define MONOWIRE_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define MONOWIRE_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t monowire_state_seq_flags;

    MONOWIRE_API size_t monowire_state_seq_get_size_ext(
            struct monowire_context * ctx,
                    monowire_seq_id   seq_id,
           monowire_state_seq_flags   flags);

    MONOWIRE_API size_t monowire_state_seq_get_data_ext(
            struct monowire_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    monowire_seq_id   seq_id,
           monowire_state_seq_flags   flags);

    MONOWIRE_API size_t monowire_state_seq_set_data_ext(
            struct monowire_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    monowire_seq_id   dest_seq_id,
           monowire_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by monowire_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    MONOWIRE_API struct monowire_batch monowire_batch_get_one(
                  monowire_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with monowire_batch_free()
    // If embd != 0, monowire_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, monowire_batch.token will be allocated to store n_tokens monowire_token
    // The rest of the monowire_batch members are allocated with size n_tokens
    // All members are left uninitialized
    MONOWIRE_API struct monowire_batch monowire_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with monowire_batch_init()
    MONOWIRE_API void monowire_batch_free(struct monowire_batch batch);

    // Process a batch of tokens.
    // In contrast to monowire_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    MONOWIRE_API int32_t monowire_encode(
            struct monowire_context * ctx,
              struct monowire_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using monowire_memory_seq_pos_min() and monowire_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    MONOWIRE_API int32_t monowire_decode(
            struct monowire_context * ctx,
              struct monowire_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    MONOWIRE_API void monowire_set_n_threads(struct monowire_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    MONOWIRE_API int32_t monowire_n_threads(struct monowire_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    MONOWIRE_API int32_t monowire_n_threads_batch(struct monowire_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with monowire_get_embeddings()
    MONOWIRE_API void monowire_set_embeddings(struct monowire_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    MONOWIRE_API void monowire_set_causal_attn(struct monowire_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during monowire_decode() to load and cache their weights.
    MONOWIRE_API void monowire_set_warmup(struct monowire_context * ctx, bool warmup);

    // Set abort callback
    MONOWIRE_API void monowire_set_abort_callback(struct monowire_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    MONOWIRE_API void monowire_synchronize(struct monowire_context * ctx);

    // Token logits obtained from the last call to monowire_decode()
    // The logits for which monowire_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which monowire_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of monowire_get_logits_ith() (ref: upstream-reference)
    MONOWIRE_API float * monowire_get_logits(struct monowire_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // monowire_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    MONOWIRE_API float * monowire_get_logits_ith(struct monowire_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == MONOWIRE_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which monowire_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of monowire_get_embeddings_ith() (ref: upstream-reference)
    MONOWIRE_API float * monowire_get_embeddings(struct monowire_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // monowire_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    MONOWIRE_API float * monowire_get_embeddings_ith(struct monowire_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is MONOWIRE_POOLING_TYPE_NONE
    // when pooling_type == MONOWIRE_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    MONOWIRE_API float * monowire_get_embeddings_seq(struct monowire_context * ctx, monowire_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the monowire_context was created with at least one monowire_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns MONOWIRE_TOKEN_NULL if no token was sampled.
    MONOWIRE_API monowire_token monowire_get_sampled_token_ith(struct monowire_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches monowire_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    MONOWIRE_API float *  monowire_get_sampled_probs_ith      (struct monowire_context * ctx, int32_t i);
    MONOWIRE_API uint32_t monowire_get_sampled_probs_count_ith(struct monowire_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    MONOWIRE_API float *  monowire_get_sampled_logits_ith      (struct monowire_context * ctx, int32_t i);
    MONOWIRE_API uint32_t monowire_get_sampled_logits_count_ith(struct monowire_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    MONOWIRE_API monowire_token * monowire_get_sampled_candidates_ith      (struct monowire_context * ctx, int32_t i);
    MONOWIRE_API uint32_t      monowire_get_sampled_candidates_count_ith(struct monowire_context * ctx, int32_t i);

    //
    // Vocab
    //

    MONOWIRE_API const char * monowire_vocab_get_text(const struct monowire_vocab * vocab, monowire_token token);

    MONOWIRE_API float monowire_vocab_get_score(const struct monowire_vocab * vocab, monowire_token token);

    MONOWIRE_API enum monowire_token_attr monowire_vocab_get_attr(const struct monowire_vocab * vocab, monowire_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    MONOWIRE_API bool monowire_vocab_is_eog(const struct monowire_vocab * vocab, monowire_token token);

    // Identify if Token Id is a control token or a render-able token
    MONOWIRE_API bool monowire_vocab_is_control(const struct monowire_vocab * vocab, monowire_token token);

    // Special tokens
    MONOWIRE_API monowire_token monowire_vocab_bos(const struct monowire_vocab * vocab); // beginning-of-sentence
    MONOWIRE_API monowire_token monowire_vocab_eos(const struct monowire_vocab * vocab); // end-of-sentence
    MONOWIRE_API monowire_token monowire_vocab_eot(const struct monowire_vocab * vocab); // end-of-turn
    MONOWIRE_API monowire_token monowire_vocab_sep(const struct monowire_vocab * vocab); // sentence separator
    MONOWIRE_API monowire_token monowire_vocab_nl (const struct monowire_vocab * vocab); // next-line
    MONOWIRE_API monowire_token monowire_vocab_pad(const struct monowire_vocab * vocab); // padding
    MONOWIRE_API monowire_token monowire_vocab_mask(const struct monowire_vocab * vocab); // mask

    MONOWIRE_API bool monowire_vocab_get_add_bos(const struct monowire_vocab * vocab);
    MONOWIRE_API bool monowire_vocab_get_add_eos(const struct monowire_vocab * vocab);
    MONOWIRE_API bool monowire_vocab_get_add_sep(const struct monowire_vocab * vocab);

    MONOWIRE_API monowire_token monowire_vocab_fim_pre(const struct monowire_vocab * vocab);
    MONOWIRE_API monowire_token monowire_vocab_fim_suf(const struct monowire_vocab * vocab);
    MONOWIRE_API monowire_token monowire_vocab_fim_mid(const struct monowire_vocab * vocab);
    MONOWIRE_API monowire_token monowire_vocab_fim_pad(const struct monowire_vocab * vocab);
    MONOWIRE_API monowire_token monowire_vocab_fim_rep(const struct monowire_vocab * vocab);
    MONOWIRE_API monowire_token monowire_vocab_fim_sep(const struct monowire_vocab * vocab);

    DEPRECATED(MONOWIRE_API const char * monowire_token_get_text(const struct monowire_vocab * vocab, monowire_token token), "use monowire_vocab_get_text instead");
    DEPRECATED(MONOWIRE_API float monowire_token_get_score(const struct monowire_vocab * vocab, monowire_token token), "use monowire_vocab_get_score instead");
    DEPRECATED(MONOWIRE_API enum monowire_token_attr monowire_token_get_attr(const struct monowire_vocab * vocab, monowire_token token), "use monowire_vocab_get_attr instead");
    DEPRECATED(MONOWIRE_API bool monowire_token_is_eog(const struct monowire_vocab * vocab, monowire_token token), "use monowire_vocab_is_eog instead");
    DEPRECATED(MONOWIRE_API bool monowire_token_is_control(const struct monowire_vocab * vocab, monowire_token token), "use monowire_vocab_is_control instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_bos(const struct monowire_vocab * vocab), "use monowire_vocab_bos instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_eos(const struct monowire_vocab * vocab), "use monowire_vocab_eos instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_eot(const struct monowire_vocab * vocab), "use monowire_vocab_eot instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_cls(const struct monowire_vocab * vocab), "use monowire_vocab_cls instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_sep(const struct monowire_vocab * vocab), "use monowire_vocab_sep instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_nl (const struct monowire_vocab * vocab), "use monowire_vocab_nl instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_pad(const struct monowire_vocab * vocab), "use monowire_vocab_pad instead");
    DEPRECATED(MONOWIRE_API bool monowire_add_bos_token(const struct monowire_vocab * vocab), "use monowire_vocab_get_add_bos instead");
    DEPRECATED(MONOWIRE_API bool monowire_add_eos_token(const struct monowire_vocab * vocab), "use monowire_vocab_get_add_eos instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_pre(const struct monowire_vocab * vocab), "use monowire_vocab_fim_pre instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_suf(const struct monowire_vocab * vocab), "use monowire_vocab_fim_suf instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_mid(const struct monowire_vocab * vocab), "use monowire_vocab_fim_mid instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_pad(const struct monowire_vocab * vocab), "use monowire_vocab_fim_pad instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_rep(const struct monowire_vocab * vocab), "use monowire_vocab_fim_rep instead");
    DEPRECATED(MONOWIRE_API monowire_token monowire_token_fim_sep(const struct monowire_vocab * vocab), "use monowire_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(MONOWIRE_API monowire_token monowire_vocab_cls(const struct monowire_vocab * vocab), // classification
            "use monowire_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    MONOWIRE_API int32_t monowire_tokenize(
        const struct monowire_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     monowire_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    MONOWIRE_API int32_t monowire_token_to_piece(
              const struct monowire_vocab * vocab,
                           monowire_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of monowire_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    MONOWIRE_API int32_t monowire_detokenize(
        const struct monowire_vocab * vocab,
               const monowire_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only supports the built-in template list.
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple monowire_chat_message
    /// @param n_msg Number of monowire_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    MONOWIRE_API int32_t monowire_chat_apply_template(
                            const char * tmpl,
       const struct monowire_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    MONOWIRE_API int32_t monowire_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = monowire_sampler_chain_default_params();
    //
    //    monowire_sampler * smpl = monowire_sampler_chain_init(sparams);
    //
    //    monowire_sampler_chain_add(smpl, monowire_sampler_init_top_k(50));
    //    monowire_sampler_chain_add(smpl, monowire_sampler_init_top_p(0.9, 1));
    //    monowire_sampler_chain_add(smpl, monowire_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    monowire_sampler_chain_add(smpl, monowire_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        monowire_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const monowire_token id = monowire_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    monowire_sampler_free(smpl);
    //

    typedef void * monowire_sampler_context_t;

    struct monowire_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom monowire_sampler
    struct monowire_sampler_i {
        const char *           (*name)  (const struct monowire_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct monowire_sampler * smpl, monowire_token token);              // can be NULL
        void                   (*apply) (      struct monowire_sampler * smpl, monowire_token_data_array * cur_p); // required
        void                   (*reset) (      struct monowire_sampler * smpl);                                 // can be NULL
        struct monowire_sampler * (*clone) (const struct monowire_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct monowire_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct monowire_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct monowire_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct monowire_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct monowire_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct monowire_sampler * smpl);
    };

    struct monowire_sampler {
        struct monowire_sampler_i * iface;

        monowire_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with monowire_context_params.samplers when possible
    MONOWIRE_API bool monowire_set_sampler(struct monowire_context * ctx, monowire_seq_id seq_id, struct monowire_sampler * smpl);

    // mirror of monowire_sampler_i:
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init  (      struct monowire_sampler_i * iface, monowire_sampler_context_t ctx);
    MONOWIRE_API const char *           monowire_sampler_name  (const struct monowire_sampler * smpl);
    MONOWIRE_API void                   monowire_sampler_accept(      struct monowire_sampler * smpl, monowire_token token);
    MONOWIRE_API void                   monowire_sampler_apply (      struct monowire_sampler * smpl, monowire_token_data_array * cur_p);
    MONOWIRE_API void                   monowire_sampler_reset (      struct monowire_sampler * smpl);
    MONOWIRE_API struct monowire_sampler * monowire_sampler_clone (const struct monowire_sampler * smpl);
    // important: do not free if the sampler has been added to a monowire_sampler_chain (via monowire_sampler_chain_add)
    MONOWIRE_API void                   monowire_sampler_free  (      struct monowire_sampler * smpl);

    // monowire_sampler_chain
    // a type of monowire_sampler that can chain multiple samplers one after another

    MONOWIRE_API struct monowire_sampler * monowire_sampler_chain_init(struct monowire_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when monowire_sampler_free is called
    MONOWIRE_API void                   monowire_sampler_chain_add(      struct monowire_sampler * chain, struct monowire_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a monowire_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    MONOWIRE_API struct monowire_sampler * monowire_sampler_chain_get(      struct monowire_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    MONOWIRE_API int                    monowire_sampler_chain_n  (const struct monowire_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    MONOWIRE_API struct monowire_sampler * monowire_sampler_chain_remove(   struct monowire_sampler * chain, int32_t i);

    // available samplers:

    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_greedy(void);

    /// seed == MONOWIRE_DEFAULT_SEED to use a random seed.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in upstream-reference
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `monowire_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `monowire_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_grammar(
            const struct monowire_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(MONOWIRE_API struct monowire_sampler * monowire_sampler_init_grammar_lazy(
            const struct monowire_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const monowire_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use monowire_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in upstream-reference
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_grammar_lazy_patterns(
        const struct monowire_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const monowire_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_dry(
            const struct monowire_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: upstream-reference
    ///
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const monowire_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    MONOWIRE_API struct monowire_sampler * monowire_sampler_init_infill(const struct monowire_vocab * vocab);

    // Returns the seed used by the sampler if applicable, MONOWIRE_DEFAULT_SEED otherwise
    MONOWIRE_API uint32_t monowire_sampler_get_seed(const struct monowire_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = monowire_get_logits_ith(ctx, idx);
    //    monowire_token_data_array cur_p = { ... init from logits ... };
    //    monowire_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    monowire_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    MONOWIRE_API monowire_token monowire_sampler_sample(struct monowire_sampler * smpl, struct monowire_context * ctx, int32_t idx);

    // TODO: extend in the future
    //MONOWIRE_API void monowire_decode_with_sampler(struct monowire_context * ctx, struct monowire_sampler * smpl, struct monowire_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          monowire_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    MONOWIRE_API int32_t monowire_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          monowire_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    MONOWIRE_API int32_t monowire_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    MONOWIRE_API const char * monowire_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    MONOWIRE_API void monowire_log_get(ggml_log_callback * log_callback, void ** user_data);
    MONOWIRE_API void monowire_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by Monowire examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct monowire_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct monowire_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    MONOWIRE_API struct monowire_perf_context_data monowire_perf_context      (const struct monowire_context * ctx);
    MONOWIRE_API void                           monowire_perf_context_print(const struct monowire_context * ctx);
    MONOWIRE_API void                           monowire_perf_context_reset(      struct monowire_context * ctx);

    // NOTE: the following work only with samplers constructed via monowire_sampler_chain_init
    MONOWIRE_API struct monowire_perf_sampler_data monowire_perf_sampler      (const struct monowire_sampler * chain);
    MONOWIRE_API void                           monowire_perf_sampler_print(const struct monowire_sampler * chain);
    MONOWIRE_API void                           monowire_perf_sampler_reset(      struct monowire_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*monowire_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    MONOWIRE_API bool monowire_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct monowire_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in monowire_context if 0

        monowire_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    MONOWIRE_API void monowire_opt_init(struct monowire_context * lctx, struct monowire_model * model, struct monowire_opt_params lopt_params);

    MONOWIRE_API void monowire_opt_epoch(
            struct monowire_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // MONOWIRE_H
