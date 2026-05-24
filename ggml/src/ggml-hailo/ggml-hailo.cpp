// Hailo-10H NPU backend.
//
// Two network groups inside the HEF:
//   "prefill" : processes the full prompt in one shot (n_tokens > 1)
//   "tbt"     : token-by-token generation (n_tokens == 1)

#include "ggml-hailo.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <optional>

#include <hailo/hailort.hpp>
using namespace hailort;

static std::string hailo_json_find_string(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t k = js.find(pat);
    if (k == std::string::npos) {
        return "";
    }
    size_t c = js.find(':', k + pat.size());
    if (c == std::string::npos) {
        return "";
    }
    size_t q1 = js.find('"', c + 1);
    if (q1 == std::string::npos) {
        return "";
    }
    size_t q2 = js.find('"', q1 + 1);
    if (q2 == std::string::npos) {
        return "";
    }
    return js.substr(q1 + 1, q2 - q1 - 1);
}

static int hailo_json_find_int(const std::string & js, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t k = js.find(pat);
    if (k == std::string::npos) {
        return -1;
    }
    size_t c = js.find(':', k + pat.size());
    if (c == std::string::npos) {
        return -1;
    }
    size_t i = c + 1;
    while (i < js.size() && std::isspace((unsigned char)js[i])) {
        i++;
    }
    if (i >= js.size() || !std::isdigit((unsigned char)js[i])) {
        return -1;
    }
    long v = 0;
    while (i < js.size() && std::isdigit((unsigned char)js[i])) {
        v = v * 10 + (js[i] - '0'); i++;
    }
    return (int)v;
}

enum class hailo_active_group {
    NONE,
    PREFILL,
    TBT
};

struct hailo_stream_params {
    float    embd_zp        = 0.0f;   // input_layer1 quant zero-point (padding fill)
    uint8_t  mask_value     = 0;      // input_layer2 "attend" byte = quantize(1.0)
    int      num_attn_heads = 0;      // input_layer3 features / head_dim
    int      num_kv_heads   = 0;      // input_layer5 features / head_dim
    int      kv_cache_size  = 0;      // input_layer2 features / num_attn_heads
    int      n_embd         = 0;      // input_layer1 features (token-embedding width)
    int      seq_len        = 0;      // input_layer1 width (prefill window; 1 for tbt)
    bool     loaded         = false;
};

struct hailo_dma_group {
    std::map<std::string, MemoryView>             views;
    std::vector<BufferPtr>                        buffers;
    std::vector<std::string>                      out_names;
    std::vector<size_t>                           out_frames;
    std::optional<ConfiguredInferModel::Bindings> bindings;
};

struct ggml_backend_hailo_context {
    std::shared_ptr<VDevice>       vdevice;

    std::shared_ptr<InferModel>    infer_model_prefill;
    ConfiguredInferModel           configured_prefill;
    hailo_dma_group                dma_prefill;

    std::shared_ptr<InferModel>    infer_model_tbt;
    ConfiguredInferModel           configured_tbt;
    hailo_dma_group                dma_tbt;

    hailo_active_group             active_group = hailo_active_group::NONE;
    int64_t                        n_past       = 0;
    std::string                    name         = "Hailo";

    hailo_stream_params            params_prefill;
    hailo_stream_params            params_tbt;

    std::vector<float>             rope_theta;
    int                            head_dim = 0;

    // RoPE input-layer roles from hailo-config.json "input_layers_names_suffixes".
    std::string                    sfx_pe_q_cos, sfx_pe_q_sin, sfx_pe_k_cos, sfx_pe_k_sin;
    int                            cfg_num_attn_heads = -1;
    int                            cfg_num_kv_heads   = -1;
    int                            cfg_kv_cache_size  = -1;

    const uint16_t * hef_embeddings = nullptr;
};

static bool hailo_is_positional_embed_layer(const std::string & name) {
    return name.find("input_layer3") != std::string::npos ||
           name.find("input_layer4") != std::string::npos ||
           name.find("input_layer5") != std::string::npos ||
           name.find("input_layer6") != std::string::npos;
}

// This function set the input and output format types to HAILO_FORMAT_TYPE_FLOAT32
// which enables the hailort driver on the host to handle quantization.
static void hailo_configure_llm_formats(std::shared_ptr<InferModel> & infer_model) {
    for (const auto & name : infer_model->get_input_names()) {
        // Only dfor the RoPE related input formats which are layers 3-6.
        if (hailo_is_positional_embed_layer(name)) {
            infer_model->input(name).expect("hailo: bad input").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        }
    }
    for (const auto & name : infer_model->get_output_names()) {
        infer_model->output(name).expect("hailo: bad output").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }
}

static void ggml_backend_hailo_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_hailo_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void hailo_alloc_dma_group(std::shared_ptr<InferModel>& model, ConfiguredInferModel& cfg, hailo_dma_group& grp) {
    for (const auto & name : model->get_input_names()) {
        size_t fs = model->input(name).expect("").get_frame_size();
        auto buf = Buffer::create_shared(fs, BufferStorageParams::create_dma()).expect("Failed DMA init");
        grp.views.emplace(name, buf->as_view());
        grp.buffers.push_back(std::move(buf));
    }

    auto target_outs = model->get_output_names();
    for (size_t i = 1; i < target_outs.size(); ++i) {
        GGML_ASSERT(target_outs[i-1] < target_outs[i] && "hailo: output streams are not in sorted order!");
    }

    for (const auto & name : target_outs) {
        size_t fs = model->output(name).expect("").get_frame_size();
        auto buf = Buffer::create_shared(fs, BufferStorageParams::create_dma()).expect("Failed DMA init");
        grp.views.emplace(name, buf->as_view());
        grp.buffers.push_back(std::move(buf));
        grp.out_names.push_back(name);
        grp.out_frames.push_back(fs);
    }
    grp.bindings = cfg.create_bindings(grp.views).expect("Failed creating bindings");
}

static void ggml_backend_hailo_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                 const void * data, size_t offset, size_t size) {
    std::memcpy((char *)tensor->data + offset, data, size);

    if (std::strcmp(tensor->name, "hailo.hef_data") != 0 || offset + size != ggml_nbytes(tensor)) {
        return;
    }

    auto * ctx = static_cast<ggml_backend_hailo_context *>(buffer->buft->device->context);
    GGML_ASSERT(ctx != nullptr && ctx->vdevice != nullptr);

    GGML_LOG_INFO("%s: initializing NPU networks from HEF (%zu bytes)\n", __func__, ggml_nbytes(tensor));
    hailort::MemoryView hef_mem{tensor->data, ggml_nbytes(tensor)};

    {
        // Extract embeddings.bin from .hef
        auto hef_exp = Hef::create(hef_mem);
        if (!hef_exp) {
            GGML_ABORT("hailo: Hef::create failed\n");
        }
        Hef hef_api = hef_exp.release();

        auto em_bin = hef_api.get_external_resources("embeddings.bin");
        if (!em_bin) {
            GGML_ABORT("hailo: embeddings.bin missing from HEF\n");
        }
        MemoryView emb = em_bin.release();
        ctx->hef_embeddings = reinterpret_cast<const uint16_t*>(emb.data());

        // Extract rope_theta_data.bin for .hef
        auto th_bin = hef_api.get_external_resources("rope_theta_data.bin");
        if (!th_bin) {
            GGML_ABORT("hailo: rope_theta_data.bin missing from HEF\n");
        }
        MemoryView theta_mv = th_bin.release();
        const float * theta_f = reinterpret_cast<const float *>(theta_mv.data());
        const size_t theta_n  = theta_mv.size() / sizeof(float);
        ctx->rope_theta.assign(theta_f, theta_f + theta_n);
        ctx->head_dim = (int)theta_n;
        GGML_LOG_INFO("%s: loaded rope_theta_data.bin (head_dim=%d)\n", __func__, ctx->head_dim);

        // Extract hailo-config.json
        auto cfg_res = hef_api.get_external_resources("hailo-config.json");
        if (cfg_res) {
            MemoryView cv = cfg_res.release();
            const std::string js(reinterpret_cast<const char *>(cv.data()), cv.size());
            ctx->sfx_pe_q_cos = hailo_json_find_string(js, "pe_q_cos");
            ctx->sfx_pe_q_sin = hailo_json_find_string(js, "pe_q_sin");
            ctx->sfx_pe_k_cos = hailo_json_find_string(js, "pe_k_cos");
            ctx->sfx_pe_k_sin = hailo_json_find_string(js, "pe_k_sin");
            ctx->cfg_num_attn_heads = hailo_json_find_int(js, "num_attention_heads");
            ctx->cfg_num_kv_heads   = hailo_json_find_int(js, "num_key_value_heads");
            ctx->cfg_kv_cache_size  = hailo_json_find_int(js, "kv_cache_size");
            GGML_LOG_INFO("%s: hailo-config.json roles q_cos=%s q_sin=%s k_cos=%s k_sin=%s "
                          "attn_heads=%d kv_heads=%d kv_cache=%d\n", __func__,
                          ctx->sfx_pe_q_cos.c_str(), ctx->sfx_pe_q_sin.c_str(),
                          ctx->sfx_pe_k_cos.c_str(), ctx->sfx_pe_k_sin.c_str(),
                          ctx->cfg_num_attn_heads, ctx->cfg_num_kv_heads, ctx->cfg_kv_cache_size);
        } else {
            GGML_LOG_INFO("%s: no hailo-config.json; using positional layer mapping\n", __func__);
        }
    }

    // Initialize prefill neural network group.
    ctx->infer_model_prefill = ctx->vdevice->create_infer_model(hef_mem, "base_model__prefill")
        .expect("Failed to create prefill infer model");
    ctx->infer_model_prefill->set_enable_kv_cache(true);
    hailo_configure_llm_formats(ctx->infer_model_prefill);
    ctx->configured_prefill = ctx->infer_model_prefill->configure()
        .expect("Failed to configure prefill model");

    // Initialize token generation neural network group.
    ctx->infer_model_tbt = ctx->vdevice->create_infer_model(hef_mem, "base_model__tbt")
        .expect("Failed to create tbt infer model");
    ctx->infer_model_tbt->set_enable_kv_cache(true);
    hailo_configure_llm_formats(ctx->infer_model_tbt);
    ctx->configured_tbt = ctx->infer_model_tbt->configure()
        .expect("Failed to configure tbt model");

    // Allocate DMA groups.
    hailo_alloc_dma_group(ctx->infer_model_prefill, ctx->configured_prefill, ctx->dma_prefill);
    hailo_alloc_dma_group(ctx->infer_model_tbt,     ctx->configured_tbt,     ctx->dma_tbt);
}

static void ggml_backend_hailo_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                                 void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
    GGML_UNUSED(buffer);
}

static const struct ggml_backend_buffer_i ggml_backend_hailo_buffer_interface = {
    ggml_backend_hailo_buffer_free_buffer,
    ggml_backend_hailo_buffer_get_base,
    nullptr, nullptr,
    ggml_backend_hailo_buffer_set_tensor,
    ggml_backend_hailo_buffer_get_tensor,
    nullptr, nullptr, nullptr, nullptr, nullptr,
};

static ggml_backend_buffer_t ggml_backend_hailo_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * mem = nullptr;
    if (posix_memalign(&mem, 32, size) != 0) {
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_hailo_buffer_interface, mem, size);
}

static const char * ggml_backend_hailo_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "Hailo_Host";
}

static size_t ggml_backend_hailo_buffer_type_get_alignment(ggml_backend_buffer_type_t) {
    return 32;
}

static bool ggml_backend_hailo_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_hailo_buffer_type_interface = {
    ggml_backend_hailo_buffer_type_get_name,
    ggml_backend_hailo_buffer_type_alloc_buffer,
    ggml_backend_hailo_buffer_type_get_alignment,
    nullptr,
    nullptr,
    ggml_backend_hailo_buffer_type_is_host,
};

static struct ggml_backend_buffer_type ggml_backend_hailo_buffer_type_obj = {
    ggml_backend_hailo_buffer_type_interface,
    nullptr,
    nullptr
};

static ggml_backend_buffer_type_t ggml_backend_hailo_buffer_type(void) {
    return &ggml_backend_hailo_buffer_type_obj;
}

static void hailo_load_stream_params(const ggml_backend_hailo_context * ctx,
                                     std::shared_ptr<InferModel> & infer_model,
                                     hailo_stream_params & p, int head_dim) {
    uint32_t mask_features = 0;

    for (const auto & name : infer_model->get_input_names()) {
        auto stream = infer_model->input(name).expect("");
        auto qi     = stream.get_quant_infos();
        const uint32_t features = stream.shape().features;

        if (name.find("input_layer1") != std::string::npos) {
            p.n_embd = (int)features;
            // input_layer1 shape is (1 x seq_len x n_embd): width is the prefill
            // window (96 for the Qwen2.5 HEF, 1 for tbt).
            p.seq_len = (int)stream.shape().width;
            if (!qi.empty()) {
                p.embd_zp = qi[0].qp_zp;
            }
        } else if (name.find("input_layer2") != std::string::npos) {
            if (!qi.empty()) {
                p.mask_value = (uint8_t)std::lround(1.0f / qi[0].qp_scale + qi[0].qp_zp);
            }
            mask_features = features;
        }
    }

    auto features_of = [&](const char * suffix) -> uint32_t {
        for (const auto & name : infer_model->get_input_names()) {
            if (name.find(suffix) != std::string::npos) {
                return infer_model->input(name).expect("").shape().features;
            }
        }
        return 0;
    };
    if (ctx->cfg_num_attn_heads > 0) {
        p.num_attn_heads = ctx->cfg_num_attn_heads;
    } else {
        p.num_attn_heads = (int)(features_of("input_layer3") / head_dim);
    }
    if (ctx->cfg_num_kv_heads > 0) {
        p.num_kv_heads = ctx->cfg_num_kv_heads;
    } else {
        p.num_kv_heads = (int)(features_of("input_layer5") / head_dim);
    }

    GGML_ASSERT(p.num_attn_heads > 0 && "hailo: num_attn_heads is 0");
    GGML_ASSERT(p.num_kv_heads   > 0 && "hailo: num_kv_heads is 0");

    p.kv_cache_size = ctx->cfg_kv_cache_size > 0
                        ? ctx->cfg_kv_cache_size
                        : (int)(mask_features / p.num_attn_heads);
    p.loaded = true;
}

static void hailo_fill_attention_mask(uint8_t * buf, int seq_len, int64_t cache_usage,
                                      int kv_cache_size, int n_heads, uint8_t mask_value) {
    const int mask_cache_usage = (int)std::min<int64_t>(cache_usage, kv_cache_size);
    const int token_rows = (int)std::min<int64_t>(mask_cache_usage, seq_len);
    const int padded_rows = seq_len - token_rows;
    const int cols        = kv_cache_size;
    const int row_stride  = cols * n_heads;

    std::memset(buf, 0, (size_t)seq_len * row_stride);

    auto set_head0 = [&](int row, int col0, int count) {
        if (count <= 0) {
            return;
        }
        std::memset(buf + (size_t)row * row_stride + col0, mask_value, (size_t)count);
    };

    // padding rows
    for (int pr = 0; pr < padded_rows; ++pr) {
        set_head0(pr, 0, cols);
    }

    const int past = mask_cache_usage - token_rows;
    const int used_cols = cols - mask_cache_usage;

    // set token rows
    for (int br = 0; br < token_rows; ++br) {
        const int row = padded_rows + br;
        set_head0(row, used_cols, past + br + 1);
    }

    for (int r = 0; r < seq_len; ++r) {
        const uint8_t * head0 = buf + (size_t)r * row_stride;
        for (int h = 1; h < n_heads; ++h) {
            std::memcpy(buf + (size_t)r * row_stride + (size_t)h * cols, head0, cols);
        }
    }
}

static void ggml_backend_hailo_compute_forward(ggml_backend_hailo_context * ctx, struct ggml_tensor * node) {
    const int32_t * token_ids  = static_cast<const int32_t *>(node->src[0]->data);
    const int64_t n_tokens = node->src[0]->ne[0];

    const bool is_tbt = (n_tokens == 1) && (ctx->n_past > 0);

    if (!is_tbt) {
        ctx->active_group = hailo_active_group::PREFILL;
        ctx->n_past = 0;
    } else {
        ctx->active_group = hailo_active_group::TBT;
    }

    ConfiguredInferModel & cfg = is_tbt ? ctx->configured_tbt : ctx->configured_prefill;
    hailo_dma_group & dma_grp = is_tbt ? ctx->dma_tbt : ctx->dma_prefill;
    std::shared_ptr<InferModel> & infer_model = is_tbt ? ctx->infer_model_tbt : ctx->infer_model_prefill;
    hailo_stream_params & params = is_tbt ? ctx->params_tbt : ctx->params_prefill;

    const int head_dim = ctx->head_dim;

    if (!params.loaded) {
        hailo_load_stream_params(ctx, infer_model, params, head_dim);

        GGML_LOG_INFO("ggml-hailo: [%s] params: head_dim=%d n_embd=%d seq_len=%d "
                      "attn_heads=%d kv_heads=%d kv_cache=%d mask_value=%u embd_zp=%.3f\n",
                      is_tbt ? "tbt" : "prefill", head_dim, params.n_embd, params.seq_len,
                      params.num_attn_heads, params.num_kv_heads, params.kv_cache_size,
                      params.mask_value, params.embd_zp);
    }

    const int64_t seq_len     = params.seq_len;
    const int n_attn_heads    = params.num_attn_heads;
    const int n_kv_heads      = params.num_kv_heads;
    const int kv_cache_size   = params.kv_cache_size;
    const int64_t n_embd      = params.n_embd;
    const int32_t embd_zp     = (int32_t)std::lround(params.embd_zp);
    const int64_t pos_base    = ctx->n_past;
    const int64_t cache_usage = pos_base + n_tokens;
    const int64_t row_offset  = seq_len - n_tokens;

    const bool have_cfg_roles = !ctx->sfx_pe_q_cos.empty() && !ctx->sfx_pe_q_sin.empty() &&
                                !ctx->sfx_pe_k_cos.empty() && !ctx->sfx_pe_k_sin.empty();
    const std::string sfx_qcos = have_cfg_roles ? ctx->sfx_pe_q_cos : "input_layer3";
    const std::string sfx_qsin = have_cfg_roles ? ctx->sfx_pe_q_sin : "input_layer4";
    const std::string sfx_kcos = have_cfg_roles ? ctx->sfx_pe_k_cos : "input_layer5";
    const std::string sfx_ksin = have_cfg_roles ? ctx->sfx_pe_k_sin : "input_layer6";

    uint16_t * embd_dst = nullptr;
    uint8_t  * mask_dst = nullptr;
    float    * qcos_dst = nullptr, * qsin_dst = nullptr, * kcos_dst = nullptr, * ksin_dst = nullptr;
    for (const auto & name : infer_model->get_input_names()) {
        void * dst = dma_grp.views[name].data();
        if      (name.find("input_layer1") != std::string::npos) embd_dst = (uint16_t *)dst;
        else if (name.find("input_layer2") != std::string::npos) mask_dst = (uint8_t  *)dst;
        else if (name.find(sfx_qcos) != std::string::npos)       qcos_dst = (float    *)dst;
        else if (name.find(sfx_qsin) != std::string::npos)       qsin_dst = (float    *)dst;
        else if (name.find(sfx_kcos) != std::string::npos)       kcos_dst = (float    *)dst;
        else if (name.find(sfx_ksin) != std::string::npos)       ksin_dst = (float    *)dst;
    }
    GGML_ASSERT(embd_dst && mask_dst && qcos_dst && qsin_dst && kcos_dst && ksin_dst &&
                "hailo: could not resolve all input layers (check config role suffixes)");

    // embeddings
    const uint16_t embd_pad = (uint16_t)std::max(0, std::min(65535, (int)embd_zp));
    std::fill(embd_dst, embd_dst + (size_t)(n_embd * seq_len), embd_pad);
    for (int64_t i = 0; i < n_tokens; ++i) {
        std::memcpy(embd_dst + (row_offset + i) * n_embd,
                    ctx->hef_embeddings + (int64_t)token_ids[i] * n_embd,
                    (size_t)n_embd * sizeof(uint16_t));
    }

    // attention mask
    hailo_fill_attention_mask(mask_dst, (int)seq_len, cache_usage, kv_cache_size, n_attn_heads, params.mask_value);

    // rope
    const size_t n_q_rope = (size_t)n_attn_heads * head_dim * seq_len;
    const size_t n_k_rope = (size_t)n_kv_heads   * head_dim * seq_len;
    std::fill(qcos_dst, qcos_dst + n_q_rope, 1.0f); std::fill(qsin_dst, qsin_dst + n_q_rope, 0.0f);
    std::fill(kcos_dst, kcos_dst + n_k_rope, 1.0f); std::fill(ksin_dst, ksin_dst + n_k_rope, 0.0f);

    const std::vector<float> & theta = ctx->rope_theta;
    std::vector<float> cos_t(head_dim), sin_t(head_dim);
    for (int64_t i = 0; i < n_tokens; ++i) {
        const float pos = (float)(pos_base + i);
        for (int l = 0; l < head_dim; ++l) {
            cos_t[l] = std::cos(theta[l] * pos);
            sin_t[l] = std::sin(theta[l] * pos);
        }
        const int64_t row = row_offset + i;
        for (int h = 0; h < n_attn_heads; ++h) {
            std::memcpy(qcos_dst + (row * n_attn_heads + h) * head_dim, cos_t.data(), head_dim * sizeof(float));
            std::memcpy(qsin_dst + (row * n_attn_heads + h) * head_dim, sin_t.data(), head_dim * sizeof(float));
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            std::memcpy(kcos_dst + (row * n_kv_heads + h) * head_dim, cos_t.data(), head_dim * sizeof(float));
            std::memcpy(ksin_dst + (row * n_kv_heads + h) * head_dim, sin_t.data(), head_dim * sizeof(float));
        }
    }

    ctx->n_past += n_tokens;

    if (pos_base == 0) {
        // This send a control command to the NPU to reset the pointers for its internal buffers.
        cfg.init_cache(0);
    }
    // Specify where the K an V vectors should be written in the internal buffers.
    cfg.update_cache_offset((int32_t) n_tokens);

    // Run inference. TODO: investigate the async api.
    auto run_s = cfg.run(*dma_grp.bindings, std::chrono::milliseconds(5000));
    GGML_ASSERT(run_s == HAILO_SUCCESS && "hailo: inference run failed");

    // reconstruct output
    const int64_t n_vocab = node->ne[0];
    const int64_t n_outputs = node->ne[1];
    float * logits = static_cast<float *>(node->data) + (n_outputs - 1) * n_vocab;

    size_t out_off = 0;
    for (size_t i = 0; i < dma_grp.out_names.size(); ++i) {
        const float* out_view = reinterpret_cast<const float*>(dma_grp.views[dma_grp.out_names[i]].data());
        const size_t n = dma_grp.out_frames[i] / sizeof(float);
        std::memcpy(logits + out_off, out_view, n * sizeof(float));
        out_off += n;
    }
}

static enum ggml_status ggml_backend_hailo_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_hailo_context *>(backend->device->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (node->op == GGML_OP_CUSTOM) {
            ggml_backend_hailo_compute_forward(ctx, node);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_hailo_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hailo_context *>(backend->device->context)->name.c_str();
}

static void ggml_backend_hailo_free(ggml_backend_t backend) {
    delete backend;
}

static const struct ggml_backend_i ggml_backend_hailo_interface = {
    ggml_backend_hailo_get_name,
    ggml_backend_hailo_free,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    ggml_backend_hailo_graph_compute,
    nullptr,
    nullptr,
    nullptr,
};

static ggml_guid_t ggml_backend_hailo_guid(void) {
    static ggml_guid guid = {
        0xc4, 0x1a, 0x94, 0x8e, 0xdb, 0xb4, 0x4e,0x84,
        0xa7, 0xd3,0x89, 0xbc, 0x67, 0x8b, 0xef, 0x6b
    };
    return &guid;
}

static const char * ggml_backend_hailo_dev_get_name(ggml_backend_dev_t) {
    return "Hailo";
}

static const char * ggml_backend_hailo_dev_get_description(ggml_backend_dev_t) {
    return "Hailo-10H NPU";
}

static void ggml_backend_hailo_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0; *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_hailo_dev_get_type(ggml_backend_dev_t) {
    // TODO: is this the correct type?
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_hailo_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hailo_dev_get_name(dev);
    props->description = ggml_backend_hailo_dev_get_description(dev);
    props->type        = ggml_backend_hailo_dev_get_type(dev);
    ggml_backend_hailo_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { false, false, false, false };
}

static ggml_backend_t ggml_backend_hailo_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    return new ggml_backend{ ggml_backend_hailo_guid(), ggml_backend_hailo_interface, dev, nullptr };
}

static ggml_backend_buffer_type_t ggml_backend_hailo_dev_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hailo_buffer_type_obj.device = dev;
    return ggml_backend_hailo_buffer_type();
}

static bool ggml_backend_hailo_dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    return op->op == GGML_OP_CUSTOM || op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE ||
           op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE;
}

static bool ggml_backend_hailo_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_hailo_buffer_type() || buft == ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_hailo_dev_offload_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    return op->op == GGML_OP_CUSTOM;
}

static const struct ggml_backend_device_i ggml_backend_hailo_device_interface = {
    ggml_backend_hailo_dev_get_name,
    ggml_backend_hailo_dev_get_description,
    ggml_backend_hailo_dev_get_memory,
    ggml_backend_hailo_dev_get_type,
    ggml_backend_hailo_dev_get_props,
    ggml_backend_hailo_dev_init_backend,
    ggml_backend_hailo_dev_get_buffer_type,
    nullptr,
    nullptr,
    ggml_backend_hailo_dev_supports_op,
    ggml_backend_hailo_dev_supports_buft,
    ggml_backend_hailo_dev_offload_op,
    nullptr,
    nullptr,
    nullptr,
};

static const char * ggml_backend_hailo_reg_get_name(ggml_backend_reg_t) {
    return "Hailo";
}

static size_t ggml_backend_hailo_reg_get_device_count(ggml_backend_reg_t) {
    return 1;
}

static ggml_backend_dev_t ggml_backend_hailo_reg_get_device(ggml_backend_reg_t reg, size_t) {
    static struct ggml_backend_device dev = { ggml_backend_hailo_device_interface, reg, reg->context };
    return &dev;
}

static const struct ggml_backend_reg_i ggml_backend_hailo_reg_interface = {
    ggml_backend_hailo_reg_get_name,
    ggml_backend_hailo_reg_get_device_count,
    ggml_backend_hailo_reg_get_device,
    nullptr,
};

ggml_backend_reg_t ggml_backend_hailo_reg(void) {
    static struct ggml_backend_reg reg = []() -> ggml_backend_reg {
        auto * ctx = new ggml_backend_hailo_context;
        auto vdevice_exp = VDevice::create();
        if (!vdevice_exp) {
            GGML_LOG_ERROR("ggml_backend_hailo_reg: failed to create VDevice: %s\n", hailo_get_status_message(vdevice_exp.status()));
            delete ctx;
            return {GGML_BACKEND_API_VERSION, ggml_backend_hailo_reg_interface, nullptr};
        }
        ctx->vdevice = std::move(vdevice_exp.value());
        return {GGML_BACKEND_API_VERSION, ggml_backend_hailo_reg_interface, ctx};
    }();
    return &reg;
}

ggml_backend_t ggml_backend_hailo_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_hailo_reg(), 0);
    return ggml_backend_hailo_dev_init_backend(dev, nullptr);
}

bool ggml_backend_is_hailo(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hailo_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_hailo_reg)
