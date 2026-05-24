#include "models.h"

#include <cstring>

void llama_model_hailo::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    type = LLM_TYPE_1_5B;
}

void llama_model_hailo::load_arch_tensors(llama_model_loader & ml) {
    const struct ggml_tensor * meta = ml.get_tensor_meta("hailo.hef_data");
    hef_data = create_tensor(tn(LLM_TENSOR_HAILO_HEF_DATA), {meta->ne[0]}, 0);
    GGML_ASSERT(hef_data);
}

std::unique_ptr<llm_graph_context> llama_model_hailo::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static void hailo_npu_forward(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith == 0) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
}

llama_model_hailo::graph::graph(const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {

    const int64_t n_vocab = model.vocab.n_tokens();

    auto inp_embd = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
    inp_embd->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_embd->tokens, "inp_tokens");
    ggml_set_input(inp_embd->tokens);
    res->t_inp_tokens = inp_embd->tokens;
    ggml_tensor * inp_tokens = inp_embd->tokens;
    res->add_input(std::move(inp_embd));

    // src[0]: token IDs (I32).
    ggml_tensor * args[] = { inp_tokens };
    ggml_tensor * logits = ggml_custom_4d(ctx0, GGML_TYPE_F32,
            n_vocab, n_outputs, 1, 1,
            args, 1,
            hailo_npu_forward, 1, nullptr);
    ggml_set_name(logits, "hailo_logits");

    res->t_logits = logits;
    ggml_build_forward_expand(gf, logits);
}
