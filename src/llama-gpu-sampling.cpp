#include "llama-gpu-sampling.h"
#include "ggml.h"
#include <cstdio>

static void llama_sampler_gpu_greedy_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    (void) smpl;
    (void) gf;

    printf("apply_ggml: Building greedy sampler using ggml_argmax\n");
    struct ggml_tensor * argmax_result = ggml_argmax(ctx, ggml_data->logits);
    ggml_set_name(argmax_result, "argmax_result");
    ggml_data->sampled_token = argmax_result;
}

static const char * llama_sampler_gpu_greedy_sampler_name(const struct llama_sampler *) {
    return "test-ggml";
}

static struct llama_sampler * llama_sampler_gpu_greedy_clone(const struct llama_sampler * smpl) {
    (void) smpl;
    return llama_sampler_gpu_init_greedy();
}

struct llama_sampler * llama_sampler_gpu_init_greedy() {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_greedy_sampler_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ llama_sampler_gpu_greedy_clone,
        /*.free        =*/ nullptr,
        /*.apply_ggml  =*/ llama_sampler_gpu_greedy_apply_ggml,
        /*.accept_ggml =*/ nullptr,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ nullptr,
    };

    return sampler;
}

struct llama_sampler_gpu_temp_ctx {
    float temp;
};

static void llama_sampler_gpu_temp_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_temp_ctx *) smpl->ctx;

    if (ctx_data->temp <= 0.0f) {
        return;
    }

    printf("gpu-temp: Applying temperature scaling with temp=%.2f\n", ctx_data->temp);

    ggml_data->logits = ggml_scale(ctx, ggml_data->logits, 1.0f / ctx_data->temp);
    ggml_set_name(ggml_data->logits, "temp_scaled_logits");

    ggml_build_forward_expand(gf, ggml_data->logits);
}

static const char * llama_sampler_gpu_temp_name(const struct llama_sampler *) {
    return "gpu-temp";
}

static void llama_sampler_gpu_temp_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_temp_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_temp_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_gpu_temp_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_temp(ctx->temp);
}

struct llama_sampler * llama_sampler_gpu_init_temp(float temp) {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_temp_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ llama_sampler_gpu_temp_clone,
        /*.free        =*/ llama_sampler_gpu_temp_free,
        /*.apply_ggml  =*/ llama_sampler_gpu_temp_apply_ggml,
        /*.accept_ggml =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_gpu_temp_ctx {
        /*.temp =*/ temp,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}

struct llama_sampler_gpu_softmax_ctx {
};

static void llama_sampler_gpu_softmax_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {
    (void) smpl;

    printf("gpu-softmax: Applying softmax to logits and writing to probs\n");

    struct ggml_tensor * softmax_result = ggml_soft_max(ctx, ggml_data->logits);
    ggml_set_name(softmax_result, "softmax_probs");
    ggml_build_forward_expand(gf, softmax_result);
    ggml_data->probs = softmax_result;
}

static const char * llama_sampler_gpu_softmax_name(const struct llama_sampler *) {
    return "gpu-softmax";
}

static void llama_sampler_gpu_softmax_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_softmax_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_softmax_clone(const struct llama_sampler * smpl) {
    (void) smpl;
    return llama_sampler_gpu_init_softmax();
}

struct llama_sampler * llama_sampler_gpu_init_softmax() {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_softmax_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ llama_sampler_gpu_softmax_clone,
        /*.free        =*/ llama_sampler_gpu_softmax_free,
        /*.apply_ggml  =*/ llama_sampler_gpu_softmax_apply_ggml,
        /*.accept_ggml =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_gpu_softmax_ctx {
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}

struct llama_sampler_gpu_top_k_ctx {
    int32_t k;
};

static void llama_sampler_gpu_top_k_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    printf("gpu top-k: Building top-k sampler with k=%d\n", ctx_data->k);

    struct ggml_tensor * top_k = ggml_top_k(ctx, ggml_data->logits, ctx_data->k);
    ggml_set_name(top_k, "top_k");
    ggml_data->filtered_ids = top_k;

    struct ggml_tensor * logits_rows = ggml_reshape_2d(ctx, ggml_data->logits, 1, ggml_data->logits->ne[0]);
    struct ggml_tensor * top_k_rows = ggml_get_rows(ctx, logits_rows, top_k);
    ggml_set_name(top_k_rows, "top_k_rows");

    ggml_data->logits = ggml_reshape_1d(ctx, top_k_rows, ctx_data->k);
    ggml_build_forward_expand(gf, ggml_data->logits);

}

static const char * llama_sampler_gpu_top_k_name(const struct llama_sampler *) {
    return "gpu-top-k";
}

static void llama_sampler_gpu_top_k_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_top_k_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_gpu_top_k_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_top_k(ctx->k);
}

struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k) {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_top_k_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ llama_sampler_gpu_top_k_clone,
        /*.free        =*/ llama_sampler_gpu_top_k_free,
        /*.apply_ggml  =*/ llama_sampler_gpu_top_k_apply_ggml,
        /*.accept_ggml =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_gpu_top_k_ctx {
        /*.k =*/ k,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}

struct llama_sampler_gpu_top_p_ctx {
    int32_t k;
};

static void llama_sampler_gpu_top_p_apply_ggml(
        struct llama_sampler           * smpl,
        struct ggml_context            * ctx,
        struct ggml_cgraph             * gf,
        struct llama_sampler_ggml_data * ggml_data) {

    auto * ctx_data = (llama_sampler_gpu_top_p_ctx *) smpl->ctx;
    printf("gpu top-p: Building top-p sampler with k=%d\n", ctx_data->k);

    struct ggml_tensor * softmax = ggml_soft_max(ctx, ggml_data->logits);
    ggml_set_name(softmax, "softmax");

    struct ggml_tensor * top_k_ids = ggml_cont(ctx, ggml_top_k(ctx, softmax, ctx_data->k));
    ggml_set_name(top_k_ids, "top_k_ids");
    ggml_data->filtered_ids = top_k_ids;

    struct ggml_tensor * prob_rows = ggml_reshape_2d(ctx, softmax, 1, ggml_data->logits->ne[0]);
    struct ggml_tensor * top_k_rows = ggml_get_rows(ctx, prob_rows, top_k_ids);
    ggml_set_name(top_k_rows, "top_k_rows");

    struct ggml_tensor * top_k = ggml_reshape_1d(ctx, top_k_rows, ctx_data->k);
    struct ggml_tensor * total = ggml_sum(ctx, top_k);
    struct ggml_tensor * norm = ggml_div(ctx, top_k, ggml_repeat(ctx, total, top_k));
    ggml_data->probs = norm;
    ggml_build_forward_expand(gf, ggml_data->probs);
}

static const char * llama_sampler_gpu_top_p_name(const struct llama_sampler *) {
    return "gpu-top-p";
}

static void llama_sampler_gpu_top_p_free(struct llama_sampler * smpl) {
    auto * ctx_data = (llama_sampler_gpu_top_p_ctx *) smpl->ctx;
    delete ctx_data;
}

static struct llama_sampler * llama_sampler_gpu_top_p_clone(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_gpu_top_p_ctx *) smpl->ctx;
    return llama_sampler_gpu_init_top_p(ctx->k);
}

struct llama_sampler * llama_sampler_gpu_init_top_p(int32_t k) {
    static const llama_sampler_i iface = {
        /*.name        =*/ llama_sampler_gpu_top_p_name,
        /*.accept      =*/ nullptr,
        /*.apply       =*/ nullptr,
        /*.reset       =*/ nullptr,
        /*.clone       =*/ llama_sampler_gpu_top_p_clone,
        /*.free        =*/ llama_sampler_gpu_top_p_free,
        /*.apply_ggml  =*/ llama_sampler_gpu_top_p_apply_ggml,
        /*.accept_ggml =*/ nullptr,
    };

    auto * ctx_data = new llama_sampler_gpu_top_p_ctx {
        /*.k =*/ k,
    };

    auto * sampler = new llama_sampler {
        /*.iface =*/ &iface,
        /*.ctx   =*/ ctx_data,
    };

    return sampler;
}
