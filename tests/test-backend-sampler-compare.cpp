#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#    define N_THREADS 1
#else
#    define N_THREADS std::thread::hardware_concurrency()
#endif

// Test result structure
struct test_result {
    std::string          sampler_name;
    const char *         backend_name;
    bool                 passed;
    std::string          error_message;
    double               max_prob_diff;
    double               max_logit_diff;
    int                  token_mismatch_count;
    std::tuple<int, int> sampled_tokens;
    std::tuple<int, int> sizes;

    test_result(ggml_backend_t dev) :
        backend_name(ggml_backend_dev_name(ggml_backend_get_device(dev))),
        passed(false),
        error_message(""),
        max_prob_diff(0.0),
        max_logit_diff(0.0),
        token_mismatch_count(0),
        sampled_tokens(std::make_tuple(0, 0)),
        sizes(std::make_tuple(0, 0)) {}
};

// Helper function to initialize token data array with random probabilities
static void init_token_data(std::vector<llama_token_data> & data, int n_vocab, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    data.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float logit   = dist(rng) * 10.0f - 5.0f;  // range [-5, 5]
        data[i].id    = i;
        data[i].logit = logit;
        data[i].p     = 0.0f;
    }
}

static llama_token_data_array infer_sampler_on_backend(ggml_backend_t                  backend,
                                                       llama_sampler *                 smpl,
                                                       int                             n_vocab,
                                                       const std::vector<float> &      logits_cpu,
                                                       std::vector<llama_token_data> & data_backend_out) {
    ggml_init_params params = {
        /*.mem_size   =*/128 * ggml_tensor_overhead() + ggml_graph_overhead() +
            n_vocab * ggml_type_size(GGML_TYPE_F32) * 10,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr ctx_ptr{ ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error("failed to create ggml context");
    }

    // Initialize backend for backend sampler
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (smpl->iface->backend_init) {
        smpl->iface->backend_init(smpl, buft);
    }

    ggml_context * ctx = ctx_ptr.get();

    llama_sampler_data data = {
        /*.logits     = */ ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab),
        /*.probs      = */ nullptr,
        /*.sampled    = */ nullptr,
        /*.candidates = */ nullptr,
    };

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * input_logits = data.logits;
    ggml_set_input(input_logits);

    smpl->iface->backend_apply(smpl, ctx, gf, &data);

    if (data.logits) {
        ggml_build_forward_expand(gf, data.logits);
        ggml_set_output(data.logits);
    }

    if (data.probs) {
        ggml_build_forward_expand(gf, data.probs);
        ggml_set_output(data.probs);
    }

    if (data.sampled) {
        ggml_build_forward_expand(gf, data.sampled);
        ggml_set_output(data.sampled);
    }

    if (data.candidates) {
        ggml_build_forward_expand(gf, data.candidates);
        ggml_set_output(data.candidates);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, gf);
    // printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    ggml_backend_buffer_t ctx_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(input_logits, logits_cpu.data(), 0, n_vocab * ggml_type_size(GGML_TYPE_F32));
    if (smpl->iface->backend_set_input) {
        smpl->iface->backend_set_input(smpl);
    }

    ggml_backend_graph_compute(backend, gf);

    std::vector<float>   logits_backend(data.logits ? ggml_nelements(data.logits) : 0);
    std::vector<float>   probs_backend(data.probs ? ggml_nelements(data.probs) : 0);
    std::vector<int32_t> candidates_backend(data.candidates ? ggml_nelements(data.candidates) : 0);
    int                  sampled_token_backend = -1;
    ggml_backend_tensor_get(data.logits, logits_backend.data(), 0, ggml_nbytes(data.logits));
    if (data.probs) {
        ggml_backend_tensor_get(data.probs, probs_backend.data(), 0, ggml_nbytes(data.probs));
    }
    if (data.candidates) {
        ggml_backend_tensor_get(data.candidates, candidates_backend.data(), 0, ggml_nbytes(data.candidates));
    }
    if (data.sampled) {
        ggml_backend_tensor_get(data.sampled, &sampled_token_backend, 0, ggml_nbytes(data.sampled));
    }

    ggml_graph_clear(gf);
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(ctx_buf);

    data_backend_out.clear();
    for (int i = 0; i < ggml_nelements(data.logits); i++) {
        if (logits_backend[i] > -1e9f) {
            // valid token
            llama_token_data td;
            td.logit = logits_backend[i];

            if (data.candidates) {
                td.id = (llama_token) candidates_backend[i];
            } else {
                td.id = i;
            }
            if (data.probs) {
                td.p = probs_backend[i];
            } else {
                td.p = 0.0f;
            }
            data_backend_out.push_back(td);
        }
    }
    bool  is_sorted = true;
    float max_logit = INFINITY;
    for (llama_token_data & td : data_backend_out) {
        if (td.logit > max_logit) {
            is_sorted = false;
            break;
        }
        max_logit = td.logit;
    }

    llama_token_data_array cur_p_backend = {
        data_backend_out.data(),
        data_backend_out.size(),
        sampled_token_backend,
        is_sorted,
    };
    return cur_p_backend;
}

// Helper to compare token data arrays
static bool compare_token_data(const llama_token_data_array & a,
                               const llama_token_data_array & b,
                               double &                       max_diff_p,
                               double &                       max_diff_l,
                               int &                          mismatch_count,
                               double                         tolerance    = 1e-5,
                               int                            size_tol     = 0,
                               int                            selected_tol = 0) {
    max_diff_p     = 0.0;
    max_diff_l     = 0.0;
    mismatch_count = 0;

    if (std::abs(static_cast<int>(a.size) - static_cast<int>(b.size)) > size_tol) {
        fprintf(stderr, "Size mismatch: %zu vs %zu\n", a.size, b.size);
        return false;
    }

    if (a.sorted != b.sorted) {
        fprintf(stderr, "Sorted flag mismatch: %d vs %d\n", a.sorted, b.sorted);
        return false;
    }

    if (a.selected != b.selected &&
        std::abs(static_cast<int>(a.selected) - static_cast<int>(b.selected)) > selected_tol) {
        fprintf(stderr, "Selected token mismatch: %ld vs %ld\n", a.selected, b.selected);
        return false;
    }

    for (size_t i = 0; i < std::min(a.size, b.size); i++) {
        // KNOWN_LIMIT: llama-cpp's sampler use std::partial_sort which is not stable,
        // so tokens with identical logits may appear in arbitrary/different order.
        if ((a.data[i].id != b.data[i].id) && (a.data[i].logit != b.data[i].logit)) {
            mismatch_count++;
        }

        double diff_p = std::abs(a.data[i].p - b.data[i].p);
        max_diff_p    = std::max(max_diff_p, diff_p);

        double diff_l = std::abs(a.data[i].logit - b.data[i].logit);
        max_diff_l    = std::max(max_diff_l, diff_l);
    }

    return mismatch_count == 0 && max_diff_p < tolerance && max_diff_l < tolerance;
}

// Test greedy sampler
static test_result test_greedy_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed) {
    test_result result(backend);
    result.sampler_name         = "greedy";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    // Initialize token data
    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    // Create samplers
    auto * sampler_cpu     = llama_sampler_init_greedy();
    auto * sampler_backend = llama_sampler_init_greedy();

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    // Create token data arrays
    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };

    // Apply samplers
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    // Compare results
    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count);

    if (!result.passed) {
        result.error_message = "Token data mismatch between CPU and backend";
        result.sampled_tokens =
            std::make_tuple(cur_p_cpu.data[cur_p_cpu.selected].id, cur_p_backend.data[cur_p_backend.selected].id);
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Test dist sampler
static test_result test_dist_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed) {
    test_result result(backend);
    result.sampler_name         = "dist";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    //auto sparams = llama_sampler_chain_default_params();
    //llama_sampler * smpl = llama_sampler_chain_init(sparams);

    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    // Use same seed for both samplers to get deterministic results
    auto * sampler_cpu     = llama_sampler_init_dist(seed);
    auto * sampler_backend = llama_sampler_init_dist(seed);

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    //KNOWN_LIMIT: Sometimes we are off-by-some (ggml thresholds softmax(logits) while llama.cpp thresholds exp(logits) based on sum(exp(logits)))
    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count, 1e-4);

    if (!result.passed) {
        result.error_message = "Token data mismatch between CPU and backend";
        result.sampled_tokens =
            std::make_tuple(cur_p_cpu.data[cur_p_cpu.selected].id, cur_p_backend.data[cur_p_backend.selected].id);
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Test top-k sampler
static test_result test_top_k_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed, int k) {
    test_result result(backend);
    result.sampler_name         = "top_k (k=" + std::to_string(k) + ")";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    auto * sampler_cpu     = llama_sampler_init_top_k(k);
    auto * sampler_backend = llama_sampler_init_top_k(k);

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    //KNOWN_LIMIT: sort top-k as llama-sampler does it that way
    std::stable_sort(cur_p_backend.data, cur_p_backend.data + cur_p_backend.size,
                     [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
    cur_p_backend.sorted = true;

    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count);

    if (!result.passed) {
        result.error_message = "Token data mismatch between CPU and backend";
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Test top-p sampler
static test_result test_top_p_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed, float p, size_t min_keep) {
    test_result result(backend);
    result.sampler_name         = "top_p (p=" + std::to_string(p) + ")";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    if (min_keep != 0) {
        throw std::runtime_error("min_keep not supported in backend sampler tests");
    }

    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    auto * sampler_cpu     = llama_sampler_init_top_p(p, min_keep);
    auto * sampler_backend = llama_sampler_init_top_p(p, min_keep);

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    //KNOWN_LIMIT: accept differences in sizes due to potential different order of FP-arithmetic for divisor-computation in softmax between llama.cpp and ggml backends
    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count, 1e-4, std::max(1, (int) std::ceil(n_vocab * 0.0006)));

    if (!result.passed) {
        result.error_message = "Token data mismatch between llama-sampler and backend";
        result.sizes = std::make_tuple(cur_p_cpu.size, cur_p_backend.size);
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Test min-p sampler
static test_result test_min_p_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed, float p, size_t min_keep) {
    test_result result(backend);
    result.sampler_name         = "min_p (p=" + std::to_string(p) + ")";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    if (min_keep != 0) {
        throw std::runtime_error("min_keep not supported in backend sampler tests");
    }

    auto * sampler_cpu     = llama_sampler_init_min_p(p, min_keep);
    auto * sampler_backend = llama_sampler_init_min_p(p, min_keep);

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count, 1e-4);

    if (!result.passed) {
        result.error_message = "Token data mismatch between CPU and backend";
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Test temp sampler
static test_result test_temp_sampler(ggml_backend_t backend, int n_vocab, uint32_t seed, float temp) {
    test_result result(backend);
    result.sampler_name         = "temp (temp=" + std::to_string(temp) + ")";
    result.passed               = false;
    result.max_prob_diff        = 0.0;
    result.max_logit_diff       = 0.0;
    result.token_mismatch_count = 0;

    std::vector<llama_token_data> data_cpu = {};
    init_token_data(data_cpu, n_vocab, seed);
    std::vector<float> logits_cpu = {};
    logits_cpu.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_cpu[i] = data_cpu[i].logit;
    }

    auto * sampler_cpu     = llama_sampler_init_temp(temp);
    auto * sampler_backend = llama_sampler_init_temp(temp);

    std::vector<llama_token_data> data_backend;
    llama_token_data_array        cur_p_backend =
        infer_sampler_on_backend(backend, sampler_backend, n_vocab, logits_cpu, data_backend);

    llama_token_data_array cur_p_cpu = {
        data_cpu.data(),
        data_cpu.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler_cpu, &cur_p_cpu);

    result.passed = compare_token_data(cur_p_cpu, cur_p_backend, result.max_prob_diff, result.max_logit_diff,
                                       result.token_mismatch_count, 1e-4);

    if (!result.passed) {
        result.error_message = "Token data mismatch between CPU and backend";
    }

    llama_sampler_free(sampler_cpu);
    llama_sampler_free(sampler_backend);

    return result;
}

// Print test results
static void print_results(const std::vector<test_result> & results) {
    printf("\n");
    printf("============================================\n");
    printf("Backend Sampler Comparison Test Results for Backend: %s\n",
           results.empty() ? "Unknown" : results[0].backend_name);
    printf("============================================\n\n");

    int passed = 0;
    int failed = 0;

    for (const auto & result : results) {
        const char * status = result.passed ? "PASS" : "FAIL";
        printf("[%s] %s\n", status, result.sampler_name.c_str());

        if (!result.passed) {
            printf("  Error: %s\n", result.error_message.c_str());
            printf("  Max prob diff: %e\n", result.max_prob_diff);
            printf("  Max logit diff: %e\n", result.max_logit_diff);
            printf("  Token mismatches: %d\n", result.token_mismatch_count);
            printf("  Selected tokens: %d, %d\n", std::get<0>(result.sampled_tokens),
                   std::get<1>(result.sampled_tokens));
            printf("  Sizes: %d, %d\n", std::get<0>(result.sizes),
                   std::get<1>(result.sizes));
            failed++;
        } else {
            printf("  Max prob diff: %e\n", result.max_prob_diff);
            printf("  Max logit diff: %e\n", result.max_logit_diff);
            passed++;
        }
        printf("\n");
    }

    printf("============================================\n");
    printf("Summary: %d/%d tests passed\n", passed, (int) results.size());
    printf("============================================\n");
}

static bool test_backend_sampler_compare(ggml_backend_t backend, int n_vocab, uint32_t seed) {
    std::vector<test_result> results;

    ggml_backend_dev_t dev          = ggml_backend_get_device(backend);
    const char *       backend_name = ggml_backend_dev_name(dev);

    printf("\n============================================\n");
    printf("Testing backend: %s\n", backend_name);
    printf("============================================\n");
    printf("Vocabulary size: %d\n", n_vocab);
    printf("Random seed: %u\n\n", seed);

    // Test greedy sampler
    printf("Testing greedy sampler...\n");
    results.push_back(test_greedy_sampler(backend, n_vocab, seed));

    // Test dist sampler
    printf("Testing dist sampler...\n");
    results.push_back(test_dist_sampler(backend, n_vocab, seed));

    // Test top-k sampler with different k values
    printf("Testing top-k sampler...\n");
    results.push_back(test_top_k_sampler(backend, n_vocab, seed, 10));
    results.push_back(test_top_k_sampler(backend, n_vocab, seed, 40));
    results.push_back(test_top_k_sampler(backend, n_vocab, seed, 100));

    // Test top-p sampler with different p values
    printf("Testing top-p sampler...\n");
    results.push_back(test_top_p_sampler(backend, n_vocab, seed, 0.5f, 0));
    results.push_back(test_top_p_sampler(backend, n_vocab, seed, 0.9f, 0));
    results.push_back(test_top_p_sampler(backend, n_vocab, seed, 0.95f, 0));

    // Test min-p sampler with different p values
    printf("Testing min-p sampler...\n");
    results.push_back(test_min_p_sampler(backend, n_vocab, seed, 0.05f, 0));
    results.push_back(test_min_p_sampler(backend, n_vocab, seed, 0.1f, 0));

    // Test temp sampler with different temperatures
    printf("Testing temp sampler...\n");
    results.push_back(test_temp_sampler(backend, n_vocab, seed, 0.8f));
    results.push_back(test_temp_sampler(backend, n_vocab, seed, 1.0f));
    results.push_back(test_temp_sampler(backend, n_vocab, seed, 1.5f));

    // Print results
    print_results(results);

    // Return exit code
    int failed_count = 0;
    for (const auto & result : results) {
        if (!result.passed) {
            failed_count++;
        }
    }

    return failed_count == 0;
}

int main(int argc, char ** argv) {
    // Default parameters
    int      n_vocab = 100;                         // Default vocabulary size
    uint32_t seed    = std::random_device()();        // Random seed by default

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0) {
            if (i + 1 < argc) {
                n_vocab = atoi(argv[++i]);
                if (n_vocab <= 0) {
                    fprintf(stderr, "Error: vocabulary size must be positive\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --vocab requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = (uint32_t)atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: --seed requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -v, --vocab <size>   Set vocabulary size (default: %d)\n", 32000);
            printf("  -s, --seed <seed>    Set random seed (default: random)\n");
            printf("  -h, --help           Show this help message\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage information\n");
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    bool all_passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev     = ggml_backend_dev_get(i);
        ggml_backend_t     backend = ggml_backend_dev_init(dev, NULL);
        if (backend == NULL) {
            fprintf(stderr, "Failed to initialize backend %zu\n", i);
            all_passed = false;
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto               ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend, N_THREADS);
        }

        bool passed = test_backend_sampler_compare(backend, n_vocab, seed);
        all_passed  = all_passed && passed;

        ggml_backend_free(backend);
    }

    return all_passed ? 0 : 1;
}
