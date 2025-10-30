#include "ggml.h"
#include "llama.h"
#include "llama-gpu-sampling.h"
#include "get-model.h"
#include "common.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct test_model_context {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    std::vector<llama_sampler_seq_config> sampler_configs;
    std::unordered_map<llama_seq_id, int32_t> seq_positions;
    std::unordered_map<llama_seq_id, int32_t> last_batch_info;

    bool setup(const char * model_path, const std::vector<llama_sampler_seq_config> & configs) {
        sampler_configs = configs;
        if (model != nullptr && ctx != nullptr) {
            return true;
        }

        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        model = llama_model_load_from_file(model_path, mparams);
        if (model == nullptr) {
            fprintf(stderr, "Warning: failed to load model '%s', skipping test\n", model_path);
            cleanup();
            return false;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        cparams.samplers = sampler_configs.data();
        cparams.n_samplers = sampler_configs.size();

        int32_t max_seq_id = 0;
        for (const auto & config : sampler_configs) {
            if (config.seq_id > max_seq_id) {
                max_seq_id = config.seq_id;
            }
        }
        cparams.n_seq_max = max_seq_id + 1;

        ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "Warning: failed to create context, skipping test\n");
            cleanup();
            return false;
        }
        llama_set_warmup(ctx, false);

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        fprintf(stderr, "Vocabulary size: %d\n", n_vocab);

        return true;
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        if (ctx == nullptr || vocab == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        last_batch_info.clear();
        llama_batch batch = llama_batch_init(512, 0, prompts.size());

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                llama_batch_free(batch);
                return false;
            }

            for (int i = 0; i < n_tokens; i++) {
                tokens.push_back(prompt_tokens[i]);
            }

            for (size_t i = 0; i < tokens.size(); i++) {
                common_batch_add(batch, tokens[i], i, { seq_id }, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = tokens.size();
        }


        printf("Batch contents:\n");
        printf("  n_tokens: %d\n", batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; i++) {
            printf("  token[%d]: tok=%-5d, pos=%d, n_seq_id=%d, seq_ids=[", i, batch.token[i], batch.pos[i], batch.n_seq_id[i]);

        for (int j = 0; j < batch.n_seq_id[i]; j++) {
            printf("%d%s", batch.seq_id[i][j], j < batch.n_seq_id[i]-1 ? ", " : "");
        }
        printf("], logits=%d\n", batch.logits[i]);
}

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            llama_batch_free(batch);
            return false;
        }

        // Build mapping from seq id to batch token idx
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id seq_id = batch.seq_id[i][0];
                last_batch_info[seq_id] = i;
                printf("seq %d : batch idx %d\n", seq_id, i);
            }
        }

        llama_batch_free(batch);
        return true;
    }

    int32_t idx_for_seq(llama_seq_id seq_id) {
        auto it = last_batch_info.find(seq_id);
        if (it == last_batch_info.end()) {
            fprintf(stderr, "Error: no batch index found for seq_id %d\n", seq_id);
            return -1;
        }
        return it->second;
    }

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        if (ctx == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        int32_t pos = seq_positions[seq_id];
        common_batch_add(batch, token, pos, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            llama_batch_free(batch);
            return false;
        }

        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }

        seq_positions[seq_id]++;
        llama_batch_free(batch);
        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        if (ctx == nullptr) {
            fprintf(stderr, "Error: context not initialized, call setup() first\n");
            return false;
        }

        llama_batch batch = llama_batch_init(seq_tokens.size(), 0, seq_tokens.size());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            common_batch_add(batch, token, pos, { seq_id }, true);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            llama_batch_free(batch);
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        last_batch_info.clear();
        for (int i = 0; i < batch.n_tokens; i++) {
            if (batch.logits[i]) {
                llama_seq_id cur_seq = batch.seq_id[i][0];
                last_batch_info[cur_seq] = i;
            }
        }

        llama_batch_free(batch);
        return true;
    }

    std::string token_to_piece(llama_token token, bool special) {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        }
        else {
            piece.resize(n_chars);
        }

        return piece;
    }

    void cleanup() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        ctx = nullptr;
        model = nullptr;
        vocab = nullptr;
    }

    ~test_model_context() {
        cleanup();
    }
};

static void test_gpu_greedy_sampling(const char * model_path) {
    test_model_context test_ctx;

    struct llama_sampler_chain_params gpu_sampler_params = llama_sampler_chain_default_params();
    struct llama_sampler * gpu_sampler_chain = llama_sampler_chain_init(gpu_sampler_params);
    GGML_ASSERT(gpu_sampler_chain->iface->apply_ggml != nullptr);

    llama_sampler_chain_add(gpu_sampler_chain, llama_sampler_gpu_init_greedy());
    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {{ 0, gpu_sampler_chain }};

    if (!test_ctx.setup(model_path, gpu_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Some"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(0);
    GGML_ASSERT(batch_idx >= 0);
    llama_token id = llama_get_sampled_token_ith(test_ctx.ctx, batch_idx);
    GGML_ASSERT(id >= 0 && id < test_ctx.n_vocab);
    printf("gpu greedy sampled id:%d, string: %s\n", id, test_ctx.token_to_piece(id, false).c_str());

    for (int i = 0; i < 1; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(0);
        GGML_ASSERT(loop_idx >= 0);
        llama_token id = llama_get_sampled_token_ith(test_ctx.ctx, loop_idx);
        GGML_ASSERT(id >= 0 && id < test_ctx.n_vocab);
        printf("Generation step %d: token id:%d, string: %s\n", i, id, test_ctx.token_to_piece(id, false).c_str());
        test_ctx.decode_token(id, 0);
    }
}

static void test_gpu_top_k_sampling(const char * model_path) {
    test_model_context test_ctx;

    const int32_t k = 8;
    struct llama_sampler_chain_params gpu_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * gpu_sampler_chain = llama_sampler_chain_init(gpu_chain_params);
    llama_sampler_chain_add(gpu_sampler_chain, llama_sampler_gpu_init_top_k(k));
    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {{ 0, gpu_sampler_chain }};

    if (!test_ctx.setup(model_path, gpu_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Hello"}})) {
        return;
    }

    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    GGML_ASSERT(chain->iface->apply_ggml != nullptr);

    llama_sampler_chain_add(chain, llama_sampler_init_dist(18));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

    llama_token id = llama_sampler_sample(chain, test_ctx.ctx, -1);

    // The sampler should select a valid token from the vocabulary
    GGML_ASSERT(id >= 0 && id < test_ctx.n_vocab);

    printf("GPU top-k sampling test PASSED\n");
    printf("Top-k GPU sampled token id:%d, string: %s\n", id, test_ctx.token_to_piece(id, false).c_str());

    llama_sampler_free(chain);
}

static void test_gpu_temp_sampling(const char * model_path) {
    test_model_context test_ctx;

    const float temp_0 = 0.8f;
    struct llama_sampler_chain_params gpu_chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * gpu_sampler_chain_0 = llama_sampler_chain_init(gpu_chain_params_0);
    llama_sampler_chain_add(gpu_sampler_chain_0, llama_sampler_gpu_init_temp(temp_0));

    const float temp_1 = 0.1f;
    struct llama_sampler_chain_params gpu_chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * gpu_sampler_chain_1 = llama_sampler_chain_init(gpu_chain_params_1);
    llama_sampler_chain_add(gpu_sampler_chain_1, llama_sampler_gpu_init_temp(temp_1));

    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {
        { 0, gpu_sampler_chain_0 },
        { 1, gpu_sampler_chain_1 }
    };

    if (!test_ctx.setup(model_path, gpu_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Some"}, {1, "Hello"}})) {
        return;
    }

    int32_t batch_idx_0 = test_ctx.idx_for_seq(0);
    GGML_ASSERT(batch_idx_0 >= 0);
    float * logits_0 = llama_get_logits_ith(test_ctx.ctx, batch_idx_0);
    printf("Sequence 0 (temp=%.2f) first 10 logits:\n", temp_0);
    for (int i = 0; i < 10; i++) {
        printf("  logit[%d] = %.6f\n", i, logits_0[i]);
    }

    int32_t batch_idx_1 = test_ctx.idx_for_seq(1);
    GGML_ASSERT(batch_idx_1 >= 0);
    float * logits_1 = llama_get_logits_ith(test_ctx.ctx, batch_idx_1);
    printf("\nSequence 1 (temp=%.2f) first 10 logits:\n", temp_1);
    for (int i = 0; i < 10; i++) {
        printf("  logit[%d] = %.6f\n", i, logits_1[i]);
    }

    // Sample from sequence 0 using CPU sampler
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * chain_0 = llama_sampler_chain_init(chain_params_0);
    llama_sampler_chain_add(chain_0, llama_sampler_init_dist(18));

    llama_token id_0 = llama_sampler_sample(chain_0, test_ctx.ctx, batch_idx_0);
    printf("\nSequence 0 sampled token id:%d, string: '%s'\n", id_0, test_ctx.token_to_piece(id_0, false).c_str());

    // Sample from sequence 1 using CPU sampler
    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * chain_1 = llama_sampler_chain_init(chain_params_1);
    llama_sampler_chain_add(chain_1, llama_sampler_init_dist(18));

    llama_token id_1 = llama_sampler_sample(chain_1, test_ctx.ctx, batch_idx_1);
    printf("Sequence 1 sampled token id:%d, string: '%s'\n", id_1, test_ctx.token_to_piece(id_1, false).c_str());

    printf("GPU temp sampling test PASSED\n");

    llama_sampler_free(chain_0);
    llama_sampler_free(chain_1);
}

static void test_gpu_softmax_sampling(const char * model_path) {
    test_model_context test_ctx;

    struct llama_sampler_chain_params gpu_chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * gpu_sampler_chain = llama_sampler_chain_init(gpu_chain_params);

    llama_sampler_chain_add(gpu_sampler_chain, llama_sampler_gpu_init_softmax());
    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {{ 0, gpu_sampler_chain }};

    if (!test_ctx.setup(model_path, gpu_sampler_configs)) {
        return;
    }

    if (!test_ctx.decode({{0, "Hello"}})) {
        return;
    }

    int32_t idx = test_ctx.idx_for_seq(0);
    GGML_ASSERT(idx >= 0);
    float * probs = llama_get_sampled_probs_ith(test_ctx.ctx, idx);
    GGML_ASSERT(probs != nullptr);

    float * logits = llama_get_logits_ith(test_ctx.ctx, idx);
    GGML_ASSERT(logits == nullptr);

    float sum = 0.0f;
    for (int i = 0; i < test_ctx.n_vocab; i++) {
        sum += probs[i];
    }
    printf("probs sum = %.6f\n", sum);

    //TODO: Enable the following verification code once normal (non-gpu) samplers
    // can handle probabilites directly without needing access to the logits.
    // Currently all the sampler implementations will calculate the probabilites
    // using softmax from logits, but in the case where the probabilities have
    // already been computed (as is the case with GPU softmax sampler), is should
    // be possible for samplers to check if the probabilities are available and
    // use them directly.
    /*
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9f, 2));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(25));

    llama_token id = llama_sampler_sample(chain, test_ctx.ctx, idx);
    printf("Sampled token id:%d, string: '%s'\n", id, test_ctx.token_to_piece(id, false).c_str());
    */

    printf("GPU softmax sampling test PASSED\n");
}

static void test_gpu_multi_sequence_sampling(const char * model_path) {
    test_model_context test_ctx;

    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    struct llama_sampler * sampler_chain_0 = llama_sampler_chain_init(chain_params_0);
    llama_sampler_chain_add(sampler_chain_0, llama_sampler_gpu_init_greedy());

    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    struct llama_sampler * sampler_chain_1 = llama_sampler_chain_init(chain_params_1);
    llama_sampler_chain_add(sampler_chain_1, llama_sampler_gpu_init_temp(0.8f));
    llama_sampler_chain_add(sampler_chain_1, llama_sampler_gpu_init_greedy());

    std::vector<llama_sampler_seq_config> gpu_sampler_configs = {
        { 0, sampler_chain_0 },
        { 1, sampler_chain_1 }
    };

    if (!test_ctx.setup(model_path, gpu_sampler_configs)) {
        return;
    }

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        return;
    }

    int32_t batch_idx_0 = test_ctx.idx_for_seq(0);
    GGML_ASSERT(batch_idx_0 >= 0);
    llama_token seq0_token = llama_get_sampled_token_ith(test_ctx.ctx, batch_idx_0);
    printf("Seq 0 sampled token id=%d, string='%s'\n",
           seq0_token, test_ctx.token_to_piece(seq0_token, false).c_str());

    int32_t batch_idx_1 = test_ctx.idx_for_seq(1);
    GGML_ASSERT(batch_idx_1 >= 0);
    llama_token seq1_token = llama_get_sampled_token_ith(test_ctx.ctx, batch_idx_1);
    printf("Seq 1 sampled token id=%d, string='%s'\n",
           seq1_token, test_ctx.token_to_piece(seq1_token, false).c_str());

    // Generate tokens for each sequence
    printf("\nMulti-sequence generation:\n");
    for (int step = 0; step < 4; step++) {
        std::map<llama_seq_id, llama_token> tokens;

        for (llama_seq_id seq_id : {0, 1}) {
            int32_t idx = test_ctx.idx_for_seq(seq_id);
            GGML_ASSERT(idx >= 0);
            llama_token token = llama_get_sampled_token_ith(test_ctx.ctx, idx);
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
            tokens[seq_id] = token;
            printf("  Seq %d, step %d: token id=%d, string='%s'\n", seq_id, step, token, test_ctx.token_to_piece(token, false).c_str());
        }

        // Decode all tokens in a single batch
        if (!test_ctx.decode_tokens(tokens)) {
            break;
        }
    }

    printf("GPU multi-sequence sampling test PASSED\n");
}

struct gpu_test_case {
    const char * name;
    void (*fn)(const char *);
    bool enabled_by_default;
};

static const gpu_test_case GPU_TESTS[] = {
    { "gpu_greedy",          test_gpu_greedy_sampling,         true  },
    { "gpu_temp",            test_gpu_temp_sampling,           true  },
    { "gpu_softmax",         test_gpu_softmax_sampling,        true  },
    { "gpu_top_k",           test_gpu_top_k_sampling,          false },
    { "gpu_multi_sequence",  test_gpu_multi_sequence_sampling, true  },
};

struct gpu_cli_args {
    const char * model = nullptr;
    const char * test = nullptr;
};

static gpu_cli_args parse_gpu_cli(int argc, char ** argv) {
    gpu_cli_args out;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--test") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--test expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.test = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--test=", 7) == 0) {
            out.test = arg + 7;
            continue;
        }
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.model = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--model=", 8) == 0) {
            out.model = arg + 8;
            continue;
        }
        if (!out.model) {
            out.model = arg;
            continue;
        }

        fprintf(stderr, "Unexpected argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    return out;
}

static std::vector<const gpu_test_case *> collect_tests_to_run(const char * requested) {
    std::vector<const gpu_test_case *> selected;

    if (requested != nullptr) {
        for (const auto & test : GPU_TESTS) {
            if (std::strcmp(test.name, requested) == 0) {
                selected.push_back(&test);
                break;
            }
        }
        if (selected.empty()) {
            fprintf(stderr, "Unknown test '%s'. Available tests:\n", requested);
            for (const auto & test : GPU_TESTS) {
                fprintf(stderr, "  %s\n", test.name);
            }
            exit(EXIT_FAILURE);
        }
    } else {
        for (const auto & test : GPU_TESTS) {
            if (test.enabled_by_default) {
                selected.push_back(&test);
            }
        }
    }

    if (selected.empty()) {
        fprintf(stderr, "No GPU sampling tests selected. Use --test=<name> to pick one.\n");
    }

    return selected;
}

static void run_tests(const std::vector<const gpu_test_case *> & tests, const char * model_path) {
    for (const auto * test : tests) {
        fprintf(stderr, "\n=== %s ===\n", test->name);
        test->fn(model_path);
    }
}


int main(int argc, char *argv[] ) {
    const gpu_cli_args args = parse_gpu_cli(argc, argv);

    std::array<char *, 2> model_argv { argv[0], const_cast<char *>(args.model) };
    const int model_argc = args.model ? 2 : 1;
    char * model_path = get_model_or_exit(model_argc, model_argv.data());

    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    ggml_time_init();

    const std::vector<const gpu_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, model_path);
    }

    return 0;
}
