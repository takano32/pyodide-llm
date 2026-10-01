// T238: what Prism ML's fork of llama.cpp computes for Ternary Bonsai 2 27B on the CPU, written out for
// tests/reference_27b.py to compare the engine with. Only the public API of the fork's include/llama.h
// (https://github.com/PrismML-Eng/llama.cpp at 88c4bc60b9c9578f134385be9535e853f2db9b9f, MIT); no line of the fork
// is in this file. tests/reference_27b.sh builds the fork's libraries and this file against them.
//
//   reference_27b_fork <model.gguf> <out directory> <threads> <seconds> <tokens to write> <prompt> [<prompt> ...]
//
// For prompt i (parse_special: <|im_start|> and the like are their tokens; the GGUF says to add no BOS):
//   <out>/fork-<i>.ids      two lines of token ids: the prompt's as the fork tokenizes it, then the tokens it wrote
//                           (greedy: the largest logit, the first of equals; a token that ends a text does not stop it)
//   <out>/fork-<i>.logits   float32, a row of the vocabulary for every position of the prompt (decoded as one batch)
//                           and then for every written token but the last (decoded one at a time): row t is what
//                           follows token t
//   <out>/fork-<i>.single   float32, the prompt's rows again, decoded one token at a time (the fork's other path
//                           through the linear-attention layers)
// and on stdout the texts, the largest logits of every written position and the seconds of every decode. The
// writing stops early once <seconds> have passed (the files then hold what was written).
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    return n < 0 ? std::string("?") : std::string(buf, n);
}

// a text as one line: the newline and the backslash escaped
static std::string shown(const std::string & text) {
    std::string out;
    for (const char c : text) {
        if (c == '\n') out += "\\n";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

// decodes tokens at positions pos0, pos0 + 1, ... as one batch, asking for the logits of every one of them
static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, int pos0) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (status != 0) fprintf(stderr, "llama_decode failed: %d\n", status);
    return status == 0;
}

static llama_token largest(const float * logits, int n) {
    return (llama_token) (std::max_element(logits, logits + n) - logits);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 7) {
        fprintf(stderr, "usage: %s <model.gguf> <out> <threads> <seconds> <tokens> <prompt> ...\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], out = argv[2];
    const int threads = atoi(argv[3]), written = atoi(argv[5]);
    const double seconds = atof(argv[4]), began = now();

    ggml_backend_load_all();
    printf("fork: %s\n", llama_print_system_info());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "the fork could not load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    printf("fork: loaded in %.1f s, a vocabulary of %d, %d threads\n", now() - began, n_vocab, threads);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 256;
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    ctx_params.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "the fork could not make a context\n");
        return 1;
    }

    double one_at_a_time = 0.0;
    int decodes = 0;
    for (int p = 6; p < argc; p++) {
        const int index = p - 6;
        const std::string prompt = argv[p];
        const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> ids(n_prompt);
        if (n_prompt <= 0 || llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), ids.data(), n_prompt, true, true) < 0) {
            fprintf(stderr, "the fork could not tokenize prompt %d\n", index);
            return 1;
        }
        printf("fork: prompt %d: %s\nfork: prompt %d ids:", index, shown(prompt).c_str(), index);
        for (const llama_token id : ids) printf(" %d", id);
        printf("\nfork: prompt %d pieces:", index);
        for (const llama_token id : ids) printf(" [%s]", shown(piece(vocab, id)).c_str());
        printf("\n");

        const std::string base = out + "/fork-" + std::to_string(index);
        FILE * logits_file = fopen((base + ".logits").c_str(), "wb");
        FILE * single_file = fopen((base + ".single").c_str(), "wb");
        FILE * ids_file = fopen((base + ".ids").c_str(), "w");
        if (!logits_file || !single_file || !ids_file) {
            fprintf(stderr, "cannot write into %s\n", out.c_str());
            return 1;
        }
        for (size_t i = 0; i < ids.size(); i++) fprintf(ids_file, "%s%d", i ? " " : "", ids[i]);
        fprintf(ids_file, "\n");

        // the prompt one token at a time
        llama_memory_clear(llama_get_memory(ctx), true);
        double t0 = now();
        for (int t = 0; t < n_prompt; t++) {
            if (!decode(ctx, { ids[t] }, t)) return 1;
            fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), n_vocab, single_file);
        }
        const double single_seconds = now() - t0;
        one_at_a_time += single_seconds;
        decodes += n_prompt;
        fclose(single_file);

        // the prompt as one batch, then the tokens it writes
        llama_memory_clear(llama_get_memory(ctx), true);
        t0 = now();
        if (!decode(ctx, ids, 0)) return 1;
        const double batch_seconds = now() - t0;
        for (int t = 0; t < n_prompt; t++) fwrite(llama_get_logits_ith(ctx, t), sizeof(float), n_vocab, logits_file);
        printf("fork: prompt %d: %d tokens, one at a time %.2f s (%.3f tokens/s), as one batch %.2f s (%.3f tokens/s)\n",
               index, n_prompt, single_seconds, n_prompt / single_seconds, batch_seconds, n_prompt / batch_seconds);

        std::vector<float> row(llama_get_logits_ith(ctx, n_prompt - 1), llama_get_logits_ith(ctx, n_prompt - 1) + n_vocab);
        std::string text;
        std::vector<llama_token> wrote;
        for (int step = 0; step < written; step++) {
            const llama_token next = largest(row.data(), n_vocab);
            wrote.push_back(next);
            text += piece(vocab, next);
            // the five largest logits of this position
            std::vector<int> order(n_vocab);
            for (int i = 0; i < n_vocab; i++) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                              [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            printf("fork: prompt %d position %d:", index, n_prompt - 1 + step);
            for (int i = 0; i < 5; i++) printf(" %d [%s] %.4f", order[i], shown(piece(vocab, order[i])).c_str(), row[order[i]]);
            printf("\n");
            if (step == written - 1) break;
            if (now() - began > seconds) {
                printf("fork: out of time after %d tokens of prompt %d\n", step + 1, index);
                break;
            }
            t0 = now();
            if (!decode(ctx, { next }, n_prompt + step)) return 1;
            one_at_a_time += now() - t0;
            decodes += 1;
            const float * logits = llama_get_logits_ith(ctx, 0);
            fwrite(logits, sizeof(float), n_vocab, logits_file);
            row.assign(logits, logits + n_vocab);
        }
        for (size_t i = 0; i < wrote.size(); i++) fprintf(ids_file, "%s%d", i ? " " : "", wrote[i]);
        fprintf(ids_file, "\n");
        fclose(ids_file);
        fclose(logits_file);
        printf("fork: prompt %d wrote: %s\n", index, shown(text).c_str());
        fflush(stdout);
    }
    printf("fork: %d tokens one at a time in %.2f s: %.3f tokens/s on %d threads\n", decodes, one_at_a_time,
           decodes / one_at_a_time, threads);
    llama_perf_context_print(ctx);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
