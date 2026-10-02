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
//
// T237's review, three switches (the environment), for a second run beside the first that computes the model with the
// ternary matrices multiplied by float32 activations, where the fork rounds them to Q8_0 first (tests/reference_27b_patch.py
// adds the float32 path to the fork: PTQ1_0_F32_ACTIVATIONS=1 asks for it):
//   KV_F32=1            the keys and values of the attention layers are float32 (the fork's own default is float16)
//   REPLAY_FROM=<dir>   decode the tokens the first run wrote, not what this one would choose: for prompt i the ids of
//                       <dir>/fork-<i>.ids (its prompt's, then what it wrote but the last), one token at a time, into
//                       <out>/fork-<i>.single (a row for each); the prompt arguments only count the prompts
//   REPLAY_BATCH=1      and the same tokens as one batch, into <out>/fork-<i>.logits
//
// T233's review, a fourth switch (the environment), for a prompt past the 4096 positions the list's context has: what the
// fork computes on a long text, with the logits of some positions only (a row is 1 MB: all of 6,000 would be 6 GB):
//   LONG_TEXT=<a text file>   decode its tokens (the prompt arguments only count: one is needed, and is not used), in batches
//                             of n_batch, asking for the logits of the positions of LONG_ROWS and of the last only; then
//                             the tokens it writes greedily, one at a time as above. parse_special is off: a text of this
//                             repository's documents has no "<|im_start|>", and a prompt of the page is cut the same way
//   LONG_ROWS=<positions>     comma separated
//   LONG_CTX=<positions>      the context (512 when it is not said)
//   LONG_TOKENS=<n>           only the first n tokens of the text
//   <out>/fork-long.ids       two lines of ids as fork-<i>.ids: the prompt's, the tokens written
//   <out>/fork-long.rows      the positions of the prompt whose logits are kept, in order, on one line
//   <out>/fork-long.logits    float32: a row of the vocabulary for each of those positions, then for each token written but
//                             the last (row t is what follows token t, as for the other texts)
// A run that runs out of <seconds> in the middle of the prompt keeps the rows it has and writes nothing after them.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

// the two lines of a fork-<i>.ids: the prompt's ids, then the ids it wrote
static bool read_ids(const std::string & path, std::vector<llama_token> & prompt, std::vector<llama_token> & wrote) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) return false;
    {
        std::istringstream words(line);
        llama_token id;
        while (words >> id) prompt.push_back(id);
    }
    if (std::getline(in, line)) {
        std::istringstream words(line);
        llama_token id;
        while (words >> id) wrote.push_back(id);
    }
    return !prompt.empty();
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream all;
    all << in.rdbuf();
    return all.str();
}

// a batch of tokens at positions pos0, pos0 + 1, ..., asking for the logits of those with keep[i] set (the batch index
// is what llama_get_logits_ith takes then)
static bool decode_rows(llama_context * ctx, const std::vector<llama_token> & tokens, int pos0, const std::vector<char> & keep) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = keep[i] ? 1 : 0;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (status != 0) fprintf(stderr, "llama_decode failed: %d\n", status);
    return status == 0;
}

// T233's review: the long prompt (the switches LONG_* above)
static int run_long(const llama_vocab * vocab, llama_context * ctx, int n_vocab, int n_batch, const std::string & out, int written,
                    double began, double seconds) {
    const std::string text = read_file(getenv("LONG_TEXT"));
    const int all = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), NULL, 0, false, false);
    std::vector<llama_token> ids(all > 0 ? all : 0);
    if (all <= 0 || llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), all, false, false) < 0) {
        fprintf(stderr, "the fork could not tokenize %s\n", getenv("LONG_TEXT"));
        return 1;
    }
    if (getenv("LONG_TOKENS") != NULL) ids.resize(std::min<size_t>(ids.size(), (size_t) atoi(getenv("LONG_TOKENS"))));
    const int n_prompt = (int) ids.size();
    std::vector<char> keep(n_prompt, 0);
    if (getenv("LONG_ROWS") != NULL) {
        std::istringstream list(getenv("LONG_ROWS"));
        std::string item;
        while (std::getline(list, item, ',')) {
            const int position = atoi(item.c_str());
            if (position >= 0 && position < n_prompt) keep[position] = 1;
        }
    }
    keep[n_prompt - 1] = 1;
    printf("fork: long: %d tokens of %s, the logits of %d positions kept, a context of %d\n", n_prompt, getenv("LONG_TEXT"),
           (int) std::count(keep.begin(), keep.end(), 1), (int) llama_n_ctx(ctx));
    const std::string base = out + "/fork-long";
    FILE * logits_file = fopen((base + ".logits").c_str(), "wb");
    FILE * ids_file = fopen((base + ".ids").c_str(), "w");
    if (!logits_file || !ids_file) {
        fprintf(stderr, "cannot write into %s\n", out.c_str());
        return 1;
    }
    for (int i = 0; i < n_prompt; i++) fprintf(ids_file, "%s%d", i ? " " : "", ids[i]);
    fprintf(ids_file, "\n");
    fflush(ids_file);

    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<int> rows;
    std::vector<float> row;
    int done = 0;
    const double t0 = now();
    double previous = t0;
    for (int at = 0; at < n_prompt; at += n_batch) {
        const int n = std::min(n_batch, n_prompt - at);
        const std::vector<llama_token> chunk(ids.begin() + at, ids.begin() + at + n);
        const std::vector<char> flags(keep.begin() + at, keep.begin() + at + n);
        if (!decode_rows(ctx, chunk, at, flags)) return 1;
        for (int i = 0; i < n; i++) {
            if (!flags[i]) continue;
            const float * logits = llama_get_logits_ith(ctx, i);
            fwrite(logits, sizeof(float), n_vocab, logits_file);
            rows.push_back(at + i);
            if (at + i == n_prompt - 1) row.assign(logits, logits + n_vocab);
        }
        done = at + n;
        const double t = now();
        printf("fork: long: %d of %d tokens in %.0f s (%.3f tokens/s in the last %d)\n", done, n_prompt, t - t0, n / (t - previous), n);
        fflush(stdout);
        previous = t;
        if (t - began > seconds && done < n_prompt) {
            printf("fork: out of time after %d of the %d tokens of the long text\n", done, n_prompt);
            break;
        }
    }
    FILE * rows_file = fopen((base + ".rows").c_str(), "w");
    if (!rows_file) {
        fprintf(stderr, "cannot write into %s\n", out.c_str());
        return 1;
    }
    for (size_t i = 0; i < rows.size(); i++) fprintf(rows_file, "%s%d", i ? " " : "", rows[i]);
    fprintf(rows_file, "\n");
    fclose(rows_file);
    printf("fork: long: the prompt in %.0f s (%.3f tokens/s), the logits of %zu positions\n", now() - t0, done / (now() - t0), rows.size());

    // the tokens it writes, one at a time (greedy, the first of equals), if the prompt was decoded whole
    std::vector<llama_token> wrote;
    std::string written_text;
    if (done == n_prompt) {
        for (int step = 0; step < written; step++) {
            const llama_token next = largest(row.data(), n_vocab);
            wrote.push_back(next);
            written_text += piece(vocab, next);
            std::vector<int> order(n_vocab);
            for (int i = 0; i < n_vocab; i++) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                              [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            printf("fork: long position %d:", n_prompt - 1 + step);
            for (int i = 0; i < 5; i++) printf(" %d [%s] %.4f", order[i], shown(piece(vocab, order[i])).c_str(), row[order[i]]);
            printf("\n");
            if (step == written - 1) break;
            if (now() - began > seconds) {
                printf("fork: out of time after %d tokens written\n", step + 1);
                break;
            }
            const double t = now();
            if (!decode(ctx, { next }, n_prompt + step)) return 1;
            const float * logits = llama_get_logits_ith(ctx, 0);
            fwrite(logits, sizeof(float), n_vocab, logits_file);
            row.assign(logits, logits + n_vocab);
            printf("fork: long: one token in %.2f s\n", now() - t);
        }
    }
    for (size_t i = 0; i < wrote.size(); i++) fprintf(ids_file, "%s%d", i ? " " : "", wrote[i]);
    fprintf(ids_file, "\n");
    fclose(ids_file);
    fclose(logits_file);
    printf("fork: long: wrote: %s\n", shown(written_text).c_str());
    fflush(stdout);
    return 0;
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
    ctx_params.n_ctx = getenv("LONG_CTX") != NULL ? atoi(getenv("LONG_CTX")) : 512;  // (T233's review: a long text has its own)
    ctx_params.n_batch = 256;
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;
    ctx_params.no_perf = false;
    const bool kv_f32 = getenv("KV_F32") != NULL;
    if (kv_f32) {
        ctx_params.type_k = GGML_TYPE_F32;
        ctx_params.type_v = GGML_TYPE_F32;
    }
    const char * replay_dir = getenv("REPLAY_FROM");
    const bool replay_batch = getenv("REPLAY_BATCH") != NULL;
    printf("fork: keys and values %s, ptq1_0 activations %s%s\n", kv_f32 ? "float32" : "float16 (the fork's default)",
           getenv("PTQ1_0_F32_ACTIVATIONS") != NULL ? "float32" : "Q8_0 (the fork's own)",
           replay_dir ? ", replaying the ids of the first run" : "");
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "the fork could not make a context\n");
        return 1;
    }

    if (getenv("LONG_TEXT") != NULL) {
        const int status = run_long(vocab, ctx, n_vocab, (int) ctx_params.n_batch, out, written, began, seconds);
        llama_perf_context_print(ctx);
        llama_free(ctx);
        llama_model_free(model);
        return status;
    }

    double one_at_a_time = 0.0;
    int decodes = 0;
    for (int p = 6; p < argc; p++) {
        const int index = p - 6;
        if (replay_dir) {
            std::vector<llama_token> first, wrote;
            const std::string from = std::string(replay_dir) + "/fork-" + std::to_string(index) + ".ids";
            if (!read_ids(from, first, wrote)) {
                fprintf(stderr, "no ids to replay in %s\n", from.c_str());
                return 1;
            }
            // what the first run decoded: the prompt, then every token it wrote but the last
            std::vector<llama_token> sequence = first;
            if (!wrote.empty()) sequence.insert(sequence.end(), wrote.begin(), wrote.end() - 1);
            const std::string base = out + "/fork-" + std::to_string(index);
            FILE * single_file = fopen((base + ".single").c_str(), "wb");
            if (!single_file) {
                fprintf(stderr, "cannot write into %s\n", out.c_str());
                return 1;
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            double t0 = now();
            size_t done = 0;
            for (size_t t = 0; t < sequence.size(); t++) {
                if (now() - began > seconds) {
                    printf("fork: out of time after %zu of the %zu tokens of text %d\n", done, sequence.size(), index);
                    break;
                }
                if (!decode(ctx, { sequence[t] }, (int) t)) return 1;
                fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), n_vocab, single_file);
                done++;
                if (t % 8 == 7) {
                    printf("fork: replay %d: %zu tokens, %.2f s\n", index, t + 1, now() - t0);
                    fflush(stdout);
                }
            }
            fclose(single_file);
            const double replay_seconds = now() - t0;
            one_at_a_time += replay_seconds;
            decodes += (int) done;
            printf("fork: replay %d: %zu tokens one at a time %.2f s (%.3f tokens/s)\n", index, done, replay_seconds,
                   done / replay_seconds);
            if (replay_batch && done == sequence.size()) {
                FILE * logits_file = fopen((base + ".logits").c_str(), "wb");
                if (!logits_file) {
                    fprintf(stderr, "cannot write into %s\n", out.c_str());
                    return 1;
                }
                llama_memory_clear(llama_get_memory(ctx), true);
                t0 = now();
                if (!decode(ctx, sequence, 0)) return 1;
                for (size_t t = 0; t < sequence.size(); t++) {
                    fwrite(llama_get_logits_ith(ctx, (int32_t) t), sizeof(float), n_vocab, logits_file);
                }
                fclose(logits_file);
                printf("fork: replay %d: as one batch %.2f s\n", index, now() - t0);
            }
            fflush(stdout);
            continue;
        }
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
