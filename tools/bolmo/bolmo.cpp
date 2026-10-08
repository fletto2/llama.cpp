// llama-bolmo: run Bolmo / Bwen byte-level models (arch "bolmo").
//
//   generate:  llama-bolmo -m model.gguf -p "prompt" [-n 256] [--temp 0] [--seed 1]
//   score:     llama-bolmo -m model.gguf --score-file text.txt [--chunk 2048]
//              per-byte NLL marginalised over the boundary flag, as bits/byte
//   dump:      llama-bolmo -m model.gguf -p "text" --dump-logits out.bin
//              n x 520 float32 logits of the teacher-forced forward, then n int8 boundary flags
//   mcq:       llama-bolmo -m model.gguf --mcq items.tsv out.tsv [--mcq-tf]
//              items: cat, stem, correct, wrong1..3 (header line first), as llm_score mcq. For every option,
//              the log-probability of the bytes of " <option>" after the stem. Default: causal, as in generation
//              (prefill the stem, then step through the option bytes; each byte ends a patch when the model
//              prefers its fused id). --mcq-tf: the teacher-forced forward over "<stem> <option>" instead, whose
//              boundaries use one byte of lookahead into the option.
//              out: item, option (0 = correct), n_bytes, n_bytes, logprob

#include "llama.h"
#include "llama-bolmo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [-p prompt | -f prompt.txt] [-n n_bytes] [--temp T] [--seed S]\n"
        "       [--score-file text.txt] [--chunk n_bytes] [--dump-logits out.bin] [--mcq items.tsv out.tsv [--mcq-tf]]\n"
        "       [-ngl n] [-t threads] [-c n_ctx_patches] [-b n_batch] [--show-patches]\n", argv0);
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static double nll_byte(const float * lg, int n_vocab, int32_t id, int32_t boff) {
    // -log p(byte) with p(byte) = p(id) + p(id + boundary)
    float mx = lg[0];
    for (int i = 1; i < n_vocab; ++i) {
        mx = std::max(mx, lg[i]);
    }
    double z = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        z += std::exp((double) lg[i] - mx);
    }
    double p = std::exp((double) lg[id] - mx);
    if (id + boff < n_vocab) {
        p += std::exp((double) lg[id + boff] - mx);
    }
    return -std::log(p / z);
}

int main(int argc, char ** argv) {
    std::string model_path, prompt, score_file, dump_path, mcq_in, mcq_out;
    bool mcq_tf = false;
    int n_predict = 128, ngl = 0, n_threads = 8, n_ctx = 4096, n_batch = 512, chunk = 2048;
    float temp = 0.0f;
    unsigned seed = 1;
    bool show_patches = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if      (a == "-m") { model_path = next(); }
        else if (a == "-p") { prompt = next(); }
        else if (a == "-f") { prompt = read_file(next()); }
        else if (a == "-n") { n_predict = std::stoi(next()); }
        else if (a == "-ngl" || a == "--n-gpu-layers") { ngl = std::stoi(next()); }
        else if (a == "-t") { n_threads = std::stoi(next()); }
        else if (a == "-c") { n_ctx = std::stoi(next()); }
        else if (a == "-b") { n_batch = std::stoi(next()); }
        else if (a == "--temp") { temp = std::stof(next()); }
        else if (a == "--seed") { seed = (unsigned) std::stoul(next()); }
        else if (a == "--score-file") { score_file = next(); }
        else if (a == "--chunk") { chunk = std::stoi(next()); }
        else if (a == "--dump-logits") { dump_path = next(); }
        else if (a == "--show-patches") { show_patches = true; }
        else if (a == "--mcq") { mcq_in = next(); mcq_out = next(); }
        else if (a == "--mcq-tf") { mcq_tf = true; }
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path.c_str());
        return 1;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_batch;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings      = true;
    cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    llama_bolmo * bolmo = llama_bolmo_init(ctx);
    if (!bolmo) {
        return 1;
    }
    const int n_vocab = llama_bolmo_n_vocab(bolmo);
    const int32_t boff = llama_bolmo_boundary_offset(bolmo);
    const int32_t eos  = llama_bolmo_token_eos(bolmo);

    auto tokenize = [&](const std::string & s) {
        std::vector<int32_t> ids(s.size() + 1);
        const int n = llama_bolmo_tokenize(bolmo, s.data(), (int32_t) s.size(), ids.data(), (int32_t) ids.size(), true);
        ids.resize(n);
        return ids;
    };

    int ret = 0;

    if (!mcq_in.empty()) {
        std::ifstream in(mcq_in);
        FILE * out = fopen(mcq_out.c_str(), "w");
        fprintf(out, "item\toption\tn_tokens\tn_bytes\tlogprob\n");
        std::string line;
        std::getline(in, line); // header
        int item = 0;
        std::vector<float> lg(n_vocab);
        const bool dbg = getenv("LLAMA_BOLMO_DEBUG") != nullptr;
        const auto t0 = ggml_time_us();
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            size_t p0 = 0;
            for (size_t p; (p = line.find('\t', p0)) != std::string::npos; p0 = p + 1) {
                f.push_back(line.substr(p0, p - p0));
            }
            f.push_back(line.substr(p0));
            if (f.size() < 6) {
                continue;
            }
            const std::string & stem = f[1];
            const auto ids_stem = tokenize(stem);
            for (int o = 0; o < 4; ++o) {
                const std::string opt = " " + f[2 + o];
                double lp = 0.0;
                if (mcq_tf) {
                    const auto ids = tokenize(stem + opt);
                    const int n = (int) ids.size();
                    std::vector<float> all((size_t) n*n_vocab);
                    std::vector<int8_t> bnd(n);
                    if (llama_bolmo_score(bolmo, ids.data(), n, all.data(), bnd.data()) < 0) {
                        fprintf(stderr, "score failed\n");
                        return 1;
                    }
                    for (int t = (int) ids_stem.size() - 1; t + 1 < n; ++t) {
                        const double nb = nll_byte(all.data() + (size_t) t*n_vocab, n_vocab, ids[t + 1], boff);
                        lp -= nb;
                        if (dbg) {
                            fprintf(stderr, "tf item %d opt %d byte '%c' nll %.4f bnd[t] %d\n", item, o, (char) llama_bolmo_id_to_byte(bolmo, ids[t + 1]), nb, bnd[t]);
                        }
                    }
                } else {
                    if (llama_bolmo_prefill(bolmo, ids_stem.data(), (int32_t) ids_stem.size(), lg.data()) < 0) {
                        fprintf(stderr, "prefill failed\n");
                        return 1;
                    }
                    for (size_t i = 0; i < opt.size(); ++i) {
                        int32_t id = 0;
                        llama_bolmo_tokenize(bolmo, opt.data() + i, 1, &id, 1, false);
                        const double nb = nll_byte(lg.data(), n_vocab, id, boff);
                        lp -= nb;
                        if (dbg) {
                            fprintf(stderr, "causal item %d opt %d byte '%c' nll %.4f fused %d\n", item, o, opt[i], nb, lg[id + boff] > lg[id]);
                        }
                        if (i + 1 < opt.size()) {
                            const int32_t tok = lg[id + boff] > lg[id] ? id + boff : id;
                            if (llama_bolmo_step(bolmo, tok, lg.data()) != 0) {
                                fprintf(stderr, "step failed\n");
                                return 1;
                            }
                        }
                    }
                }
                fprintf(out, "%d\t%d\t%zu\t%zu\t%.6f\n", item, o, opt.size(), opt.size(), lp);
            }
            item++;
            if (item % 20 == 0) {
                fprintf(stderr, "\r%d items (%.1f s)", item, (ggml_time_us() - t0) / 1e6);
            }
        }
        fprintf(stderr, "\n");
        fclose(out);
    } else if (!score_file.empty()) {
        const std::string text = read_file(score_file);
        double nll = 0.0;
        long n_bytes = 0, n_patch = 0;
        const auto t0 = ggml_time_us();
        for (size_t off = 0; off < text.size(); off += chunk) {
            const auto ids = tokenize(text.substr(off, chunk));
            const int n = (int) ids.size();
            std::vector<float> lg((size_t) n*n_vocab);
            const int P = llama_bolmo_score(bolmo, ids.data(), n, lg.data(), nullptr);
            if (P < 0) {
                fprintf(stderr, "score failed (%d)\n", P);
                ret = 1;
                break;
            }
            for (int t = 0; t + 1 < n; ++t) {
                nll += nll_byte(lg.data() + (size_t) t*n_vocab, n_vocab, ids[t + 1], boff);
                n_bytes++;
            }
            n_patch += P;
        }
        const double dt = (ggml_time_us() - t0) / 1e6;
        printf("bytes %ld  patches %ld  bytes/patch %.3f  nll/byte %.5f  bits/byte %.5f  ppl/byte %.4f  (%.1f s)\n",
                n_bytes, n_patch, (double) n_bytes / std::max(1L, n_patch), nll / n_bytes, nll / n_bytes / M_LN2,
                std::exp(nll / n_bytes), dt);
    } else if (!dump_path.empty()) {
        const auto ids = tokenize(prompt);
        const int n = (int) ids.size();
        std::vector<float> lg((size_t) n*n_vocab);
        std::vector<int8_t> bnd(n);
        const int P = llama_bolmo_score(bolmo, ids.data(), n, lg.data(), bnd.data());
        if (P < 0) {
            fprintf(stderr, "score failed (%d)\n", P);
            ret = 1;
        } else {
            FILE * f = fopen(dump_path.c_str(), "wb");
            fwrite(lg.data(), sizeof(float), lg.size(), f);
            fwrite(bnd.data(), 1, bnd.size(), f);
            fclose(f);
            fprintf(stderr, "%d bytes, %d patches -> %s\n", n, P, dump_path.c_str());
        }
    } else {
        const auto ids = tokenize(prompt);
        std::vector<float> lg(n_vocab);
        const auto t0 = ggml_time_us();
        const int P = llama_bolmo_prefill(bolmo, ids.data(), (int32_t) ids.size(), lg.data());
        if (P < 0) {
            fprintf(stderr, "prefill failed (%d)\n", P);
            return 1;
        }
        const auto t1 = ggml_time_us();
        std::mt19937 rng(seed);
        fputs(prompt.c_str(), stdout);
        fflush(stdout);
        int n_gen = 0;
        for (; n_gen < n_predict; ++n_gen) {
            int32_t tok = 0;
            if (temp <= 0.0f) {
                for (int i = 1; i < n_vocab; ++i) {
                    if (lg[i] > lg[tok]) {
                        tok = i;
                    }
                }
            } else {
                float mx = lg[0];
                for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, lg[i]);
                std::vector<double> p(n_vocab);
                for (int i = 0; i < n_vocab; ++i) p[i] = std::exp((lg[i] - mx) / temp);
                std::discrete_distribution<int> d(p.begin(), p.end());
                tok = d(rng);
            }
            if (tok == eos || tok == eos + boff) {
                break;
            }
            const int32_t byte = llama_bolmo_id_to_byte(bolmo, tok);
            if (byte < 0) {
                fprintf(stderr, "\n[special token %d]\n", tok);
                break;
            }
            putchar(byte);
            if (show_patches && tok >= boff) {
                fputs("\xc2\xa6", stdout); // broken bar after a patch-final byte
            }
            fflush(stdout);
            if (llama_bolmo_step(bolmo, tok, lg.data()) != 0) {
                fprintf(stderr, "step failed\n");
                ret = 1;
                break;
            }
        }
        const auto t2 = ggml_time_us();
        fprintf(stderr, "\n\nprompt: %zu bytes, %d patches, %.2f s; generated %d bytes in %.2f s (%.1f bytes/s), %d patches total\n",
                ids.size(), P, (t1 - t0) / 1e6, n_gen, (t2 - t1) / 1e6, n_gen / ((t2 - t1) / 1e6), llama_bolmo_n_patches(bolmo));
    }

    llama_bolmo_free(bolmo);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ret;
}
