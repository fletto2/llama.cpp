// llama-bolmo: run Bolmo / Bwen byte-level models (arch "bolmo").
//
//   generate:  llama-bolmo -m model.gguf -p "prompt" [-n 256] [--temp 0] [--seed 1]
//   score:     llama-bolmo -m model.gguf --score-file text.txt [--chunk 2048]
//              per-byte NLL marginalised over the boundary flag, as bits/byte
//   dump:      llama-bolmo -m model.gguf -p "text" --dump-logits out.bin
//              n x 520 float32 logits of the teacher-forced forward, then n int8 boundary flags
//   server:    llama-bolmo -m model.gguf --server [--host 127.0.0.1] [--port 8090]
//              POST /v1/completions {"prompt", "max_tokens" (bytes, default 256), "temperature" (0 = greedy),
//              "seed", "stop": [strings]} -> {"choices": [{"text", "finish_reason"}], "usage"}; POST /score {"text"}
//              -> {"bytes", "patches", "nll_per_byte", "bits_per_byte"}; GET /health. One request at a time.
//   mcq:       llama-bolmo -m model.gguf --mcq items.tsv out.tsv [--mcq-tf]
//              items: cat, stem, correct, wrong1..3 (header line first), as llm_score mcq. For every option,
//              the log-probability of the bytes of " <option>" after the stem. Default: causal, as in generation
//              (prefill the stem, then step through the option bytes; each byte ends a patch when the model
//              prefers its fused id). --mcq-tf: the teacher-forced forward over "<stem> <option>" instead, whose
//              boundaries use one byte of lookahead into the option.
//              --mcq-exact: the option's log-probability summed over both boundary choices at every byte (the
//              stem's last byte included), depth-first with snapshots; choices below --mcq-tau (default 1e-3) of
//              their byte's probability are skipped. Causal and teacher-forced scores bracket this one.
//              out: item, option (0 = correct), n_bytes, n_bytes, logprob

#include "llama.h"
#include "llama-bolmo.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>
#include <mutex>

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
        "       [--score-file text.txt] [--chunk n_bytes] [--dump-logits out.bin] [--mcq items.tsv out.tsv [--mcq-tf | --mcq-exact [--mcq-tau t]]]\n"
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

// log p(id) and log p(id + boundary) under the softmax of lg
static void logp_pair(const float * lg, int n_vocab, int32_t id, int32_t boff, double & l0, double & l1) {
    float mx = lg[0];
    for (int i = 1; i < n_vocab; ++i) {
        mx = std::max(mx, lg[i]);
    }
    double z = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        z += std::exp((double) lg[i] - mx);
    }
    l0 = (double) lg[id] - mx - std::log(z);
    l1 = (double) lg[id + boff] - mx - std::log(z);
}

static double logaddexp(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// Sum over the boundary flag of every byte of p(bytes[i..], flags), accumulated into total (log space).
// lg predicts bytes[i] in the current state; prefix = log-probability of the path so far. Depth-first, the likelier
// flag first; a branch is skipped when its prefix probability is below mass_eps of the mass found so far (it cannot
// add more than its prefix), or when its flag has less than tau of its byte's probability. budget caps the steps.
struct marg_ctx {
    llama_bolmo * bolmo;
    const std::vector<int32_t> * bytes;
    int n_vocab;
    int32_t boff;
    double log_tau, log_mass_eps;
    long steps = 0, budget = 0;
    bool truncated = false;
};

static void bolmo_marginal(marg_ctx & m, size_t i, const std::vector<float> & lg, double prefix, double & total) {
    const auto & bytes = *m.bytes;
    double l[2];
    logp_pair(lg.data(), m.n_vocab, bytes[i], m.boff, l[0], l[1]);
    const double lb = logaddexp(l[0], l[1]);
    if (i + 1 == bytes.size()) {
        total = logaddexp(total, prefix + lb);
        return;
    }
    const int order[2] = { l[1] > l[0] ? 1 : 0, l[1] > l[0] ? 0 : 1 };
    llama_bolmo_snapshot * snap = nullptr;
    std::vector<float> lg2(m.n_vocab);
    bool stepped = false;
    for (int k = 0; k < 2; ++k) {
        const int f = order[k];
        if (l[f] - lb < m.log_tau) {
            continue;
        }
        if (total > -INFINITY && prefix + l[f] < total + m.log_mass_eps) {
            continue;
        }
        if (m.steps >= m.budget) {
            m.truncated = true;
            continue;
        }
        if (stepped) {
            llama_bolmo_snapshot_restore(m.bolmo, snap);
        } else if (k == 0) {
            snap = llama_bolmo_snapshot_take(m.bolmo); // needed if the second flag is explored too
        }
        if (llama_bolmo_step(m.bolmo, bytes[i] + f*m.boff, lg2.data()) != 0) {
            fprintf(stderr, "step failed\n");
            exit(1);
        }
        m.steps++;
        stepped = true;
        bolmo_marginal(m, i + 1, lg2, prefix + l[f], total);
    }
    if (snap) {
        llama_bolmo_snapshot_free(snap);
    }
}

int main(int argc, char ** argv) {
    std::string model_path, prompt, score_file, dump_path, mcq_in, mcq_out;
    bool mcq_tf = false, mcq_exact = false, test_snapshot = false, server = false;
    std::string host = "127.0.0.1";
    int port = 8090;
    double mcq_tau = 1e-3;
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
        else if (a == "--mcq-exact") { mcq_exact = true; }
        else if (a == "--test-snapshot") { test_snapshot = true; }
        else if (a == "--server") { server = true; }
        else if (a == "--host") { host = next(); }
        else if (a == "--port") { port = std::stoi(next()); }
        else if (a == "--mcq-tau") { mcq_tau = std::stod(next()); }
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

    if (server) {
        using json = nlohmann::json;
        std::mutex mu;
        httplib::Server srv;
        auto sample = [&](const std::vector<float> & lg, float t, std::mt19937 & rng) {
            int32_t tok = 0;
            if (t <= 0.0f) {
                for (int i = 1; i < n_vocab; ++i) {
                    if (lg[i] > lg[tok]) {
                        tok = i;
                    }
                }
                return tok;
            }
            float mx = lg[0];
            for (int i = 1; i < n_vocab; ++i) mx = std::max(mx, lg[i]);
            std::vector<double> p(n_vocab);
            for (int i = 0; i < n_vocab; ++i) p[i] = std::exp((lg[i] - mx) / t);
            std::discrete_distribution<int> d(p.begin(), p.end());
            return (int32_t) d(rng);
        };
        srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        srv.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
            json in;
            try { in = json::parse(req.body); } catch (...) { res.status = 400; return; }
            const std::string p = in.value("prompt", std::string());
            const int n_max = in.value("max_tokens", 256);
            const float t = in.value("temperature", 0.0f);
            std::mt19937 rng(in.value("seed", 1u));
            std::vector<std::string> stops;
            if (in.contains("stop")) {
                if (in["stop"].is_string()) stops.push_back(in["stop"]);
                else for (auto & x : in["stop"]) stops.push_back(x);
            }
            std::lock_guard<std::mutex> lock(mu);
            const auto ids = tokenize(p);
            std::vector<float> lg(n_vocab);
            json out;
            if (ids.size() < 2 || llama_bolmo_prefill(bolmo, ids.data(), (int32_t) ids.size(), lg.data()) < 0) {
                res.status = 500;
                res.set_content("{\"error\":\"prefill failed (the prompt needs at least one byte)\"}", "application/json");
                return;
            }
            std::string text;
            std::string reason = "length";
            for (int g = 0; g < n_max; ++g) {
                const int32_t tok = sample(lg, t, rng);
                if (tok == eos || tok == eos + boff) { reason = "stop"; break; }
                const int32_t byte = llama_bolmo_id_to_byte(bolmo, tok);
                if (byte < 0) { reason = "stop"; break; }
                text.push_back((char) byte);
                bool hit = false;
                for (auto & st : stops) {
                    if (!st.empty() && text.size() >= st.size() && text.compare(text.size() - st.size(), st.size(), st) == 0) {
                        text.resize(text.size() - st.size());
                        hit = true;
                    }
                }
                if (hit) { reason = "stop"; break; }
                if (llama_bolmo_step(bolmo, tok, lg.data()) != 0) { reason = "error"; break; }
            }
            out["object"] = "text_completion";
            out["choices"] = json::array({ json{{"index", 0}, {"text", text}, {"finish_reason", reason}} });
            out["usage"] = json{{"prompt_bytes", (int) p.size()}, {"completion_bytes", (int) text.size()},
                                {"patches", llama_bolmo_n_patches(bolmo)}};
            res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        });
        srv.Post("/score", [&](const httplib::Request & req, httplib::Response & res) {
            json in;
            try { in = json::parse(req.body); } catch (...) { res.status = 400; return; }
            const std::string txt = in.value("text", std::string());
            std::lock_guard<std::mutex> lock(mu);
            const auto ids = tokenize(txt);
            const int n = (int) ids.size();
            std::vector<float> all((size_t) n*n_vocab);
            const int P = llama_bolmo_score(bolmo, ids.data(), n, all.data(), nullptr);
            if (P < 0) { res.status = 500; return; }
            double nll = 0;
            for (int i = 0; i + 1 < n; ++i) nll += nll_byte(all.data() + (size_t) i*n_vocab, n_vocab, ids[i + 1], boff);
            const int nb = std::max(1, n - 1);
            json out{{"bytes", n - 1}, {"patches", P}, {"nll_per_byte", nll / nb}, {"bits_per_byte", nll / nb / M_LN2}};
            res.set_content(out.dump(), "application/json");
        });
        fprintf(stderr, "llama-bolmo: listening on http://%s:%d\n", host.c_str(), port);
        if (!srv.listen(host, port)) {
            fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
            ret = 1;
        }
    } else if (test_snapshot) {
        // snapshot, run 8 bytes (each fused, so the global model steps too), restore, run them again: same logits
        const auto ids = tokenize(prompt);
        std::vector<float> lg(n_vocab), a(n_vocab), bb(n_vocab);
        if (llama_bolmo_prefill(bolmo, ids.data(), (int32_t) ids.size(), lg.data()) < 0) {
            return 1;
        }
        const int32_t seq[8] = { 'a', 'b', 'c', ' ', 'd', 'e', 'f', ' ' };
        llama_bolmo_snapshot * snap = llama_bolmo_snapshot_take(bolmo);
        const int n0 = llama_bolmo_n_patches(bolmo);
        double maxd = 0;
        std::vector<std::vector<float>> first;
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {
                llama_bolmo_snapshot_restore(bolmo, snap);
            }
            for (int k = 0; k < 8; ++k) {
                int32_t id = 0;
                const char ch = (char) seq[k];
                llama_bolmo_tokenize(bolmo, &ch, 1, &id, 1, false);
                llama_bolmo_step(bolmo, id + (k % 2 ? boff : 0), a.data());
                if (pass == 0) {
                    first.push_back(a);
                } else {
                    for (int i = 0; i < n_vocab; ++i) {
                        maxd = std::max(maxd, (double) std::fabs(a[i] - first[k][i]));
                    }
                }
            }
        }
        printf("patches %d -> %d; max |logit difference| after restore: %g\n", n0, llama_bolmo_n_patches(bolmo), maxd);
        llama_bolmo_snapshot_free(snap);
        GGML_UNUSED(bb);
    } else if (!mcq_in.empty()) {
        std::ifstream in(mcq_in);
        FILE * out = fopen(mcq_out.c_str(), "w");
        fprintf(out, "item\toption\tn_tokens\tn_bytes\tlogprob\n");
        std::string line;
        std::getline(in, line); // header
        int item = 0;
        std::vector<float> lg(n_vocab);
        const bool dbg = getenv("LLAMA_BOLMO_DEBUG") != nullptr;
        long mcq_steps = 0, mcq_trunc = 0;
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
                } else if (mcq_exact) {
                    // the stem's last byte is the first one whose boundary flag is summed over
                    if (llama_bolmo_prefill_ext(bolmo, ids_stem.data(), (int32_t) ids_stem.size(), lg.data(), false) < 0) {
                        fprintf(stderr, "prefill failed\n");
                        return 1;
                    }
                    std::vector<int32_t> bytes = { ids_stem.back() };
                    for (char ch : opt) {
                        int32_t id = 0;
                        llama_bolmo_tokenize(bolmo, &ch, 1, &id, 1, false);
                        bytes.push_back(id);
                    }
                    double l0, l1;
                    logp_pair(lg.data(), n_vocab, bytes[0], boff, l0, l1);
                    marg_ctx mc;
                    mc.bolmo = bolmo; mc.bytes = &bytes; mc.n_vocab = n_vocab; mc.boff = boff;
                    mc.log_tau = std::log(mcq_tau); mc.log_mass_eps = std::log(1e-4); mc.budget = 64*(long) bytes.size();
                    double total = -INFINITY;
                    bolmo_marginal(mc, 0, lg, 0.0, total);
                    mcq_steps += mc.steps;
                    mcq_trunc += mc.truncated;
                    // condition on the stem: p(option | stem) = p(last stem byte, option) / p(last stem byte)
                    lp = total - logaddexp(l0, l1);
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
                fprintf(stderr, "\r%d items (%.1f s, %ld branch steps, %ld options truncated)", item, (ggml_time_us() - t0) / 1e6, mcq_steps, mcq_trunc);
            }
        }
        fprintf(stderr, "\n%d items in %.1f s", item, (ggml_time_us() - t0) / 1e6);
        if (mcq_exact) {
            fprintf(stderr, ", %ld branch steps, %ld options truncated", mcq_steps, mcq_trunc);
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
