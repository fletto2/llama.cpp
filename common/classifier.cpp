#include "classifier.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>

// ---- names -------------------------------------------------------------------------------------------

const char * common_classifier_type_name(common_classifier_type type) {
    switch (type) {
        case COMMON_CLASSIFIER_NOUL:   return "noul";
        case COMMON_CLASSIFIER_CHOICE: return "choice";
        case COMMON_CLASSIFIER_SCORE:  return "score";
    }
    return "?";
}

bool common_classifier_type_from_name(const std::string & name, common_classifier_type & type) {
    if (name == "noul")   { type = COMMON_CLASSIFIER_NOUL;   return true; }
    if (name == "choice") { type = COMMON_CLASSIFIER_CHOICE; return true; }
    if (name == "score")  { type = COMMON_CLASSIFIER_SCORE;  return true; }
    return false;
}

const char * common_classifier_pooling_name(common_classifier_pooling pooling) {
    return pooling == COMMON_CLASSIFIER_POOL_LAST ? "last" : "mean";
}

bool common_classifier_pooling_from_name(const std::string & name, common_classifier_pooling & pooling) {
    if (name == "mean") { pooling = COMMON_CLASSIFIER_POOL_MEAN; return true; }
    if (name == "last") { pooling = COMMON_CLASSIFIER_POOL_LAST; return true; }
    return false;
}

// ---- head ------------------------------------------------------------------------------------------

int32_t common_classifier_head::n_classes() const {
    return type == COMMON_CLASSIFIER_NOUL ? 2 : (int32_t) options.size();
}

int32_t common_classifier_head::n_out() const {
    return type == COMMON_CLASSIFIER_CHOICE ? (int32_t) options.size() : 1;
}

int32_t common_classifier_head::n_bias() const {
    switch (type) {
        case COMMON_CLASSIFIER_NOUL:   return 1;
        case COMMON_CLASSIFIER_CHOICE: return (int32_t) options.size();
        case COMMON_CLASSIFIER_SCORE:  return (int32_t) options.size() - 1;
    }
    return 0;
}

std::string common_classifier_head::validate() const {
    if (n_embd <= 0) {
        return "n_embd is not set";
    }
    if (layer < 1) {
        return "classifier.layer must be >= 1";
    }
    if (type != COMMON_CLASSIFIER_NOUL && options.size() < 2) {
        return std::string(common_classifier_type_name(type)) + " needs at least 2 classifier.options";
    }
    if ((int64_t) weight.size() != (int64_t) n_out() * n_embd) {
        return "classifier.weight has " + std::to_string(weight.size()) + " values, expected " +
               std::to_string(n_out()) + " x " + std::to_string(n_embd);
    }
    if ((int32_t) bias.size() != n_bias()) {
        return "classifier.bias has " + std::to_string(bias.size()) + " values, expected " + std::to_string(n_bias());
    }
    for (float v : weight) {
        if (!std::isfinite(v)) {
            return "classifier.weight is not finite";
        }
    }
    for (size_t i = 0; i < bias.size(); i++) {
        if (!std::isfinite(bias[i])) {
            return "classifier.bias is not finite";
        }
        if (type == COMMON_CLASSIFIER_SCORE && i > 0 && bias[i] < bias[i - 1]) {
            return "score thresholds (classifier.bias) must be non-decreasing";
        }
    }
    return "";
}

bool common_classifier_head_load(const std::string & path, common_classifier_head & head, std::string & err) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params ip = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_data,
    };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), ip);
    if (!ctx) {
        err = "cannot read " + path;
        return false;
    }
    head = common_classifier_head();
    head.path = path;
    bool ok = true;
    auto fail = [&](const std::string & msg) { if (ok) { err = msg; ok = false; } };
    auto get_str = [&](const char * key, const std::string & def) -> std::string {
        const int64_t id = gguf_find_key(ctx, key);
        if (id < 0) {
            return def;
        }
        if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
            fail(std::string(key) + " is not a string");
            return def;
        }
        return gguf_get_val_str(ctx, id);
    };

    if (get_str("general.type", "") != "classifier") {
        fail("general.type is not 'classifier'");
    }
    const int64_t id_layer = gguf_find_key(ctx, "classifier.layer");
    if (id_layer < 0) {
        fail("classifier.layer is missing");
    } else if (gguf_get_kv_type(ctx, id_layer) == GGUF_TYPE_UINT32) {
        head.layer = (int32_t) gguf_get_val_u32(ctx, id_layer);
    } else if (gguf_get_kv_type(ctx, id_layer) == GGUF_TYPE_INT32) {
        head.layer = gguf_get_val_i32(ctx, id_layer);
    } else {
        fail("classifier.layer is not a 32-bit integer");
    }
    head.question_id = get_str("classifier.question_id", "relevant");
    const std::string type = get_str("classifier.type", "noul");
    if (!common_classifier_type_from_name(type, head.type)) {
        fail("unsupported classifier.type '" + type + "' (noul, choice, score)");
    }
    const std::string pooling = get_str("classifier.pooling", "mean");
    if (!common_classifier_pooling_from_name(pooling, head.pooling)) {
        fail("unsupported classifier.pooling '" + pooling + "' (mean, last)");
    }
    const int64_t id_opt = gguf_find_key(ctx, "classifier.options");
    if (id_opt >= 0) {
        if (gguf_get_kv_type(ctx, id_opt) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id_opt) != GGUF_TYPE_STRING) {
            fail("classifier.options is not a string array");
        } else {
            for (size_t i = 0; i < gguf_get_arr_n(ctx, id_opt); i++) {
                head.options.push_back(gguf_get_arr_str(ctx, id_opt, i));
            }
        }
    }
    ggml_tensor * w = ok ? ggml_get_tensor(ctx_data, "classifier.weight") : nullptr;
    ggml_tensor * b = ok ? ggml_get_tensor(ctx_data, "classifier.bias")   : nullptr;
    if (ok && (!w || !b || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32)) {
        fail("classifier.weight / classifier.bias (F32) are missing");
    }
    if (ok) {
        head.n_embd = (int32_t) w->ne[0];
        head.weight.assign((const float *) w->data, (const float *) w->data + ggml_nelements(w));
        head.bias.assign((const float *) b->data, (const float *) b->data + ggml_nelements(b));
        const std::string v = head.validate();
        if (!v.empty()) {
            fail(v);
        }
    }
    gguf_free(ctx);
    ggml_free(ctx_data);
    if (!ok) {
        err = path + ": " + err;
    }
    return ok;
}

bool common_classifier_head_save(const std::string & path, const common_classifier_head & head, std::string & err,
                                 const std::vector<std::pair<std::string, std::string>> & extra) {
    const std::string v = head.validate();
    if (!v.empty()) {
        err = v;
        return false;
    }
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", "classifier");
    gguf_set_val_str(ctx, "general.type", "classifier");
    gguf_set_val_u32(ctx, "classifier.layer", (uint32_t) head.layer);
    gguf_set_val_str(ctx, "classifier.question_id", head.question_id.c_str());
    gguf_set_val_str(ctx, "classifier.type", common_classifier_type_name(head.type));
    gguf_set_val_str(ctx, "classifier.pooling", common_classifier_pooling_name(head.pooling));
    if (!head.options.empty()) {
        std::vector<const char *> opts;
        for (const auto & o : head.options) {
            opts.push_back(o.c_str());
        }
        gguf_set_arr_str(ctx, "classifier.options", opts.data(), opts.size());
    }
    for (const auto & kv : extra) {
        gguf_set_val_str(ctx, ("classifier.info." + kv.first).c_str(), kv.second.c_str());
    }
    ggml_init_params gp = {
        /*.mem_size   =*/ (head.weight.size() + head.bias.size()) * sizeof(float) + 4 * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, head.n_embd, head.n_out());
    ggml_tensor * b = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, head.n_bias());
    ggml_set_name(w, "classifier.weight");
    ggml_set_name(b, "classifier.bias");
    memcpy(w->data, head.weight.data(), head.weight.size() * sizeof(float));
    memcpy(b->data, head.bias.data(), head.bias.size() * sizeof(float));
    gguf_add_tensor(ctx, w);
    gguf_add_tensor(ctx, b);
    const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    ggml_free(gctx);
    if (!ok) {
        err = "cannot write " + path;
    }
    return ok;
}

static double sigmoid(double z) {
    return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
}

static double dot(const float * a, const float * b, int32_t n) {
    double s = 0.0;
    for (int32_t i = 0; i < n; i++) {
        s += (double) a[i] * b[i];
    }
    return s;
}

// class probabilities from the linear outputs: z (n_out values) and, for score, the thresholds
static void probs_from_z(common_classifier_type type, int32_t n_classes, const double * z, const float * thr, double * p) {
    switch (type) {
        case COMMON_CLASSIFIER_NOUL: {
            const double t = sigmoid(z[0]);
            p[0] = 1.0 - t;
            p[1] = t;
        } break;
        case COMMON_CLASSIFIER_CHOICE: {
            double m = z[0];
            for (int32_t k = 1; k < n_classes; k++) m = std::max(m, z[k]);
            double s = 0.0;
            for (int32_t k = 0; k < n_classes; k++) { p[k] = std::exp(z[k] - m); s += p[k]; }
            for (int32_t k = 0; k < n_classes; k++) p[k] /= s;
        } break;
        case COMMON_CLASSIFIER_SCORE: {
            double prev = 0.0;
            for (int32_t k = 0; k < n_classes; k++) {
                const double c = k < n_classes - 1 ? sigmoid((double) thr[k] - z[0]) : 1.0;
                p[k] = std::max(0.0, c - prev);
                prev = c;
            }
        } break;
    }
}

std::vector<double> common_classifier_probs(const common_classifier_head & head, const float * x) {
    const int32_t K = head.n_classes();
    std::vector<double> z(head.n_out()), p(K);
    for (int32_t o = 0; o < head.n_out(); o++) {
        z[o] = dot(head.weight.data() + (size_t) o * head.n_embd, x, head.n_embd);
        if (head.type != COMMON_CLASSIFIER_SCORE) {
            z[o] += head.bias[o];
        }
    }
    probs_from_z(head.type, K, z.data(), head.bias.data(), p.data());
    return p;
}

// ---- capture ---------------------------------------------------------------------------------------

void common_classifier_capture::set_layers(int32_t n_embd, const std::vector<int32_t> & layers) {
    this->n_embd = n_embd;
    this->layers = layers;
    sum.assign(layers.size(), std::vector<double>(n_embd, 0.0));
    last.assign(layers.size(), std::vector<float>(n_embd, 0.0f));
    rows.assign(layers.size(), 0);
    error.clear();
}

void common_classifier_capture::reset() {
    for (auto & s : sum) std::fill(s.begin(), s.end(), 0.0);
    std::fill(rows.begin(), rows.end(), 0);
    error.clear();
}

int common_classifier_capture::index_of(int32_t layer) const {
    for (size_t i = 0; i < layers.size(); i++) {
        if (layers[i] == layer) {
            return (int) i;
        }
    }
    return -1;
}

bool common_classifier_capture::get(int32_t layer, common_classifier_pooling pooling, std::vector<float> & out) const {
    const int i = index_of(layer);
    if (i < 0 || rows[i] == 0) {
        return false;
    }
    out.resize(n_embd);
    for (int32_t j = 0; j < n_embd; j++) {
        out[j] = pooling == COMMON_CLASSIFIER_POOL_LAST ? last[i][j] : (float) (sum[i][j] / (double) rows[i]);
    }
    return true;
}

// "l_out-<il>" -> il + 1 (the feature after il + 1 layers), else -1
static int32_t tapped_layer(const char * name) {
    if (strncmp(name, "l_out-", 6) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + 6, &end, 10);
    return (end && *end == '\0' && il >= 0) ? (int32_t) il + 1 : -1;
}

bool common_classifier_capture::cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (common_classifier_capture *) user_data;
    const int i = self->index_of(tapped_layer(t->name));
    if (ask) {
        return i >= 0;
    }
    if (i < 0) {
        return true;
    }
    const int64_t n_embd = self->n_embd;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != n_embd || !ggml_is_contiguous(t)) {
        self->error = std::string("classifier: unexpected layout of ") + t->name;
        return false; // stops the graph
    }
    const int64_t n_rows = ggml_nelements(t) / n_embd;
    try {
        self->buf.resize(ggml_nelements(t));
    } catch (const std::exception & e) {
        self->error = std::string("classifier: ") + e.what();
        return false;
    }
    ggml_backend_tensor_get(t, self->buf.data(), 0, ggml_nbytes(t));
    double * s = self->sum[i].data();
    for (int64_t r = 0; r < n_rows; r++) {
        const float * row = self->buf.data() + r * n_embd;
        for (int64_t j = 0; j < n_embd; j++) {
            s[j] += row[j];
        }
    }
    if (n_rows > 0) {
        memcpy(self->last[i].data(), self->buf.data() + (n_rows - 1) * n_embd, n_embd * sizeof(float));
    }
    self->rows[i] += n_rows;
    return true;
}

// ---- fitting -----------------------------------------------------------------------------------------

namespace {

struct problem {
    const float *                  X;   // standardized, n x d
    int64_t                        n;
    int32_t                        d;
    const int32_t *                y;
    std::vector<double>            sw;  // per-row weights
    double                         sw_sum = 0.0;
    common_classifier_type         type;
    int32_t                        K;   // classes
    double                         reg; // 1 / (C n)
    int32_t                        n_threads;

    int32_t n_out() const { return type == COMMON_CLASSIFIER_CHOICE ? K : 1; }
    // parameters: weights (n_out x d), then noul: b; choice: b[K]; score: theta0, log-gaps[K-2]
    int64_t n_params() const { return (int64_t) n_out() * d + (type == COMMON_CLASSIFIER_SCORE ? K - 1 : n_out()); }

    void thresholds(const double * p, std::vector<double> & thr) const {
        const double * t = p + (int64_t) d;
        thr.resize(K - 1);
        thr[0] = t[0];
        for (int32_t k = 1; k < K - 1; k++) thr[k] = thr[k - 1] + std::exp(t[k]);
    }

    double eval(const std::vector<double> & p, std::vector<double> & g) const {
        const int64_t P = n_params();
        const int32_t no = n_out();
        std::vector<double> thr;
        if (type == COMMON_CLASSIFIER_SCORE) thresholds(p.data(), thr);
        const int nt = std::max(1, std::min<int>(n_threads, (int) std::max<int64_t>(1, n / 64)));
        std::vector<std::vector<double>> gs(nt, std::vector<double>(P, 0.0));
        std::vector<std::vector<double>> gthr(nt, std::vector<double>(std::max(0, K - 1), 0.0));
        std::vector<double> losses(nt, 0.0);
        auto work = [&](int tid) {
            std::vector<double> z(no), pr(K), dz(no);
            std::vector<double> & gt = gs[tid];
            for (int64_t i = tid; i < n; i += nt) {
                const float * x = X + i * d;
                for (int32_t o = 0; o < no; o++) {
                    double s = 0.0;
                    const double * w = p.data() + (int64_t) o * d;
                    for (int32_t j = 0; j < d; j++) s += w[j] * x[j];
                    z[o] = s;
                }
                const int32_t yi = y[i];
                const double si = sw[i];
                double loss = 0.0;
                if (type == COMMON_CLASSIFIER_NOUL) {
                    const double zz = z[0] + p[d];
                    loss = (zz > 0 ? zz + std::log1p(std::exp(-zz)) : std::log1p(std::exp(zz))) - (yi ? zz : 0.0);
                    dz[0] = sigmoid(zz) - (yi ? 1.0 : 0.0);
                    gt[d] += si * dz[0];
                } else if (type == COMMON_CLASSIFIER_CHOICE) {
                    for (int32_t k = 0; k < K; k++) z[k] += p[(int64_t) K * d + k];
                    probs_from_z(type, K, z.data(), nullptr, pr.data());
                    loss = -std::log(std::max(pr[yi], 1e-300));
                    for (int32_t k = 0; k < K; k++) {
                        dz[k] = pr[k] - (k == yi ? 1.0 : 0.0);
                        gt[(int64_t) K * d + k] += si * dz[k];
                    }
                } else {
                    const double s = z[0];
                    const double a = yi < K - 1 ? sigmoid(thr[yi] - s) : 1.0;
                    const double b = yi > 0 ? sigmoid(thr[yi - 1] - s) : 0.0;
                    const double pp = std::max(a - b, 1e-300);
                    loss = -std::log(pp);
                    const double da = yi < K - 1 ? a * (1.0 - a) : 0.0;
                    const double db = yi > 0 ? b * (1.0 - b) : 0.0;
                    dz[0] = (da - db) / pp;            // dloss/ds
                    if (yi < K - 1) gthr[tid][yi]     += si * (-da / pp);
                    if (yi > 0)     gthr[tid][yi - 1] += si * ( db / pp);
                }
                losses[tid] += si * loss;
                for (int32_t o = 0; o < no; o++) {
                    const double c = si * dz[o];
                    if (c == 0.0) continue;
                    double * gw = gt.data() + (int64_t) o * d;
                    for (int32_t j = 0; j < d; j++) gw[j] += c * x[j];
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; t++) th.emplace_back(work, t);
        work(0);
        for (auto & t : th) t.join();

        g.assign(P, 0.0);
        double f = 0.0;
        std::vector<double> dthr(std::max(0, K - 1), 0.0);
        for (int t = 0; t < nt; t++) {
            f += losses[t];
            for (int64_t q = 0; q < P; q++) g[q] += gs[t][q];
            for (int32_t k = 0; k < K - 1; k++) dthr[k] += gthr[t][k];
        }
        f /= sw_sum;
        for (int64_t q = 0; q < P; q++) g[q] /= sw_sum;
        if (type == COMMON_CLASSIFIER_SCORE) {
            for (int32_t k = 0; k < K - 1; k++) dthr[k] /= sw_sum;
            // theta_k = theta_0 + sum_{j=1..k} exp(t_j)
            double tail = 0.0;
            for (int32_t k = K - 2; k >= 0; k--) {
                tail += dthr[k];
                if (k == 0) g[d] = tail;
                else        g[d + k] = tail * std::exp(p[d + k]);
            }
        }
        const int64_t nw = (int64_t) no * d;
        for (int64_t q = 0; q < nw; q++) {
            f    += 0.5 * reg * p[q] * p[q];
            g[q] += reg * p[q];
        }
        return f;
    }
};

// limited-memory BFGS with backtracking (Armijo) line search
static double lbfgs(const problem & pb, std::vector<double> & x, int32_t max_iter, double tol, int32_t & iters) {
    const int m = 10;
    const int64_t P = (int64_t) x.size();
    std::vector<double> g, gn, xn(P), dir(P);
    double f = pb.eval(x, g);
    std::vector<std::vector<double>> S, Y;
    std::vector<double> rho;
    iters = 0;
    for (int it = 0; it < max_iter; it++) {
        double gmax = 0.0;
        for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax < tol) break;
        // two-loop recursion
        dir = g;
        std::vector<double> alpha(S.size());
        for (int i = (int) S.size() - 1; i >= 0; i--) {
            double a = 0.0;
            for (int64_t q = 0; q < P; q++) a += S[i][q] * dir[q];
            a *= rho[i];
            alpha[i] = a;
            for (int64_t q = 0; q < P; q++) dir[q] -= a * Y[i][q];
        }
        double gamma = 1.0;
        if (!S.empty()) {
            double sy = 0.0, yy = 0.0;
            for (int64_t q = 0; q < P; q++) { sy += S.back()[q] * Y.back()[q]; yy += Y.back()[q] * Y.back()[q]; }
            gamma = yy > 0 ? sy / yy : 1.0;
        } else {
            double gn2 = 0.0;
            for (double v : g) gn2 += v * v;
            gamma = 1.0 / std::max(1.0, std::sqrt(gn2));
        }
        for (int64_t q = 0; q < P; q++) dir[q] *= gamma;
        for (size_t i = 0; i < S.size(); i++) {
            double b = 0.0;
            for (int64_t q = 0; q < P; q++) b += Y[i][q] * dir[q];
            b *= rho[i];
            for (int64_t q = 0; q < P; q++) dir[q] += S[i][q] * (alpha[i] - b);
        }
        for (int64_t q = 0; q < P; q++) dir[q] = -dir[q];
        double gd = 0.0;
        for (int64_t q = 0; q < P; q++) gd += g[q] * dir[q];
        if (gd >= 0) { // not a descent direction: restart from the gradient
            S.clear(); Y.clear(); rho.clear();
            double gn2 = 0.0;
            for (double v : g) gn2 += v * v;
            const double sc = 1.0 / std::max(1.0, std::sqrt(gn2));
            for (int64_t q = 0; q < P; q++) dir[q] = -g[q] * sc;
            gd = -gn2 * sc;
        }
        double step = 1.0, fn = f;
        bool accepted = false;
        for (int ls = 0; ls < 40; ls++) {
            for (int64_t q = 0; q < P; q++) xn[q] = x[q] + step * dir[q];
            fn = pb.eval(xn, gn);
            if (std::isfinite(fn) && fn <= f + 1e-4 * step * gd) { accepted = true; break; }
            step *= 0.5;
        }
        if (!accepted) break;
        std::vector<double> s(P), yv(P);
        double sy = 0.0;
        for (int64_t q = 0; q < P; q++) { s[q] = xn[q] - x[q]; yv[q] = gn[q] - g[q]; sy += s[q] * yv[q]; }
        const double df = f - fn;
        x.swap(xn); g.swap(gn); f = fn;
        iters = it + 1;
        if (sy > 1e-12) {
            S.push_back(std::move(s)); Y.push_back(std::move(yv)); rho.push_back(1.0 / sy);
            if ((int) S.size() > m) { S.erase(S.begin()); Y.erase(Y.begin()); rho.erase(rho.begin()); }
        }
        if (df >= 0 && df < 1e-12 * std::max(1.0, std::fabs(f))) break;
    }
    return f;
}

} // namespace

common_classifier_fit_result common_classifier_fit(const std::vector<float> & X, int64_t n, int32_t d,
                                                   const std::vector<int32_t> & y, const common_classifier_fit_params & params) {
    common_classifier_fit_result res;
    const int32_t K = params.type == COMMON_CLASSIFIER_NOUL ? 2 : params.n_classes;
    // standardize
    std::vector<double> mu(d, 0.0), sd(d, 0.0);
    for (int64_t i = 0; i < n; i++) for (int32_t j = 0; j < d; j++) mu[j] += X[i * d + j];
    for (int32_t j = 0; j < d; j++) mu[j] /= (double) n;
    for (int64_t i = 0; i < n; i++) for (int32_t j = 0; j < d; j++) { const double v = X[i * d + j] - mu[j]; sd[j] += v * v; }
    for (int32_t j = 0; j < d; j++) { sd[j] = std::sqrt(sd[j] / (double) n); if (sd[j] < 1e-8) sd[j] = 1.0; }
    std::vector<float> Xs((size_t) n * d);
    for (int64_t i = 0; i < n; i++) for (int32_t j = 0; j < d; j++) Xs[i * d + j] = (float) ((X[i * d + j] - mu[j]) / sd[j]);

    problem pb;
    pb.X = Xs.data(); pb.n = n; pb.d = d; pb.y = y.data(); pb.type = params.type; pb.K = K;
    pb.reg = 1.0 / (params.C * (double) n);
    pb.n_threads = params.n_threads;
    std::vector<int64_t> count(K, 0);
    for (int64_t i = 0; i < n; i++) count[y[i]]++;
    pb.sw.assign(n, 1.0);
    if (params.balanced) {
        int present = 0;
        for (int32_t k = 0; k < K; k++) present += count[k] > 0;
        for (int64_t i = 0; i < n; i++) pb.sw[i] = (double) n / ((double) present * (double) count[y[i]]);
    }
    pb.sw_sum = std::accumulate(pb.sw.begin(), pb.sw.end(), 0.0);

    // start from the class priors
    std::vector<double> x(pb.n_params(), 0.0);
    auto logit = [](double q) { q = std::min(std::max(q, 1e-6), 1.0 - 1e-6); return std::log(q / (1.0 - q)); };
    if (params.type == COMMON_CLASSIFIER_NOUL) {
        x[d] = logit((double) count[1] / (double) n);
    } else if (params.type == COMMON_CLASSIFIER_CHOICE) {
        for (int32_t k = 0; k < K; k++) x[(int64_t) K * d + k] = std::log(std::max<double>(count[k], 0.5) / (double) n);
    } else {
        double cum = 0.0, prev = 0.0;
        for (int32_t k = 0; k < K - 1; k++) {
            cum += (double) count[k] / (double) n;
            const double t = logit(cum);
            if (k == 0) x[d] = t;
            else        x[d + k] = std::log(std::max(t - prev, 1e-3));
            prev = k == 0 ? t : prev + std::exp(x[d + k]);
        }
    }
    res.loss = lbfgs(pb, x, params.max_iter, params.tol, res.iters);

    // fold the standardization into weights for raw features
    const int32_t no = pb.n_out();
    res.weight.assign((size_t) no * d, 0.0f);
    std::vector<double> shift(no, 0.0); // -(w . mu / sd)
    for (int32_t o = 0; o < no; o++) {
        for (int32_t j = 0; j < d; j++) {
            const double w = x[(int64_t) o * d + j] / sd[j];
            res.weight[(size_t) o * d + j] = (float) w;
            shift[o] -= w * mu[j];
        }
    }
    if (params.type == COMMON_CLASSIFIER_NOUL) {
        res.bias = { (float) (x[d] + shift[0]) };
    } else if (params.type == COMMON_CLASSIFIER_CHOICE) {
        res.bias.resize(K);
        for (int32_t k = 0; k < K; k++) res.bias[k] = (float) (x[(int64_t) K * d + k] + shift[k]);
    } else {
        // theta_k - s_std = theta_k - (w_raw . x - shift) => raw threshold = theta_k + shift... with s_std = w_raw . x + shift
        std::vector<double> thr;
        pb.thresholds(x.data(), thr);
        res.bias.resize(K - 1);
        for (int32_t k = 0; k < K - 1; k++) res.bias[k] = (float) (thr[k] - shift[0]);
    }
    return res;
}

common_classifier_metrics common_classifier_metrics_of(const std::vector<double> & probs, int32_t K,
                                                      const std::vector<int32_t> & y, common_classifier_type type) {
    common_classifier_metrics m;
    const int64_t n = (int64_t) y.size();
    m.n = n;
    if (n == 0) return m;
    double ll = 0.0, mae = 0.0;
    int64_t correct = 0;
    for (int64_t i = 0; i < n; i++) {
        const double * p = probs.data() + i * K;
        ll += -std::log(std::max(p[y[i]], 1e-15));
        const int32_t am = (int32_t) (std::max_element(p, p + K) - p);
        correct += am == y[i];
        if (type == COMMON_CLASSIFIER_SCORE) {
            double e = 0.0;
            for (int32_t k = 0; k < K; k++) e += k * p[k];
            mae += std::fabs(e - y[i]);
        }
    }
    m.log_loss = ll / n;
    m.accuracy = (double) correct / n;
    if (type == COMMON_CLASSIFIER_SCORE) m.mae = mae / n;
    if (type == COMMON_CLASSIFIER_NOUL) {
        std::vector<int64_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return probs[a * 2 + 1] < probs[b * 2 + 1]; });
        double rank_pos = 0.0;
        int64_t n_pos = 0;
        for (int64_t i = 0; i < n;) {
            int64_t j = i;
            while (j + 1 < n && probs[idx[j + 1] * 2 + 1] == probs[idx[i] * 2 + 1]) j++;
            const double avg = 0.5 * (double) (i + j) + 1.0;
            for (int64_t k = i; k <= j; k++) if (y[idx[k]] == 1) { rank_pos += avg; n_pos++; }
            i = j + 1;
        }
        const int64_t n_neg = n - n_pos;
        if (n_pos > 0 && n_neg > 0) {
            m.auc = (rank_pos - (double) n_pos * (n_pos + 1) / 2.0) / ((double) n_pos * n_neg);
        }
    }
    return m;
}

common_classifier_metrics common_classifier_cv(const std::vector<float> & X, int64_t n, int32_t d,
                                               const std::vector<int32_t> & y, const common_classifier_fit_params & params,
                                               int32_t folds, uint32_t seed) {
    const int32_t K = params.type == COMMON_CLASSIFIER_NOUL ? 2 : params.n_classes;
    // stratified fold assignment
    std::vector<int32_t> fold(n, 0);
    std::mt19937 rng(seed);
    for (int32_t k = 0; k < K; k++) {
        std::vector<int64_t> rows;
        for (int64_t i = 0; i < n; i++) if (y[i] == k) rows.push_back(i);
        std::shuffle(rows.begin(), rows.end(), rng);
        for (size_t r = 0; r < rows.size(); r++) fold[rows[r]] = (int32_t) (r % folds);
    }
    std::vector<double> probs((size_t) n * K, 0.0);
    common_classifier_head h;
    h.type = params.type; h.n_embd = d; h.layer = 1;
    if (params.type != COMMON_CLASSIFIER_NOUL) h.options.assign(K, "");
    for (int32_t f = 0; f < folds; f++) {
        std::vector<float> Xtr;
        std::vector<int32_t> ytr;
        for (int64_t i = 0; i < n; i++) {
            if (fold[i] == f) continue;
            Xtr.insert(Xtr.end(), X.begin() + i * d, X.begin() + (i + 1) * d);
            ytr.push_back(y[i]);
        }
        if (ytr.empty()) continue;
        const auto fit = common_classifier_fit(Xtr, (int64_t) ytr.size(), d, ytr, params);
        h.weight = fit.weight;
        h.bias   = fit.bias;
        for (int64_t i = 0; i < n; i++) {
            if (fold[i] != f) continue;
            const auto p = common_classifier_probs(h, X.data() + i * d);
            std::copy(p.begin(), p.end(), probs.begin() + i * K);
        }
    }
    return common_classifier_metrics_of(probs, K, y, params.type);
}

// ---- training -----------------------------------------------------------------------------------------

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

bool common_classifier_encode_labels(const std::vector<std::string> & labels, common_classifier_type type,
                                     std::vector<std::string> & options, std::vector<int32_t> & y, std::string & err) {
    y.clear();
    if (type == COMMON_CLASSIFIER_NOUL) {
        for (const auto & l : labels) {
            const std::string s = lower(l);
            int32_t v = -1;
            if (s == "true" || s == "yes") v = 1;
            else if (s == "false" || s == "no") v = 0;
            else {
                char * end = nullptr;
                const double x = std::strtod(s.c_str(), &end);
                if (!s.empty() && end && *end == '\0') v = x != 0.0 ? 1 : 0;
            }
            if (v < 0) { err = "noul label must be true/false: '" + l + "'"; return false; }
            y.push_back(v);
        }
        return true;
    }
    if (type == COMMON_CLASSIFIER_CHOICE && options.empty()) {
        options = labels;
        std::sort(options.begin(), options.end());
        options.erase(std::unique(options.begin(), options.end()), options.end());
    }
    if (type == COMMON_CLASSIFIER_SCORE && options.empty()) {
        err = "score needs options: one name per level";
        return false;
    }
    if (options.size() < 2) { err = "at least 2 options are needed"; return false; }
    for (const auto & l : labels) {
        int32_t v = -1;
        for (size_t k = 0; k < options.size(); k++) if (options[k] == l) v = (int32_t) k;
        if (v < 0 && type == COMMON_CLASSIFIER_SCORE) {
            char * end = nullptr;
            const long x = std::strtol(l.c_str(), &end, 10);
            if (!l.empty() && end && *end == '\0' && x >= 0 && x < (long) options.size()) v = (int32_t) x;
        }
        if (v < 0) { err = "label '" + l + "' is not one of the options" + (type == COMMON_CLASSIFIER_SCORE ? " or a level index" : ""); return false; }
        y.push_back(v);
    }
    return true;
}

std::vector<int32_t> common_classifier_auto_layers(int32_t n_layer) {
    std::vector<int32_t> L;
    const int32_t lo = std::max(1, n_layer / 6), hi = std::max(lo, n_layer - 2);
    const int32_t step = std::max(1, (hi - lo) / 11);
    for (int32_t l = lo; l <= hi; l += step) L.push_back(l);
    return L;
}

std::string common_classifier_metrics_str(const common_classifier_metrics & m, common_classifier_type type) {
    char b[256];
    if (type == COMMON_CLASSIFIER_NOUL)       snprintf(b, sizeof b, "log-loss %.4f  AUC %.4f  accuracy %.4f", m.log_loss, m.auc, m.accuracy);
    else if (type == COMMON_CLASSIFIER_SCORE) snprintf(b, sizeof b, "log-loss %.4f  MAE %.4f  accuracy %.4f", m.log_loss, m.mae, m.accuracy);
    else                                      snprintf(b, sizeof b, "log-loss %.4f  accuracy %.4f", m.log_loss, m.accuracy);
    return b;
}

bool common_classifier_train(const std::vector<std::vector<std::vector<float>>> & feats, int64_t n, int32_t d,
                             const std::vector<int32_t> & layers, const std::vector<common_classifier_pooling> & pools,
                             const std::vector<int32_t> & y, const common_classifier_train_params & params,
                             common_classifier_train_result & res, std::string & err) {
    const int32_t K = params.fit.n_classes;
    if ((int64_t) y.size() != n || n == 0) { err = "no items or a label count mismatch"; return false; }
    if (layers.empty() || pools.empty() || params.Cs.empty()) { err = "no layers, poolings or C values"; return false; }
    if (feats.size() != layers.size()) { err = "feature/layer count mismatch"; return false; }
    if (params.folds < 2) { err = "folds must be at least 2"; return false; }
    std::vector<int64_t> count(K, 0);
    for (int32_t v : y) {
        if (v < 0 || v >= K) { err = "label index outside 0.." + std::to_string(K - 1); return false; }
        count[v]++;
    }
    for (int32_t k = 0; k < K; k++) {
        if (count[k] < params.folds) {
            err = "class " + std::to_string(k) + " has " + std::to_string(count[k]) + " items, fewer than the " +
                  std::to_string(params.folds) + " folds";
            return false;
        }
    }
    common_classifier_fit_params fp = params.fit;
    double best = INFINITY;
    size_t bl = 0, bp = 0;
    res.log.clear();
    for (size_t l = 0; l < layers.size(); l++) {
        for (size_t p = 0; p < pools.size(); p++) {
            for (double C : params.Cs) {
                fp.C = C;
                const auto m = common_classifier_cv(feats[l][p], n, d, y, fp, params.folds, params.seed);
                char b[96];
                snprintf(b, sizeof b, "layer %3d  %-4s  C %-6g  ", layers[l], common_classifier_pooling_name(pools[p]), C);
                res.log.push_back(b + common_classifier_metrics_str(m, fp.type));
                if (m.log_loss < best) { best = m.log_loss; bl = l; bp = p; res.C = C; res.cv = m; }
            }
        }
    }
    fp.C = res.C;
    const auto fit = common_classifier_fit(feats[bl][bp], n, d, y, fp);
    res.layer   = layers[bl];
    res.pooling = pools[bp];
    res.weight  = fit.weight;
    res.bias    = fit.bias;
    res.iters   = fit.iters;
    return true;
}
