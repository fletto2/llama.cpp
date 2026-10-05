#include "server-classifier.h"

#include "../../src/llama-ext.h" // staging API: llama_set_n_layer_exit

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

// one early-exit context and the mean of the residual after its last layer
struct server_classifier_ctx {
    int32_t         layer = 0;
    std::string     tap_name; // l_out-<layer-1>
    llama_context * ctx   = nullptr;

    std::vector<double> sum;
    std::vector<float>  buf;
    int64_t             rows = 0;
    std::string         error; // set by the callback (exceptions must not cross the ggml C code)

    ~server_classifier_ctx() {
        llama_free(ctx);
    }

    static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
        auto * self = (server_classifier_ctx *) user_data;
        if (ask) {
            return self->tap_name == t->name;
        }
        if (self->tap_name != t->name) {
            return true;
        }
        const int64_t n_embd = (int64_t) self->sum.size();
        if (t->type != GGML_TYPE_F32 || t->ne[0] != n_embd || !ggml_is_contiguous(t)) {
            self->error = std::string("classifier: unexpected layout of ") + t->name;
            return false; // stop the graph
        }
        const int64_t n_rows = ggml_nelements(t) / n_embd;
        self->buf.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, self->buf.data(), 0, ggml_nbytes(t));
        for (int64_t r = 0; r < n_rows; r++) {
            const float * row = self->buf.data() + r*n_embd;
            for (int64_t i = 0; i < n_embd; i++) {
                self->sum[i] += row[i];
            }
        }
        self->rows += n_rows;
        return true;
    }
};

server_classifier::server_classifier()  = default;
server_classifier::~server_classifier() = default;

static server_classifier_head load_head(const std::string & path) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params ip = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_data,
    };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), ip);
    if (!ctx) {
        throw std::runtime_error("classifier: cannot read " + path);
    }
    auto get_str = [&](const char * key, const char * def) -> std::string {
        const int64_t id = gguf_find_key(ctx, key);
        return id < 0 ? std::string(def) : std::string(gguf_get_val_str(ctx, id));
    };

    server_classifier_head head;
    head.path = path;
    try {
        if (get_str("general.type", "") != "classifier") {
            throw std::runtime_error("general.type is not 'classifier'");
        }
        const int64_t id_layer = gguf_find_key(ctx, "classifier.layer");
        if (id_layer < 0) {
            throw std::runtime_error("classifier.layer is missing");
        }
        head.layer       = (int32_t) gguf_get_val_u32(ctx, id_layer);
        head.question_id = get_str("classifier.question_id", "relevant");
        head.type        = get_str("classifier.type", "noul");
        if (head.type != "noul") {
            throw std::runtime_error("unsupported classifier.type '" + head.type + "'");
        }
        ggml_tensor * w = ggml_get_tensor(ctx_data, "classifier.weight");
        ggml_tensor * b = ggml_get_tensor(ctx_data, "classifier.bias");
        if (!w || !b || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || ggml_nelements(b) != 1) {
            throw std::runtime_error("classifier.weight / classifier.bias (F32) are missing");
        }
        head.weight.assign((const float *) w->data, (const float *) w->data + ggml_nelements(w));
        head.bias = *(const float *) b->data;
    } catch (const std::exception & e) {
        gguf_free(ctx);
        ggml_free(ctx_data);
        throw std::runtime_error("classifier: " + path + ": " + e.what());
    }
    gguf_free(ctx);
    ggml_free(ctx_data);
    return head;
}

void server_classifier::init(llama_model * model, const std::vector<std::string> & paths, int32_t n_ctx, int32_t n_threads) {
    this->model  = model;
    this->n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_model_n_layer(model);

    for (const auto & path : paths) {
        server_classifier_head head = load_head(path);
        if ((int32_t) head.weight.size() != n_embd) {
            throw std::runtime_error("classifier: " + path + ": weight has " + std::to_string(head.weight.size()) +
                                     " values, the model has n_embd = " + std::to_string(n_embd));
        }
        if (head.layer < 1 || head.layer > n_layer) {
            throw std::runtime_error("classifier: " + path + ": layer " + std::to_string(head.layer) +
                                     " is outside 1.." + std::to_string(n_layer));
        }
        heads.push_back(std::move(head));
    }

    for (const auto & head : heads) {
        bool have = false;
        for (const auto & c : ctxs) {
            have |= c->layer == head.layer;
        }
        if (have) {
            continue;
        }
        auto c = std::make_unique<server_classifier_ctx>();
        c->layer    = head.layer;
        c->tap_name = "l_out-" + std::to_string(head.layer - 1);
        c->sum.assign(n_embd, 0.0);

        llama_context_params cp = llama_context_default_params();
        // embeddings mode: every token is an output, so the graph keeps all rows up to the exit layer
        // (otherwise the rows of non-output tokens are dropped before the last computed layer) and no LM
        // head is computed; pooling needs the whole input in one ubatch
        cp.embeddings        = true;
        cp.pooling_type      = LLAMA_POOLING_TYPE_MEAN;
        cp.n_ctx             = n_ctx;
        cp.n_batch           = n_ctx;
        cp.n_ubatch          = n_ctx;
        cp.n_seq_max         = 1;
        cp.n_threads         = n_threads;
        cp.n_threads_batch   = n_threads;
        cp.cb_eval           = server_classifier_ctx::cb_eval;
        cp.cb_eval_user_data = c.get();
        c->ctx = llama_init_from_model(model, cp);
        if (!c->ctx) {
            throw std::runtime_error("classifier: cannot create a context for layer " + std::to_string(head.layer));
        }
        llama_set_n_layer_exit(c->ctx, head.layer);
        ctxs.push_back(std::move(c));
    }
}

json server_classifier::classify(const llama_tokens & tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("classifier: empty input");
    }
    json answers = json::object();
    for (auto & c : ctxs) {
        if (tokens.size() > llama_n_ctx(c->ctx)) {
            throw std::invalid_argument("classifier: input has " + std::to_string(tokens.size()) +
                                        " tokens, the classifier context holds " + std::to_string(llama_n_ctx(c->ctx)));
        }
        std::fill(c->sum.begin(), c->sum.end(), 0.0);
        c->rows = 0;
        c->error.clear();
        llama_memory_clear(llama_get_memory(c->ctx), true);

        llama_tokens ids = tokens;
        const int32_t ret = llama_decode(c->ctx, llama_batch_get_one(ids.data(), (int32_t) ids.size()));
        if (!c->error.empty()) {
            throw std::runtime_error(c->error);
        }
        if (ret != 0) {
            throw std::runtime_error("classifier: decode failed");
        }
        llama_synchronize(c->ctx);
        if (c->rows != (int64_t) tokens.size()) {
            throw std::runtime_error("classifier: read " + std::to_string(c->rows) + " rows for " +
                                     std::to_string(tokens.size()) + " tokens");
        }

        for (const auto & head : heads) {
            if (head.layer != c->layer) {
                continue;
            }
            double z = head.bias;
            for (int32_t i = 0; i < n_embd; i++) {
                z += head.weight[i] * (c->sum[i] / (double) c->rows);
            }
            const double p = 1.0 / (1.0 + std::exp(-z));
            answers[head.question_id] = json {
                {"type", head.type},
                {head.type, p},
            };
        }
    }
    return answers;
}

json server_classifier::info() const {
    json out = json::array();
    for (const auto & head : heads) {
        out.push_back({
            {"path",        head.path},
            {"question_id", head.question_id},
            {"type",        head.type},
            {"layer",       head.layer},
        });
    }
    return out;
}
