#include "server-classifier.h"

#include "../../src/llama-ext.h" // staging API: llama_set_n_layer_exit

#include <algorithm>
#include <stdexcept>

// one early-exit context and the capture of its tapped layers
struct server_classifier_ctx {
    llama_context *           ctx = nullptr;
    common_classifier_capture cap;

    ~server_classifier_ctx() {
        llama_free(ctx);
    }
};

server_classifier::server_classifier()  = default;
server_classifier::~server_classifier() = default;

bool server_classifier::features_enabled() const {
    return feat != nullptr;
}

static std::unique_ptr<server_classifier_ctx> make_ctx(llama_model * model, int32_t n_ctx, int32_t n_threads,
                                                       int32_t n_embd, const std::vector<int32_t> & layers, int32_t exit) {
    auto c = std::make_unique<server_classifier_ctx>();
    c->cap.set_layers(n_embd, layers);
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
    cp.cb_eval           = common_classifier_capture::cb_eval;
    cp.cb_eval_user_data = &c->cap;
    c->ctx = llama_init_from_model(model, cp);
    if (!c->ctx) {
        throw std::runtime_error("classifier: cannot create a context");
    }
    llama_set_n_layer_exit(c->ctx, exit);
    return c;
}

void server_classifier::init(llama_model * model, const std::vector<std::string> & paths, int32_t n_ctx, int32_t n_threads, bool features) {
    this->model    = model;
    this->n_embd   = llama_model_n_embd(model);
    this->n_ctx    = n_ctx;
    this->n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    this->n_layer_ = llama_model_n_layer(model);
    this->n_threads = n_threads;

    for (const auto & path : paths) {
        common_classifier_head head;
        std::string err;
        if (!common_classifier_head_load(path, head, err)) {
            throw std::runtime_error("classifier: " + err);
        }
        if (head.n_embd != n_embd) {
            throw std::runtime_error("classifier: " + path + ": the head is for n_embd = " + std::to_string(head.n_embd) +
                                     ", the model has n_embd = " + std::to_string(n_embd));
        }
        if (head.layer < 1 || head.layer > n_layer_) {
            throw std::runtime_error("classifier: " + path + ": layer " + std::to_string(head.layer) +
                                     " is outside 1.." + std::to_string(n_layer_));
        }
        for (const auto & other : heads) {
            if (other.question_id == head.question_id) {
                throw std::runtime_error("classifier: " + path + ": question_id '" + head.question_id + "' is already used by " + other.path);
            }
        }
        if (!head.model.empty()) {
            const std::string here = common_classifier_model_desc(model);
            if (head.model != here) {
                fprintf(stderr, "classifier: warning: %s was fitted on '%s', this model is '%s'; its answers may be off\n",
                        path.c_str(), head.model.c_str(), here.c_str());
            }
        }
        heads.push_back(std::move(head));
    }

    std::vector<int32_t> layers;
    for (const auto & head : heads) {
        if (std::find(layers.begin(), layers.end(), head.layer) == layers.end()) {
            layers.push_back(head.layer);
        }
    }
    for (int32_t L : layers) {
        ctxs.push_back(make_ctx(model, n_ctx, n_threads, n_embd, {L}, L));
    }
    if (features) {
        feat = make_ctx(model, n_ctx, n_threads, n_embd, {}, 0);
    }
}

void server_classifier::check_tokens(const llama_tokens & tokens) const {
    if (tokens.empty()) {
        throw std::invalid_argument("classifier: empty input");
    }
    if ((int64_t) tokens.size() > n_ctx) {
        throw std::invalid_argument("classifier: input has " + std::to_string(tokens.size()) +
                                    " tokens, at most " + std::to_string(n_ctx) + " are allowed (--classifier-ctx)");
    }
    for (const llama_token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            throw std::invalid_argument("classifier: token id " + std::to_string(t) + " is outside the vocabulary");
        }
    }
}

void server_classifier::decode(server_classifier_ctx & c, const llama_tokens & tokens) {
    c.cap.reset();
    llama_memory_clear(llama_get_memory(c.ctx), true);
    llama_tokens ids = tokens;
    const int32_t ret = llama_decode(c.ctx, llama_batch_get_one(ids.data(), (int32_t) ids.size()));
    if (!c.cap.error.empty()) {
        throw std::runtime_error(c.cap.error);
    }
    if (ret != 0) {
        throw std::runtime_error("classifier: decode failed");
    }
    llama_synchronize(c.ctx);
    for (size_t i = 0; i < c.cap.layers.size(); i++) {
        if (c.cap.rows[i] != (int64_t) tokens.size()) {
            throw std::runtime_error("classifier: read " + std::to_string(c.cap.rows[i]) + " rows of layer " +
                                     std::to_string(c.cap.layers[i]) + " for " + std::to_string(tokens.size()) + " tokens");
        }
    }
}

static json server_classifier_answer_body(const common_classifier_head & head, const std::vector<double> & p);

json server_classifier_answer(const common_classifier_head & head, const std::vector<double> & p) {
    json j = server_classifier_answer_body(head, p);
    if (!head.version.empty()) {
        j["version"] = head.version;   // which fit produced the answer (cache keys, stale-head detection)
    }
    return j;
}

static json server_classifier_answer_body(const common_classifier_head & head, const std::vector<double> & p) {
    switch (head.type) {
        case COMMON_CLASSIFIER_NOUL:
            return json {{"type", "noul"}, {"noul", p[1]}};
        case COMMON_CLASSIFIER_CHOICE: {
            const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
            json probs = json::object();
            for (size_t k = 0; k < p.size(); k++) {
                probs[head.options[k]] = p[k];
            }
            return json {{"type", "choice"}, {"choice", head.options[best]}, {"confidence", p[best]}, {"probabilities", probs}};
        }
        case COMMON_CLASSIFIER_SCORE: {
            double expected = 0.0;
            json probs  = json::object();
            json legend = json::object();
            for (size_t k = 0; k < p.size(); k++) {
                expected += (double) k * p[k];
                probs[std::to_string(k)]  = p[k];
                legend[std::to_string(k)] = head.options[k];
            }
            // "score" is the expected level (a float); "level" is the most likely level (an integer) and "label"
            // its description, for callers that need a whole level
            const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
            return json {{"type", "score"}, {"score", expected}, {"level", (int) best}, {"label", head.options[best]},
                         {"confidence", p[best]}, {"legend", legend}, {"probabilities", probs}};
        }
    }
    return json();
}

json server_classifier::classify(const llama_tokens & tokens) {
    check_tokens(tokens);
    json answers = json::object();
    std::vector<float> x;
    for (auto & c : ctxs) {
        decode(*c, tokens);
        for (const auto & head : heads) {
            if (head.layer != c->cap.layers[0]) {
                continue;
            }
            c->cap.get(head.layer, head.pooling, x);
            answers[head.question_id] = server_classifier_answer(head, common_classifier_probs(head, x.data()));
        }
    }
    return answers;
}

std::vector<float> server_classifier::features_raw(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                                                  const std::vector<common_classifier_pooling> & poolings) {
    if (!feat) {
        throw std::invalid_argument("features are disabled, start the server with --features");
    }
    check_tokens(tokens);
    if (layers.empty() || poolings.empty()) {
        throw std::invalid_argument("features: no layers or poolings requested");
    }
    for (int32_t L : layers) {
        if (L < 1 || L > n_layer_) {
            throw std::invalid_argument("features: layer " + std::to_string(L) + " is outside 1.." + std::to_string(n_layer_));
        }
    }
    feat->cap.set_layers(n_embd, layers);
    // compute only as deep as the deepest requested layer (the memory is cleared for every input anyway)
    llama_set_n_layer_exit(feat->ctx, *std::max_element(layers.begin(), layers.end()));
    decode(*feat, tokens);
    std::vector<float> out;
    out.reserve(layers.size() * poolings.size() * n_embd);
    std::vector<float> x;
    for (int32_t L : layers) {
        for (auto pool : poolings) {
            feat->cap.get(L, pool, x);
            out.insert(out.end(), x.begin(), x.end());
        }
    }
    return out;
}

void server_classifier::add_head(const common_classifier_head & head_in) {
    common_classifier_head head = head_in;
    if (head.model.empty() && model) {
        head.model = common_classifier_model_desc(model);   // live-trained heads: the model they were fitted on
    }
    const std::string err = head.validate();
    if (!err.empty()) {
        throw std::invalid_argument("classifier: " + err);
    }
    if (head.n_embd != n_embd || head.layer < 1 || head.layer > n_layer_) {
        throw std::invalid_argument("classifier: the head does not fit the model");
    }
    bool have_ctx = false;
    for (const auto & c : ctxs) {
        have_ctx = have_ctx || c->cap.layers[0] == head.layer;
    }
    if (!have_ctx) {
        ctxs.push_back(make_ctx(model, n_ctx, n_threads, n_embd, {head.layer}, head.layer));
    }
    bool replaced = false;
    for (auto & h : heads) {
        if (h.question_id == head.question_id) {
            h = head;
            replaced = true;
        }
    }
    if (!replaced) {
        heads.push_back(head);
    }
    // a replaced head may have left its old layer without heads
    ctxs.erase(std::remove_if(ctxs.begin(), ctxs.end(), [&](const std::unique_ptr<server_classifier_ctx> & c) {
        return std::none_of(heads.begin(), heads.end(), [&](const common_classifier_head & h) { return h.layer == c->cap.layers[0]; });
    }), ctxs.end());
}

json server_classifier::features(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                                 const std::vector<common_classifier_pooling> & poolings) {
    const std::vector<float> raw = features_raw(tokens, layers, poolings);
    json out = json::object();
    size_t off = 0;
    for (int32_t L : layers) {
        json per = json::object();
        for (auto pool : poolings) {
            per[common_classifier_pooling_name(pool)] = std::vector<float>(raw.begin() + off, raw.begin() + off + n_embd);
            off += n_embd;
        }
        out[std::to_string(L)] = per;
    }
    return out;
}

json server_classifier::info() const {
    json out = json::array();
    for (const auto & head : heads) {
        json h = {
            {"path",        head.path},
            {"question_id", head.question_id},
            {"type",        common_classifier_type_name(head.type)},
            {"pooling",     common_classifier_pooling_name(head.pooling)},
            {"layer",       head.layer},
        };
        if (!head.version.empty()) h["version"] = head.version;
        if (!head.model.empty())   h["model"]   = head.model;
        if (!head.options.empty()) {
            h["options"] = head.options;
        }
        out.push_back(h);
    }
    return out;
}
