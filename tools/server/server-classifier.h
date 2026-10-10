#pragma once

// Classifier heads that share the loaded model with generation (llama_set_n_layer_exit), and pooled
// hidden-state features (POST /features).
//
// Heads (--classifier head.gguf, format in common/classifier.h) answer noul, choice and score questions
// from the residual after `layer` layers, mean- or last-token-pooled. Heads that read the same layer share
// one early-exit context. With --features, one more context returns pooled residuals of any layers.
// Heads are written by llama-classifier (tools/classifier) or trained live (POST /classify/train).

#include "llama.h"
#include "classifier.h"
#include "server-common.h"

#include <memory>
#include <string>
#include <vector>

struct server_classifier_ctx;

// most inputs of one /classify request decoded together (one sequence each)
#define SERVER_CLASSIFIER_MAX_SEQ 64

struct server_classifier {
    server_classifier();
    ~server_classifier();

    // load the heads and create one early-exit context per layer, plus the features context if asked for;
    // throws on error
    void init(llama_model * model, const std::vector<std::string> & paths, int32_t n_ctx, int32_t n_threads, bool features);

    bool enabled()          const { return !heads.empty(); }
    bool features_enabled() const;

    // typed-decision answers for one input: {"<question_id>": {"type": ..., ...}, ...}
    json classify(const llama_tokens & tokens);

    // the answers for several inputs, decoded in one batch when they fit the context together (at most
    // SERVER_CLASSIFIER_MAX_SEQ inputs, plain-attention models); otherwise one by one
    std::vector<json> classify_batch(const std::vector<llama_tokens> & items);

    // pooled residuals: {"<layer>": {"mean": [...], "last": [...]}} for the requested layers and poolings
    json features(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                  const std::vector<common_classifier_pooling> & poolings);

    // the same as float arrays: layers x poolings x n_embd
    std::vector<float> features_raw(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                                    const std::vector<common_classifier_pooling> & poolings);

    // register a head (replaces a head with the same question_id); creates its layer's context if needed
    void add_head(const common_classifier_head & head);

    json info() const;

    std::vector<common_classifier_head> heads;

    int32_t n_layer() const { return n_layer_; }

private:
    void check_tokens(const llama_tokens & tokens) const;
    void decode(server_classifier_ctx & c, const llama_tokens & tokens);
    void decode_batch(server_classifier_ctx & c, const std::vector<llama_tokens> & items, int64_t n_total);

    llama_model * model    = nullptr;
    int32_t       n_embd   = 0;
    int32_t       n_ctx    = 0; // requested maximum input length (the context itself is padded to a multiple of 256)
    int32_t       n_vocab  = 0;
    int32_t       n_layer_ = 0;
    int32_t       n_threads = 1;
    int32_t       n_seq_batch = 1; // inputs per batch (1 for hybrid / recurrent models)
    std::vector<std::unique_ptr<server_classifier_ctx>> ctxs;
    std::unique_ptr<server_classifier_ctx>              feat; // --features
};

// the answer of one head in the typed-decision shape
json server_classifier_answer(const common_classifier_head & head, const std::vector<double> & probs);
