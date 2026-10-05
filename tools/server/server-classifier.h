#pragma once

// Classifier heads that share the loaded model with generation (llama_set_n_layer_exit).
//
// A head is a small GGUF file (general.type = "classifier"):
//   classifier.layer        u32  the feature is the residual after this many layers (graph tensor l_out-<layer-1>),
//                                mean-pooled over the input tokens
//   classifier.question_id  str  key of the answer in the response
//   classifier.type         str  "noul" (yes/no probability)
//   classifier.weight       F32 [n_embd], classifier.bias F32 [1]: p = sigmoid(weight . feature + bias)
// Heads that read the same layer share one early-exit context.

#include "llama.h"
#include "server-common.h"

#include <memory>
#include <string>
#include <vector>

struct server_classifier_head {
    std::string        path;
    std::string        question_id;
    std::string        type;
    int32_t            layer = 0;
    std::vector<float> weight;
    float              bias  = 0.0f;
};

struct server_classifier_ctx;

struct server_classifier {
    server_classifier();
    ~server_classifier();

    // load the heads and create one early-exit context per layer; throws on error
    void init(llama_model * model, const std::vector<std::string> & paths, int32_t n_ctx, int32_t n_threads);

    bool enabled() const { return !heads.empty(); }

    // typed-decision answers for one input: {"<question_id>": {"type": "noul", "noul": p}, ...}
    json classify(const llama_tokens & tokens);

    json info() const;

    std::vector<server_classifier_head> heads;

private:
    llama_model * model = nullptr;
    int32_t n_embd  = 0;
    int32_t n_ctx   = 0; // requested maximum input length (the context itself is padded to a multiple of 256)
    int32_t n_vocab = 0;
    std::vector<std::unique_ptr<server_classifier_ctx>> ctxs;
};
