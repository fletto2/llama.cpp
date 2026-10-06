#pragma once

// Classifier heads on a model's hidden states (typed decisions), shared by llama-server (/classify,
// /features) and the llama-classifier tool (features, train, eval).
//
// A head reads the residual stream after `layer` layers (graph tensor l_out-<layer-1>), pooled over the
// input tokens, and answers one question of one of three types:
//   noul    yes/no:            p(true) = sigmoid(w . x + b)
//   choice  one of K options:  softmax(W x + b), W is K x n_embd
//   score   one of K ordered levels (proportional odds): P(y <= k) = sigmoid(theta_k - w . x),
//           theta non-decreasing (K-1 thresholds); the answer reports the expected level
// Head file (GGUF, general.type = "classifier"):
//   classifier.layer        u32   layer count (1..n_layer)
//   classifier.question_id  str   key of the answer
//   classifier.type         str   noul | choice | score            (default noul)
//   classifier.pooling      str   mean | last                      (default mean)
//   classifier.options      [str] choice: option ids; score: level descriptions (the legend)
//   classifier.weight       F32   [n_embd, n_out] (n_out = K for choice, else 1)
//   classifier.bias         F32   noul [1], choice [K], score [K-1] (the thresholds)
// The answers follow the typed-decision shape: {"type": "noul", "noul": p},
// {"type": "choice", "choice": id, "confidence": p, "probabilities": {...}},
// {"type": "score", "score": E[level], "confidence": p, "legend": {...}, "probabilities": {...}}.

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_classifier_type {
    COMMON_CLASSIFIER_NOUL,
    COMMON_CLASSIFIER_CHOICE,
    COMMON_CLASSIFIER_SCORE,
};

enum common_classifier_pooling {
    COMMON_CLASSIFIER_POOL_MEAN,
    COMMON_CLASSIFIER_POOL_LAST,
};

const char * common_classifier_type_name(common_classifier_type type);
bool         common_classifier_type_from_name(const std::string & name, common_classifier_type & type);
const char * common_classifier_pooling_name(common_classifier_pooling pooling);
bool         common_classifier_pooling_from_name(const std::string & name, common_classifier_pooling & pooling);

struct common_classifier_head {
    std::string               path;
    std::string               question_id = "relevant";
    common_classifier_type    type        = COMMON_CLASSIFIER_NOUL;
    common_classifier_pooling pooling     = COMMON_CLASSIFIER_POOL_MEAN;
    int32_t                   layer       = 0;
    int32_t                   n_embd      = 0;
    std::vector<std::string>  options;    // choice: option ids; score: level descriptions
    std::vector<float>        weight;     // n_out x n_embd, row-major
    std::vector<float>        bias;       // noul 1, choice K, score K-1

    int32_t n_classes() const; // noul 2 (index 1 = true), choice / score K
    int32_t n_out()     const; // rows of weight: choice K, else 1
    int32_t n_bias()    const;
    // shape and value checks; returns an empty string when the head is consistent
    std::string validate() const;
};

// load / save a head file; return false and set err on failure. `extra` key/value strings are stored
// as classifier.info.<key> (training metrics and settings).
bool common_classifier_head_load(const std::string & path, common_classifier_head & head, std::string & err);
bool common_classifier_head_save(const std::string & path, const common_classifier_head & head, std::string & err,
                                 const std::vector<std::pair<std::string, std::string>> & extra = {});

// class probabilities for one pooled feature vector of n_embd floats:
// noul -> {p(false), p(true)}, choice / score -> K values
std::vector<double> common_classifier_probs(const common_classifier_head & head, const float * x);

// Pooled hidden states of chosen layers, captured during llama_decode through the graph eval callback.
// Set cb_eval = common_classifier_capture::cb_eval and cb_eval_user_data = &capture on the context; the
// context should use embeddings mode (every token an output, so all rows reach the tapped layers) and
// decode one input per call.
struct common_classifier_capture {
    int32_t              n_embd = 0;
    std::vector<int32_t> layers;   // feature after L layers = tensor l_out-<L-1>

    std::vector<std::vector<double>> sum;  // per layer
    std::vector<std::vector<float>>  last; // per layer: the last row seen
    std::vector<int64_t>             rows; // per layer
    std::vector<float>               buf;
    std::string                      error; // set by the callback; exceptions must not cross ggml

    void set_layers(int32_t n_embd, const std::vector<int32_t> & layers);
    void reset();
    int  index_of(int32_t layer) const;
    // pooled feature of one layer; false if the layer was not captured
    bool get(int32_t layer, common_classifier_pooling pooling, std::vector<float> & out) const;

    static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
};

// ---- fitting -----------------------------------------------------------------------------------------
// X: n rows of d floats (pooled features), y: class index (noul: 1 = true, 0 = false; choice: option
// index; score: level index). The features are standardized inside the fit and the returned weights are
// for raw features. The loss is the mean (optionally class-balanced) log-loss plus (1 / (2 C n)) |w|^2,
// as in scikit-learn's LogisticRegression(C); biases and thresholds are not penalized.

struct common_classifier_fit_params {
    common_classifier_type type      = COMMON_CLASSIFIER_NOUL;
    int32_t                n_classes = 2;
    double                 C         = 1.0;
    bool                   balanced  = false;
    int32_t                max_iter  = 300;
    double                 tol       = 1e-5;   // stop when the gradient max-norm falls below tol
    int32_t                n_threads = 4;
};

struct common_classifier_fit_result {
    std::vector<float> weight; // n_out x d for raw features
    std::vector<float> bias;
    double             loss  = 0.0; // final training objective
    int32_t            iters = 0;
};

common_classifier_fit_result common_classifier_fit(const std::vector<float> & X, int64_t n, int32_t d,
                                                   const std::vector<int32_t> & y, const common_classifier_fit_params & params);

struct common_classifier_metrics {
    double log_loss = 0.0; // mean, unweighted
    double accuracy = 0.0; // argmax (noul: p > 0.5)
    double auc      = -1.0; // noul only
    double mae      = -1.0; // score only: |E[level] - true level|
    int64_t n       = 0;
};

// metrics of predicted class probabilities (n x n_classes) against y
common_classifier_metrics common_classifier_metrics_of(const std::vector<double> & probs, int32_t n_classes,
                                                      const std::vector<int32_t> & y, common_classifier_type type);

// stratified k-fold cross-validation of one setting; the probabilities of every row come from the fold
// model that did not see it
common_classifier_metrics common_classifier_cv(const std::vector<float> & X, int64_t n, int32_t d,
                                               const std::vector<int32_t> & y, const common_classifier_fit_params & params,
                                               int32_t folds, uint32_t seed);

// ---- training: label encoding and the choice of layer, pooling and C ---------------------------------

// labels as strings -> class indices. noul: true/false, yes/no or a number (non-zero = true); choice: an
// option id (options are filled from the sorted distinct labels when empty); score: a level name from
// options or the level index 0..K-1. Returns false and sets err on a bad label.
bool common_classifier_encode_labels(const std::vector<std::string> & labels, common_classifier_type type,
                                     std::vector<std::string> & options, std::vector<int32_t> & y, std::string & err);

// "auto" candidate layers: about 12 spread over the depth, skipping the first sixth and the last 2 layers
std::vector<int32_t> common_classifier_auto_layers(int32_t n_layer);

// one line per metric set, e.g. "log-loss 0.3012  AUC 0.9213  accuracy 0.8700"
std::string common_classifier_metrics_str(const common_classifier_metrics & m, common_classifier_type type);

struct common_classifier_train_params {
    common_classifier_fit_params fit;          // type, n_classes, balanced, threads (C is chosen from Cs)
    std::vector<double>          Cs    = {0.01, 0.1, 1.0};
    int32_t                      folds = 5;
    uint32_t                     seed  = 42;
};

struct common_classifier_train_result {
    int32_t                   layer   = 0;
    common_classifier_pooling pooling = COMMON_CLASSIFIER_POOL_MEAN;
    double                    C       = 0.0;
    common_classifier_metrics cv;      // held-out metrics of the chosen setting
    std::vector<float>        weight;  // refit on all items with the chosen setting
    std::vector<float>        bias;
    int32_t                   iters   = 0;
    std::vector<std::string>  log;     // one line per cross-validated setting
};

// feats[l][p]: n x d features of layers[l] with pools[p]. Every layer x pooling x C setting is
// cross-validated; the one with the lowest held-out log-loss is refit on all items. Checks that every
// class has at least `folds` items; returns false and sets err otherwise.
bool common_classifier_train(const std::vector<std::vector<std::vector<float>>> & feats, int64_t n, int32_t d,
                             const std::vector<int32_t> & layers, const std::vector<common_classifier_pooling> & pools,
                             const std::vector<int32_t> & y, const common_classifier_train_params & params,
                             common_classifier_train_result & res, std::string & err);
