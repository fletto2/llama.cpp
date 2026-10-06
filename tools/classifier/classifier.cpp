// llama-classifier: features, training and evaluation of classifier heads (common/classifier.h) on a model's
// hidden states. The heads it writes are served by llama-server (--classifier head.gguf, POST /classify).
//
//   llama-classifier train    -m model.gguf --data train.jsonl --type noul|choice|score --out head.gguf
//                             [--question-id ID] [--options a,b,c] [--layers auto|L,L,..] [--pooling auto|mean|last]
//                             [--C 0.01,0.1,1] [--folds 5] [--balanced] [--seed N]
//   llama-classifier eval     -m model.gguf --head head.gguf --data test.jsonl [--predictions out.jsonl]
//   llama-classifier features -m model.gguf --data data.jsonl --layers L,L,..|all --out prefix [--pooling mean|last]
//   llama-classifier fit      --features prefix.L12.mean.f32 --labels prefix.labels.txt --layer 12 --type ... --out head.gguf
//
// Data (JSONL), one item per line: {"text": "...", "label": ...} or {"tokens": [ids], "label": ...}.
// Labels: noul true/false (also 1/0, "yes"/"no"); choice an option id (the options are --options, or the
// sorted distinct labels); score a level index 0..K-1, or a level name from --options.
// Common: -ngl N (default 0), -t N threads, -c N max tokens per item (default 512; longer items are cut).
// Features are taken exactly as llama-server's /classify takes them: text tokenized with BOS and special
// tokens, all tokens outputs (embeddings mode), the residual l_out-<L-1> pooled over the item.

#include "arg.h"
#include "classifier.h"
#include "common.h"
#include "llama.h"

#include "../../src/llama-ext.h" // staging API: llama_set_n_layer_exit

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

struct item {
    std::string  text;
    llama_tokens tokens;
    json         label;
};

struct args_t {
    std::string cmd, model, data, out, head, type = "noul", question_id, options, layers = "auto", pooling = "auto",
                Cs = "0.01,0.1,1", features, labels, predictions;
    int32_t ngl = 0, threads = 0, n_ctx = 512, folds = 5, layer = 0;
    uint32_t seed = 42;
    bool balanced = false;
};

[[noreturn]] static void fatal(const std::string & msg) {
    fprintf(stderr, "llama-classifier: %s\n", msg.c_str());
    exit(1);
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, sep)) {
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

static void usage() {
    fprintf(stderr,
        "usage:\n"
        "  llama-classifier train    -m model.gguf --data train.jsonl --type noul|choice|score --out head.gguf\n"
        "                            [--question-id ID] [--options a,b,c] [--layers auto|L,L,..] [--pooling auto|mean|last]\n"
        "                            [--C 0.01,0.1,1] [--folds 5] [--balanced] [--seed N]\n"
        "  llama-classifier eval     -m model.gguf --head head.gguf --data test.jsonl [--predictions out.jsonl]\n"
        "  llama-classifier features -m model.gguf --data data.jsonl --layers L,L,..|all --out prefix [--pooling mean|last]\n"
        "  llama-classifier fit      --features x.f32 --labels labels.txt --layer L --type ... --out head.gguf\n"
        "common: -ngl N (0), -t N (threads), -c N (max tokens per item, 512)\n"
        "data: JSONL lines {\"text\": ..., \"label\": ...} or {\"tokens\": [...], \"label\": ...}\n");
    exit(1);
}

static args_t parse(int argc, char ** argv) {
    if (argc < 2) usage();
    args_t a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) fatal("missing value for " + k); return argv[++i]; };
        if      (k == "-m" || k == "--model")   a.model = v();
        else if (k == "--data")                 a.data = v();
        else if (k == "--out" || k == "-o")     a.out = v();
        else if (k == "--head")                 a.head = v();
        else if (k == "--type")                 a.type = v();
        else if (k == "--question-id")          a.question_id = v();
        else if (k == "--options")              a.options = v();
        else if (k == "--layers")               a.layers = v();
        else if (k == "--layer")                a.layer = std::stoi(v());
        else if (k == "--pooling")              a.pooling = v();
        else if (k == "--C")                    a.Cs = v();
        else if (k == "--folds")                a.folds = std::stoi(v());
        else if (k == "--balanced")             a.balanced = true;
        else if (k == "--seed")                 a.seed = (uint32_t) std::stoul(v());
        else if (k == "-ngl" || k == "--n-gpu-layers") a.ngl = std::stoi(v());
        else if (k == "-t" || k == "--threads") a.threads = std::stoi(v());
        else if (k == "-c" || k == "--ctx")     a.n_ctx = std::stoi(v());
        else if (k == "--features")             a.features = v();
        else if (k == "--labels")               a.labels = v();
        else if (k == "--predictions")          a.predictions = v();
        else fatal("unknown option " + k);
    }
    if (a.threads <= 0) a.threads = std::max(1, (int) std::thread::hardware_concurrency());
    return a;
}

static std::vector<item> read_items(const std::string & path, bool need_label) {
    std::ifstream f(path);
    if (!f) fatal("cannot read " + path);
    std::vector<item> items;
    std::string line;
    int64_t ln = 0;
    while (std::getline(f, line)) {
        ln++;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        json j;
        try { j = json::parse(line); } catch (const std::exception & e) { fatal(path + ":" + std::to_string(ln) + ": " + e.what()); }
        item it;
        if (j.contains("tokens")) it.tokens = j.at("tokens").get<llama_tokens>();
        else if (j.contains("text")) it.text = j.at("text").get<std::string>();
        else fatal(path + ":" + std::to_string(ln) + ": needs \"text\" or \"tokens\"");
        if (j.contains("label")) it.label = j.at("label");
        else if (need_label) fatal(path + ":" + std::to_string(ln) + ": needs \"label\"");
        items.push_back(std::move(it));
    }
    if (items.empty()) fatal(path + ": no items");
    return items;
}

// label -> class index (common_classifier_encode_labels); fills options for choice when not given
static std::vector<int32_t> encode_labels(const std::vector<item> & items, common_classifier_type type, std::vector<std::string> & options) {
    std::vector<std::string> labels;
    for (const auto & it : items) {
        const json & l = it.label;
        labels.push_back(l.is_string() ? l.get<std::string>() : l.dump());
    }
    std::vector<int32_t> y;
    std::string err;
    if (!common_classifier_encode_labels(labels, type, options, y, err)) fatal(err + (type == COMMON_CLASSIFIER_SCORE ? " (score: --options with one name per level)" : ""));
    return y;
}

// ---- feature extraction --------------------------------------------------------------------------

struct extractor {
    llama_model *             model = nullptr;
    llama_context *           ctx   = nullptr;
    const llama_vocab *       vocab = nullptr;
    common_classifier_capture cap;
    int32_t                   n_ctx = 0;
    int64_t                   n_cut = 0;

    extractor(const args_t & a, const std::vector<int32_t> & layers) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = a.ngl;
        model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) fatal("cannot load " + a.model);
        vocab = llama_model_get_vocab(model);
        const int32_t n_layer = llama_model_n_layer(model);
        for (int32_t L : layers) if (L < 1 || L > n_layer) fatal("layer " + std::to_string(L) + " is outside 1.." + std::to_string(n_layer));
        cap.set_layers(llama_model_n_embd(model), layers);
        n_ctx = a.n_ctx;
        llama_context_params cp = llama_context_default_params();
        cp.embeddings        = true;   // every token an output: all rows reach the tapped layers
        cp.pooling_type      = LLAMA_POOLING_TYPE_MEAN;
        cp.n_ctx             = n_ctx;
        cp.n_batch           = n_ctx;
        cp.n_ubatch          = n_ctx;
        cp.n_seq_max         = 1;
        cp.n_threads         = a.threads;
        cp.n_threads_batch   = a.threads;
        cp.cb_eval           = common_classifier_capture::cb_eval;
        cp.cb_eval_user_data = &cap;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) fatal("cannot create a context");
        llama_set_n_layer_exit(ctx, *std::max_element(layers.begin(), layers.end()));
    }
    ~extractor() { llama_free(ctx); llama_model_free(model); }

    llama_tokens tokens_of(const item & it) {
        llama_tokens t = it.tokens.empty() ? common_tokenize(vocab, it.text, true, true) : it.tokens;
        if ((int32_t) t.size() > n_ctx) { t.resize(n_ctx); n_cut++; }
        return t;
    }

    // run one item; afterwards cap.get(layer, pooling, out) returns its features
    void run(const item & it) {
        llama_tokens t = tokens_of(it);
        if (t.empty()) fatal("an item has no tokens");
        cap.reset();
        llama_memory_clear(llama_get_memory(ctx), true);
        const int32_t ret = llama_decode(ctx, llama_batch_get_one(t.data(), (int32_t) t.size()));
        if (!cap.error.empty()) fatal(cap.error);
        if (ret != 0) fatal("decode failed");
        llama_synchronize(ctx);
    }
};

// features of all items for the given layers and poolings: feats[layer index][pooling] = n x d
static std::vector<std::vector<std::vector<float>>> extract(extractor & ex, const std::vector<item> & items,
                                                            const std::vector<common_classifier_pooling> & pools) {
    const size_t nl = ex.cap.layers.size();
    std::vector<std::vector<std::vector<float>>> feats(nl, std::vector<std::vector<float>>(pools.size()));
    std::vector<float> v;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < items.size(); i++) {
        ex.run(items[i]);
        for (size_t l = 0; l < nl; l++) {
            for (size_t p = 0; p < pools.size(); p++) {
                ex.cap.get(ex.cap.layers[l], pools[p], v);
                feats[l][p].insert(feats[l][p].end(), v.begin(), v.end());
            }
        }
        if ((i + 1) % 50 == 0 || i + 1 == items.size()) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "\rfeatures: %zu/%zu items (%.1f items/s)", i + 1, items.size(), (i + 1) / std::max(s, 1e-9));
        }
    }
    fprintf(stderr, "\n");
    if (ex.n_cut) fprintf(stderr, "warning: %lld items were cut to -c %d tokens\n", (long long) ex.n_cut, ex.n_ctx);
    return feats;
}

static std::vector<int32_t> parse_layers(const std::string & spec, int32_t n_layer) {
    std::vector<int32_t> L;
    if (spec == "all") {
        for (int32_t l = 1; l <= n_layer; l++) L.push_back(l);
    } else if (spec == "auto") {
        L = common_classifier_auto_layers(n_layer);
    } else {
        for (const auto & s : split(spec, ',')) L.push_back(std::stoi(s));
    }
    if (L.empty()) fatal("no layers");
    return L;
}

static int32_t model_n_layer(const std::string & path) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) fatal("cannot load " + path);
    const int32_t n = llama_model_n_layer(m);
    llama_model_free(m);
    return n;
}

static common_classifier_type parse_type(const std::string & s) {
    common_classifier_type t;
    if (!common_classifier_type_from_name(s, t)) fatal("--type must be noul, choice or score");
    return t;
}

static std::string fmt_metrics(const common_classifier_metrics & m, common_classifier_type type) {
    return common_classifier_metrics_str(m, type);
}

static int cmd_train(const args_t & a) {
    if (a.model.empty() || a.data.empty() || a.out.empty()) usage();
    const common_classifier_type type = parse_type(a.type);
    auto items = read_items(a.data, true);
    std::vector<std::string> options = split(a.options, ',');
    const std::vector<int32_t> y = encode_labels(items, type, options);
    const int32_t K = type == COMMON_CLASSIFIER_NOUL ? 2 : (int32_t) options.size();
    std::vector<int64_t> count(K, 0);
    for (int32_t v : y) count[v]++;
    fprintf(stderr, "train: %zu items, type %s, %d classes:", items.size(), a.type.c_str(), K);
    for (int32_t k = 0; k < K; k++) fprintf(stderr, " %lld", (long long) count[k]);
    fprintf(stderr, "\n");
    // checked again by common_classifier_train, here to fail before the extraction pass
    for (int32_t k = 0; k < K; k++) if (count[k] < a.folds) fatal("class " + std::to_string(k) + " has fewer items than --folds");

    const std::vector<int32_t> layers = parse_layers(a.layers, model_n_layer(a.model));
    std::vector<common_classifier_pooling> pools;
    if (a.pooling == "auto" || a.pooling == "mean") pools.push_back(COMMON_CLASSIFIER_POOL_MEAN);
    if (a.pooling == "auto" || a.pooling == "last") pools.push_back(COMMON_CLASSIFIER_POOL_LAST);
    if (pools.empty()) fatal("--pooling must be auto, mean or last");
    std::vector<double> Cs;
    for (const auto & s : split(a.Cs, ',')) Cs.push_back(std::stod(s));

    extractor ex(a, layers);
    const auto feats = extract(ex, items, pools);
    const int32_t d = ex.cap.n_embd;
    const int64_t n = (int64_t) items.size();

    common_classifier_train_params tp;
    tp.fit.type = type; tp.fit.n_classes = K; tp.fit.balanced = a.balanced; tp.fit.n_threads = a.threads;
    tp.Cs = Cs; tp.folds = a.folds; tp.seed = a.seed;
    common_classifier_train_result tr;
    std::string err;
    fprintf(stderr, "cross-validation (%d folds), held-out metrics:\n", a.folds);
    if (!common_classifier_train(feats, n, d, layers, pools, y, tp, tr, err)) fatal(err);
    for (const auto & line : tr.log) fprintf(stderr, "  %s\n", line.c_str());
    const double bC = tr.C;
    const common_classifier_metrics & bm = tr.cv;
    fprintf(stderr, "chosen: layer %d, %s pooling, C %g: %s\n", tr.layer, common_classifier_pooling_name(tr.pooling), bC,
            fmt_metrics(bm, type).c_str());

    common_classifier_head h;
    h.question_id = a.question_id.empty() ? "relevant" : a.question_id;
    h.type = type; h.pooling = tr.pooling; h.layer = tr.layer; h.n_embd = d;
    h.options = options; h.weight = tr.weight; h.bias = tr.bias;
    char cvb[64];
    std::vector<std::pair<std::string, std::string>> info = {
        {"model",      a.model.substr(a.model.find_last_of('/') + 1)},
        {"n_items",    std::to_string(n)},
        {"C",          std::to_string(bC)},
        {"balanced",   a.balanced ? "true" : "false"},
        {"folds",      std::to_string(a.folds)},
        {"cv",         fmt_metrics(bm, type)},
    };
    snprintf(cvb, sizeof cvb, "%.6f", bm.log_loss); info.push_back({"cv_log_loss", cvb});
    if (!common_classifier_head_save(a.out, h, err, info)) fatal(err);
    fprintf(stderr, "head -> %s (layer %d, %s, %s, %d iterations)\n", a.out.c_str(), h.layer,
            common_classifier_type_name(h.type), common_classifier_pooling_name(h.pooling), tr.iters);
    return 0;
}

static int cmd_eval(const args_t & a) {
    if (a.model.empty() || a.head.empty() || a.data.empty()) usage();
    common_classifier_head h;
    std::string err;
    if (!common_classifier_head_load(a.head, h, err)) fatal(err);
    auto items = read_items(a.data, true);
    std::vector<std::string> options = h.options;
    const std::vector<int32_t> y = encode_labels(items, h.type, options);
    extractor ex(a, {h.layer});
    if (ex.cap.n_embd != h.n_embd) fatal("the head is for n_embd " + std::to_string(h.n_embd) + ", the model has " + std::to_string(ex.cap.n_embd));
    const int32_t K = h.n_classes();
    std::vector<double> probs;
    std::ofstream pred;
    if (!a.predictions.empty()) pred.open(a.predictions);
    std::vector<float> v;
    for (size_t i = 0; i < items.size(); i++) {
        ex.run(items[i]);
        ex.cap.get(h.layer, h.pooling, v);
        const auto p = common_classifier_probs(h, v.data());
        probs.insert(probs.end(), p.begin(), p.end());
        if (pred) pred << json({{"index", i}, {"label", items[i].label}, {"probabilities", p}}).dump() << "\n";
    }
    const auto m = common_classifier_metrics_of(probs, K, y, h.type);
    printf("%s: %zu items, %s\n", a.head.c_str(), items.size(), fmt_metrics(m, h.type).c_str());
    return 0;
}

static int cmd_features(const args_t & a) {
    if (a.model.empty() || a.data.empty() || a.out.empty()) usage();
    auto items = read_items(a.data, false);
    const std::vector<int32_t> layers = parse_layers(a.layers == "auto" ? "all" : a.layers, model_n_layer(a.model));
    std::vector<common_classifier_pooling> pools;
    if (a.pooling == "auto" || a.pooling == "mean") pools.push_back(COMMON_CLASSIFIER_POOL_MEAN);
    if (a.pooling == "auto" || a.pooling == "last") pools.push_back(COMMON_CLASSIFIER_POOL_LAST);
    extractor ex(a, layers);
    const auto feats = extract(ex, items, pools);
    for (size_t l = 0; l < layers.size(); l++) {
        for (size_t p = 0; p < pools.size(); p++) {
            const std::string fn = a.out + ".L" + std::to_string(layers[l]) + "." + common_classifier_pooling_name(pools[p]) + ".f32";
            std::ofstream f(fn, std::ios::binary);
            f.write((const char *) feats[l][p].data(), feats[l][p].size() * sizeof(float));
        }
    }
    std::ofstream fl(a.out + ".labels.txt");
    for (const auto & it : items) fl << (it.label.is_null() ? "" : (it.label.is_string() ? it.label.get<std::string>() : it.label.dump())) << "\n";
    fprintf(stderr, "features: %zu items x %d values, layers %zu x %zu poolings -> %s.L<layer>.<pooling>.f32 (+ .labels.txt)\n",
            items.size(), ex.cap.n_embd, layers.size(), pools.size(), a.out.c_str());
    return 0;
}

static int cmd_fit(const args_t & a) {
    if (a.features.empty() || a.labels.empty() || a.out.empty() || a.layer < 1) usage();
    const common_classifier_type type = parse_type(a.type);
    std::vector<item> items;
    {
        std::ifstream f(a.labels);
        std::string line;
        while (std::getline(f, line)) {
            item it;
            try { it.label = json::parse(line); } catch (...) { it.label = line; }
            items.push_back(it);
        }
    }
    std::vector<std::string> options = split(a.options, ',');
    const std::vector<int32_t> y = encode_labels(items, type, options);
    std::ifstream f(a.features, std::ios::binary | std::ios::ate);
    if (!f) fatal("cannot read " + a.features);
    const int64_t bytes = f.tellg();
    f.seekg(0);
    const int64_t n = (int64_t) items.size();
    if (bytes % (n * (int64_t) sizeof(float)) != 0) fatal("the feature file is not n_items x d floats");
    const int32_t d = (int32_t) (bytes / (n * (int64_t) sizeof(float)));
    std::vector<float> X((size_t) n * d);
    f.read((char *) X.data(), bytes);
    common_classifier_fit_params fp;
    fp.type = type; fp.n_classes = type == COMMON_CLASSIFIER_NOUL ? 2 : (int32_t) options.size();
    fp.balanced = a.balanced; fp.n_threads = a.threads; fp.C = std::stod(split(a.Cs, ',').front());
    const auto m = common_classifier_cv(X, n, d, y, fp, a.folds, a.seed);
    fprintf(stderr, "cross-validation (%d folds): %s\n", a.folds, fmt_metrics(m, type).c_str());
    const auto fit = common_classifier_fit(X, n, d, y, fp);
    common_classifier_head h;
    h.question_id = a.question_id.empty() ? "relevant" : a.question_id;
    h.type = type; h.layer = a.layer; h.n_embd = d; h.options = options; h.weight = fit.weight; h.bias = fit.bias;
    common_classifier_pooling pool = COMMON_CLASSIFIER_POOL_MEAN;
    if (a.pooling != "auto" && !common_classifier_pooling_from_name(a.pooling, pool)) fatal("--pooling must be mean or last");
    h.pooling = pool;
    std::string err;
    if (!common_classifier_head_save(a.out, h, err, {{"cv", fmt_metrics(m, type)}, {"C", std::to_string(fp.C)}})) fatal(err);
    fprintf(stderr, "head -> %s\n", a.out.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    const args_t a = parse(argc, argv);
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
    int ret = 1;
    if      (a.cmd == "train")    ret = cmd_train(a);
    else if (a.cmd == "eval")     ret = cmd_eval(a);
    else if (a.cmd == "features") ret = cmd_features(a);
    else if (a.cmd == "fit")      ret = cmd_fit(a);
    else usage();
    llama_backend_free();
    return ret;
}
