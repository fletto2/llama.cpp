#include "server-lora-train.h"

#include "../../src/llama-ext.h" // staging API: LoRA training

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

json server_lora_train_job::to_json() const {
    json j = {
        {"id",         id},
        {"name",       name},
        {"status",     status},
        {"step",       step},
        {"n_steps",    n_steps},
        {"rank",       rank},
        {"alpha",      alpha},
        {"lr",         lr},
        {"epochs",     epochs},
        {"n_ctx",      n_ctx},
        {"priority",   shared ? "shared" : "idle"},
        {"register",   do_register},
        {"n_tokens",   (int64_t) tokens.size()},
        {"loss_first", loss_first},
        {"loss_last",  loss_last},
        {"loss_epoch", loss_epoch},
    };
    if (!path.empty()) {
        j["path"] = path;
    }
    if (!error.empty()) {
        j["error"] = error;
    }
    if (adapter_id >= 0) {
        j["adapter_id"] = adapter_id;
    }
    if (t_start_us > 0) {
        j["seconds"] = ((t_end_us > 0 ? t_end_us : ggml_time_us()) - t_start_us) / 1e6;
    }
    return j;
}

server_lora_train::~server_lora_train() {
    if (job) {
        finish("cancelled");
    }
}

// windows of n_ctx tokens at stride n_ctx/2, labels = the next token (as llama-finetune)
static int64_t n_windows(const server_lora_train_job & j) {
    const int64_t stride = j.n_ctx / 2;
    return ((int64_t) j.tokens.size() - j.n_ctx - 1) / stride + 1;
}

int server_lora_train::start(llama_model * model, const llama_vocab * vocab, const json & body, int32_t n_threads) {
    if (job) {
        throw std::invalid_argument("a LoRA training job is already running (id " + std::to_string(job->id) + ")");
    }
    // ops without a backward pass would abort the process inside ggml
    if (llama_model_is_recurrent(model) || llama_model_is_hybrid(model) || llama_model_is_diffusion(model) ||
        llama_model_n_expert(model) > 0) {
        throw std::invalid_argument("this model cannot be trained: recurrent, hybrid (SSM / DeltaNet), MoE and diffusion "
                                    "models use ops without a backward pass");
    }
    auto j = std::make_unique<server_lora_train_job>();
    j->name    = json_value(body, "name",    std::string());
    const std::string file = json_value(body, "file", std::string());
    if (!file.empty() && !fs_validate_filename(file)) {
        throw std::invalid_argument("\"file\" must be a plain file name; it is written to the --lora-train-dir directory");
    }
    j->do_register = json_value(body, "register", true);
    j->rank    = json_value(body, "rank",    j->rank);
    j->alpha   = json_value(body, "alpha",   (float) (2*j->rank));
    j->lr      = json_value(body, "lr",      j->lr);
    j->epochs  = json_value(body, "epochs",  j->epochs);
    j->n_ctx   = json_value(body, "n_ctx",   j->n_ctx);
    j->targets = json_value(body, "targets", std::string());
    j->seed    = json_value(body, "seed",    j->seed);
    const std::string priority = json_value(body, "priority", std::string("idle"));
    if (priority != "idle" && priority != "shared") {
        throw std::invalid_argument("priority must be \"idle\" or \"shared\"");
    }
    j->shared = priority == "shared";

    if (j->rank < 1 || j->rank > 1024 || j->epochs < 1 || !(j->lr > 0.0f) || j->alpha <= 0.0f) {
        throw std::invalid_argument("invalid rank, alpha, lr or epochs");
    }
    // llama.cpp pads the context to a multiple of 256
    if (j->n_ctx < 256 || j->n_ctx % 256 != 0) {
        throw std::invalid_argument("n_ctx must be a positive multiple of 256");
    }
    if (j->rank > limits.max_rank) {
        throw std::invalid_argument("rank " + std::to_string(j->rank) + " is over the server limit of " +
                                    std::to_string(limits.max_rank) + " (--lora-train-max-rank)");
    }
    if (j->n_ctx > limits.max_ctx) {
        throw std::invalid_argument("n_ctx " + std::to_string(j->n_ctx) + " is over the server limit of " +
                                    std::to_string(limits.max_ctx) + " (--lora-train-max-ctx)");
    }
    if (body.contains("tokens")) {
        if (body.at("tokens").is_array() && (int64_t) body.at("tokens").size() > limits.max_tokens) {
            throw std::invalid_argument("the training data has " + std::to_string(body.at("tokens").size()) +
                                        " tokens, over the server limit of " + std::to_string(limits.max_tokens) + " (--lora-train-max-tokens)");
        }
        j->tokens = body.at("tokens").get<llama_tokens>();
    } else if (body.contains("text")) {
        const std::string text = body.at("text").get<std::string>();
        if ((int64_t) text.size() > 16 * limits.max_tokens) {
            throw std::invalid_argument("the training text is over 16 bytes x --lora-train-max-tokens");
        }
        j->tokens = common_tokenize(vocab, text, /*add_special*/ true, /*parse_special*/ false);
    } else {
        throw std::invalid_argument("\"text\" or \"tokens\" is required");
    }
    if ((int64_t) j->tokens.size() > limits.max_tokens) {
        throw std::invalid_argument("the training data has " + std::to_string(j->tokens.size()) +
                                    " tokens, over the server limit of " + std::to_string(limits.max_tokens) + " (--lora-train-max-tokens)");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (const llama_token t : j->tokens) {
        if (t < 0 || t >= n_vocab) {
            throw std::invalid_argument("token id " + std::to_string(t) + " is outside the vocabulary (0.." + std::to_string(n_vocab - 1) + ")");
        }
    }
    if ((int64_t) j->tokens.size() < j->n_ctx + 1) {
        throw std::invalid_argument("the training data has " + std::to_string(j->tokens.size()) +
                                    " tokens, at least n_ctx + 1 = " + std::to_string(j->n_ctx + 1) + " are needed");
    }
    j->n_steps = n_windows(*j) * j->epochs;
    if (j->n_steps > limits.max_steps) {
        throw std::invalid_argument("the job needs " + std::to_string(j->n_steps) + " optimizer steps (" +
                                    std::to_string(n_windows(*j)) + " windows x " + std::to_string(j->epochs) +
                                    " epochs), over the server limit of " + std::to_string(limits.max_steps) + " (--lora-train-max-steps)");
    }
    j->id = next_id++;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    j->path = (std::filesystem::path(dir) / (file.empty()
        ? "lora-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" + std::to_string(j->id) + ".gguf"
        : file)).string();

    this->model     = model;
    this->n_threads = n_threads;
    job = std::move(j);
    return job->id;
}

void server_lora_train::setup() {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = job->n_ctx;
    cp.n_batch         = job->n_ctx;
    cp.n_ubatch        = job->n_ctx; // the K and V projections only get gradients when n_ubatch == n_ctx
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.type_k          = GGML_TYPE_F32; // OUT_PROD has no F16 support
    cp.type_v          = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // FLASH_ATTN_EXT has no backward pass
    ctx = llama_init_from_model(model, cp);
    if (!ctx || llama_n_ctx(ctx) != (uint32_t) job->n_ctx) {
        throw std::runtime_error("cannot create the training context");
    }

    const llama_adapter_lora_train_params lp = {
        /*rank    =*/ job->rank,
        /*alpha   =*/ job->alpha,
        /*targets =*/ job->targets.empty() ? nullptr : job->targets.c_str(),
        /*seed    =*/ job->seed,
    };
    adapter = llama_adapter_lora_init_trainable(model, lp);
    if (!adapter) {
        throw std::runtime_error("cannot create the adapter (check targets)");
    }
    float scale = 1.0f;
    llama_set_adapters_lora(ctx, &adapter, 1, &scale);

    opt_pars = ggml_opt_get_default_optimizer_params(nullptr);
    opt_pars.adamw.alpha = job->lr;
    opt_pars.sgd.alpha   = job->lr;
    llama_opt_params lop = {
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ llama_opt_param_filter_lora,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/ &opt_pars,
        /*optimizer_type  =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
    llama_opt_init(ctx, model, lop);

    result_train = ggml_opt_result_init();
    result_eval  = ggml_opt_result_init();
    epoch_loss_sum = 0.0;
    epoch_steps    = 0;
    job->status     = "running";
    job->t_start_us = ggml_time_us();
}

llama_adapter_lora * server_lora_train::step() {
    if (!job) {
        return nullptr;
    }
    try {
        if (job->status == "queued") {
            setup();
        }

        const int64_t nw     = n_windows(*job);
        const int64_t w      = job->step % nw;
        const int64_t stride = job->n_ctx / 2;

        ggml_opt_dataset_t ds = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, job->n_ctx, job->n_ctx, 1, 1);
        llama_token * data   = (llama_token *) ggml_opt_dataset_data(ds)->data;
        llama_token * labels = (llama_token *) ggml_opt_dataset_labels(ds)->data;
        for (int32_t i = 0; i < job->n_ctx; i++) {
            data  [i] = job->tokens[w*stride + i];
            labels[i] = job->tokens[w*stride + i + 1];
        }
        ggml_opt_result_reset(result_train);
        llama_opt_epoch(ctx, ds, result_train, result_eval, /*idata_split*/ 1, nullptr, nullptr);
        ggml_opt_dataset_free(ds);

        double loss, unc;
        ggml_opt_result_loss(result_train, &loss, &unc);
        if (!std::isfinite(loss)) {
            throw std::runtime_error("the loss is not finite (lr too high?)");
        }
        job->step++;
        job->loss_last = loss;
        if (job->loss_first < 0.0) {
            job->loss_first = loss;
        }
        epoch_loss_sum += loss;
        if (++epoch_steps == nw) {
            job->loss_epoch = epoch_loss_sum / epoch_steps;
            epoch_loss_sum = 0.0;
            epoch_steps    = 0;
        }

        if (job->step < job->n_steps) {
            return nullptr;
        }

        if (!llama_adapter_lora_save(adapter, job->path.c_str())) {
            throw std::runtime_error("cannot write " + job->path);
        }
        llama_adapter_lora * done = adapter;
        adapter = nullptr;
        const bool do_register = job->do_register;
        finish("done"); // frees the training context, which clears the adapter's parameter flags
        if (!do_register) {
            llama_adapter_lora_free(done); // saved only
            return nullptr;
        }
        return done; // now owned by the server's adapter list
    } catch (const std::exception & e) {
        finish("error", e.what());
        return nullptr;
    }
}

void server_lora_train::finish(const std::string & status, const std::string & error) {
    if (result_train) { ggml_opt_result_free(result_train); result_train = nullptr; }
    if (result_eval)  { ggml_opt_result_free(result_eval);  result_eval  = nullptr; }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (adapter) {
        llama_adapter_lora_free(adapter);
        adapter = nullptr;
    }
    job->status   = status;
    job->error    = error;
    job->t_end_us = ggml_time_us();
    job->tokens.clear();
    job->tokens.shrink_to_fit();
    history.push_back(std::move(*job));
    job.reset();
}

void server_lora_train::set_adapter_id(int adapter_id) {
    if (!history.empty()) {
        history.back().adapter_id = adapter_id;
    }
}

bool server_lora_train::cancel(int id) {
    if (!job || job->id != id) {
        return false;
    }
    finish("cancelled");
    return true;
}

json server_lora_train::status() const {
    json out = json::array();
    for (const auto & j : history) {
        out.push_back(j.to_json());
    }
    if (job) {
        out.push_back(job->to_json());
    }
    return out;
}
