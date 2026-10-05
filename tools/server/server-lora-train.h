#pragma once

// LoRA training on the loaded model, between requests (POST /lora/train).
//
// A job trains a new adapter (llama_adapter_lora_init_trainable) in its own training context on the
// model that the server already holds, one optimizer step per iteration of the server loop, so that
// generation continues meanwhile. When it finishes, the adapter is saved to <--lora-train-dir>/<file>
// and, unless "register": false, added to the server's LoRA adapters with scale 0; requests select it
// with "lora": [{"id": <id>, "scale": 1}].
//
// Only models whose ops all have a backward pass can be trained: not recurrent / hybrid (SSM, DeltaNet)
// models, not MoE models (MUL_MAT_ID) and not diffusion models; such jobs are rejected.

#include "llama.h"
#include "ggml-opt.h"
#include "server-common.h"

#include <memory>
#include <string>
#include <vector>

struct server_lora_train_job {
    int         id     = 0;
    std::string name;
    std::string path;            // output file: <dir>/<file>; also lets a registered adapter be reloaded after sleeping
    bool        do_register = true; // add the adapter to the server's adapters (false: only save it)
    std::string status = "queued"; // queued, running, done, error, cancelled
    std::string error;

    // parameters
    int32_t     rank   = 8;
    float       alpha  = 16.0f;
    float       lr     = 1e-4f;
    int32_t     epochs = 1;
    int32_t     n_ctx  = 256;
    std::string targets;
    uint32_t    seed   = 42;
    bool        shared = false;  // "priority": "idle" (default) trains only while no request is processed,
                                 // "shared" also between the decode rounds of running requests

    // progress
    int64_t     step    = 0;
    int64_t     n_steps = 0;
    double      loss_first = -1.0;
    double      loss_last  = -1.0;
    double      loss_epoch = -1.0; // mean loss of the last finished epoch
    int64_t     t_start_us = 0;
    int64_t     t_end_us   = 0;
    int         adapter_id = -1;  // index in the server's LoRA list once done

    llama_tokens tokens;

    json to_json() const;
};

struct server_lora_train {
    explicit server_lora_train(std::string dir) : dir(std::move(dir)) {}
    ~server_lora_train();

    // validate a request and queue a job; returns the job id. Throws std::invalid_argument on bad input.
    int start(llama_model * model, const llama_vocab * vocab, const json & body, int32_t n_threads);

    bool busy() const { return job != nullptr; }

    // whether a step may run while requests are being processed
    bool shared() const { return job && job->shared; }

    // run one optimizer step of the current job. Returns the finished adapter when the job completes
    // successfully (the caller registers it), nullptr otherwise.
    llama_adapter_lora * step();

    bool cancel(int id);

    // record the server's id of the adapter of the last finished job
    void set_adapter_id(int adapter_id);

    json status() const;

private:
    void setup();
    void finish(const std::string & status, const std::string & error = "");

    std::string dir; // where adapters are written (--lora-train-dir)
    llama_model * model = nullptr;
    int32_t n_threads = 0;
    int next_id = 1;

    std::unique_ptr<server_lora_train_job> job;
    std::vector<server_lora_train_job> history;

    llama_context * ctx = nullptr;
    llama_adapter_lora * adapter = nullptr;
    ggml_opt_result_t result_train = nullptr;
    ggml_opt_result_t result_eval  = nullptr;
    ggml_opt_optimizer_params opt_pars = {};
    double epoch_loss_sum = 0.0;
    int64_t epoch_steps = 0;
};
