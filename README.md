# llama.cpp-classifier

This is [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a few small patches:

- **Per-context early exit:** a classifier and text generation can share one loaded model.
- **LoRA training:** train a LoRA adapter in C/C++ on a frozen base model, including a quantized one, on CPU or GPU, and save it as a GGUF adapter that `--lora` loads.
- **llama-server:** classifier heads (`--classifier`, `POST /classify`) and LoRA training on the loaded model between requests (`--lora-train`, `POST /lora/train`).

Everything else is unchanged upstream code. For building, models, tools and the full documentation, see the [official llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

## Early exit (classifier)

```c
#include "llama-ext.h"   // in src/

// n > 0: this context builds its graph only up to n trunk layers (then output_norm,
// pooling / LM head as usual). 0 = all layers (default).
void llama_set_n_layer_exit(struct llama_context * ctx, int32_t n);
```

The weights belong to `llama_model`, and a context owns only its KV cache and compute buffers. So one model can serve two contexts:

- **Generation context:** full depth, normal decoding.
- **Classifier context:** `embeddings = true`, `llama_set_n_layer_exit(ctx, L)`. Its graph stops after `L` layers and computes no LM head. That makes it much cheaper than a full forward pass.

To use it as a classifier:

1. Load the model once and create both contexts from it.
2. Read the residual after layer `L-1` (graph tensor `l_out-<L-1>`) with an eval callback (`cb_eval`), and mean-pool it over the item's tokens. You can also set `pooling_type = LLAMA_POOLING_TYPE_MEAN` and read the pooled embedding. That output passes through `output_norm`, so it is the normalised version of the same feature.
3. Train a linear head, e.g. logistic regression, on those features: `p = sigmoid(w·x + b)`.
4. At run time, send items to the classifier context. The generation context's prompt prefill passes through the same layer, so a program using the C API can also classify a generation request from its own prefill at no extra cost. (llama-server's `/classify` always runs the input through the classifier context.)

The middle layers usually make the best classifier features. Pick `L` by cross-validation on your own data.

## LoRA training

```sh
llama-finetune -m base.gguf -f train.txt -c 256 -b 256 -ub 256 \
    --lora-rank 8 --lora-alpha 16 [--lora-targets attn_q,attn_v] -lr 1e-4 -epochs 1 -o adapter.gguf
llama-cli -m base.gguf --lora adapter.gguf      # or llama-server --lora, llama-export-lora to merge
```

- **What gets adapted:** `--lora-rank` trains a new adapter on the attention and FFN matrices of every layer, with the base model frozen. The base can be F32 or quantized (tested: F32 and Q4_K_M), on CPU or GPU. On the GPU, quantized weights are dequantized for the backward pass.
- **Initialisation:** as PEFT does it. A is uniform in ±1/sqrt(n_in) and B = 0, so the adapter starts as a no-op. The output is scaled by alpha / rank.
- **API** (`src/llama-ext.h`):
  - `llama_adapter_lora_init_trainable()` creates the adapter.
  - `llama_set_adapters_lora()` attaches it to a context.
  - `llama_opt_init()` with `llama_opt_param_filter_lora` sets up training of only the adapter.
  - `llama_opt_epoch()` trains, and `llama_adapter_lora_save()` writes the adapter.
- **Checked against PyTorch** (Qwen2.5-0.5B in F32, same initial adapter, same 256-token windows, same optimizer):
  - With SGD, the adapter weights after two steps agree to 0.2%.
  - With AdamW, the losses agree to about 1e-4 for the first 20 steps. After that, ordinary float32 differences grow, as they also do between two PyTorch runs started 1e-6 apart.

- **Example result:** Qwen2.5-0.5B Q4_K_M base, 8k tokens of synthetic disassembly, rank 8, 1 epoch, lr 2e-4. Perplexity on held-out text from the same source fell from 6.60 to 5.43. With 2 epochs at lr 1e-3 it overfits (held-out 9.26).

It also fixes two problems in upstream training code:

- **ggml-opt (correctness):** with dynamic graphs, which `llama_opt_epoch` uses, the gradients were never reset between optimizer steps, so every step applied the sum of all gradients so far. This affected full fine-tuning too.
- **CPU `OUT_PROD` with quantized weights (speed):** each weight row was dequantized once per token. Doing it once per row and thread gives a 5.6× faster training step on a Q4_K_M base, with bit-identical results.

**Limits:**
- There's no backward pass for flash attention (it's switched off during training), for `MUL_MAT_ID` (no MoE), or for the Gated DeltaNet / SSM ops.
- The GELU family has a backward pass in this fork: GELU (tanh form), GELU-erf and GELU-quick, plus their split GEGLU forms, built from existing ops (erf via the Abramowitz–Stegun approximation, error 1.5e-7), so Gemma-family models train. Gemma 3 270M F32 matches PyTorch within 2e-4 in loss and 0.3% in the trained weights after two SGD steps, on CPU and GPU.
- Training needs `n_ubatch == n_ctx`, or the K and V projections get no gradient.

## llama-server

```sh
llama-server -m model.gguf --classifier head.gguf [--classifier head2.gguf] --lora-train
```

### Classifier

**Head format.** A head is a small GGUF file with:
- `general.type = classifier`
- `classifier.layer` (u32): the exit layer L
- `classifier.question_id`
- `classifier.type = noul`
- `classifier.weight` F32 [n_embd] and `classifier.bias` F32 [1]

The feature is the residual after layer L, mean-pooled over the input. Heads that read the same layer share one early-exit context on the loaded model.

**Memory:** each distinct layer costs one context of `--classifier-ctx` tokens (default 4096). Its KV cache covers all layers, because early exit doesn't shrink it, plus a compute buffer of that size. Lower `--classifier-ctx` when inputs are short.

**Request:** `POST /classify` with `{"input": "text"}`, a token array, or an array of either. The response:

```json
{"model": "...", "answers": {"<question_id>": {"type": "noul", "noul": 0.97}}, "usage": {"input_tokens": 35, "output_tokens": 0}}
```

### LoRA training

`POST /lora/train` starts a job on the loaded model. The body fields:
- `text` or `tokens`
- `rank`, `alpha` (default 2·rank, unlike `llama-finetune`, where the default is rank), `lr`, `epochs`, `n_ctx` (a multiple of 256), `targets`, `name`, `seed`
- `file`: a plain file name inside `--lora-train-dir` (default `<temp dir>/llama-lora-train`)
- `register`: `false` saves the adapter without adding it to the server
- `priority`:
  - `idle` (default): train only while no request is being processed
  - `shared`: also train between decode rounds

Training runs in its own context, one optimizer step per server-loop iteration.

**When the job finishes:**
- The adapter is saved to the training dir. A registered adapter is reloaded from that file after the server sleeps, so keep the file.
- It is added to `/lora-adapters` with scale 0. A request uses it with `"lora": [{"id": <adapter_id>, "scale": 1}]`.

`GET /lora/train` lists the jobs (step, losses, adapter id). `POST /lora/train/cancel` with `{"id": n}` cancels one.

With `--lora-train`, weight repacking is turned off.

Recurrent, hybrid (SSM / DeltaNet), MoE and diffusion models are rejected: some of their ops have no backward pass.

The training context (F32 KV cache, AdamW state, the backward graph) needs memory on top of the served model.
