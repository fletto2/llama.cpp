# llama.cpp-classifier

This is [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a few small patches:

- **Per-context early exit:** a classifier and text generation can share one loaded model.
- **LoRA training:** train a LoRA adapter in C/C++ on a frozen base model, including a quantized one, and save it as a GGUF adapter that `--lora` loads.

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
4. At run time, send items to the classifier context. The generation context's prompt prefill passes through the same layer, so a generation request can also be classified at no extra cost.

The middle layers usually make the best classifier features. Pick `L` by cross-validation on your own data.

## LoRA training

```sh
llama-finetune -m base.gguf -f train.txt -c 256 -b 256 -ub 256 \
    --lora-rank 8 --lora-alpha 16 [--lora-targets attn_q,attn_v] -lr 1e-4 -epochs 1 -o adapter.gguf
llama-cli -m base.gguf --lora adapter.gguf      # or llama-server --lora, llama-export-lora to merge
```

- **What gets adapted:** `--lora-rank` trains a new adapter on the attention and FFN matrices of every layer, with the base model frozen. The base can be F32 or quantized (tested: F32 and Q4_K_M); the backward pass reads the quantized weights directly, on the CPU.
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
- On CUDA, `OUT_PROD` supports F32 weights only.
- Training needs `n_ubatch == n_ctx`, or the K and V projections get no gradient.
