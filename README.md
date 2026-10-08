# llama.cpp-classifier

This is [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a few small patches:

- **Per-context early exit:** a classifier and text generation can share one loaded model.
- **LoRA training:** train a LoRA adapter in C/C++ on a frozen base model, including a quantized one, on CPU or GPU, and save it as a GGUF adapter that `--lora` loads.
- **Classifier heads** on the hidden states of the loaded model: yes/no (`noul`), multiple-choice (`choice`) and ordinal (`score`) questions. `llama-classifier` extracts features and trains heads. llama-server answers with them (`POST /classify`), returns pooled hidden states (`POST /features`) and trains new heads live (`POST /classify/train`).
- **llama-server LoRA training** on the loaded model between requests (`--lora-train`, `POST /lora/train`).
- **CPU + integrated GPU co-processing** of Q4_0 matrix products for prompt processing on boards whose GPU shares memory with the CPU (`GGML_CPU_COPROC_VULKAN`).
- **Bolmo / Bwen byte-level models** (arch `bolmo`): conversion, quantization and `llama-bolmo` for generation and per-byte scoring.

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
3. Train a linear head on those features (`llama-classifier`, below).
4. At run time, send items to the classifier context. The generation context's prompt prefill passes through the same layer, so a program using the C API can also classify a generation request from its own prefill at no extra cost. (llama-server's `/classify` always runs the input through the classifier context.)

The middle layers usually make the best classifier features. Pick `L` by cross-validation on your own data; `llama-classifier train` does that.

`common/classifier.h` holds the shared parts: the head file, the feature capture through `cb_eval`, and the fitting code.

## Classifier heads

A head reads the residual after `L` layers, pooled over the input (`mean`, or the `last` token), and answers one question:

| type | model | answer |
|---|---|---|
| `noul` | logistic regression, `p = sigmoid(w·x + b)` | `{"type": "noul", "noul": 0.97}` |
| `choice` | softmax over K options, `softmax(W x + b)` | `{"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"a": 0.1, "b": 0.8, "c": 0.1}}` |
| `score` | proportional odds over K ordered levels, `P(y ≤ k) = sigmoid(θ_k − w·x)` | `{"type": "score", "score": 2.4, "confidence": 0.6, "legend": {"0": "none", ...}, "probabilities": {"0": 0.02, ...}}` |

`score` is the expected level, `Σ k·p_k`. `confidence` is the probability of the most likely option or level.

**Head file** (GGUF, `general.type = classifier`):
- `classifier.layer` (u32): L
- `classifier.question_id` (str)
- `classifier.type`: `noul` | `choice` | `score` (default `noul`)
- `classifier.pooling`: `mean` | `last` (default `mean`)
- `classifier.options` ([str]): the option ids (`choice`) or level descriptions (`score`)
- `classifier.weight` F32 [n_embd, n_out] (n_out = K for `choice`, else 1)
- `classifier.bias` F32: [1] (`noul`), [K] (`choice`) or the K−1 non-decreasing thresholds θ (`score`)
- `classifier.info.*` (str): training settings and cross-validated metrics, written by the trainers

**Training** minimises the mean log-loss plus `‖w‖² / (2·C·n)`, on standardised features, with L-BFGS. That is the same objective as scikit-learn's `LogisticRegression(C)`; the stored weights are for raw features. On a 579-item, 4096-dimensional set it matches scikit-learn's probabilities within 2e-4, for `noul` and for `choice` (multinomial). `--balanced` weights the classes by inverse frequency.

```sh
# data: one JSON object per line, {"text": "...", "label": ...} or {"tokens": [...], "label": ...}
# labels: true/false (noul), an option id (choice), a level name or index 0..K-1 (score)
llama-classifier train -m model.gguf --data train.jsonl --type choice --question-id topic --out topic.gguf \
    [--options a,b,c] [--layers auto|L,L,..] [--pooling auto|mean|last] [--C 0.01,0.1,1] [--folds 5] [--balanced]
llama-classifier eval     -m model.gguf --head topic.gguf --data test.jsonl [--predictions out.jsonl]
llama-classifier features -m model.gguf --data data.jsonl --layers 12,18 --out feats    # raw f32 matrices + labels
llama-classifier fit      --features feats.L18.mean.f32 --labels feats.labels.txt --layer 18 --type noul --out head.gguf
```

`train` extracts the features of every candidate layer and both poolings in one pass. It cross-validates every layer × pooling × C setting (stratified k-fold), picks the one with the lowest held-out log-loss, and refits it on all items. `--layers auto` tries about 12 layers spread over the depth.

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
llama-server -m model.gguf --classifier head.gguf --lora-train
```

### Classifier

```sh
llama-server -m model.gguf --classifier topic.gguf [--classifier h2.gguf] [--features] [--classifier-train [--classifier-dir DIR]]
```

Heads that read the same layer share one early-exit context on the loaded model.

**Memory:** each distinct layer costs one context of `--classifier-ctx` tokens (default 4096), and so does `--features`. Its KV cache covers all layers, because early exit doesn't shrink it, plus a compute buffer of that size. Lower `--classifier-ctx` when inputs are short.

**`POST /classify`** with `{"input": "text"}`, a token array, or an array of either. One answer per head:

```json
{"model": "...", "answers": {"relevant": {"type": "noul", "noul": 0.97}, "topic": {"type": "choice", ...}}, "usage": {"input_tokens": 35, "output_tokens": 0}}
```

**`POST /features`** (`--features`): pooled hidden states, for training heads elsewhere. The body: `input` as above, `layers` (a number or an array, 1..n_layer, default the last layer), `pooling` (`"mean"`, `"last"` or both, default both). The response: `{"model": ..., "n_tokens": 35, "features": {"18": {"mean": [...], "last": [...]}}}`. The context computes only up to the deepest requested layer.

**`POST /classify/train`** (`--classifier-train`, which also turns on `--features`): fits a head on labelled inputs and adds it to `/classify` at once. A head with the same `question_id` is replaced. The body:
- `question_id`: letters, digits, `_`, `-`, `.`
- `type`: `noul` (default), `choice` or `score`; `options` as in `llama-classifier` (required for `score`)
- `data`: `[{"input": text or tokens, "label": ...}, ...]`; inputs longer than `--classifier-ctx` are cut
- `layers` (`"auto"` or numbers), `pooling` (`auto`, `mean`, `last`), `C` (a number or an array, default 0.01, 0.1, 1), `folds` (5), `seed`, `balanced`
- `save`: write `<question_id>.gguf` to `--classifier-dir`

The response has the chosen `layer`, `pooling` and `C`, the cross-validated metrics (`cv`: log-loss, accuracy, AUC for `noul`, MAE for `score`), one line per tried setting (`grid`), the saved path and the list of heads. The features are computed in the server loop one input at a time, so other requests are served in between; the cross-validation and the fit run on the HTTP thread. The result is identical to `llama-classifier train` on the same data and settings.

These endpoints also work on a native decision model; its own decision endpoint gives the same answers with them turned on.

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

## CPU + GPU co-processing (`GGML_CPU_COPROC_VULKAN`)

On a board whose integrated GPU shares memory with the CPU, the GPU can compute part of each prompt-sized Q4_0 matrix product while the CPU threads compute the rest. Decode is bound by memory bandwidth, which the two share, so only prompt processing gains.

- Build the CPU backend with `-DGGML_CPU_COPROC_VULKAN=ON` (needs the Vulkan loader and headers, `glslc`, and the CPU repack path, which is on by default). It works without the Vulkan backend.
- `GGML_CPU_COPROC=<fraction>` turns it on: the starting share of each Q4_0 weight's rows (in blocks of 128) that the GPU computes. When the repack buffer loads the weights, up to `GGML_CPU_COPROC_MAX` of the rows (default: the share + 0.15) are also copied into GPU buffers.
- The share adapts per weight: one block more when the GPU finished before the CPU, one block fewer when the CPU waited more than 3% of the product's time. This follows the CPU's speed as it throttles. `GGML_CPU_COPROC_ADAPT=0` keeps the share fixed.
- `GGML_CPU_COPROC_MIN_N` (default 32): only products with at least this many tokens use the GPU. `GGML_CPU_COPROC_DEVICE` picks the Vulkan device. `GGML_CPU_COPROC_STATS=1` prints timing totals at exit.
- The GPU kernel (`ggml/src/ggml-cpu/coproc-q4_0.comp`) uses the packed int8 dot product (Vulkan 1.3 `shaderIntegerDotProduct`) on Q4_0 weights and Q8_1 activations.

Measured with Qwen3-1.7B Q4_0, 4 threads, prompt 512 tokens, on a 4-core single-board computer whose integrated GPU uses the Mesa v3dv driver, no fan: 92.0 tok/s on the CPU alone, 104.5 tok/s with `GGML_CPU_COPROC=0.3` (+13.5%); perplexity unchanged within its error. That driver needs a fix to run compute workgroups concurrently: Mesa leaves the dispatch's maximum supergroup ID at 0, so its workgroups run one at a time and the GPU is about 4x slower.


## Bolmo / Bwen byte-level models (arch `bolmo`)

[Bolmo](https://huggingface.co/allenai/Bolmo-1B) and Bwen (Minixhofer et al., Nature 2026) turn a subword LLM into a byte-level one. The subword transformer becomes the "global" model over patches. Around it: a one-layer mLSTM encoder over bytes (byte embedding plus the embedding of the longest subword ending at each byte), a boundary predictor with one byte of lookahead, last-byte pooling into patches, and a four-layer mLSTM decoder with a 520-way head (byte, or byte with a fused patch boundary).

- `convert_hf_to_gguf.py` converts `BolmoForCausalLM` checkpoints (OLMo 2 global blocks for Bolmo-1B/7B, Qwen 3 / Llama 3 pre-norm blocks for Bwen-8B). It needs the source subword tokenizer (`original_identifier` in the config) to store the suffix table; `BOLMO_SUBWORD_TOKENIZER=<path>` points to a local copy. `llama-quantize` works as usual.
- `llama_decode` on a bolmo context runs only the global transformer: the batch carries patch embeddings, and the context returns the decoder's normalised input per patch as embeddings (`embeddings = true`, pooling none).
- `include/llama-bolmo.h` runs the whole model: `llama_bolmo_prefill` / `llama_bolmo_step` follow `BolmoForCausalLM.generate()` (the prompt's last byte is force-decoded to decide whether it ends a patch; the global model steps only when a byte ends a patch), and `llama_bolmo_score` is the teacher-forced forward.
- `llama-bolmo -m model.gguf -p "..." [-n N] [--temp T] [--show-patches]` generates; `--score-file text.txt` prints the per-byte NLL (marginalised over the boundary flag) as bits/byte; `--dump-logits` writes the teacher-forced logits and boundaries; `--mcq items.tsv out.tsv` scores multiple-choice options byte by byte, as generation would see them (`--mcq-tf`: teacher-forced, with the predictor's one byte of lookahead).
- The mLSTM cell is built from standard ggml ops, so it runs on any backend: the parallel form for a prompt, the recurrent step for generation. It drops the kernels' max-stabiliser, which is safe in float32 because the gates are soft-capped (input gate at most e^15, forget gate at most 1). One sequence at a time.

Checked against the Hugging Face implementation (transformers 4.57.3, xlstm 2.0.4, CPU fp32) on four prompts (English, x86 assembly, Python): Bolmo-1B F32 and Bwen-8B F16 GGUFs give the same patch boundaries, a 520-way softmax within KL 4e-6 at every byte (2.4e-5 with CUDA), and byte-identical greedy generations (51 to 91 bytes). Bolmo-1B Q8_0 on an RTX 5090: 660 bytes/s greedy generation.
