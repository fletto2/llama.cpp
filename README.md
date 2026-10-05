# llama.cpp-classifier

This is [llama.cpp](https://github.com/ggml-org/llama.cpp) plus one small patch: **per-context early exit**. With it, a classifier and text generation can share one loaded model. Everything else is unchanged upstream code. For building, models, tools and the full documentation, see the [official llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

## What the patch adds

```c
#include "llama-ext.h"   // in src/

// n > 0: this context builds its graph only up to n trunk layers (then output_norm,
// pooling / LM head as usual). 0 = all layers (default).
void llama_set_n_layer_exit(struct llama_context * ctx, int32_t n);
```

The weights belong to `llama_model`, and a context owns only its KV cache and compute buffers. So one model can serve two contexts:

- **Generation context:** full depth, normal decoding.
- **Classifier context:** `embeddings = true`, `llama_set_n_layer_exit(ctx, L)`. Its graph stops after `L` layers and computes no LM head. That makes it much cheaper than a full forward pass.

## Using it as a classifier

1. Load the model once and create both contexts from it.
2. Read the residual after layer `L-1` (graph tensor `l_out-<L-1>`) with an eval callback (`cb_eval`), and mean-pool it over the item's tokens. You can also set `pooling_type = LLAMA_POOLING_TYPE_MEAN` and read the pooled embedding. That output passes through `output_norm`, so it is the normalised version of the same feature.
3. Train a linear head, e.g. logistic regression, on those features: `p = sigmoid(w·x + b)`.
4. At run time, send items to the classifier context. The generation context's prompt prefill passes through the same layer, so a generation request can also be classified at no extra cost.

The middle layers usually make the best classifier features. Pick `L` by cross-validation on your own data.

The change touches only `src/llama-context.{h,cpp}`, `src/llama-cparams.h`, `src/llama-graph.{h,cpp}` and `src/llama-ext.h`.
