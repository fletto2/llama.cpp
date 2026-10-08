#pragma once

// Bolmo / Bwen byte-level models (arch "bolmo").
//
// llama_decode on a bolmo context runs only the global transformer over patch embeddings.
// This API runs the whole model: the local mLSTM encoder over bytes, the boundary predictor,
// pooling into patches, the global transformer (through llama_decode on the given context)
// and the local mLSTM decoder with its 520-way head.
//
// Token ids (520): 0 <pad>, 1 <bos> (= eos), 2 <eos>, 3 <bpe_token_end>, 4..259 the bytes 0..255;
// ids + 260 are the same tokens with a fused patch boundary ("this byte ends a patch").
//
// The context must be created from a bolmo model with embeddings = true and
// pooling_type = LLAMA_POOLING_TYPE_NONE. One sequence at a time.

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_bolmo;

    LLAMA_API struct llama_bolmo * llama_bolmo_init(struct llama_context * ctx);
    LLAMA_API void                 llama_bolmo_free(struct llama_bolmo * bolmo);

    LLAMA_API int32_t llama_bolmo_n_vocab   (const struct llama_bolmo * bolmo); // 520
    LLAMA_API int32_t llama_bolmo_token_eos (const struct llama_bolmo * bolmo);
    LLAMA_API int32_t llama_bolmo_token_bos (const struct llama_bolmo * bolmo);
    LLAMA_API int32_t llama_bolmo_boundary_offset(const struct llama_bolmo * bolmo); // 260

    // text -> byte ids (with <bos> first if add_bos); returns the number of ids, or -n if n_max is too small
    LLAMA_API int32_t llama_bolmo_tokenize(const struct llama_bolmo * bolmo, const char * text, int32_t len,
                                           int32_t * ids, int32_t n_max, bool add_bos);

    // byte value of an id (fused or not), or -1 for the special tokens
    LLAMA_API int32_t llama_bolmo_id_to_byte(const struct llama_bolmo * bolmo, int32_t id);

    // Teacher-forced forward over a whole sequence (what the HF model computes without a cache):
    // boundaries come from the predictor with its byte of lookahead, the last byte is never a boundary.
    // logits: n x 520 (may be NULL); boundaries: n flags (may be NULL). Returns the number of patches, < 0 on error.
    // Resets the sequence state.
    LLAMA_API int32_t llama_bolmo_score(struct llama_bolmo * bolmo, const int32_t * ids, int32_t n,
                                        float * logits, int8_t * boundaries);

    // Generation, as BolmoForCausalLM.generate(): prefill the prompt (resets the state; n >= 2),
    // decide whether its last byte ends a patch (greedy between the plain and the fused id),
    // and write the logits (520) for the first new token. Returns the number of prompt patches, < 0 on error.
    LLAMA_API int32_t llama_bolmo_prefill(struct llama_bolmo * bolmo, const int32_t * ids, int32_t n, float * logits);

    // Feed the token chosen from the last logits (a byte id or a fused byte id) and write the logits
    // for the next one. Returns 0 on success.
    LLAMA_API int32_t llama_bolmo_step(struct llama_bolmo * bolmo, int32_t token, float * logits);

    // As llama_bolmo_prefill, but with decide_last = false the logits are those that predict the prompt's last
    // byte, and the caller feeds it with llama_bolmo_step (plain or fused id) to choose its boundary flag.
    LLAMA_API int32_t llama_bolmo_prefill_ext(struct llama_bolmo * bolmo, const int32_t * ids, int32_t n, float * logits,
                                              bool decide_last);

    // number of patches (global positions) in the current sequence
    LLAMA_API int32_t llama_bolmo_n_patches(const struct llama_bolmo * bolmo);

    // Save and restore the sequence state (mLSTM states, the global KV cache length, the byte history), e.g. to
    // explore both boundary choices of a byte. A snapshot restores a state with the same or fewer patches.
    struct llama_bolmo_snapshot;
    LLAMA_API struct llama_bolmo_snapshot * llama_bolmo_snapshot_take   (const struct llama_bolmo * bolmo);
    LLAMA_API void                          llama_bolmo_snapshot_restore(struct llama_bolmo * bolmo, const struct llama_bolmo_snapshot * snap);
    LLAMA_API void                          llama_bolmo_snapshot_free   (struct llama_bolmo_snapshot * snap);

#ifdef __cplusplus
}
#endif
