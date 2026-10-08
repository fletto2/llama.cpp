#include "models.h"

void llama_model_bolmo::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key("bolmo.reordered_norm",               reordered_norm);
    ml.get_key("bolmo.use_qk_norm",                  use_qk_norm);
    ml.get_key("bolmo.head_qk_norm",                 head_qk_norm);
    ml.get_key("bolmo.local.head_count",             n_local_head);
    ml.get_key("bolmo.local.feed_forward_length",    n_local_ff);
    ml.get_key("bolmo.local.rms_eps",                local_eps);
    ml.get_key("bolmo.local.encoder_layer_count",    n_enc_layer);
    ml.get_key("bolmo.local.decoder_layer_count",    n_dec_layer);
    ml.get_key("bolmo.boundary.lookahead",           lookahead);
    ml.get_key("bolmo.subword_vocab_size",           n_subword);
    ml.get_key("bolmo.mlstm.qk_dim_factor",          qk_dim_factor, false);
    ml.get_key("bolmo.mlstm.v_dim_factor",           v_dim_factor,  false);
    ml.get_key("bolmo.mlstm.gate_soft_cap",          gate_soft_cap, false);
    ml.get_key("bolmo.mlstm.norm_eps",               mlstm_norm_eps, false);

    uint32_t v = 0;
    if (ml.get_key("bolmo.token.bos",     v, false)) { tok_bos     = v; }
    if (ml.get_key("bolmo.token.eos",     v, false)) { tok_eos     = v; }
    if (ml.get_key("bolmo.token.pad",     v, false)) { tok_pad     = v; }
    if (ml.get_key("bolmo.token.bpe_end", v, false)) { tok_bpe_end = v; }
    if (ml.get_key("bolmo.token.offset",  v, false)) { tok_offset  = v; }

    if (n_subword > 0) {
        ml.get_arr("bolmo.expand_table", expand_table);
    }

    switch (hparams.n_layer()) {
        case 16: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bolmo::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_qk = (int64_t) (n_embd * qk_dim_factor);
    const int64_t n_v  = (int64_t) (n_embd * v_dim_factor);
    const int64_t n_lh = n_local_head;

    byte_embd = create_tensor(tn(LLM_TENSOR_BOLMO_BYTE_EMBD, "weight"), {n_embd, n_vocab}, 0);
    if (n_subword > 0) {
        subword_embd = create_tensor(tn(LLM_TENSOR_BOLMO_SUBWORD_EMBD, "weight"), {n_embd, (int64_t) n_subword}, 0);
    }

    enc_out_norm = create_tensor(tn(LLM_TENSOR_BOLMO_ENC_OUT_NORM, "weight"), {n_embd}, 0);
    enc_out_w    = create_tensor(tn(LLM_TENSOR_BOLMO_ENC_OUT_PROJ, "weight"), {n_embd, n_embd}, 0);
    enc_out_b    = create_tensor(tn(LLM_TENSOR_BOLMO_ENC_OUT_PROJ, "bias"),   {n_embd}, 0);
    bnd_q        = create_tensor(tn(LLM_TENSOR_BOLMO_BND_Q, "weight"), {n_embd, n_embd}, 0);
    bnd_k        = create_tensor(tn(LLM_TENSOR_BOLMO_BND_K, "weight"), {n_embd, n_embd}, 0);
    dec_in_norm  = create_tensor(tn(LLM_TENSOR_BOLMO_DEC_IN_NORM, "weight"), {n_embd}, 0);
    dec_in_w     = create_tensor(tn(LLM_TENSOR_BOLMO_DEC_IN_PROJ, "weight"), {n_embd, n_embd}, 0);
    dec_in_b     = create_tensor(tn(LLM_TENSOR_BOLMO_DEC_IN_PROJ, "bias"),   {n_embd}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    local.resize(n_enc_layer + n_dec_layer);
    for (int i = 0; i < (int) local.size(); ++i) {
        auto & l = local[i];
        l.xlstm_norm = create_tensor(tn(LLM_TENSOR_BOLMO_XLSTM_NORM,  "weight", i), {n_embd}, 0);
        l.wq         = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_Q,     "weight", i), {n_embd, n_qk}, 0);
        l.wk         = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_K,     "weight", i), {n_embd, n_qk}, 0);
        l.wv         = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_V,     "weight", i), {n_embd, n_v}, 0);
        l.wo_gate    = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_OGATE, "weight", i), {n_embd, n_v}, 0);
        l.wi_gate    = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_IGATE, "weight", i), {n_embd, n_lh}, 0);
        l.bi_gate    = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_IGATE, "bias",   i), {n_lh}, 0);
        l.wf_gate    = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_FGATE, "weight", i), {n_embd, n_lh}, 0);
        l.bf_gate    = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_FGATE, "bias",   i), {n_lh}, 0);
        l.h_norm     = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_HNORM, "weight", i), {n_v}, 0);
        l.w_out      = create_tensor(tn(LLM_TENSOR_BOLMO_MLSTM_OUT,   "weight", i), {n_v, n_embd}, 0);
        l.ffn_norm   = create_tensor(tn(LLM_TENSOR_BOLMO_LFFN_NORM,   "weight", i), {n_embd}, 0);
        l.ffn_gate   = create_tensor(tn(LLM_TENSOR_BOLMO_LFFN_GATE,   "weight", i), {n_embd, (int64_t) n_local_ff}, 0);
        l.ffn_up     = create_tensor(tn(LLM_TENSOR_BOLMO_LFFN_UP,     "weight", i), {n_embd, (int64_t) n_local_ff}, 0);
        l.ffn_down   = create_tensor(tn(LLM_TENSOR_BOLMO_LFFN_DOWN,   "weight", i), {(int64_t) n_local_ff, n_embd}, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        if (use_qk_norm) {
            if (head_qk_norm) {
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
            } else {
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k * n_head}, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_k_gqa}, 0);
            }
        }

        if (reordered_norm) {
            layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, 0);
        } else {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_bolmo::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// the global transformer over patch embeddings
llama_model_bolmo::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & bm = static_cast<const llama_model_bolmo &>(model);

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // the token path is never used for real input (the batch carries patch embeddings);
    // the byte table only gives it a valid shape for graph reservation
    ggml_tensor * inpL = build_inp_embd(bm.byte_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpSA = inpL;

        cur = bm.reordered_norm ? inpL : build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur,
                    n_embd_head, n_head,
                    n_embd_head, n_head_kv,
                    n_embd_head, n_head_kv,
                    il, false);

            if (bm.use_qk_norm && !bm.head_qk_norm) {
                // OLMo 2: normalize the flat projections before splitting heads
                Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
                Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            if (bm.use_qk_norm && bm.head_qk_norm) {
                // Qwen 3: normalize each head
                Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
                Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            }
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        if (bm.reordered_norm) {
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = bm.reordered_norm ? ffn_inp : build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (bm.reordered_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // the local decoder's initial_norm (plain RMSNorm with the local epsilon)
    cur = ggml_rms_norm(ctx0, inpL, bm.local_eps);
    cur = ggml_mul(ctx0, cur, bm.dec_in_norm);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
