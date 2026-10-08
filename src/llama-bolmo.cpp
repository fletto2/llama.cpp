// Bolmo / Bwen runtime: local mLSTM encoder/decoder, boundary predictor and the byte loop
// around the global transformer (llama_decode on a bolmo context). See include/llama-bolmo.h.
//
// Reference: modeling_bolmo.py / utils_bolmo.py / tokenization_bolmo.py (allenai/Bolmo-1B,
// allenai/Bwen-8B) and the mLSTM recurrent step of mlstm_kernels (native_step.py).

#include "llama-bolmo.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

// reversed byte trie over the subword vocabulary: the longest subword ending at a byte
struct bolmo_trie {
    std::vector<int32_t> tok { -1 };                // token id per node (-1: none)
    std::unordered_map<uint64_t, int32_t> child;    // (node << 16 | byte id) -> node

    int32_t next(int32_t node, int32_t b) const {
        auto it = child.find(((uint64_t) node << 16) | (uint32_t) b);
        return it == child.end() ? -1 : it->second;
    }

    void build(const std::vector<int32_t> & table) {
        size_t i = 0;
        while (i + 2 <= table.size()) {
            const int32_t id = table[i];
            const int32_t n  = table[i + 1];
            i += 2;
            int32_t node = 0;
            for (int32_t j = n - 1; j >= 0; --j) { // stored from the back
                const int32_t b = table[i + j];
                int32_t nx = next(node, b);
                if (nx < 0) {
                    nx = (int32_t) tok.size();
                    tok.push_back(-1);
                    child[((uint64_t) node << 16) | (uint32_t) b] = nx;
                }
                node = nx;
            }
            tok[node] = id; // later entries overwrite, as in the python dict
            i += n;
        }
    }
};

} // namespace

struct llama_bolmo {
    llama_context * ctx = nullptr;
    const llama_model_bolmo * m = nullptr;

    int64_t D = 0, n_qk = 0, n_v = 0, H = 0, DK = 0, DV = 0, n_vocab = 0;
    float eps_g = 1e-6f;

    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;

    // per local layer: the mLSTM memory C [DK, DV, H] and normalizer n [DK, 1, H], unstabilized (see build_mlstm),
    // next to the layer's weights
    ggml_context * ctx_state = nullptr;
    std::vector<ggml_backend_buffer_t> buf_state;
    std::vector<ggml_tensor *> state_c;
    std::vector<ggml_tensor *> state_n;
    ggml_cgraph * gf_cur = nullptr; // graph being built (receives the state writes)

    std::vector<uint8_t> meta;
    bolmo_trie trie;

    // sequence state
    std::vector<int32_t> hist;          // byte ids fed so far (no fused ids)
    int32_t n_patches = 0;
    std::vector<float> last_glob;       // decoder input from the last patch (normalized global output)
    std::vector<float> pending_enc;     // encoder output of the prompt's last byte (already in the encoder state)
    bool has_pending = false;

    ~llama_bolmo() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        for (auto * b : backends) {
            ggml_backend_free(b);
        }
        for (auto * b : buf_state) {
            ggml_backend_buffer_free(b);
        }
        if (ctx_state) {
            ggml_free(ctx_state);
        }
    }

    int32_t expand_at(int32_t i) const;

    ggml_context * new_graph_ctx() {
        ggml_init_params p = {
            /*.mem_size   =*/ meta.size(),
            /*.mem_buffer =*/ meta.data(),
            /*.no_alloc   =*/ true,
        };
        return ggml_init(p);
    }

    ggml_tensor * build_mlstm(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * ig, ggml_tensor * fg, int il);
    ggml_tensor * build_local_layer(ggml_context * c, ggml_tensor * x, int il);
    bool compute(ggml_cgraph * gf);

    bool run_encoder(const int32_t * ids, const int32_t * sub, int32_t T, bool want_bnd,
                     std::vector<float> & enc_h, std::vector<float> * bq, std::vector<float> * bk);
    bool run_proj(const float * h, int32_t P, std::vector<float> & out);
    bool run_global(const float * patches, int32_t P, std::vector<float> & out);
    bool run_decoder(const float * enc_h, int32_t T, const float * glob, int32_t P, const int32_t * plug,
                     bool last_only, float * logits);

    void reset();
};

int32_t llama_bolmo::expand_at(int32_t i) const {
    // BolmoTokenizer.expand_byte_ids for one position
    int32_t node = 0;
    int32_t best = -1;
    for (int32_t j = i; j >= 0; --j) {
        int32_t b = hist[j];
        if (b == m->tok_bpe_end) {
            continue;
        }
        if (b >= m->tok_offset + 256) {
            b -= m->tok_offset + 256;
        }
        const int32_t nx = trie.next(node, b);
        if (nx < 0) {
            break;
        }
        node = nx;
        if (trie.tok[node] >= 0) {
            best = trie.tok[node];
        }
    }
    return best < 0 ? 0 : best;
}

// The mLSTM cell (xLSTM mLSTMLayer, mlstm_kernels backend) without the max-stabilizer m:
//   C_t = f_t C_{t-1} + i_t k_t v_t^T,  n_t = f_t n_{t-1} + i_t k_t,  f_t = sigmoid(f~), i_t = exp(i~)
//   h_t = (q_t/sqrt(DK))^T C_t / (max(|(q_t/sqrt(DK))^T n_t|, 1) + eps)
// The kernels carry C, n scaled by exp(-m) and use max(|q.n|, exp(-m)); both forms give the same h up to eps.
// Unstabilized is safe in float32 here because the gates are soft-capped: i~ <= 15 and log f <= 0, so no
// weight exceeds exp(15). Prefill (T > 1, from the zero state) uses the parallel form, generation (T = 1)
// the recurrent step; both are plain ggml ops, so they run on any backend.
ggml_tensor * llama_bolmo::build_mlstm(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                       ggml_tensor * ig, ggml_tensor * fg, int il) {
    const int64_t T = q->ne[1];
    const float scale = 1.0f / sqrtf((float) DK);
    const float eps   = m->mlstm_norm_eps;

    // max(|x|, 1) + eps
    auto denom = [&](ggml_tensor * x) {
        x = ggml_relu(c, ggml_scale_bias(c, ggml_abs(c, x), 1.0f, -1.0f));
        return ggml_scale_bias(c, x, 1.0f, 1.0f + eps);
    };

    if (T == 1) {
        ggml_tensor * C = state_c[il];
        ggml_tensor * N = state_n[il];
        ggml_tensor * q3 = ggml_reshape_3d(c, ggml_cont(c, q), DK, 1, H);
        ggml_tensor * k3 = ggml_reshape_3d(c, ggml_cont(c, k), DK, 1, H);
        ggml_tensor * fa = ggml_reshape_3d(c, ggml_sigmoid(c, fg), 1, 1, H);
        ggml_tensor * ia = ggml_reshape_3d(c, ggml_exp(c, ig), 1, 1, H);

        // outer products k v^T per head: [1, DK, H] x [1, DV, H] -> [DK, DV, H]
        ggml_tensor * kv = ggml_mul_mat(c, ggml_reshape_3d(c, k3, 1, DK, H), ggml_reshape_3d(c, ggml_cont(c, v), 1, DV, H));
        ggml_tensor * Cn = ggml_add(c, ggml_mul(c, C, fa), ggml_mul(c, kv, ia));
        ggml_tensor * Nn = ggml_add(c, ggml_mul(c, N, fa), ggml_mul(c, k3, ia));
        ggml_build_forward_expand(gf_cur, ggml_cpy(c, Cn, C));
        ggml_build_forward_expand(gf_cur, ggml_cpy(c, Nn, N));

        ggml_tensor * num = ggml_scale(c, ggml_mul_mat(c, Cn, q3), scale); // [DV, 1, H]
        ggml_tensor * qn  = ggml_scale(c, ggml_mul_mat(c, Nn, q3), scale); // [1, 1, H]
        ggml_tensor * h = ggml_div(c, num, denom(qn));
        return ggml_reshape_3d(c, h, DV, H, 1);
    }

    // parallel form over T positions from the zero state
    ggml_tensor * Q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, ggml_cont(c, q), DK, H, T), 0, 2, 1, 3)); // [DK, T, H]
    ggml_tensor * K = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, ggml_cont(c, k), DK, H, T), 0, 2, 1, 3)); // [DK, T, H]
    ggml_tensor * Vt = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, ggml_cont(c, v), DV, H, T), 1, 2, 0, 3)); // [T, DV, H]
    ggml_tensor * it = ggml_cont(c, ggml_transpose(c, ig)); // [T, H]
    ggml_tensor * ft = ggml_cont(c, ggml_transpose(c, fg)); // [T, H]

    // F_t = sum_{u <= t} log sigmoid(f_u);  log D[s, t] = F_t - F_s + i_s for s <= t
    ggml_tensor * F  = ggml_cumsum(c, ggml_neg(c, ggml_softplus(c, ggml_neg(c, ft)))); // [T, H]
    ggml_tensor * Bs = ggml_reshape_3d(c, ggml_sub(c, it, F), T, 1, H); // s on ne0
    ggml_tensor * At = ggml_reshape_3d(c, F, 1, T, H);                  // t on ne1
    ggml_tensor * lD = ggml_add(c, ggml_repeat_4d(c, Bs, T, T, H, 1), At);
    lD = ggml_tri(c, lD, GGML_TRI_TYPE_LOWER_DIAG); // s > t would overflow exp: zero first, mask again after
    ggml_tensor * Dm = ggml_tri(c, ggml_exp(c, lD), GGML_TRI_TYPE_LOWER_DIAG); // [T_s, T_t, H]

    ggml_tensor * S = ggml_mul(c, ggml_scale(c, ggml_mul_mat(c, K, Q), scale), Dm); // [T_s, T_t, H]
    ggml_tensor * num = ggml_mul_mat(c, Vt, S);                                    // [DV, T_t, H]
    ggml_tensor * h = ggml_div(c, num, denom(ggml_sum_rows(c, S)));                // [DV, T, H]

    // state after the last position: weights w_s = D[s, T-1]
    ggml_tensor * w  = ggml_view_3d(c, Dm, T, 1, H, Dm->nb[1], Dm->nb[2], (T - 1)*Dm->nb[1]);
    ggml_tensor * Kw = ggml_mul(c, K, ggml_reshape_3d(c, ggml_cont(c, w), 1, T, H)); // [DK, T, H]
    ggml_tensor * KwT = ggml_cont(c, ggml_transpose(c, Kw));                        // [T, DK, H]
    ggml_build_forward_expand(gf_cur, ggml_cpy(c, ggml_mul_mat(c, KwT, Vt), state_c[il]));
    ggml_build_forward_expand(gf_cur, ggml_cpy(c, ggml_reshape_3d(c, ggml_sum_rows(c, KwT), DK, 1, H), state_n[il]));

    return ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3)); // [DV, H, T]
}

ggml_tensor * llama_bolmo::build_local_layer(ggml_context * c, ggml_tensor * x, int il) {
    const auto & L = m->local[il];
    const int64_t T = x->ne[1];
    const float cap = m->gate_soft_cap;

    ggml_tensor * xn = ggml_mul(c, ggml_rms_norm(c, x, eps_g), L.xlstm_norm);

    ggml_tensor * q = ggml_mul_mat(c, L.wq, xn);
    ggml_tensor * k = ggml_mul_mat(c, L.wk, xn);
    ggml_tensor * v = ggml_mul_mat(c, L.wv, xn);
    ggml_tensor * o = ggml_mul_mat(c, L.wo_gate, xn);

    auto soft_cap = [&](ggml_tensor * t) {
        return ggml_scale(c, ggml_tanh(c, ggml_scale(c, t, 1.0f/cap)), cap);
    };
    ggml_tensor * ig = soft_cap(ggml_add(c, ggml_mul_mat(c, L.wi_gate, xn), L.bi_gate));
    ggml_tensor * fg = soft_cap(ggml_add(c, ggml_mul_mat(c, L.wf_gate, xn), L.bf_gate));

    ggml_tensor * h = build_mlstm(c, q, k, v, ig, fg, il); // [DV, H, T]

    // MultiHeadLayerNorm: a LayerNorm over each head, weight over all heads, no bias
    h = ggml_norm(c, h, m->mlstm_norm_eps);
    h = ggml_reshape_2d(c, h, n_v, T);
    h = ggml_mul(c, h, L.h_norm);
    h = ggml_mul(c, ggml_sigmoid(c, o), h);

    x = ggml_add(c, x, ggml_mul_mat(c, L.w_out, h));

    ggml_tensor * fn = ggml_mul(c, ggml_rms_norm(c, x, eps_g), L.ffn_norm);
    ggml_tensor * ff = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.ffn_gate, fn)), ggml_mul_mat(c, L.ffn_up, fn));
    return ggml_add(c, x, ggml_mul_mat(c, L.ffn_down, ff));
}

bool llama_bolmo::compute(ggml_cgraph * gf) {
    const int n_threads = (int) llama_n_threads(ctx);
    for (auto * b : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
        auto * fn  = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) {
            fn(b, n_threads);
        }
    }
    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    static const bool dbg = getenv("LLAMA_BOLMO_DEBUG") != nullptr;
    if (dbg) {
        LLAMA_LOG_INFO("%s: %d nodes, %d splits\n", __func__, ggml_graph_n_nodes(gf), ggml_backend_sched_get_n_splits(sched));
    }
    return ok;
}

static void set_i32(ggml_tensor * t, const int32_t * v) {
    ggml_backend_tensor_set(t, v, 0, ggml_nbytes(t));
}

bool llama_bolmo::run_encoder(const int32_t * ids, const int32_t * sub, int32_t T, bool want_bnd,
                              std::vector<float> & enc_h, std::vector<float> * bq, std::vector<float> * bk) {
    ggml_context * c = new_graph_ctx();
    ggml_cgraph * gf = ggml_new_graph_custom(c, 4096, false);
    gf_cur = gf;

    ggml_tensor * t_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(t_ids);
    ggml_tensor * x = ggml_get_rows(c, m->byte_embd, t_ids);
    ggml_tensor * t_sub = nullptr;
    if (m->subword_embd) {
        t_sub = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        ggml_set_input(t_sub);
        x = ggml_add(c, x, ggml_get_rows(c, m->subword_embd, t_sub));
    }
    for (uint32_t il = 0; il < m->n_enc_layer; ++il) {
        x = build_local_layer(c, x, il);
    }
    ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, m->local_eps), m->enc_out_norm);
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_tensor * q = nullptr, * k = nullptr;
    if (want_bnd) {
        q = ggml_mul_mat(c, m->bnd_q, h);
        k = ggml_mul_mat(c, m->bnd_k, h);
        ggml_set_output(q);
        ggml_set_output(k);
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, k);
    }

    ggml_backend_sched_reset(sched);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        set_i32(t_ids, ids);
        if (t_sub) {
            set_i32(t_sub, sub);
        }
        ok = compute(gf);
    }
    if (ok) {
        enc_h.resize((size_t) D*T);
        ggml_backend_tensor_get(h, enc_h.data(), 0, ggml_nbytes(h));
        if (want_bnd) {
            bq->resize((size_t) D*T);
            bk->resize((size_t) D*T);
            ggml_backend_tensor_get(q, bq->data(), 0, ggml_nbytes(q));
            ggml_backend_tensor_get(k, bk->data(), 0, ggml_nbytes(k));
        }
    }
    ggml_free(c);
    return ok;
}

bool llama_bolmo::run_proj(const float * hp, int32_t P, std::vector<float> & out) {
    ggml_context * c = new_graph_ctx();
    ggml_cgraph * gf = ggml_new_graph_custom(c, 64, false);

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, P);
    ggml_set_input(x);
    ggml_tensor * y = ggml_add(c, ggml_mul_mat(c, m->enc_out_w, x), m->enc_out_b);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_reset(sched);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        ggml_backend_tensor_set(x, hp, 0, ggml_nbytes(x));
        ok = compute(gf);
    }
    if (ok) {
        out.resize((size_t) D*P);
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    }
    ggml_free(c);
    return ok;
}

bool llama_bolmo::run_global(const float * patches, int32_t P, std::vector<float> & out) {
    out.resize((size_t) D*P);
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    for (int32_t i0 = 0; i0 < P; i0 += n_batch) {
        const int32_t nb = std::min(n_batch, P - i0);
        llama_batch b = llama_batch_init(nb, (int32_t) D, 1);
        memcpy(b.embd, patches + (size_t) i0*D, sizeof(float)*D*nb);
        for (int32_t i = 0; i < nb; ++i) {
            b.pos[i]       = n_patches + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 1;
        }
        b.n_tokens = nb;
        const int r = llama_decode(ctx, b);
        if (r != 0) {
            LLAMA_LOG_ERROR("%s: llama_decode failed (%d)\n", __func__, r);
            llama_batch_free(b);
            return false;
        }
        for (int32_t i = 0; i < nb; ++i) {
            const float * e = llama_get_embeddings_ith(ctx, i);
            if (!e) {
                llama_batch_free(b);
                return false;
            }
            memcpy(out.data() + (size_t) (i0 + i)*D, e, sizeof(float)*D);
        }
        llama_batch_free(b);
        n_patches += nb;
    }
    return true;
}

bool llama_bolmo::run_decoder(const float * enc_h, int32_t T, const float * glob, int32_t P, const int32_t * plug,
                              bool last_only, float * logits) {
    ggml_context * c = new_graph_ctx();
    ggml_cgraph * gf = ggml_new_graph_custom(c, 4096, false);
    gf_cur = gf;

    ggml_tensor * eh = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    ggml_set_input(eh);
    ggml_tensor * gl = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, P);
    ggml_set_input(gl);
    ggml_tensor * pl = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(pl);

    ggml_tensor * x = ggml_add(c, ggml_mul_mat(c, m->dec_in_w, eh), m->dec_in_b);
    x = ggml_add(c, x, ggml_get_rows(c, gl, pl));
    for (uint32_t il = 0; il < m->n_dec_layer; ++il) {
        x = build_local_layer(c, x, m->n_enc_layer + il);
    }
    if (last_only && T > 1) {
        x = ggml_view_2d(c, x, D, 1, x->nb[1], (T - 1)*x->nb[1]);
    }
    x = ggml_mul(c, ggml_rms_norm(c, x, eps_g), m->output_norm);
    ggml_tensor * lg = ggml_mul_mat(c, m->output, x);
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);

    ggml_backend_sched_reset(sched);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        ggml_backend_tensor_set(eh, enc_h, 0, ggml_nbytes(eh));
        ggml_backend_tensor_set(gl, glob,  0, ggml_nbytes(gl));
        set_i32(pl, plug);
        ok = compute(gf);
    }
    if (ok && logits) {
        ggml_backend_tensor_get(lg, logits, 0, ggml_nbytes(lg));
    }
    ggml_free(c);
    return ok;
}

void llama_bolmo::reset() {
    llama_memory_clear(llama_get_memory(ctx), true);
    for (auto * buf : buf_state) {
        ggml_backend_buffer_clear(buf, 0);
    }
    hist.clear();
    n_patches = 0;
    last_glob.assign(D, 0.0f);
    has_pending = false;
}

// boundary predictor (lookahead 1, threshold sample:0) on the encoder's projections
static void bolmo_boundaries(const std::vector<float> & bq, const std::vector<float> & bk, int64_t D, int32_t n,
                             std::vector<int8_t> & bnd) {
    bnd.assign(n, 0);
    for (int32_t t = 0; t + 1 < n; ++t) {
        const float * q = bq.data() + (size_t) t*D;
        const float * k = bk.data() + (size_t) (t + 1)*D;
        double qq = 0, kk = 0, qk = 0;
        for (int64_t j = 0; j < D; ++j) {
            qq += (double) q[j]*q[j];
            kk += (double) k[j]*k[j];
            qk += (double) q[j]*k[j];
        }
        const float cs = (float) (qk / (std::max(sqrt(qq), 1e-12) * std::max(sqrt(kk), 1e-12)));
        const float lp = log1pf(-std::min(cs, 1.0f - 1e-3f)) - (float) M_LN2;
        bnd[t] = lp > logf(0.5f) ? 1 : 0;
    }
    if (n > 0) {
        bnd[0] = 1; // log-prob 0 at the sequence start
    }
}

static void bolmo_patches(const std::vector<float> & enc_h, const std::vector<int8_t> & bnd, int32_t n, int64_t D,
                          std::vector<float> & hp, std::vector<int32_t> & plug) {
    hp.clear();
    plug.resize(n);
    int32_t P = 0;
    for (int32_t t = 0; t < n; ++t) {
        if (bnd[t]) {
            hp.insert(hp.end(), enc_h.begin() + (size_t) t*D, enc_h.begin() + (size_t) (t + 1)*D);
            P++;
        }
        plug[t] = std::max(P - 1, 0);
    }
}

//
// API
//

llama_bolmo * llama_bolmo_init(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const auto * bm = dynamic_cast<const llama_model_bolmo *>(model);
    if (!bm) {
        LLAMA_LOG_ERROR("%s: not a bolmo model\n", __func__);
        return nullptr;
    }

    auto * b = new llama_bolmo();
    b->ctx = ctx;
    b->m   = bm;
    b->D       = bm->hparams.n_embd;
    b->n_qk    = (int64_t) (b->D * bm->qk_dim_factor);
    b->n_v     = (int64_t) (b->D * bm->v_dim_factor);
    b->H       = bm->n_local_head;
    b->DK      = b->n_qk / b->H;
    b->DV      = b->n_v  / b->H;
    b->n_vocab = bm->vocab.n_tokens();
    b->eps_g   = bm->hparams.f_norm_rms_eps;

    // backends: the model's devices, then the CPU
    for (const auto & dev : bm->devices) {
        if (dev.is_meta) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev.dev, nullptr);
        if (be) {
            b->backends.push_back(be);
        }
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        delete b;
        return nullptr;
    }
    b->backends.push_back(cpu);
    b->sched = ggml_backend_sched_new(b->backends.data(), nullptr, (int) b->backends.size(), 8192, false, true);

    b->meta.resize(ggml_tensor_overhead()*8192 + ggml_graph_overhead_custom(8192, false));

    // mLSTM states, each in a plain buffer of the device that holds its layer's weights
    const size_t n_local = bm->local.size();
    ggml_init_params ps = { ggml_tensor_overhead()*2*(n_local + 1), nullptr, true };
    b->ctx_state = ggml_init(ps);
    for (size_t i = 0; i < n_local; ++i) {
        ggml_tensor * sc = ggml_new_tensor_3d(b->ctx_state, GGML_TYPE_F32, b->DK, b->DV, b->H);
        ggml_tensor * sn = ggml_new_tensor_3d(b->ctx_state, GGML_TYPE_F32, b->DK, 1, b->H);
        ggml_format_name(sc, "bolmo_mlstm_c_%zu", i);
        ggml_format_name(sn, "bolmo_mlstm_n_%zu", i);
        b->state_c.push_back(sc);
        b->state_n.push_back(sn);

        ggml_backend_buffer_t wbuf = bm->local[i].wq->buffer;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(wbuf));
        if (!dev || ggml_backend_buffer_is_host(wbuf)) {
            dev = ggml_backend_get_device(cpu);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        const size_t sz = ggml_backend_buft_get_alloc_size(buft, sc) + ggml_backend_buft_get_alloc_size(buft, sn) + 2*ggml_backend_buft_get_alignment(buft);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, sz);
        ggml_tallocr ta = ggml_tallocr_new(buf);
        ggml_tallocr_alloc(&ta, sc);
        ggml_tallocr_alloc(&ta, sn);
        b->buf_state.push_back(buf);
    }

    b->trie.build(bm->expand_table);
    b->reset();

    LLAMA_LOG_INFO("%s: D = %lld, local heads = %lld (qk %lld, v %lld), %u enc + %u dec layers, trie nodes = %zu\n", __func__,
            (long long) b->D, (long long) b->H, (long long) b->DK, (long long) b->DV, bm->n_enc_layer, bm->n_dec_layer, b->trie.tok.size());
    return b;
}

void llama_bolmo_free(llama_bolmo * bolmo) {
    delete bolmo;
}

int32_t llama_bolmo_n_vocab(const llama_bolmo * b) { return (int32_t) b->n_vocab; }
int32_t llama_bolmo_token_eos(const llama_bolmo * b) { return b->m->tok_eos; }
int32_t llama_bolmo_token_bos(const llama_bolmo * b) { return b->m->tok_bos; }
int32_t llama_bolmo_boundary_offset(const llama_bolmo * b) { return b->m->tok_offset + 256; }
int32_t llama_bolmo_n_patches(const llama_bolmo * b) { return b->n_patches; }

int32_t llama_bolmo_tokenize(const llama_bolmo * b, const char * text, int32_t len, int32_t * ids, int32_t n_max, bool add_bos) {
    const int32_t n = len + (add_bos ? 1 : 0);
    if (n > n_max) {
        return -n;
    }
    int32_t k = 0;
    if (add_bos) {
        ids[k++] = b->m->tok_bos;
    }
    for (int32_t i = 0; i < len; ++i) {
        ids[k++] = b->m->tok_offset + (uint8_t) text[i];
    }
    return n;
}

int32_t llama_bolmo_id_to_byte(const llama_bolmo * b, int32_t id) {
    const int32_t off = b->m->tok_offset;
    if (id >= off + 256) {
        id -= off + 256;
    }
    return id >= off && id < off + 256 ? id - off : -1;
}

int32_t llama_bolmo_score(llama_bolmo * b, const int32_t * ids, int32_t n, float * logits, int8_t * boundaries) {
    if (n < 1) {
        return -1;
    }
    b->reset();
    b->hist.assign(ids, ids + n);
    std::vector<int32_t> sub(n);
    for (int32_t i = 0; i < n; ++i) {
        sub[i] = b->expand_at(i);
    }

    std::vector<float> enc_h, bq, bk;
    if (!b->run_encoder(ids, sub.data(), n, true, enc_h, &bq, &bk)) {
        return -2;
    }
    std::vector<int8_t> bnd;
    bolmo_boundaries(bq, bk, b->D, n, bnd);
    if (boundaries) {
        memcpy(boundaries, bnd.data(), n);
    }

    std::vector<float> hp, patches, glob;
    std::vector<int32_t> plug;
    bolmo_patches(enc_h, bnd, n, b->D, hp, plug);
    const int32_t P = (int32_t) (hp.size() / b->D);
    if (!b->run_proj(hp.data(), P, patches) || !b->run_global(patches.data(), P, glob)) {
        return -3;
    }

    std::vector<float> lg;
    float * out = logits;
    if (!out) {
        lg.resize((size_t) n*b->n_vocab);
        out = lg.data();
    }
    if (!b->run_decoder(enc_h.data(), n, glob.data(), P, plug.data(), false, out)) {
        return -4;
    }
    return P;
}

int32_t llama_bolmo_prefill(llama_bolmo * b, const int32_t * ids, int32_t n, float * logits) {
    if (n < 2) {
        return -1;
    }
    b->reset();
    b->hist.assign(ids, ids + n);
    std::vector<int32_t> sub(n);
    for (int32_t i = 0; i < n; ++i) {
        sub[i] = b->expand_at(i);
    }

    // encoder over the whole prompt: the boundary of byte n-2 needs byte n-1 as lookahead.
    // The encoder state then already includes byte n-1, whose output is kept for the first step.
    std::vector<float> enc_h, bq, bk;
    if (!b->run_encoder(ids, sub.data(), n, true, enc_h, &bq, &bk)) {
        return -2;
    }
    std::vector<int8_t> bnd;
    bolmo_boundaries(bq, bk, b->D, n, bnd);

    // global model and decoder over bytes 0..n-2
    const int32_t n1 = n - 1;
    std::vector<float> hp, patches, glob;
    std::vector<int32_t> plug;
    bolmo_patches(enc_h, bnd, n1, b->D, hp, plug);
    const int32_t P = (int32_t) (hp.size() / b->D);
    if (!b->run_proj(hp.data(), P, patches) || !b->run_global(patches.data(), P, glob)) {
        return -3;
    }
    std::vector<float> lg(b->n_vocab);
    if (!b->run_decoder(enc_h.data(), n1, glob.data(), P, plug.data(), true, lg.data())) {
        return -4;
    }
    b->last_glob.assign(glob.end() - b->D, glob.end());
    b->pending_enc.assign(enc_h.end() - b->D, enc_h.end());
    b->has_pending = true;

    // forced decoding of the last prompt byte: only its plain and fused ids compete
    const int32_t x  = ids[n - 1];
    const int32_t xb = x + b->m->tok_offset + 256;
    const int32_t tok = lg[xb] > lg[x] ? xb : x;

    if (llama_bolmo_step(b, tok, logits) != 0) {
        return -5;
    }
    return P;
}

int32_t llama_bolmo_step(llama_bolmo * b, int32_t token, float * logits) {
    const int32_t boff = b->m->tok_offset + 256;
    const bool is_bnd = token >= boff;
    const int32_t x = is_bnd ? token - boff : token;
    if (x < b->m->tok_offset || x >= b->m->tok_offset + 256) {
        LLAMA_LOG_ERROR("%s: token %d is not a byte\n", __func__, token);
        return -1;
    }

    std::vector<float> h;
    if (b->has_pending) {
        h = b->pending_enc; // the prompt's last byte, already in hist and in the encoder state
        b->has_pending = false;
    } else {
        b->hist.push_back(x);
        const int32_t sub = b->expand_at((int32_t) b->hist.size() - 1);
        if (!b->run_encoder(&x, &sub, 1, false, h, nullptr, nullptr)) {
            return -2;
        }
    }

    if (is_bnd) {
        // this byte ends a patch: the global model takes a step
        std::vector<float> patch, glob;
        if (!b->run_proj(h.data(), 1, patch) || !b->run_global(patch.data(), 1, glob)) {
            return -3;
        }
        b->last_glob = glob;
    }

    const int32_t plug = 0;
    if (!b->run_decoder(h.data(), 1, b->last_glob.data(), 1, &plug, true, logits)) {
        return -4;
    }
    return 0;
}
