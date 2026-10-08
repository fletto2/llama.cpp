from __future__ import annotations

import os
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


def _bytes_to_unicode() -> dict[int, str]:
    # GPT-2 byte <-> printable character table (the one byte-level BPE vocabularies are written in)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@ModelBase.register("BolmoForCausalLM")
@ModelBase.example("allenai/Bolmo-1B")
class BolmoModel(TextModel):
    """Bolmo / Bwen byte-level models (byteified subword LLMs).

    The global transformer keeps the source model's layers (OLMo 2 "reordered_norm" or
    Qwen 3 / Llama 3 pre-norm). Around it: a local mLSTM encoder over bytes with a boundary
    predictor, last-byte pooling into patches, and a local mLSTM decoder with a 520-way
    byte(+boundary) head. The subword table of the source tokenizer is kept as an input
    embedding of the longest subword ending at each byte; its byte sequences are stored in
    `bolmo.expand_table` so the runtime can rebuild the suffix trie.
    """
    model_arch = gguf.MODEL_ARCH.BOLMO

    def set_vocab(self):
        # byte-level: 4 specials + 256 bytes, each with a fused-boundary twin
        self.gguf_writer.add_tokenizer_model("none")
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        tc = self.hparams["tokenizer_config"]
        self.gguf_writer.add_uint32("bolmo.token.bos", tc["bos_token_id"])
        self.gguf_writer.add_uint32("bolmo.token.eos", tc["eos_token_id"])
        self.gguf_writer.add_uint32("bolmo.token.pad", tc["pad_token_id"])
        self.gguf_writer.add_uint32("bolmo.token.bpe_end", tc["bpe_token_end_id"])
        assert tc.get("special_tokens_first", True), "only special_tokens_first tokenizers are supported"
        offset = len(tc["special_tokens"])
        self.gguf_writer.add_uint32("bolmo.token.offset", offset)
        if self.hparams.get("add_expanded_embeddings", True):
            self._add_expand_table(tc, offset)

    def _add_expand_table(self, tc: dict, offset: int):
        from transformers import AutoTokenizer
        ident = os.environ.get("BOLMO_SUBWORD_TOKENIZER", tc["original_identifier"])
        logger.info(f"bolmo: subword tokenizer {ident}")
        hf_tok = AutoTokenizer.from_pretrained(ident)
        b2u = _bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}
        specials = tc["special_tokens"]
        flat: list[int] = []
        n = 0
        # same rules as BolmoTokenizer.__init__ (tokenization_bolmo.py)
        for key, value in hf_tok.get_vocab().items():
            if key in specials:
                seq = [specials.index(key)]
            elif value == hf_tok.eos_token_id and tc["eos_token_id"] is not None:
                seq = [tc["eos_token_id"]]
            elif value == hf_tok.bos_token_id and tc["bos_token_id"] is not None:
                seq = [tc["bos_token_id"]]
            elif value == hf_tok.pad_token_id and tc["pad_token_id"] is not None:
                seq = [tc["pad_token_id"]]
            else:
                seq = [offset + u2b[c] for c in key]
            flat.append(value)
            flat.append(len(seq))
            flat.extend(seq)
            n += 1
        logger.info(f"bolmo: expand table {n} entries, {len(flat)} ints")
        self.gguf_writer.add_uint32("bolmo.expand_count", n)
        self.gguf_writer.add_array("bolmo.expand_table", flat)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        assert hp.get("sliding_window") is None, "sliding-window global layers are not supported"
        assert hp.get("boundary_threshold", "sample:0") == "sample:0", "only boundary_threshold sample:0 is supported"
        self.gguf_writer.add_bool("bolmo.reordered_norm", hp.get("block_type", "reordered_norm") == "reordered_norm")
        self.gguf_writer.add_bool("bolmo.use_qk_norm", hp.get("use_qk_norm", True))
        self.gguf_writer.add_bool("bolmo.head_qk_norm", hp.get("use_head_qk_norm", False))
        self.gguf_writer.add_uint32("bolmo.local.head_count", hp["num_local_heads"])
        self.gguf_writer.add_uint32("bolmo.local.feed_forward_length", hp["local_intermediate_size"])
        self.gguf_writer.add_float32("bolmo.local.rms_eps", hp["local_rms_norm_eps"])
        self.gguf_writer.add_uint32("bolmo.local.encoder_layer_count", hp["num_local_encoder_layers"])
        self.gguf_writer.add_uint32("bolmo.local.decoder_layer_count", hp["num_local_decoder_layers"])
        self.gguf_writer.add_uint32("bolmo.boundary.lookahead", hp.get("boundary_predictor_lookahead", 1))
        self.gguf_writer.add_uint32("bolmo.subword_vocab_size", hp["subword_vocab_size"] if hp.get("add_expanded_embeddings", True) else 0)
        # xLSTM-large defaults used by BolmoXLSTMLayer (mLSTMLayerConfig)
        self.gguf_writer.add_float32("bolmo.mlstm.qk_dim_factor", 0.5)
        self.gguf_writer.add_float32("bolmo.mlstm.v_dim_factor", 1.0)
        self.gguf_writer.add_float32("bolmo.mlstm.gate_soft_cap", 15.0)
        self.gguf_writer.add_float32("bolmo.mlstm.norm_eps", 1e-6)

    _LOCAL = {
        "pre_xlstm_layernorm.weight":   (gguf.MODEL_TENSOR.BOLMO_XLSTM_NORM,  ".weight"),
        "xlstm.q.weight":               (gguf.MODEL_TENSOR.BOLMO_MLSTM_Q,     ".weight"),
        "xlstm.k.weight":               (gguf.MODEL_TENSOR.BOLMO_MLSTM_K,     ".weight"),
        "xlstm.v.weight":               (gguf.MODEL_TENSOR.BOLMO_MLSTM_V,     ".weight"),
        "xlstm.ogate_preact.weight":    (gguf.MODEL_TENSOR.BOLMO_MLSTM_OGATE, ".weight"),
        "xlstm.igate_preact.weight":    (gguf.MODEL_TENSOR.BOLMO_MLSTM_IGATE, ".weight"),
        "xlstm.igate_preact.bias":      (gguf.MODEL_TENSOR.BOLMO_MLSTM_IGATE, ".bias"),
        "xlstm.fgate_preact.weight":    (gguf.MODEL_TENSOR.BOLMO_MLSTM_FGATE, ".weight"),
        "xlstm.fgate_preact.bias":      (gguf.MODEL_TENSOR.BOLMO_MLSTM_FGATE, ".bias"),
        "xlstm.multihead_norm.weight":  (gguf.MODEL_TENSOR.BOLMO_MLSTM_HNORM, ".weight"),
        "xlstm.out_proj.weight":        (gguf.MODEL_TENSOR.BOLMO_MLSTM_OUT,   ".weight"),
        "pre_feedforward_layernorm.weight": (gguf.MODEL_TENSOR.BOLMO_LFFN_NORM, ".weight"),
        "mlp.gate_proj.weight":         (gguf.MODEL_TENSOR.BOLMO_LFFN_GATE,   ".weight"),
        "mlp.up_proj.weight":           (gguf.MODEL_TENSOR.BOLMO_LFFN_UP,     ".weight"),
        "mlp.down_proj.weight":         (gguf.MODEL_TENSOR.BOLMO_LFFN_DOWN,   ".weight"),
    }

    _SINGLE = {
        "model.local_encoder.byte_embedding.weight":    (gguf.MODEL_TENSOR.BOLMO_BYTE_EMBD,    ".weight"),
        "model.local_encoder.subword_embedding.weight": (gguf.MODEL_TENSOR.BOLMO_SUBWORD_EMBD, ".weight"),
        "model.local_encoder.post_last_block_norm.weight": (gguf.MODEL_TENSOR.BOLMO_ENC_OUT_NORM, ".weight"),
        "model.local_encoder.out_projection.weight":    (gguf.MODEL_TENSOR.BOLMO_ENC_OUT_PROJ, ".weight"),
        "model.local_encoder.out_projection.bias":      (gguf.MODEL_TENSOR.BOLMO_ENC_OUT_PROJ, ".bias"),
        "model.local_encoder.boundary_predictor_module.q_proj_layer.weight": (gguf.MODEL_TENSOR.BOLMO_BND_Q, ".weight"),
        "model.local_encoder.boundary_predictor_module.k_proj_layer.weight": (gguf.MODEL_TENSOR.BOLMO_BND_K, ".weight"),
        "model.local_decoder.initial_norm.weight":      (gguf.MODEL_TENSOR.BOLMO_DEC_IN_NORM,  ".weight"),
        "model.local_decoder.in_projection.weight":     (gguf.MODEL_TENSOR.BOLMO_DEC_IN_PROJ,  ".weight"),
        "model.local_decoder.in_projection.bias":       (gguf.MODEL_TENSOR.BOLMO_DEC_IN_PROJ,  ".bias"),
        "model.norm.weight":                            (gguf.MODEL_TENSOR.OUTPUT_NORM,        ".weight"),
        "lm_head.weight":                               (gguf.MODEL_TENSOR.OUTPUT,             ".weight"),
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in self._SINGLE:
            key, suffix = self._SINGLE[name]
            yield self.format_tensor_name(key, None, suffix), data_torch
            return

        n_enc = self.hparams["num_local_encoder_layers"]
        for part, base in (("model.local_encoder.layers.", 0), ("model.local_decoder.layers.", n_enc)):
            if name.startswith(part):
                idx_s, rest = name[len(part):].split(".", 1)
                key, suffix = self._LOCAL[rest]
                # encoder layers come first, then decoder layers: local.<i>.*
                yield self.format_tensor_name(key, base + int(idx_s), suffix), data_torch
                return

        if name.startswith("model.layers."):
            assert bid is not None
            rest = name.split(".", 3)[3]
            reordered = self.hparams.get("block_type", "reordered_norm") == "reordered_norm"
            norms = {
                "input_layernorm.weight":            gguf.MODEL_TENSOR.ATTN_NORM,
                "post_attention_layernorm.weight":   gguf.MODEL_TENSOR.ATTN_POST_NORM if reordered else gguf.MODEL_TENSOR.FFN_NORM,
                "post_feedforward_layernorm.weight": gguf.MODEL_TENSOR.FFN_POST_NORM,
            }
            other = {
                "self_attn.q_proj.weight": gguf.MODEL_TENSOR.ATTN_Q,
                "self_attn.k_proj.weight": gguf.MODEL_TENSOR.ATTN_K,
                "self_attn.v_proj.weight": gguf.MODEL_TENSOR.ATTN_V,
                "self_attn.o_proj.weight": gguf.MODEL_TENSOR.ATTN_OUT,
                "self_attn.q_norm.weight": gguf.MODEL_TENSOR.ATTN_Q_NORM,
                "self_attn.k_norm.weight": gguf.MODEL_TENSOR.ATTN_K_NORM,
                "mlp.gate_proj.weight":    gguf.MODEL_TENSOR.FFN_GATE,
                "mlp.up_proj.weight":      gguf.MODEL_TENSOR.FFN_UP,
                "mlp.down_proj.weight":    gguf.MODEL_TENSOR.FFN_DOWN,
            }
            key = norms.get(rest) or other.get(rest)
            if key is None:
                raise ValueError(f"Can not map tensor {name!r}")
            yield self.format_tensor_name(key, bid), data_torch
            return

        raise ValueError(f"Can not map tensor {name!r}")
