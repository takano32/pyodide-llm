# A Gemma 2 (T612: Google's, transformers' model_type "gemma2"): a Gemma 3's layers (a norm behind the attention and the
# FFN as well as in front, a GELU in the gate, the embedding multiplied by the root of the width, local layers that
# attend a window of the last positions) without the norms of the heads of q and k, and with what a Gemma 3 has not:
# the attention's scores capped by a tanh (attn_logit_softcapping: cap * tanh(scores / cap)), the logits too
# (final_logit_softcapping), and one local layer of every two (the first, then every other). The local layers turn by
# the same base of RoPE as the others. The scores are divided by the root of query_pre_attn_scalar, where the engine
# divides them by the root of a head's size: the conversion multiplies q by the ratio (the family's scale(), as a
# Granite's, T253): the 2B's is 224 of heads of 256, the 9B's 256, which is 1. llama.cpp divides a 2B's scores by
# 16 all the same (its f_attention_scale), and transformers by the root of 224. The formulas are transformers' (Gemma2Config
# and modeling_gemma2.py at 28034545, the commit engine/layers.py's Gemma 3 is read from) and llama.cpp's (gemma2.cpp).
import math

from convert.families.family import Family, positive, refuse
from convert.families.gemma3 import GEMMA3, gemma3_sources
from convert.families.llama import gguf_config
from convert.families.local import FULL, SLIDING, local_agrees, local_check, local_options

# (what transformers' Gemma2Config says where a config.json says nothing; llama.cpp's too)
THETA, WINDOW, HEAD, SCORES_CAP, LOGITS_CAP = 10000.0, 4096, 256, 50.0, 30.0


def gemma2_config(config):
    """T612: a Gemma 2's config.json by the names the rest of the converter reads, as transformers' Gemma2Config reads it:
    the kinds of the layers listed (layer_types, or one of every two, from the first, a sliding-attention one, which is
    what a config.json of 2024 leaves unsaid), and what it takes where nothing is said. Its RoPE is one model's
    (normalize() reads a rope_parameters after this). Twice is once."""
    layers, kinds = config.get("num_hidden_layers"), config.get("layer_types")
    if kinds is None and isinstance(layers, int):
        kinds = [SLIDING if (layer + 1) % 2 else FULL for layer in range(layers)]
    size = config.get("head_dim") or HEAD
    return {**config, "layer_types": kinds, "sliding_window": config.get("sliding_window", WINDOW),
            "tie_word_embeddings": config.get("tie_word_embeddings", True),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6), "head_dim": size,
            "query_pre_attn_scalar": config.get("query_pre_attn_scalar", HEAD),
            "attn_logit_softcapping": config.get("attn_logit_softcapping", SCORES_CAP),
            "final_logit_softcapping": config.get("final_logit_softcapping", LOGITS_CAP)}


def gemma2_after(config):
    """the local layers turn by the one base (local.py reads it as rope_local_base_freq)"""
    return {**config, "rope_local_base_freq": config.get("rope_theta", THETA)}


def gemma2_check(config):
    """T612: what transformers' Gemma2 computes and the engine has not (another activation, biases), caps that are no
    numbers, a scale of the scores that is none, and a config.json that does not say its layers."""
    if config.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        refuse(f"its activation is {config['hidden_activation']}, not gelu_pytorch_tanh")
    if config.get("attention_bias"):
        refuse("its layers have biases")
    for key in ("attn_logit_softcapping", "final_logit_softcapping"):
        if config.get(key) is not None and not positive(config[key]):
            refuse(f"its config.json has no usable {key}")
    if not positive(config.get("query_pre_attn_scalar")):
        refuse("its config.json has no usable query_pre_attn_scalar")
    local_check(config)


def gemma2_scale(config):
    """what q is multiplied by: the engine divides the scores by the root of a head's size, the model by the root of
    query_pre_attn_scalar (no norm of q's heads to undo it, no bias of q): 1 where the two are one (what Gemma2Config
    takes for either where config.json says nothing)"""
    return math.sqrt((config.get("head_dim") or HEAD) / config.get("query_pre_attn_scalar", HEAD))


def gemma2_options(config):
    """What the file cannot say: the local layers (a Gemma 3's), and the caps of the scores and of the logits."""
    said = local_options(config)
    if config.get("final_logit_softcapping"):
        said["softcap"] = float(config["final_logit_softcapping"])
    if config.get("attn_logit_softcapping"):
        said["score_softcap"] = float(config["attn_logit_softcapping"])
    return said


def gemma2_sources(d, prefix, rotary):
    """A Gemma 3's names (convert/families/gemma3.py: every norm's 1, q and k turned, the root of the width in the table
    and out of the last norm), without the norms of the heads."""
    sources = gemma3_sources(d, prefix, rotary)
    del sources["q_norm"], sources["k_norm"]
    return sources


# ---- a GGUF of a Gemma 2
def gemma2_gguf_config(key, tensors):
    """T612: what llama.cpp says of a Gemma 2 (convert_hf_to_gguf.py's Gemma2Model and src/llama-model.cpp's
    LLM_ARCH_GEMMA2), by config.json's names: the window as attention.sliding_window (4096 where it says none), one layer
    of every two a sliding-attention one (llama.cpp's set_swa_pattern(2): no key says it), the caps (50 and 30 where it
    says none), the one base of RoPE. llama.cpp divides the scores by the root of a head's size, or of dim / heads in a
    model of 46 layers (the 27B, whose query_pre_attn_scalar is that): what it computes is what this says."""
    config = gguf_config(key, tensors)
    layers, window = key("block_count") or 0, key("attention.sliding_window", WINDOW)
    size = config.get("head_dim") or HEAD
    config.update(layer_types=[SLIDING if layer % 2 == 0 else FULL for layer in range(layers)], sliding_window=window,
                  rope_local_base_freq=config["rope_theta"], head_dim=size,
                  query_pre_attn_scalar=config["hidden_size"] // config["num_attention_heads"] if layers == 46 else size,
                  attn_logit_softcapping=key("attn_logit_softcapping", SCORES_CAP),
                  final_logit_softcapping=key("final_logit_softcapping", LOGITS_CAP), hidden_activation="gelu_pytorch_tanh")
    del config["hidden_act"]
    return config


# llama.cpp writes a Gemma 2's norms with the 1 added (Gemma2Model.modify_tensors: every tensor whose name ends in
# norm.weight), as a Gemma 3's; q and k in Hugging Face's order, the embedding as it is
GEMMA2_NORMS = ("output_norm.weight", "attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                "post_ffw_norm.weight")


def gemma2_gguf_marks(config):
    return {name: {"done": "one"} for name in GEMMA2_NORMS}


def gemma2_agrees(own, config):
    # which layers are local ones, their window and their base, which are no tensor; and the two caps
    cap = lambda c, key: c.get(key) or None
    return [local_agrees(own, config),
            ("cap of the scores", cap(own, "attn_logit_softcapping"), cap(config, "attn_logit_softcapping")),
            ("cap of the logits", cap(own, "final_logit_softcapping"), cap(config, "final_logit_softcapping"))]


GEMMA3_OUTSIDE, GEMMA3_LAYER, GEMMA3_NAMES = GEMMA3.gguf_names
GEMMA2 = Family("Gemma 2", "gemma2", gemma2_sources, gemma2_check, before=gemma2_config, after=gemma2_after,
                options=gemma2_options, scale=gemma2_scale, free_heads=True,
                gguf="gemma2", gguf_config=gemma2_gguf_config, gguf_marks=gemma2_gguf_marks, agrees=gemma2_agrees,
                gguf_names=(GEMMA3_OUTSIDE, GEMMA3_LAYER,
                            {ours: theirs for ours, theirs in GEMMA3_NAMES.items() if ours not in ("attn_q_norm", "attn_k_norm")}))
