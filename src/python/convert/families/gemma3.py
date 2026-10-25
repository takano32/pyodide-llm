# A Gemma 3, the text model (T.261: Google's, transformers' model_type "gemma3_text"): a Qwen3's attention and a gated
# FFN with a norm behind each as well as in front, a GELU in the gate, and local layers (engine/layers.py has the rule).
import math

from convert.families.family import Family, f32, head_size, refuse
from convert.families.llama import gguf_config

# the two kinds of a Gemma 3's layers, by config.json's names: one that attends a window of the last positions, and
# one that attends them all
SLIDING, FULL = "sliding_attention", "full_attention"
# (what transformers' Gemma3TextConfig says where a config.json says nothing)
THETA, LOCAL_THETA, PATTERN = 1000000.0, 10000.0, 6


def gemma3_config(config):
    """T.261: a Gemma 3's config.json by the names the rest of the converter reads, as transformers' Gemma3TextConfig
    reads it (28034545: configuration_gemma3.py, lines 74 to 159). layer_types, or every sliding_window_pattern-th
    layer (6 where none is said) a full-attention one and the others sliding-attention ones: Gemma 3 1B's config.json
    has the pattern and no list. The two bases of RoPE as rope_theta (the full-attention layers') and
    rope_local_base_freq, which transformers 5 writes as one rope_parameters with a dict for each kind of layer
    (normalize() reads a rope_parameters as one model's, and would find no theta there). A kind of RoPE that is not
    the plain one stays as a rope_scaling, which check_config() refuses. The embedding is the classifier unless the
    config.json says otherwise, the epsilon of every RMSNorm is 1e-6 and a head is 256 where none is said.
    Twice is once: what it writes it reads back as it is."""
    layers, kinds = config.get("num_hidden_layers"), config.get("layer_types")
    if kinds is None and isinstance(layers, int):
        pattern = config.get("sliding_window_pattern", config.get("_sliding_window_pattern", PATTERN))
        if isinstance(pattern, int) and not isinstance(pattern, bool) and pattern > 0:
            kinds = [SLIDING if (layer + 1) % pattern else FULL for layer in range(layers)]
    rope = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    of = lambda kind: rope.get(kind) if isinstance(rope.get(kind), dict) else {}
    scaling = next(({**of(kind)} for kind in (FULL, SLIDING)
                    if of(kind).get("rope_type", of(kind).get("type", "default")) != "default"), config.get("rope_scaling"))
    rest = {key: value for key, value in config.items() if key not in ("rope_parameters", "rope_scaling")}
    return {**rest, "layer_types": kinds, **({"rope_scaling": scaling} if scaling else {}),
            "rope_theta": of(FULL).get("rope_theta", config.get("rope_theta", THETA)),
            "rope_local_base_freq": of(SLIDING).get("rope_theta", config.get("rope_local_base_freq", LOCAL_THETA)),
            "tie_word_embeddings": config.get("tie_word_embeddings", True),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6), "head_dim": config.get("head_dim") or 256}


def local(config):
    """The engine's option "local" from a Gemma 3's (normalized) config.json: the sliding-attention layers, their
    window and the base RoPE turns them by (engine/layout.py's local_layers()). None for a model without any."""
    layers = [layer for layer, kind in enumerate(config["layer_types"]) if kind == SLIDING]
    return {"layers": layers, "window": config["sliding_window"], "rope_theta": float(config["rope_local_base_freq"])} \
        if layers else None


def gemma3_options(config):
    # the file does not say which layers are local ones (only where there are any)
    said = local(config)
    return {"local": said} if said else {}


def gemma3_check(config):
    """T.261: what transformers' Gemma3 computes and the engine has not (scores capped by a tanh, or the logits;
    attention in both directions; scores scaled by another number than the size of a head, which is what the engine
    divides them by: Gemma 3 27B's 168), and a config.json that does not say its layers."""
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    if config.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        refuse(f"its activation is {config['hidden_activation']}, not gelu_pytorch_tanh")
    if config.get("attention_bias"):
        refuse("its layers have biases")
    if config.get("attn_logit_softcapping") or config.get("final_logit_softcapping"):
        refuse("it caps its attention's scores or its logits")
    if config.get("use_bidirectional_attention"):
        refuse("it attends in both directions")
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"] or set(kinds) - {SLIDING, FULL}:
        refuse("its layers are not sliding-attention layers and full-attention layers, one kind for every layer")
    window = config.get("sliding_window")
    if SLIDING in kinds and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
        refuse("its config.json has no usable sliding_window")
    if not number(config.get("rope_theta")) or not number(config.get("rope_local_base_freq")):
        refuse("its config.json has no usable base of RoPE")
    if config.get("query_pre_attn_scalar", head_size(config)) != head_size(config):
        refuse(f"it scales its attention's scores by {config['query_pre_attn_scalar']}, not by the size of its heads")


def gemma3_sources(d, prefix, rotary):
    """T.261: by transformers' names of a Gemma3ForCausalLM. Every norm's weight is stored around zero (the model
    multiplies by 1 + weight); q and k, and the norms of their heads, are turned as a Qwen3's.
    The model multiplies the embedding's row by sqrt(dim), and the engine has no such step: the table is written
    multiplied (in float32, as transformers multiplies it, a piece at a time). Where the table is the classifier too,
    the logits would then be sqrt(dim) times the model's, and the last norm's weight is divided by it:
    (sqrt(dim) E) (n (1 + w) / sqrt(dim)) = E (n (1 + w)), the same logits. (T.253 refused a Granite whose embedding has
    a multiplier, since no scaling of the one table is right for both of its uses; with the norm in between it is.)"""
    one, scale = ("one",), math.sqrt(d.dim)
    of = lambda name, transform=None: ("model.layers.{}." + name + ".weight", transform)
    head_norm = (("permute", 1), one)
    return {"token_embedding_table": ("model.embed_tokens.weight", ("scale", scale)),
            "rms_att_weight": of("input_layernorm", one),
            "wq": of("self_attn.q_proj", ("permute", d.n_heads)), "wk": of("self_attn.k_proj", ("permute", d.n_kv_heads)),
            "wv": of("self_attn.v_proj"), "wo": of("self_attn.o_proj"),
            "q_norm": of("self_attn.q_norm", head_norm), "k_norm": of("self_attn.k_norm", head_norm),
            "rms_att_post_weight": of("post_attention_layernorm", one),
            "rms_ffn_weight": of("pre_feedforward_layernorm", one),
            "w1": of("mlp.gate_proj"), "w2": of("mlp.down_proj"), "w3": of("mlp.up_proj"),
            "rms_ffn_post_weight": of("post_feedforward_layernorm", one),
            "rms_final_weight": ("model.norm.weight", (one, ("scale", 1.0 / scale)) if d.shared else one),
            "wcls": ("lm_head.weight", None)}


# ---- a GGUF of a Gemma 3
def gemma3_gguf_config(key, tensors):
    """T.261: what llama.cpp says of a Gemma 3's layers (conversion/gemma.py's Gemma3Model and src/models/gemma3.cpp at
    4413a192), by config.json's names: the window as attention.sliding_window (none: every layer attends all the
    positions), which layers have it as attention.sliding_window_pattern (a list of yes and no, a layer each, or a
    number, 6 where the GGUF says none: every sixth layer is a full-attention one), the full-attention layers' base of
    RoPE as rope.freq_base and the others' as rope.freq_base_swa (1e4 where it says none), a cap of the logits where it
    has one. llama.cpp divides the scores by the root of a head's size (but in a model of 62 layers, the 27B)."""
    config = gguf_config(key, tensors)
    layers, window, pattern = key("block_count") or 0, key("attention.sliding_window"), key("attention.sliding_window_pattern", PATTERN)
    if isinstance(pattern, list):
        kinds = [SLIDING if here else FULL for here in pattern]
    else:
        kinds = [SLIDING if window and (layer % pattern) < pattern - 1 else FULL for layer in range(layers)]
    config.update(layer_types=kinds, sliding_window=window, rope_theta=float(key("rope.freq_base", THETA)),
                  rope_local_base_freq=float(key("rope.freq_base_swa", LOCAL_THETA)),
                  final_logit_softcapping=key("final_logit_softcapping"), hidden_activation="gelu_pytorch_tanh")
    del config["hidden_act"]
    return config


def gemma3_gguf_stored(config):
    def stored(entry, info, target, parts):
        # T.261: llama.cpp writes a Gemma 3's norms with the 1 added that the model adds to them (Gemma3Model's
        # norm_shift: every tensor whose name ends in norm.weight), a step of the plan that is done already. Nothing
        # else of a Gemma 3 is stored another way than the original has it: q and k in Hugging Face's order, and the
        # embedding as it is (llama.cpp multiplies the row at run time)
        if target.endswith("norm.weight"):
            entry["done"] = "one"
    return stored


def gemma3_agrees(own, config):
    # T.261: which layers are local ones, their window and their base, which are no tensor; and a cap of the logits,
    # which the engine has not (a GGUF with one would go through as the original without)
    said = lambda c: local(c) and {**local(c), "rope_theta": f32(local(c)["rope_theta"])}
    return [("sliding-attention layers", said(own), said(config)),
            ("cap of the logits", own.get("final_logit_softcapping") or None, config.get("final_logit_softcapping") or None)]


# T.261: its GGUF by transformers' names of it. llama.cpp calls the norm behind the attention post_attention_norm and the
# one behind the FFN post_ffw_norm; ffn_norm is the one in front of the FFN (transformers' pre_feedforward_layernorm,
# where a Llama's ffn_norm is its post_attention_layernorm)
GEMMA3 = Family("Gemma 3", "gemma3", gemma3_sources, gemma3_check, before=gemma3_config, options=gemma3_options,
                free_heads=True,
                gguf="gemma3", gguf_config=gemma3_gguf_config, gguf_stored=gemma3_gguf_stored, agrees=gemma3_agrees,
                gguf_names=({"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
                             "output.weight": "lm_head.weight"},
                            "model.layers.{}.",
                            {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
                             "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
                             "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                             "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
                             "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
                             "ffn_down": "mlp.down_proj"}))
