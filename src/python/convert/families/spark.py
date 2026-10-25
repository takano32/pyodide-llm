# A Spark-X2.5 (T.342: XHToken's, transformers' model_type "spark2_5", by the model's own modeling_spark.py): a Llama's
# layers, three of every four of which attend a window of the last positions and turn their whole heads by a base of
# their own while the others turn a quarter of each head; q, k and v in one matrix; a gate of each head of the
# attention's output; and the exact GELU in the gate of its FFN (engine/layers.py has the rule,
# docs/notes/t339-spark-x-2026-10-15.md the model).
import math

from convert.families.family import Family, f32, head_size, refuse
from convert.families.llama import gguf_config

# the two kinds of a Spark-X2.5's layers, by config.json's names
SLIDING, FULL = "sliding_attention", "full_attention"
# (what Spark2_5Config.get_rope_theta() and get_partial_rotary_factor() say where a kind says none)
THETA, PART = 10000.0, 1.0


def spark_config(config):
    """A Spark-X2.5's config.json by the names the rest of the converter reads: its rope_parameters has a dict for
    each kind of layer, which normalize() would read as one model's and find no theta in. The full-attention layers'
    base and part of a head as rope_theta and rotary_pct (a GPT-NeoX's name: rotary_dim() reads it), the
    sliding-attention layers' as rope_local_base_freq and local_rotary_pct. Twice is once."""
    rope = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    of = lambda kind: rope.get(kind) if isinstance(rope.get(kind), dict) else {}
    scaling = next(({**of(kind)} for kind in (FULL, SLIDING)
                    if of(kind).get("rope_type", of(kind).get("type", "default")) != "default"), config.get("rope_scaling"))
    rest = {key: value for key, value in config.items() if key not in ("rope_parameters", "rope_scaling")}
    return {**rest, **({"rope_scaling": scaling} if scaling else {}),
            "rope_theta": of(FULL).get("rope_theta", config.get("rope_theta", THETA)),
            "rotary_pct": of(FULL).get("partial_rotary_factor", config.get("rotary_pct", PART)),
            "rope_local_base_freq": of(SLIDING).get("rope_theta", config.get("rope_local_base_freq", THETA)),
            "local_rotary_pct": of(SLIDING).get("partial_rotary_factor", config.get("local_rotary_pct", PART)),
            "tie_word_embeddings": config.get("tie_word_embeddings", False), "rms_norm_eps": config.get("rms_norm_eps", 1e-6)}


def local(config):
    """The engine's option "local" from a Spark-X2.5's (normalized) config.json: the sliding-attention layers, their
    window and the base RoPE turns them by (engine/layout.py's local_layers()). None for a model without any."""
    layers = [layer for layer, kind in enumerate(config["layer_types"]) if kind == SLIDING]
    return {"layers": layers, "window": config["sliding_window"], "rope_theta": float(config["rope_local_base_freq"])} \
        if layers else None


def spark_options(config):
    said = local(config)
    return {"local": said} if said else {}


def turns(config):
    """How many values of each layer's heads RoPE turns: the whole head of a sliding-attention layer, the full-attention
    layers' part of the others (rotary_dim(), which the options say as "rotary")."""
    size = head_size(config)
    part = int(size * float(config["rotary_pct"])) // 2 * 2
    return tuple(size if kind == SLIDING else part for kind in config["layer_types"])


def spark_check(config):
    """T.342: what Spark2_5 computes and the engine has not: an activation other than the exact GELU, a gate of the
    heads other than a sigmoid's or none, biases, a local layer that turns part of its heads, a config.json that does
    not say its layers."""
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    if config.get("hidden_act", "gelu") != "gelu":
        refuse(f"its activation is {config['hidden_act']}, not gelu")
    if config.get("attention_bias") or config.get("mlp_bias"):
        refuse("its layers have biases")
    if not config.get("headwise_attn_output_gate") or config.get("gate_attn_act_mode", "sigmoid") != "sigmoid":
        refuse("it gates the heads of its attention otherwise than by a sigmoid of a value a head")
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"] or set(kinds) - {SLIDING, FULL}:
        refuse("its layers are not sliding-attention layers and full-attention layers, one kind for every layer")
    window = config.get("sliding_window")
    if SLIDING in kinds and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
        refuse("its config.json has no usable sliding_window")
    if not number(config.get("rope_theta")) or not number(config.get("rope_local_base_freq")):
        refuse("its config.json has no usable base of RoPE")
    if config.get("local_rotary_pct") != 1.0 or not number(config.get("rotary_pct")) or config["rotary_pct"] > 1 \
            or min(turns(config)) < 2:
        refuse("it turns its heads by RoPE in a way the engine does not know")


def spark_sources(d, prefix, rotary):
    """T.342: by the model's names. q, k and v are one matrix (q_k_v_proj: q's rows, then k's, then v's), which is cut;
    q and k are interleaved as a Llama's are, over the values RoPE turns of that layer's heads (rotary: a number for
    each layer, Family.turns(); one number for all, or 0 for the whole head, where nothing says which layer is which:
    a plan that only sizes a model)."""
    of = lambda name: "model.layers.{}." + name + ".weight"
    width = (lambda layer: rotary[layer]) if isinstance(rotary, tuple) else (lambda layer: rotary or d.head_size)
    rows, keys = d.q_dim + 2 * d.kv_dim, d.q_dim + d.kv_dim
    cut = lambda start, stop, heads: lambda layer: (("rows", start, stop, rows), ("heads", 1, 0, heads, width(layer)))
    return {"token_embedding_table": ("model.embedding.weight", None),
            "rms_att_weight": (of("input_layernorm"), None),
            "wq": (of("self_attn.q_k_v_proj"), cut(0, d.q_dim, d.n_heads)),
            "wk": (of("self_attn.q_k_v_proj"), cut(d.q_dim, keys, d.n_kv_heads)),
            "wv": (of("self_attn.q_k_v_proj"), ("rows", keys, rows, rows)),
            "wo": (of("self_attn.out_proj"), None), "head_gate": (of("self_attn.g_proj"), None),
            "rms_ffn_weight": (of("post_attention_layernorm"), None),
            "w1": (of("mlp.gate_proj"), None), "w2": (of("mlp.down_proj"), None), "w3": (of("mlp.up_proj"), None),
            "rms_final_weight": ("model.norm.weight", None), "wcls": ("lm_head.weight", None)}


# ---- a GGUF of a Spark-X2.5
def spark_gguf_config(key, tensors):
    """T.342: what llama.cpp says of a Spark-X2.5 (XHToken's Q8_0 GGUF of the 1.7B: docs/notes/t339-spark-x-2026-10-15.md),
    by config.json's names: which layers have the window as attention.sliding_window_pattern (a yes for each
    sliding-attention layer), the window, the full-attention layers' base and part of a head as rope.freq_base and
    rope.dimension_count, the others' as rope.freq_base_swa and rope.dimension_count_swa."""
    config = gguf_config(key, tensors)
    pattern, size = key("attention.sliding_window_pattern"), config["head_dim"] or 0
    kinds = [SLIDING if here else FULL for here in pattern] if isinstance(pattern, list) else [FULL] * (key("block_count") or 0)
    config.update(layer_types=kinds, sliding_window=key("attention.sliding_window"), hidden_act="gelu",
                  headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid",
                  rope_local_base_freq=float(key("rope.freq_base_swa", THETA)),
                  rotary_pct=key("rope.dimension_count", size) / size if size else None,
                  local_rotary_pct=key("rope.dimension_count_swa", size) / size if size else None)
    return config


def spark_agrees(own, config):
    # T.342: which layers are local ones, their window and their base, and how much of a full-attention layer's heads
    # RoPE turns: no tensor shows them
    said = lambda c: local(c) and {**local(c), "rope_theta": f32(local(c)["rope_theta"])}
    return [("sliding-attention layers", said(own), said(config)),
            ("part of a head RoPE turns", turns(own), turns(config))]


# T.342: llama.cpp's names of it (as XHToken's GGUF has them: q, k and v one matrix as the original's, in Hugging Face's
# order, and the gate of the heads, attn_gate). Nothing is stored otherwise than the original has it
SPARK = Family("Spark-X2.5", "spark", spark_sources, spark_check, before=spark_config, options=spark_options,
               free_heads=True, turns=turns,
               gguf="spark2_5", gguf_config=spark_gguf_config, agrees=spark_agrees,
               gguf_names=({"token_embd.weight": "model.embedding.weight", "output_norm.weight": "model.norm.weight",
                            "output.weight": "lm_head.weight"},
                           "model.layers.{}.",
                           {"attn_norm": "input_layernorm", "attn_qkv": "self_attn.q_k_v_proj",
                            "attn_gate": "self_attn.g_proj", "attn_output": "self_attn.out_proj",
                            "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
                            "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}))
