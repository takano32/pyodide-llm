# A Qwen3.5 and a Qwen3.8 (T359, from T229): a Qwen3 most of whose layers are linear-attention ones, with a gate beside
# q, RoPE over part of each head and norms stored around zero.
from engine.layout import linear_form
from convert.families.family import Family, refuse, rotary_dim, same
from convert.families.llama import GGUF_LAYER, GGUF_NAMES, check, gguf_config

# transformers' Qwen3_5TextConfig, where config.json leaves one out
LINEAR_DEFAULTS = {"linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}


def linear(config):
    """FORM's "linear" from a Qwen3.5's (normalized) config.json: which layers attend over all positions, and the
    heads and the convolution of the others (T229). layer_types is every full_attention_interval-th layer a
    full-attention one in every published model; any other order is refused, for the file is laid out by that one
    number."""
    kinds, layers = config.get("layer_types"), config.get("num_hidden_layers")
    every = config.get("full_attention_interval")
    if every is None:
        every = kinds.index("full_attention") + 1 if isinstance(kinds, list) and "full_attention" in kinds else 4
    numbers = {key: config.get(key, default) for key, default in LINEAR_DEFAULTS.items()}
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in [every, *numbers.values()]) or every < 2:
        raise ValueError("This model cannot be converted: its config.json has no usable linear-attention layers.")
    if kinds is not None and kinds != ["linear_attention" if (l + 1) % every else "full_attention" for l in range(layers)]:
        raise ValueError(f"This model cannot be converted: its layers are not every {every}th a full-attention one.")
    return {"every": every, "key_heads": numbers["linear_num_key_heads"], "value_heads": numbers["linear_num_value_heads"],
            "key_dim": numbers["linear_key_head_dim"], "value_dim": numbers["linear_value_head_dim"],
            "conv": numbers["linear_conv_kernel_dim"]}


def language_model(config):
    """T229: Qwen3.5 is a vision-language model, and the language model's config is one level down (the vision
    model's is not read: text alone). It names no BOS: every text here begins with one (T131), and that is the
    end-of-text token, as the config.json of Qwen3.8-27B says (bos_token_id 248044, its eos_token_id) and as a
    Qwen2.5's and a Qwen3's do. A config without a text_config stays what it is, and is refused by its name."""
    text = config.get("text_config")
    if not isinstance(text, dict):
        return config
    config = {**text, "model_type": "qwen3_5_text"}
    end = text.get("eos_token_id")
    end = end[0] if isinstance(end, list) and end else end
    if config.get("bos_token_id") is None and isinstance(end, int):
        config["bos_token_id"] = end
    return config


def defaults(config):
    """(the review of T229) what transformers' Qwen3_5TextConfig says where a config.json leaves it out: RoPE over a
    quarter of a head (partial_rotary_factor, at the top where there is no rope_parameters: it is the config's
    own name for it) and heads of 256. Every published Qwen3.5 and Qwen3.8 config.json says both, but a config
    that did not would have run with whole heads turning (and sizes dim / heads), without a word."""
    config = {**config, "rotary_pct": config.get("rotary_pct", config.get("partial_rotary_factor", 0.25))}
    if config.get("head_dim") is None:
        config["head_dim"] = 256
    return config


def qwen35_check(config):
    check(config)
    if config["num_hidden_layers"] < linear_form(linear(config))["every"]:
        refuse("it has no full-attention layer")
    if rotary_dim(config) < 2:
        refuse("it rotates none of each head")
    if config.get("mlp_only_layers") or config.get("attn_output_gate") is False:
        # what transformers' Qwen3_5 does not read either: a model that says so is another model
        refuse("its layers are not the ones of a Qwen3.5")
    if config.get("output_gate_type", "silu") not in ("silu", "swish"):
        # the activation of the gate that a Gated DeltaNet layer's norm multiplies by (vLLM's and Modular's readers of the
        # field; transformers' does not read it): Qwen3.5's config has none and Qwen3.8's says "swish", which is silu
        refuse(f"its linear-attention layers gate their norm with {config['output_gate_type']}, not silu")


def qwen35_prefix(source):
    """"model.language_model." (the vision-language checkpoint) or "model." (the language model saved alone)."""
    return "model.language_model." if "model.language_model.embed_tokens.weight" in source else "model."


def qwen35_sources(d, prefix, rotary):
    # T229: RoPE turns the first rotary rows of each head of q and k (and so of the norms of their heads); the
    # gate's rows are taken as they are
    layer = prefix + "layers.{}."
    one, head_norm = ("one",), (("heads", 1, 0, 1, rotary), ("one",))
    return {"token_embedding_table": (prefix + "embed_tokens.weight", None),
            "rms_att_weight": (layer + "input_layernorm.weight", one),
            "wq": (layer + "self_attn.q_proj.weight", ("heads", 2, 0, d.n_heads, rotary)),
            "wg": (layer + "self_attn.q_proj.weight", ("heads", 2, 1, d.n_heads, 0)),
            "wk": (layer + "self_attn.k_proj.weight", ("heads", 1, 0, d.n_kv_heads, rotary)),
            "wv": (layer + "self_attn.v_proj.weight", None), "wo": (layer + "self_attn.o_proj.weight", None),
            "q_norm": (layer + "self_attn.q_norm.weight", head_norm),
            "k_norm": (layer + "self_attn.k_norm.weight", head_norm),
            "wqkv": (layer + "linear_attn.in_proj_qkv.weight", None), "wz": (layer + "linear_attn.in_proj_z.weight", None),
            "wb": (layer + "linear_attn.in_proj_b.weight", None), "wa": (layer + "linear_attn.in_proj_a.weight", None),
            "conv": (layer + "linear_attn.conv1d.weight", ("taps",)), "dt_bias": (layer + "linear_attn.dt_bias", None),
            "decay": (layer + "linear_attn.A_log", ("decay",)), "delta_norm": (layer + "linear_attn.norm.weight", None),
            "wout": (layer + "linear_attn.out_proj.weight", None),
            "rms_ffn_weight": (layer + "post_attention_layernorm.weight", one),
            "w1": (layer + "mlp.gate_proj.weight", None), "w2": (layer + "mlp.down_proj.weight", None),
            "w3": (layer + "mlp.up_proj.weight", None),
            "rms_final_weight": (prefix + "norm.weight", one), "wcls": ("lm_head.weight", None)}


# ---- a GGUF of a Qwen3.5
# T245: the tensors of a Qwen3.5's linear-attention layer that have the value heads along an axis, as (whether q and k
# stand before them, whether a head has value_dim entries there or one, the axis). llama.cpp's conversion/qwen.py at
# dcd387a4, _LinearAttentionVReorderBase.modify_tensors (lines 584 to 633), reorders these and no other (the norm of a
# value head, linear_attn.norm, is one for all the heads)
QWEN35_TILED = {"in_proj_qkv.weight": (True, True, 0), "in_proj_z.weight": (False, True, 0),
                "in_proj_a.weight": (False, False, 0), "in_proj_b.weight": (False, False, 0),
                "dt_bias": (False, False, 0), "A_log": (False, False, 0),
                "conv1d.weight": (True, True, 0), "out_proj.weight": (False, True, 1)}


def qwen35_gguf_config(key, tensors):
    """T236: what config.json's text_config says of the linear-attention layers, by its names (llama.cpp's are a
    state-space model's: the state is a key head, the groups the key heads, the rank the value heads), and how much
    of a head turns, as GPT-NeoX's. One the GGUF leaves out is left out: linear() has transformers' defaults, and
    gguf_agrees() holds the whole to the original's."""
    config = gguf_config(key, tensors)
    head, values, inner = key("attention.key_length"), key("ssm.time_step_rank"), key("ssm.inner_size")
    said = {"full_attention_interval": key("full_attention_interval"), "linear_conv_kernel_dim": key("ssm.conv_kernel"),
            "linear_key_head_dim": key("ssm.state_size"), "linear_num_key_heads": key("ssm.group_count"),
            "linear_num_value_heads": values, "linear_value_head_dim": inner // values if inner and values else None,
            "rotary_pct": key("rope.dimension_count", 0) / head if head else None}
    config.update({name: value for name, value in said.items() if value is not None})
    heads = linear(config)
    if heads["value_heads"] % heads["key_heads"]:
        raise ValueError(f"This GGUF holds a Qwen3.5 of {heads['value_heads']} value heads to "
                         f"{heads['key_heads']} key heads, which is not as many to each.")
    return config


def qwen35_gguf_stored(config):
    heads = linear(config)

    def stored(entry, info, target, parts):
        # T236: llama.cpp writes a Qwen3.5's norms with the 1 added that the model adds to them (all but the norm
        # of a linear-attention layer's value heads, which has none), A_log as -exp(A_log), and the convolution
        # (channels, 1, taps) without its axis of one. The first two are steps of the plan (transformed()'s "one"
        # and "decay") that are done already: no float32 comes back from them to the bit, so they are not undone
        # to be done again, as a turned q is
        if target.endswith("norm.weight") and not target.endswith("linear_attn.norm.weight"):
            entry["done"] = "one"
        if target.endswith("linear_attn.A_log"):
            entry["done"] = "decay"
        if target.endswith("linear_attn.conv1d.weight") and len(info["shape"]) == 2:
            entry["shape"] = [info["shape"][0], 1, info["shape"][1]]
        # T245: and where a key head has more value heads than one (the 4B and up: 2 or 3), it writes the value
        # heads in another order (untiled()) in every tensor that has them along an axis. Put back whole, as a
        # turned q is: it is a move of values, which comes back to the bit
        per, part = heads["value_heads"] // heads["key_heads"], target.rsplit("linear_attn.", 1)[-1]
        if per > 1 and "linear_attn." in target and part in QWEN35_TILED:
            after_keys, of_a_head, axis = QWEN35_TILED[part]
            entry["tiled"] = (2 * heads["key_heads"] * heads["key_dim"] if after_keys else 0, heads["key_heads"],
                              per, heads["value_dim"] if of_a_head else 1, axis)
    return stored


def qwen35_agrees(own, config):
    # T236: how much of a head turns, and the linear-attention layers: which layers they are and their heads
    # (the tensors show the products only: 16 key heads of 128 are 8 of 256 to them)
    return [("number of rotated values of a head", rotary_dim(own), rotary_dim(config)),
            ("linear-attention layers", linear(own), linear(config))]


# T236: its GGUF by the names of the language model saved alone ("model." in front). llama.cpp calls the second norm
# post_attention_norm here, the linear-attention layer's q, k and v attn_qkv, its z attn_gate, and the rest ssm_* after
# the state-space models it shares code with. A name with a dot is all of a tensor's name after its layer (llama.cpp
# writes dt_bias as ssm_dt.bias, and A_log as ssm_a without a ".weight")
QWEN35 = Family("Qwen3.5", "qwen35", qwen35_sources, qwen35_check, after=defaults, prefix=qwen35_prefix,
                form={"linear": linear}, free_heads=True, rotatable=True, rms_norm=True, partly=True,
                gguf="qwen35", gguf_config=qwen35_gguf_config, gguf_stored=qwen35_gguf_stored, agrees=qwen35_agrees,
                gguf_names=(GGUF_NAMES, "model.layers.{}.",
                            {**GGUF_LAYER, "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
                             "post_attention_norm": "post_attention_layernorm", "attn_qkv": "linear_attn.in_proj_qkv",
                             "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a",
                             "ssm_beta": "linear_attn.in_proj_b", "ssm_conv1d": "linear_attn.conv1d",
                             "ssm_norm": "linear_attn.norm", "ssm_out": "linear_attn.out_proj",
                             "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log"}))
# config.json of the vision-language model as it is published: its text_config is the Qwen3.5 above
QWEN35_WHOLE = QWEN35._replace(renamed=True, before=language_model, after=same, gguf=None)
