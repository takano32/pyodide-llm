# An LFM2 and an LFM2.5 (T359, from T260: Liquid AI's): a Qwen3 some of whose layers are convolution layers.
from convert.families.family import Family, refuse
from convert.families.llama import check, gguf_config


def lfm2_config(config):
    """T260: an LFM2's config.json by the names the rest of the converter reads, as transformers' Lfm2Config and Lfm2MLP
    read it (7cd73d9d: configuration_lfm2.py, lines 81 to 92; modeling_lfm2.py, 119 to 133). layer_types, or every
    layer of full_attn_idxs an attention layer and the others convolution layers. The FFN's inside: block_ff_dim where
    there is one, else intermediate_size; with block_auto_adjust_ff_dim (on unless said off) two thirds of it, times
    block_ffn_dim_multiplier, up to a multiple of block_multiple_of (LFM2.5-350M: 6656 becomes 4608). tie_embedding
    wins over tie_word_embeddings, the epsilon of every RMSNorm is norm_eps, and theta is 1e6 where none is said.
    Twice is once: what it writes it reads back as it is."""
    layers, kinds = config.get("num_hidden_layers"), config.get("layer_types")
    if kinds is None and isinstance(layers, int):
        attending = config.get("full_attn_idxs")
        attending = range(layers) if attending is None else attending
        kinds = ["full_attention" if layer in attending else "conv" for layer in range(layers)]
    hidden = config.get("block_ff_dim", config.get("intermediate_size"))
    if config.get("block_auto_adjust_ff_dim", True) and isinstance(hidden, int):
        hidden = int(2 * hidden / 3)
        multiplier, multiple = config.get("block_ffn_dim_multiplier", 1.0), config.get("block_multiple_of", 256)
        if multiplier is not None:
            hidden = int(multiplier * hidden)
            hidden = multiple * ((hidden + multiple - 1) // multiple)
    rope = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    rest = {key: value for key, value in config.items() if key not in ("block_ff_dim", "full_attn_idxs")}
    return {**rest, "layer_types": kinds, "intermediate_size": hidden, "block_auto_adjust_ff_dim": False,
            "rope_theta": rope.get("rope_theta", config.get("rope_theta", 1000000.0)),
            "rms_norm_eps": config.get("norm_eps", config.get("rms_norm_eps", 1e-5)),
            "tie_word_embeddings": config.get("tie_embedding", config.get("tie_word_embeddings", True))}


def convolution(config):
    """FORM's "convolution" from an LFM2's (normalized) config.json: a letter for every layer, "c" for a convolution
    layer and "a" for one that attends, and the taps of the convolution (conv_L_cache, 3 where none is said, as
    transformers' Lfm2Config has it)."""
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config.get("num_hidden_layers") or set(kinds) - {"conv", "full_attention"}:
        raise ValueError("This model cannot be converted: its layers are not convolution layers and attention layers, "
                         "one kind for every layer.")
    taps = config.get("conv_L_cache", 3)
    if not isinstance(taps, int) or isinstance(taps, bool) or taps < 2:
        raise ValueError("This model cannot be converted: its config.json has no usable conv_L_cache.")
    return {"layers": "".join("c" if kind == "conv" else "a" for kind in kinds), "taps": taps}


def lfm2_check(config):
    """T260: what transformers' Lfm2 has and the engine has not (a bias in the convolution layers: conv_bias, which
    no published model sets), and what conversion_plan() cannot tell apart (a single layer of a kind is one
    tensor to it, as the only layer of a Llama is: the review of T229)."""
    check(config)
    layers = convolution(config)["layers"]
    if config.get("conv_bias"):
        refuse("its convolution layers have biases")
    if min(layers.count("c"), layers.count("a")) < 2:
        refuse("it has fewer than two convolution layers or fewer than two attention layers")


def lfm2_sources(d, prefix, rotary):
    # T260: by transformers' names of an Lfm2. q and k (and the norms of their heads) are turned as a Qwen3's; the
    # taps as a Qwen3.5's convolution's
    of = lambda name, transform=None: ("model.layers.{}." + name + ".weight", transform)
    return {"token_embedding_table": ("model.embed_tokens.weight", None), "rms_att_weight": of("operator_norm"),
            "wq": of("self_attn.q_proj", ("permute", d.n_heads)), "wk": of("self_attn.k_proj", ("permute", d.n_kv_heads)),
            "wv": of("self_attn.v_proj"), "wo": of("self_attn.out_proj"),
            "q_norm": of("self_attn.q_layernorm", ("permute", 1)), "k_norm": of("self_attn.k_layernorm", ("permute", 1)),
            "win": of("conv.in_proj"), "conv": of("conv.conv", ("taps",)), "wout": of("conv.out_proj"),
            "rms_ffn_weight": of("ffn_norm"),
            "w1": of("feed_forward.w1"), "w2": of("feed_forward.w2"), "w3": of("feed_forward.w3"),
            "rms_final_weight": ("model.embedding_norm.weight", None), "wcls": ("lm_head.weight", None)}


# ---- a GGUF of an LFM2
def lfm2_gguf_config(key, tensors):
    """T260: llama.cpp says an LFM2's layers by their key-value heads, one number a layer and 0 for a convolution
    layer (conversion/lfm2.py's set_gguf_parameters at f1cee994), the taps as shortconv.l_cache, and the FFN's
    inside as it is (no two thirds left to take). By config.json's names, for lfm2_config()."""
    config = gguf_config(key, tensors)
    groups = key("attention.head_count_kv")
    groups = groups if isinstance(groups, list) else [groups] * (key("block_count") or 0)
    attending = sorted({count for count in groups if count})
    if len(attending) != 1 or not all(isinstance(count, int) for count in groups):
        raise ValueError(f"This GGUF holds an LFM2 whose attention layers have {attending or 'no'} key-value "
                         f"heads, not one number for all of them.")
    config.update(num_key_value_heads=attending[0], block_auto_adjust_ff_dim=False,
                  layer_types=["full_attention" if count else "conv" for count in groups],
                  norm_eps=config["rms_norm_eps"], rope_theta=float(key("rope.freq_base", 1000000.0)),
                  **({"conv_L_cache": key("shortconv.l_cache")} if key("shortconv.l_cache") is not None else {}))
    return config


def lfm2_gguf_stored(config):
    def stored(entry, info, target, parts):
        if target.endswith("conv.conv.weight") and len(info["shape"]) == 2:
            # T260: an LFM2's convolution comes without its axis of one too, (channels, taps) for Hugging Face's
            # (channels, 1, taps); nothing else of an LFM2 is stored another way than the original has it
            entry["shape"] = [info["shape"][0], 1, info["shape"][1]]
    return stored


def lfm2_agrees(own, config):
    # T260: which layers are convolution layers and their taps, and the FFN's inside as the config's rule makes it
    # (the tensors' sizes would show it too, but only once they stream past)
    return [("convolution layers", convolution(own), convolution(config)),
            ("size of the FFN", own.get("intermediate_size"), config.get("intermediate_size"))]


# T260: its GGUF by transformers' names of it. llama.cpp calls the last norm token_embd_norm (the model's own name
# for it is embedding_norm), the convolution layer's three tensors shortconv.*, and writes q and k in Hugging
# Face's order, as a Qwen3's (its converter is no child of the Llama's: conversion/lfm2.py at f1cee994)
LFM2 = Family("LFM2", "lfm2", lfm2_sources, lfm2_check, before=lfm2_config, rms_norm=True,
              form={"convolution": convolution},
              gguf="lfm2", gguf_config=lfm2_gguf_config, gguf_stored=lfm2_gguf_stored, agrees=lfm2_agrees,
              gguf_names=({"token_embd.weight": "model.embed_tokens.weight",
                           "token_embd_norm.weight": "model.embedding_norm.weight", "output.weight": "lm_head.weight"},
                          "model.layers.{}.",
                          {"attn_norm": "operator_norm", "ffn_norm": "ffn_norm", "attn_q": "self_attn.q_proj",
                           "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.out_proj",
                           "attn_q_norm": "self_attn.q_layernorm", "attn_k_norm": "self_attn.k_layernorm",
                           "ffn_gate": "feed_forward.w1", "ffn_up": "feed_forward.w3", "ffn_down": "feed_forward.w2",
                           "shortconv.in_proj.weight": "conv.in_proj.weight", "shortconv.conv.weight": "conv.conv.weight",
                           "shortconv.out_proj.weight": "conv.out_proj.weight"}))
