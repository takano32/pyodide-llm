# A GPT-2 and a GPT-NeoX (T359, from T65 and T72): LayerNorm, a bias after every projection, an FFN of two matrices
# and GELU. A GPT-2 has a learned table of positions and stores its matrices the other way round (Conv1D), q, k and v
# in one; a GPT-NeoX rotates part of each head, holds q, k and v of every head in turn, and may run its two branches
# in parallel.
from convert.families.family import Family, f32, refuse, rotary_dim


def gpt2_prefix(source):
    """openai-community/gpt2 publishes its tensors as wte.weight and h.0...., other GPT-2 models put
    transformer. in front of them. Both are the same model."""
    return "" if "wte.weight" in source else "transformer."


def gpt2_sources(d, prefix, rotary):
    of = lambda name, transform=None: (prefix + "h.{}." + name, transform)
    return {"token_embedding_table": (prefix + "wte.weight", None), "positions": (prefix + "wpe.weight", None),
            "rms_att_weight": of("ln_1.weight"), "ln_att_bias": of("ln_1.bias"),
            # c_attn holds q, k and v side by side, the other way round (Conv1D)
            "wq": of("attn.c_attn.weight", ("part", 0, 3)), "wk": of("attn.c_attn.weight", ("part", 1, 3)),
            "wv": of("attn.c_attn.weight", ("part", 2, 3)),
            "bq": of("attn.c_attn.bias", ("row", 0, 3)), "bk": of("attn.c_attn.bias", ("row", 1, 3)),
            "bv": of("attn.c_attn.bias", ("row", 2, 3)),
            "wo": of("attn.c_proj.weight", ("transpose",)), "bo": of("attn.c_proj.bias"),
            "rms_ffn_weight": of("ln_2.weight"), "ln_ffn_bias": of("ln_2.bias"),
            "w1": of("mlp.c_fc.weight", ("transpose",)), "b1": of("mlp.c_fc.bias"),
            "w2": of("mlp.c_proj.weight", ("transpose",)), "b2": of("mlp.c_proj.bias"),
            "rms_final_weight": (prefix + "ln_f.weight", None), "ln_final_bias": (prefix + "ln_f.bias", None),
            "wcls": ("lm_head.weight", None)}


def neox_sources(d, prefix, rotary):
    of = lambda name, transform=None: ("gpt_neox.layers.{}." + name, transform)
    # query_key_value holds q, k and v of every head: only q and k are rotated (rotary of each head, from the config),
    # so only they are interleaved; v is taken as it is
    fused = lambda what, i: of("attention.query_key_value." + what, ("heads", 3, i, d.n_heads, rotary if i < 2 else 0))
    return {"token_embedding_table": ("gpt_neox.embed_in.weight", None),
            "rms_att_weight": of("input_layernorm.weight"), "ln_att_bias": of("input_layernorm.bias"),
            "wq": fused("weight", 0), "wk": fused("weight", 1), "wv": fused("weight", 2),
            "bq": fused("bias", 0), "bk": fused("bias", 1), "bv": fused("bias", 2),
            "wo": of("attention.dense.weight"), "bo": of("attention.dense.bias"),
            "rms_ffn_weight": of("post_attention_layernorm.weight"), "ln_ffn_bias": of("post_attention_layernorm.bias"),
            "w1": of("mlp.dense_h_to_4h.weight"), "b1": of("mlp.dense_h_to_4h.bias"),
            "w2": of("mlp.dense_4h_to_h.weight"), "b2": of("mlp.dense_4h_to_h.bias"),
            "rms_final_weight": ("gpt_neox.final_layer_norm.weight", None),
            "ln_final_bias": ("gpt_neox.final_layer_norm.bias", None), "wcls": ("embed_out.weight", None)}


def gpt2_names(config):
    """GPT-2 spells its config.json differently: give it the names the rest of the converter uses."""
    dim = config.get("n_embd")
    return {**config, "hidden_size": dim, "intermediate_size": config.get("n_inner") or (4 * dim if dim else None),
            "num_hidden_layers": config.get("n_layer"), "num_attention_heads": config.get("n_head"),
            "max_position_embeddings": config.get("n_positions") or config.get("n_ctx"),
            "hidden_act": "gelu", "tie_word_embeddings": config.get("tie_word_embeddings", True)}


def neox_names(config):
    # GPT-NeoX has the Llama names already; only the angles are spelled differently
    return {**config, "rope_theta": config.get("rotary_emb_base", config.get("rope_theta", 10000.0)),
            "tie_word_embeddings": config.get("tie_word_embeddings", False)}


def gpt2_check(config):
    # GPT-2 has one kind of everything; only the activation could be something the GELU kernel is not
    # gelu_fast (T126: rinna/japanese-gpt-1b) is gelu_new's tanh approximation written another way
    if config.get("activation_function", "gelu_new") not in ("gelu_new", "gelu", "gelu_pytorch_tanh", "gelu_fast"):
        refuse(f"its activation is {config['activation_function']}, and only GELU is supported")
    if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
        refuse("it has grouped-query attention, which GPT-2 models do not")


def neox_check(config):
    if config.get("hidden_act", "gelu") not in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        refuse(f"its activation is {config['hidden_act']}, and only GELU is supported")
    if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
        refuse("it has grouped-query attention, which GPT-NeoX models do not")
    if rotary_dim(config) < 2:
        refuse("it rotates none of each head")


# ---- their GGUFs (T136's third stage), by the names of their own safetensors (openai-community/gpt2's, without
# "transformer.")
def gpt2_gguf_config(key, tensors):
    # config.json's own spelling, which normalize() reads. GPT-2 always shares its classifier with the embedding:
    # llama.cpp writes a copy of it as output.weight, which the conversion leaves unread
    return {"n_embd": key("embedding_length"), "n_inner": key("feed_forward_length"), "n_layer": key("block_count"),
            "n_head": key("attention.head_count"), "n_positions": key("context_length"),
            "layer_norm_epsilon": key("attention.layer_norm_epsilon"), "tie_word_embeddings": True}


def neox_gguf_config(key, tensors):
    dim, heads = key("embedding_length"), key("attention.head_count")
    return {"hidden_size": dim, "intermediate_size": key("feed_forward_length"),
            "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
            "max_position_embeddings": key("context_length"),
            "rotary_emb_base": float(key("rope.freq_base", 10000.0)),
            # llama.cpp says the rotated part as a number of values, config.json as a share of the head
            "rotary_pct": key("rope.dimension_count", 0) / (dim // heads) if dim and heads else None,
            "use_parallel_residual": bool(key("use_parallel_residual", True)),
            "layer_norm_eps": key("attention.layer_norm_epsilon"), "hidden_act": "gelu",
            "tie_word_embeddings": "output.weight" not in tensors}


def gpt2_gguf_stored(config):
    def stored(entry, info, target, parts):
        if len(parts) == 4 and parts[-1] == "weight" and parts[2] in ("attn_qkv", "attn_output", "ffn_up", "ffn_down"):
            # GPT-2's matrices are Conv1D, (in, out): llama.cpp stores them the other way round, as every other
            # model's. Back to Hugging Face's, so that the plan transposes them once, as it does a safetensors' own
            entry["shape"], entry["transposed"] = list(reversed(info["shape"])), True
    return stored


def neox_gguf_stored(config):
    def stored(entry, info, target, parts):
        if len(parts) == 4 and parts[2] == "attn_qkv":
            # GPT-NeoX's query_key_value holds q, k and v of every head in turn; llama.cpp stores all of q, then k,
            # then v (the matrix and its bias). Back to Hugging Face's order, like the turned q and k of a Llama
            entry["split"] = config["num_attention_heads"]
    return stored


def layer_norm_agrees(own, config):
    # transformers' default where config.json says none (the engine's LayerNorm takes 1e-5 whatever it says)
    layer_norm_eps = lambda c: c.get("layer_norm_eps", c.get("layer_norm_epsilon", 1e-5))
    return [("LayerNorm epsilon", f32(layer_norm_eps(own)), f32(layer_norm_eps(config)))]


def neox_agrees(own, config):
    # also how much of each head turns and whether the two branches run in parallel, which the options say (no
    # tensor does)
    return [*layer_norm_agrees(own, config),
            ("number of rotated values of a head", rotary_dim(own), rotary_dim(config)),
            ("parallel residual", own.get("use_parallel_residual", True), config.get("use_parallel_residual", True))]


GPT2 = Family("GPT-2", "gpt2", gpt2_sources, gpt2_check, after=gpt2_names, prefix=gpt2_prefix, fixed_context=True,
              gguf="gpt2", gguf_config=gpt2_gguf_config, gguf_stored=gpt2_gguf_stored, agrees=layer_norm_agrees,
              gguf_names=({"token_embd.weight": "wte.weight", "position_embd.weight": "wpe.weight",
                           "output_norm.weight": "ln_f.weight", "output_norm.bias": "ln_f.bias",
                           "output.weight": "lm_head.weight"}, "h.{}.",
                          {"attn_norm": "ln_1", "attn_qkv": "attn.c_attn", "attn_output": "attn.c_proj", "ffn_norm": "ln_2",
                           "ffn_up": "mlp.c_fc", "ffn_down": "mlp.c_proj"}))
NEOX = Family("GPT-NeoX", "neox", neox_sources, neox_check, after=neox_names, classifier="embed_out.weight", partly=True,
              # GPT-NeoX may run its two branches in parallel: the file does not say
              options=lambda config: {"parallel_residual": bool(config.get("use_parallel_residual", True))},
              gguf="gptneox", gguf_config=neox_gguf_config, gguf_stored=neox_gguf_stored, agrees=neox_agrees,
              gguf_names=({"token_embd.weight": "gpt_neox.embed_in.weight",
                           "output_norm.weight": "gpt_neox.final_layer_norm.weight",
                           "output_norm.bias": "gpt_neox.final_layer_norm.bias", "output.weight": "embed_out.weight"},
                          "gpt_neox.layers.{}.",
                          {"attn_norm": "input_layernorm", "attn_qkv": "attention.query_key_value",
                           "attn_output": "attention.dense", "ffn_norm": "post_attention_layernorm",
                           "ffn_up": "mlp.dense_h_to_4h", "ffn_down": "mlp.dense_4h_to_h"}))
