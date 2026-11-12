# A Phi-3 (T258: Microsoft's Phi-3 mini and Phi-4 mini, transformers' model_type "phi3"): a Llama whose q, k and v are
# one matrix (qkv_proj: q's rows, then k's, then v's) and whose gate and up of the FFN are another (gate_up_proj: the
# gate's rows, then the up's), which the converter cuts as it does a PLaMo 3's. Its RoPE may turn the first values of
# each head only (partial_rotary_factor, Phi-4 mini's 0.75; transformers' Phi3RotaryEmbedding and apply_rotary_pos_emb
# of 5.16.1, the form), so its layout is the Llama's with that (engine/layout.py's "phi3"). transformers masks a
# sliding window of the last sliding_window positions where config.json says one (Phi-3 mini 4k's 2047): a context no
# longer than the window attends the very same positions, so the context is cut to it, as a Mistral v0.1's is.
from convert.families.family import Family, refuse
from convert.families.llama import check


def phi3_config(config):
    """How much of a head RoPE turns, by the name rotary_dim() reads (partial_rotary_factor at the top of config.json,
    as Phi3Config reads it; 1 where it says none: a Phi-3 mini's). Twice is once."""
    return {**config, "rotary_pct": config.get("rotary_pct", config.get("partial_rotary_factor", 1.0))}


def phi3_windowed(config):
    """The context cut to the sliding window, where config.json says one (Phi-3 mini 4k: 2047 of 4096). Twice is once."""
    window, context = config.get("sliding_window"), config.get("max_position_embeddings")
    if isinstance(window, int) and not isinstance(window, bool) and window > 0 and isinstance(context, int):
        return {**config, "max_position_embeddings": min(context, window)}
    return config


def phi3_check(config):
    """A Llama's (silu, no biases), and the bias of the classifier the model's own code may have (lm_head_bias:
    Phi-4 mini's says false)."""
    check(config)
    if config.get("lm_head_bias"):
        refuse("its layers have biases")


def phi3_sources(d, prefix, rotary):
    """By transformers' names (Phi3ForCausalLM's). q, k and v are cut out of qkv_proj, and the first rotary rows of
    each head of q and k interleaved (as a Qwen3.5's, where RoPE turns part of a head; ("heads", 1, …) of the whole head
    is the Llama's "permute"); the gate and the up of gate_up_proj."""
    layer = "model.layers.{}."
    rows, keys, ffn = d.q_dim + 2 * d.kv_dim, d.q_dim + d.kv_dim, d.hidden_dim
    of = lambda name, transform=None: (layer + name + ".weight", transform)
    return {"token_embedding_table": ("model.embed_tokens.weight", None),
            "rms_att_weight": of("input_layernorm"),
            "wq": of("self_attn.qkv_proj", (("rows", 0, d.q_dim, rows), ("heads", 1, 0, d.n_heads, rotary))),
            "wk": of("self_attn.qkv_proj", (("rows", d.q_dim, keys, rows), ("heads", 1, 0, d.n_kv_heads, rotary))),
            "wv": of("self_attn.qkv_proj", ("rows", keys, rows, rows)), "wo": of("self_attn.o_proj"),
            "rms_ffn_weight": of("post_attention_layernorm"),
            "w1": of("mlp.gate_up_proj", ("rows", 0, ffn, 2 * ffn)), "w2": of("mlp.down_proj"),
            "w3": of("mlp.gate_up_proj", ("rows", ffn, 2 * ffn, 2 * ffn)),
            "rms_final_weight": ("model.norm.weight", None), "wcls": ("lm_head.weight", None)}


# (no GGUF: llama.cpp's phi3 keeps qkv and the gate and up as one tensor each, and longrope as tensors of its own)
PHI3 = Family("Phi-3", "phi3", phi3_sources, phi3_check, before=phi3_config, after=phi3_windowed)
