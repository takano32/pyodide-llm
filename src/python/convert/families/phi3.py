# A Phi-3 (T258: Microsoft's Phi-3 mini and Phi-4 mini, transformers' model_type "phi3"): a Llama whose q, k and v are
# one matrix (qkv_proj: q's rows, then k's, then v's) and whose gate and up of the FFN are another (gate_up_proj: the
# gate's rows, then the up's), which the converter cuts as it does a PLaMo 3's. Its RoPE may turn the first values of
# each head only (partial_rotary_factor, Phi-4 mini's 0.75; transformers' Phi3RotaryEmbedding and apply_rotary_pos_emb
# of 5.16.1, the form), so its layout is the Llama's with that (engine/layout.py's "phi3"). transformers masks a
# sliding window of the last sliding_window positions where config.json says one (Phi-3 mini 4k's 2047): a context no
# longer than the window attends the very same positions, so the context is cut to it, as a Mistral v0.1's is.
import math

from convert.families.family import Family, positive, refuse, rotary_dim
from convert.families.llama import check


def phi3_config(config):
    """How much of a head RoPE turns, by the name rotary_dim() reads (partial_rotary_factor at the top of config.json,
    as Phi3Config reads it; 1 where it says none: a Phi-3 mini's). Twice is once."""
    return {**config, "rotary_pct": config.get("rotary_pct", config.get("partial_rotary_factor", 1.0))}


def phi3_windowed(config):
    """The context cut to the sliding window, where config.json says one (Phi-3 mini 4k: 2047 of 4096), and (T258.2) a
    longrope RoPE scaling made what the engine's tables read (phi3_longrope()). Twice is once."""
    window, context = config.get("sliding_window"), config.get("max_position_embeddings")
    if isinstance(window, int) and not isinstance(window, bool) and window > 0 and isinstance(context, int):
        config = {**config, "max_position_embeddings": min(context, window)}
    return phi3_longrope(config)


def phi3_longrope(config):
    """T258.2: a longrope RoPE scaling (Phi-4 mini's, Phi-3.5 mini's) as transformers 5.16.1 reads it
    (_compute_longrope_parameters, standardize_rope_params): the original context is config.json's own
    original_max_position_embeddings where it has one (Phi3Config's field, over the scaling's), the factor the
    scaling's or the ratio of the context to the original one, and the attention factor, by which cos and sin are
    multiplied at every length, the scaling's or sqrt(1 + ln(factor) / ln(original)) (1 where the factor is not past 1:
    Phi-4 mini's 32 makes 1.190). transformers turns by the short factors while the text is within the original context
    and by the long ones past it, the whole text then: the context is cut to the original one, and the scaling the
    engine gets is the short factors, the attention factor and the original context. Twice is once (the attention
    factor is kept from the first time, when the context was not cut yet)."""
    scaling, context = config.get("rope_scaling"), config.get("max_position_embeddings")
    if not isinstance(scaling, dict) or scaling.get("rope_type", scaling.get("type")) != "longrope":
        return config
    original = config.get("original_max_position_embeddings") or scaling.get("original_max_position_embeddings") or context
    if not (positive(original) and positive(context)):
        return config  # (phi3_check() refuses it)
    factor = scaling.get("factor") or context / original
    magnitude = scaling.get("attention_factor")
    if magnitude is None and positive(factor):
        magnitude = 1.0 if factor <= 1 else math.sqrt(1 + math.log(factor) / math.log(original))
    made = {"rope_type": "longrope", "short_factor": scaling.get("short_factor"), "attention_factor": magnitude,
            "original_max_position_embeddings": original}
    return {**config, "rope_scaling": made, "max_position_embeddings": min(context, original),
            "original_max_position_embeddings": original}


def phi3_check(config):
    """A Llama's (silu, no biases), and the bias of the classifier the model's own code may have (lm_head_bias:
    Phi-4 mini's says false). T258.2: and a longrope scaling the engine's tables can read: a short factor for every
    pair RoPE turns, each a number above 0, and an attention factor above 0."""
    check(config)
    if config.get("lm_head_bias"):
        refuse("its layers have biases")
    scaling = config.get("rope_scaling") or {}
    if scaling.get("rope_type", scaling.get("type")) == "longrope":
        factors = scaling.get("short_factor")
        if not isinstance(factors, list) or len(factors) != rotary_dim(config) // 2 or not all(positive(f) for f in factors):
            refuse("its longrope RoPE scaling has no short factor for every pair it turns")
        if not positive(scaling.get("attention_factor")) or not positive(scaling.get("original_max_position_embeddings")):
            refuse("its longrope RoPE scaling names no attention factor or no original context")


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
PHI3 = Family("Phi-3", "phi3", phi3_sources, phi3_check, before=phi3_config, after=phi3_windowed, scaled=True,
              scalings=("longrope",))
