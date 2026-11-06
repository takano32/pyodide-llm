# A PLaMo 3 (T649.2: Preferred Networks', model_type "plamo3", by the model's own modeling_plamo.py, which transformers
# has no class of): a Gemma 3's layers (a norm in front of the attention and of the FFN and another behind each, the
# norms of the heads of q and k, local layers that attend a window of the last positions and turn by a base of their
# own) with a Llama's SiLU in the gate of the FFN, one key and value head, and yarn in the RoPE of the layers that attend
# all the positions. q, k and v are one matrix, the gate and the up of the FFN another. Every norm multiplies by
# c + weight, and c is not 1 behind the attention (1/5) and the FFN (5^-1.5): what the converter adds to the weight.
from convert.families.family import Family, f32, refuse
from convert.families.llama import gguf_config
from convert.families.local import FULL, SLIDING, local_agrees, local_check, local_options, without_rope

# (what Plamo3Config says where a config.json says nothing)
THETA, LOCAL_THETA, PATTERN, WINDOW = 150000.0, 10000.0, 8, 2048
# what each norm adds to its weight (Plamo3DecoderLayer's offsets; the heads' and the last norm's are 1)
BEHIND_ATTENTION, BEHIND_FFN = 1.0 / 5, 1.0 / 5 ** 1.5
# Plamo3RotaryEmbedding's yarn: the numbers of turns of the two bounds, unrounded
BETA_FAST, BETA_SLOW = 32.0, 1.0


def plamo3_config(config):
    """A PLaMo 3's config.json by the names the rest of the converter reads, as the model's Plamo3Config reads it: the
    kinds of the layers by sliding_window_pattern (every pattern-th layer, counted from 1, a full-attention one), the
    window as window_size, the full-attention layers' base as rope_theta and the others' as rope_local_theta, yarn where
    rope_scaling_factor is not 1 (from initial_context_length positions, the bounds unrounded). The model reads none of
    config.json's layer_types and rope_parameters (its own properties make them of these): nor does this. Twice is
    once: what it writes it reads back as it is."""
    layers, pattern = config.get("num_hidden_layers"), config.get("sliding_window_pattern", PATTERN)
    kinds = [SLIDING if (layer + 1) % pattern else FULL for layer in range(layers)] \
        if isinstance(layers, int) and isinstance(pattern, int) and not isinstance(pattern, bool) and pattern > 0 else None
    factor = config.get("rope_scaling_factor", 1)
    scaling = None if factor == 1 else {"rope_type": "yarn", "factor": factor, "beta_fast": BETA_FAST, "beta_slow": BETA_SLOW,
                                        "original_max_position_embeddings": config.get("initial_context_length"),
                                        "truncate": False}
    return {**without_rope(config), "layer_types": kinds, "sliding_window": config.get("window_size", WINDOW),
            **({"rope_scaling": scaling} if scaling else {}),
            "rope_theta": config.get("rope_theta", THETA), "rope_local_base_freq": config.get("rope_local_theta", LOCAL_THETA),
            "tie_word_embeddings": config.get("tie_word_embeddings", True), "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
            "head_dim": config.get("head_dim", 128), "num_key_value_heads": config.get("num_key_value_heads", 4)}


def plamo3_check(config):
    """What the model's code computes and the engine has not: an embedding multiplied by the root of the width
    (scale_embedding), weights of another kind than plain matrices (linear_type: its FP8), a config.json that does not
    say its layers."""
    if config.get("scale_embedding"):
        refuse("it multiplies its embedding by the root of its width")
    if config.get("linear_type", "normal") != "normal":
        refuse(f"its matrices are of the {config['linear_type']} kind, not normal ones")
    local_check(config)


def padded(vocabulary):
    """the rows of the model's lm_head: its vocabulary rounded up to 16 (Plamo3ForCausalLM), the logits past it cut"""
    return (vocabulary + 15) // 16 * 16


def plamo3_sources(d, prefix, rotary):
    """By the model's names (Plamo3ForCausalLM's: the layers are model.layers.layers). q, k and v are one matrix
    (qkv_proj: q's rows, then k's, then v's), cut; q and k, and the norms of their heads, are turned as a Qwen3's. The
    gate and the up of the FFN are one matrix too (gate_up_proj: the gate's rows, then the up's). Every norm's weight is
    stored around zero: the converter adds what the model adds to it."""
    of = lambda name, transform=None: ("model.layers.layers.{}." + name + ".weight", transform)
    rows, keys, ffn = d.q_dim + 2 * d.kv_dim, d.q_dim + d.kv_dim, d.hidden_dim
    head_norm = (("permute", 1), ("one",))
    vocabulary = d.vocab_size
    return {"token_embedding_table": ("model.embed_tokens.weight", None),
            "rms_att_weight": of("pre_mixer_norm", ("one",)),
            "wq": of("mixer.qkv_proj", (("rows", 0, d.q_dim, rows), ("permute", d.n_heads))),
            "wk": of("mixer.qkv_proj", (("rows", d.q_dim, keys, rows), ("permute", d.n_kv_heads))),
            "wv": of("mixer.qkv_proj", ("rows", keys, rows, rows)), "wo": of("mixer.o_proj"),
            "q_norm": of("mixer.q_norm", head_norm), "k_norm": of("mixer.k_norm", head_norm),
            "rms_att_post_weight": of("post_mixer_norm", ("plus", BEHIND_ATTENTION)),
            "rms_ffn_weight": of("pre_mlp_norm", ("one",)),
            "w1": of("mlp.gate_up_proj", ("rows", 0, ffn, 2 * ffn)), "w2": of("mlp.down_proj"),
            "w3": of("mlp.gate_up_proj", ("rows", ffn, 2 * ffn, 2 * ffn)),
            "rms_ffn_post_weight": of("post_mlp_norm", ("plus", BEHIND_FFN)),
            "rms_final_weight": ("model.norm.weight", ("one",)),
            "wcls": ("lm_head.weight", None if padded(vocabulary) == vocabulary
                     else ("rows", 0, vocabulary, padded(vocabulary)))}


# ---- a GGUF of a PLaMo 3 (T649.4)
def plamo3_gguf_config(key, tensors):
    """What llama.cpp says of a PLaMo 3 (conversion/plamo.py's Plamo3Model and src/models/plamo3.cpp at 67107ac3), by
    the flat names of config.json that plamo3_config() reads: the window as attention.sliding_window, every pattern-th
    layer a full-attention one as attention.sliding_window_pattern (a number), the two bases as rope.freq_base and
    rope.freq_base_swa, yarn's factor and original context. llama.cpp writes yarn's bounds as Plamo3Config makes them
    (32 and 1, not truncated); a GGUF that says others, or another kind of scaling, is refused: plamo3_config() would
    make the model's own of it and say nothing."""
    config = gguf_config(key, tensors)
    scaling = config.pop("rope_scaling", None) or {}
    if scaling and (scaling.get("type") != "yarn" or f32(scaling.get("beta_fast", BETA_FAST)) != f32(BETA_FAST)
                    or f32(scaling.get("beta_slow", BETA_SLOW)) != f32(BETA_SLOW) or key("rope.scaling.yarn_truncate", False)
                    or set(scaling) - {"type", "factor", "original_max_position_embeddings", "beta_fast", "beta_slow"}):
        refuse(f"its GGUF scales its RoPE otherwise than a PLaMo 3: {scaling}")
    config.update(sliding_window_pattern=key("attention.sliding_window_pattern", PATTERN),
                  window_size=key("attention.sliding_window", WINDOW),
                  rope_local_theta=float(key("rope.freq_base_swa", LOCAL_THETA)),
                  rope_scaling_factor=scaling.get("factor", 1),
                  initial_context_length=scaling.get("original_max_position_embeddings"))
    return config


# T649.4: llama.cpp writes every norm of a PLaMo 3 with what the model adds to it (Plamo3Model.modify_tensors): 1 to
# the norms in front of the attention and the FFN, the heads' and the last one, and 1/5 and 5^-1.5 to the two behind,
# which the plan's own "plus" steps are. The two behind are named without a ".weight" (src/models/plamo3.cpp:
# tn(LLM_TENSOR_ATTN_POST_NORM, i)). q, k and v stay one matrix in Hugging Face's order (attn_qkv: llama.cpp turns a
# PLaMo 3 as NeoX does), and the gate and the up of the FFN one too (ffn_up: its swiglu takes the gate as the first half)
PLAMO3_NORMS = ("output_norm.weight", "attn_norm.weight", "ffn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")


def plamo3_gguf_marks(config):
    return {**{name: {"done": "one"} for name in PLAMO3_NORMS},
            "post_attention_norm": {"done": "plus"}, "post_ffw_norm": {"done": "plus"}}


def plamo3_agrees(own, config):
    # which layers are local ones, their window and their base, which are no tensor
    return [local_agrees(own, config)]


PLAMO3 = Family("PLaMo 3", "plamo3", plamo3_sources, plamo3_check, before=plamo3_config, options=local_options,
                free_heads=True, scaled=True,
                gguf="plamo3", gguf_config=plamo3_gguf_config, gguf_marks=plamo3_gguf_marks, agrees=plamo3_agrees,
                gguf_names=({"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
                             "output.weight": "lm_head.weight"},
                            "model.layers.layers.{}.",
                            {"attn_norm": "pre_mixer_norm", "attn_qkv": "mixer.qkv_proj", "attn_q_norm": "mixer.q_norm",
                             "attn_k_norm": "mixer.k_norm", "attn_output": "mixer.o_proj",
                             "post_attention_norm": "post_mixer_norm.weight", "ffn_norm": "pre_mlp_norm",
                             "ffn_up": "mlp.gate_up_proj", "ffn_down": "mlp.down_proj",
                             "post_ffw_norm": "post_mlp_norm.weight"}))
