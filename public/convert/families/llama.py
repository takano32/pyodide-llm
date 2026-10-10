# The Llamas (T359): a Llama, and the families that are one but for a little. A Mistral is a Llama by another name
# (T125). A Qwen2 is a Llama with a bias on q, k and v: the converter writes those three vectors per layer, the engine
# adds them after the projections (T64); everything else about it is the same. A Qwen3 is a Llama that normalizes every
# head of q and k (T124): two vectors per layer, the same way. A Granite (T253) is a Llama whose scores are scaled
# otherwise, which the conversion puts into q. A SmolLM3 (T255) is a Llama some of whose layers RoPE leaves alone.
# Whether a file has the biases or the norms is the source's to say, not config.json's (convert/plan.py's has_bias()
# and has_qk_norm()), so a Qwen2 and a Qwen3 differ from a Llama here by what is checked and how a GGUF holds them.
import math

from convert.families.family import Family, as_stored, f32, head_size, refuse


# What each row of a layout is made of, by the row's name (so that a row moved in the file takes its source with it):
# (the Hugging Face tensor's name, with {} for the layer where the row is a stack; the transform). A source for a row
# the model has not (a classifier of its own, a Qwen2's biases, a Qwen3's norms of the heads) is asked for by nothing.
def llama_sources(d, prefix, rotary):
    layer = "model.layers.{}."
    return {"token_embedding_table": ("model.embed_tokens.weight", None),
            "rms_att_weight": (layer + "input_layernorm.weight", None),
            "wq": (layer + "self_attn.q_proj.weight", ("permute", d.n_heads)),
            "wk": (layer + "self_attn.k_proj.weight", ("permute", d.n_kv_heads)),
            "wv": (layer + "self_attn.v_proj.weight", None), "wo": (layer + "self_attn.o_proj.weight", None),
            "rms_ffn_weight": (layer + "post_attention_layernorm.weight", None),
            "w1": (layer + "mlp.gate_proj.weight", None), "w2": (layer + "mlp.down_proj.weight", None),
            "w3": (layer + "mlp.up_proj.weight", None),
            "rms_final_weight": ("model.norm.weight", None), "wcls": ("lm_head.weight", None),
            "bq": (layer + "self_attn.q_proj.bias", ("permute", d.n_heads)),
            "bk": (layer + "self_attn.k_proj.bias", ("permute", d.n_kv_heads)),
            "bv": (layer + "self_attn.v_proj.bias", None),
            # one weight for every head, over the rows of a head: interleaved like the rows it multiplies
            "q_norm": (layer + "self_attn.q_norm.weight", ("permute", 1)),
            "k_norm": (layer + "self_attn.k_norm.weight", ("permute", 1))}


def check(config, biased=False):
    """What a Llama's layers must be, and those of the families whose layers are a Llama's with more (a Qwen3.5's, an
    LFM2's): a gated FFN of silu, no biases (biased: but the attention's, a Qwen2's), no sliding window."""
    if config.get("hidden_act", "silu") != "silu":
        refuse(f"its activation is {config['hidden_act']}, not silu")
    if config.get("mlp_bias") or (config.get("attention_bias") and not biased):
        refuse("its layers have biases")
    if config.get("use_sliding_window"):
        refuse("it uses a sliding window of attention")


def windowed(config):
    """T125: a Mistral is a Llama by another name (the same tensors, names and forward). v0.1 and some of its
    descendants attend a sliding window of the last sliding_window positions: a context no longer than the
    window attends the very same positions, so the context is cut to it."""
    window, context = config.get("sliding_window"), config.get("max_position_embeddings")
    config = {**config, "model_type": "llama"}
    if isinstance(window, int) and window > 0 and isinstance(context, int):
        config["max_position_embeddings"] = min(context, window)
    return config


# ---- a GGUF of a Llama
GGUF_LAYER = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
              "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
              "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
GGUF_NAMES = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
              "output.weight": "lm_head.weight"}


def gguf_config(key, tensors):
    """config.json of a Llama as its GGUF's metadata says it, and of every family llama.cpp writes with a Llama's keys
    (a Qwen3.5 and an LFM2 too, which add their own to this)."""
    heads = key("attention.head_count")
    config = {"hidden_size": key("embedding_length"), "intermediate_size": key("feed_forward_length"),
              "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
              "num_key_value_heads": key("attention.head_count_kv", heads),
              "max_position_embeddings": key("context_length"), "rope_theta": float(key("rope.freq_base", 10000.0)),
              "tie_word_embeddings": "output.weight" not in tensors, "hidden_act": "silu",
              # a head of another size than dim / heads (T124): llama.cpp says it as the length of a key
              "head_dim": key("attention.key_length"), "rms_norm_eps": key("attention.layer_norm_rms_epsilon")}
    if key("rope.scaling.type", "none") not in ("none", None):
        config["rope_scaling"] = {"type": key("rope.scaling.type"), "factor": key("rope.scaling.factor", 1.0)}
        if key("rope.scaling.type") == "yarn":
            # T235: yarn's other numbers, by config.json's names. llama.cpp takes the trained context where the
            # GGUF names no original one; what else a GGUF may say of yarn, check_config() refuses by these names
            # and gguf_agrees() where the original's config.json has it not. The keys are llama.cpp's own (the
            # fork's src/llama-arch.cpp: yarn_log_multiplier, which only a DeepSeek-V2 GGUF has; the review of
            # T235 found "yarn_log_mul" here, a name no GGUF has)
            config["rope_scaling"]["original_max_position_embeddings"] = \
                key("rope.scaling.original_context_length", key("context_length"))
            for name, ours in (("attn_factor", "attention_factor"), ("yarn_log_multiplier", "mscale_all_dim")):
                if key(f"rope.scaling.{name}") is not None:
                    config["rope_scaling"][ours] = key(f"rope.scaling.{name}")
    return config


def gguf_turned(config):
    """llama.cpp turns q and k of a Llama (and their biases) into llama2.c's order; a Qwen2 it leaves alone (it
    rotates the other way at run time). tests/gguf_check.py found SmolLM2's turned. A Granite's as a Llama's (T253:
    its converter is the Llama's), and a SmolLM3's (T255)."""
    turns = {"attn_q": config["num_attention_heads"], "attn_k": config.get("num_key_value_heads")}

    def stored(entry, info, target, parts):
        if len(parts) == 4 and parts[2] in turns:
            entry["turned"] = turns[parts[2]]
    return stored


LLAMA = Family("Llama", "llama", llama_sources, check, free_heads=True, scaled=True, rotatable=True, rms_norm=True,
               gguf="llama", gguf_names=(GGUF_NAMES, "model.layers.{}.", GGUF_LAYER), gguf_config=gguf_config,
               gguf_stored=gguf_turned)
# (a Mistral's GGUF says "llama")
MISTRAL = LLAMA._replace(title="Mistral", renamed=True, after=windowed, gguf=None)
QWEN2 = LLAMA._replace(title="Qwen2", check=lambda config: check(config, biased=True), gguf="qwen2", gguf_stored=as_stored)
# T203 (T136's fourth stage): a Qwen3 is a Qwen2 without the biases that normalizes each head of q and k (T124).
# llama.cpp leaves q, k and the two norms in Hugging Face's order, as a Qwen2's; the head's size is key_length
QWEN3 = QWEN2._replace(title="Qwen3", check=check, gguf="qwen3", gguf_names=(
    GGUF_NAMES, "model.layers.{}.", {**GGUF_LAYER, "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}))


# ---- a Granite
# T253: a Granite (IBM's, transformers' model_type "granite") is a Llama but for four numbers of its config.json. Its
# attention multiplies the scores by attention_multiplier where a Llama divides them by the root of the head's size
# (transformers' GraniteAttention: matmul(query, key^T) * config.attention_multiplier), its embedding is multiplied by
# embedding_multiplier, each branch by residual_multiplier before it joins the stream, and its logits are divided by
# logits_scaling. The engine has none of the four; the first goes into the weights (granite_scale()), and a model whose
# other three are not 1 is refused (Granite 3.x and 4.1: 12, 0.22 and a scaling of the logits, with the embedding and
# the classifier one table, which no scaling of that table makes right for both).
GRANITE_ONES = ("embedding_multiplier", "residual_multiplier", "logits_scaling")


def granite_scale(config):
    """What the conversion multiplies a Granite's q by (T253). The engine's score is q·k / sqrt(head), a Granite's
    q·k * attention_multiplier: with q multiplied by attention_multiplier * sqrt(head) the engine computes the
    Granite's. Nothing stands between the matrix and the score that is not linear in q (RoPE turns it; a norm of the
    heads, which would undo the scale, a Granite has not and conversion_plan() refuses with it), so it is the same
    model, and its file and its options are a Llama's: no kernel, no shader and no option knows of it. Granite 4.2
    3B's is 1/64 * 8, a power of two, which changes no bit of a value but its exponent."""
    return float(config.get("attention_multiplier", 1.0)) * math.sqrt(head_size(config))


def granite_check(config):
    """T253: what the engine has not (see GRANITE_ONES), and a multiplier of the scores that is no number to scale
    q by. transformers' Granite has heads of dim / heads only (its config has no head_dim)."""
    check(config)
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    for key in GRANITE_ONES:
        if not number(config.get(key, 1.0)) or config.get(key, 1.0) != 1.0:
            refuse(f"its {key} is {config[key]}, and of a Granite's multipliers the engine has the attention's only")
    if not number(config.get("attention_multiplier", 1.0)) or config.get("attention_multiplier", 1.0) <= 0:
        refuse("its config.json has no usable attention_multiplier")
    dim, heads = config["hidden_size"], config["num_attention_heads"]
    if dim % heads or head_size(config) != dim // heads:
        refuse("its attention heads do not divide the hidden size the way a Granite's do")


def granite_gguf_config(key, tensors):
    """T253: a Granite's four multipliers by config.json's names. llama.cpp keeps the scores' in the metadata
    (attention.scale) and multiplies at run time: q is not scaled in the file, and the conversion scales it once, as it
    does a safetensors' (granite_scale()). Where a GGUF names none llama.cpp divides by the root of the head's size, a
    Llama's score; the other three it leaves out of the computation where they are missing or 0 (logit_scale it
    requires)."""
    config, heads = gguf_config(key, tensors), key("attention.head_count")
    size = key("embedding_length") // heads if key("embedding_length") and heads else 0
    config["attention_multiplier"] = key("attention.scale") or (1.0 / math.sqrt(size) if size else None)
    for name, ours in (("embedding_scale", "embedding_multiplier"), ("residual_scale", "residual_multiplier"),
                       ("logit_scale", "logits_scaling")):
        config[ours] = key(name) or 1.0
    return config


def granite_agrees(own, config):
    """T253: a Granite's multipliers, which are no tensor: the scores' goes into q from config.json's (a GGUF that
    says another would be scaled by the wrong one), and the three the engine has not must be 1 in both."""
    multipliers = lambda c: {key: f32(c.get(key, 1.0)) for key in ("attention_multiplier", *GRANITE_ONES)
                             if isinstance(c.get(key, 1.0), (int, float))}
    return [("Granite's multipliers", multipliers(own), multipliers(config))]


# T253: a Granite, a Llama to the name of every tensor (llama.cpp's GraniteModel is its LlamaModel with four numbers
# more in the metadata, and turns q and k as that does)
GRANITE = LLAMA._replace(title="Granite", check=granite_check, scale=granite_scale, gguf="granite",
                         gguf_config=granite_gguf_config, agrees=granite_agrees)


# ---- a SmolLM3
def smollm3_unturned(config):
    """T255: the layers of a SmolLM3 (transformers' model_type "smollm3", a Llama otherwise) whose q and k RoPE does not
    turn, in order: where config.json's no_rope_layers has a 0 (the name is transformers': a 1 is a layer that has
    RoPE), and where it has no such list every no_rope_layer_interval-th layer, as transformers' SmolLM3Config makes
    the list (and llama.cpp, whose interval is always 4). The file is a Llama's and the same either way (turning the
    rows of q and k into llama2.c's order changes no score of a layer that turns nothing): the options say it (unturned)."""
    said = config.get("no_rope_layers")
    if said is None:
        interval = config.get("no_rope_layer_interval", 4)
        said = [int((layer + 1) % interval != 0) for layer in range(config["num_hidden_layers"])]
    return [layer for layer, turns in enumerate(said) if not turns]


def smollm3_options(config):
    # the file does not say which layers RoPE leaves alone (only where there are any)
    left = smollm3_unturned(config)
    return {"unturned": left} if left else {}


def smollm3_check(config):
    """T255: a list of another length than the layers, or an interval that is no number, names no layers."""
    check(config)
    said, interval = config.get("no_rope_layers"), config.get("no_rope_layer_interval", 4)
    if said is None and (not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0):
        refuse("its config.json has no usable no_rope_layer_interval")
    if said is not None and (not isinstance(said, list) or len(said) != config["num_hidden_layers"]):
        refuse("its no_rope_layers does not name every layer")


def smollm3_gguf_config(key, tensors):
    # T255: llama.cpp leaves every fourth layer's q and k unturned, whatever the GGUF says (it says nothing:
    # src/models/smollm3.cpp at 71ad0590 sets n_no_rope_layer_step to 4), which is what config.json's
    # interval of 4 says. gguf_agrees() compares the layers with the original's
    return {**gguf_config(key, tensors), "no_rope_layer_interval": 4}


# T255: a SmolLM3, a Llama to the name of every tensor (llama.cpp's SmolLM3Model is its LlamaModel by another name,
# and turns q and k as that does; conversion/llama.py at 71ad0590). What a GGUF and the original must agree on: the
# layers RoPE leaves alone, which are no tensor (llama.cpp's are every fourth, always)
SMOLLM3 = LLAMA._replace(title="SmolLM3", check=smollm3_check, options=smollm3_options, gguf="smollm3",
                         gguf_config=smollm3_gguf_config,
                         agrees=lambda own, config: [("layers without RoPE", smollm3_unturned(own), smollm3_unturned(config))])
