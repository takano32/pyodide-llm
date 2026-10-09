# config.json as the converter reads it: the architectures it takes, what it refuses, and what the checkpoint's form
# is for each (what the file itself does not say).
import math

from llama2_numpy import linear_form


def architecture(config):
    """Which set of tensors and which forward: "llama" (Qwen2 is a Llama with biases), "gpt2", or "neox"
    (GPT-NeoX: a GPT-2 with RoPE over part of each head, and optionally the two branches in parallel), or "qwen35"
    (T229: Qwen3.5 and Qwen3.8, a Qwen3 most of whose layers are linear-attention ones), or "lfm2" (T260: Liquid AI's
    LFM2 and LFM2.5, a Qwen3 some of whose layers are convolution layers)."""
    return {"gpt2": "gpt2", "gpt_neox": "neox", "qwen3_5": "qwen35", "qwen3_5_text": "qwen35",
            "lfm2": "lfm2"}.get(config.get("model_type"), "llama")


def head_size(config):
    """The size of an attention head: config.json's head_dim where it says one (Qwen3 0.6B: 128 in a dim of 1024,
    T124), else dim / heads. "head_dim": null says as much as no head_dim at all (cyberagent/CAT-Translate-7b)."""
    return config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]


def rotary_dim(config):
    """How many of each head's values GPT-NeoX rotates (rotary_pct of them, an even number), and Qwen3.5 (T229: its
    partial_rotary_factor, which normalize() gives the same name)."""
    return int(head_size(config) * float(config.get("rotary_pct", 1.0))) // 2 * 2


# T253: a Granite (IBM's, transformers' model_type "granite") is a Llama but for four numbers of its config.json. Its
# attention multiplies the scores by attention_multiplier where a Llama divides them by the root of the head's size
# (transformers' GraniteAttention: matmul(query, key^T) * config.attention_multiplier), its embedding is multiplied by
# embedding_multiplier, each branch by residual_multiplier before it joins the stream, and its logits are divided by
# logits_scaling. The engine has none of the four; the first goes into the weights (query_scale()), and a model whose
# other three are not 1 is refused (Granite 3.x and 4.1: 12, 0.22 and a scaling of the logits, with the embedding and
# the classifier one table, which no scaling of that table makes right for both).
GRANITE_ONES = ("embedding_multiplier", "residual_multiplier", "logits_scaling")


def query_scale(config):
    """What the conversion multiplies q by (T253), 1.0 for every model but a Granite. The engine's score is q·k /
    sqrt(head), a Granite's q·k * attention_multiplier: with q multiplied by attention_multiplier * sqrt(head) the
    engine computes the Granite's. Nothing stands between the matrix and the score that is not linear in q (RoPE turns
    it; a norm of the heads, which would undo the scale, a Granite has not and conversion_plan() refuses with it), so
    it is the same model, and its file and its options are a Llama's: no kernel, no shader and no option knows of it.
    Granite 4.2 3B's is 1/64 * 8, a power of two, which changes no bit of a value but its exponent."""
    if config.get("model_type") != "granite":
        return 1.0
    return float(config.get("attention_multiplier", 1.0)) * math.sqrt(head_size(config))


# the architectures that turn part of each head only: the options say how much (rotary)
PARTLY_TURNED = ("neox", "qwen35")


def unturned_layers(config):
    """T255: the layers of a SmolLM3 (transformers' model_type "smollm3", a Llama otherwise) whose q and k RoPE does not
    turn, in order: where config.json's no_rope_layers has a 0 (the name is transformers': a 1 is a layer that has
    RoPE), and where it has no such list every no_rope_layer_interval-th layer, as transformers' SmolLM3Config makes
    the list (and llama.cpp, whose interval is always 4). The file is a Llama's and the same either way (turning the
    rows of q and k into llama2.c's order changes no score of a layer that turns nothing): the options say it (unturned).
    None of another model's."""
    if config.get("model_type") != "smollm3":
        return []
    said = config.get("no_rope_layers")
    if said is None:
        interval = config.get("no_rope_layer_interval", 4)
        said = [int((layer + 1) % interval != 0) for layer in range(config["num_hidden_layers"])]
    return [layer for layer, turns in enumerate(said) if not turns]
# transformers' Qwen3_5TextConfig, where config.json leaves one out
LINEAR_DEFAULTS = {"linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                   "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}


def linear_layers(config):
    """FORM's "linear" from a Qwen3.5's (normalized) config.json: which layers attend over all positions, and the
    heads and the convolution of the others (T229). layer_types is every full_attention_interval-th layer a
    full-attention one in every published model; any other order is refused, for the file is laid out by that one
    number. None for the other architectures."""
    if architecture(config) != "qwen35":
        return None
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


def lfm2_config(config):
    """T260: an LFM2's config.json by the names the rest of this file reads, as transformers' Lfm2Config and Lfm2MLP
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


def convolution_layers(config):
    """FORM's "convolution" from an LFM2's (normalized) config.json: a letter for every layer, "c" for a convolution
    layer and "a" for one that attends, and the taps of the convolution (conv_L_cache, 3 where none is said, as
    transformers' Lfm2Config has it). None for the other architectures."""
    if architecture(config) != "lfm2":
        return None
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config.get("num_hidden_layers") or set(kinds) - {"conv", "full_attention"}:
        raise ValueError("This model cannot be converted: its layers are not convolution layers and attention layers, "
                         "one kind for every layer.")
    taps = config.get("conv_L_cache", 3)
    if not isinstance(taps, int) or isinstance(taps, bool) or taps < 2:
        raise ValueError("This model cannot be converted: its config.json has no usable conv_L_cache.")
    return {"layers": "".join("c" if kind == "conv" else "a" for kind in kinds), "taps": taps}


def yarn(config):
    """What a config.json whose RoPE scaling is yarn says of it besides its kind (T235), else None."""
    scaling = config.get("rope_scaling") or {}
    if scaling.get("rope_type", scaling.get("type")) != "yarn":
        return None
    return {key: value for key, value in scaling.items() if key not in ("rope_type", "type") and value is not None}


def normalize(config):
    """GPT-2 spells its config.json differently: give it the names the rest of this file uses."""
    if config.get("model_type") == "qwen3_5" and isinstance(config.get("text_config"), dict):
        # T229: Qwen3.5 is a vision-language model, and the language model's config is one level down (the vision
        # model's is not read: text alone). It names no BOS: every text here begins with one (T131), and that is the
        # end-of-text token, as the config.json of Qwen3.8-27B says (bos_token_id 248044, its eos_token_id) and as a
        # Qwen2.5's and a Qwen3's do
        text = config["text_config"]
        config = {**text, "model_type": "qwen3_5_text"}
        end = text.get("eos_token_id")
        end = end[0] if isinstance(end, list) and end else end
        if config.get("bos_token_id") is None and isinstance(end, int):
            config["bos_token_id"] = end
    if config.get("model_type") == "lfm2":
        config = lfm2_config(config)  # T260
    rope = config.get("rope_parameters")
    if isinstance(rope, dict):
        # transformers 5 writes rope_theta, rope_scaling and GPT-NeoX's rotary_pct as one rope_parameters. Unread,
        # such a config.json ran at theta 10000, unscaled and rotating whole heads, and wrote nonsense (the review
        # of T106): the old names are what this file reads
        scaling = {key: value for key, value in rope.items() if key not in ("rope_theta", "partial_rotary_factor")}
        config = {**config, "rope_theta": rope.get("rope_theta", config.get("rope_theta", 10000.0))}
        if scaling.get("rope_type", scaling.get("type", "default")) != "default":
            config["rope_scaling"] = scaling
        if "partial_rotary_factor" in rope:
            config["rotary_pct"] = rope["partial_rotary_factor"]
    if config.get("model_type") == "qwen3_5_text":
        # (the review of T229) what transformers' Qwen3_5TextConfig says where a config.json leaves it out: RoPE over a
        # quarter of a head (partial_rotary_factor, at the top where there is no rope_parameters: it is the config's
        # own name for it) and heads of 256. Every published Qwen3.5 and Qwen3.8 config.json says both, but a config
        # that did not would have run with whole heads turning (and sizes dim / heads), without a word
        config = {**config, "rotary_pct": config.get("rotary_pct", config.get("partial_rotary_factor", 0.25))}
        if config.get("head_dim") is None:
            config["head_dim"] = 256
    if config.get("model_type") == "mistral":
        # T125: a Mistral is a Llama by another name (the same tensors, names and forward). v0.1 and some of its
        # descendants attend a sliding window of the last sliding_window positions: a context no longer than the
        # window attends the very same positions, so the context is cut to it
        window, context = config.get("sliding_window"), config.get("max_position_embeddings")
        config = {**config, "model_type": "llama"}
        if isinstance(window, int) and window > 0 and isinstance(context, int):
            config["max_position_embeddings"] = min(context, window)
        return config
    if config.get("model_type") == "gpt_neox":
        # GPT-NeoX has the Llama names already; only the angles are spelled differently
        return {**config, "rope_theta": config.get("rotary_emb_base", config.get("rope_theta", 10000.0)),
                "tie_word_embeddings": config.get("tie_word_embeddings", False)}
    if config.get("model_type") != "gpt2":
        return config
    dim = config.get("n_embd")
    return {**config, "hidden_size": dim, "intermediate_size": config.get("n_inner") or (4 * dim if dim else None),
            "num_hidden_layers": config.get("n_layer"), "num_attention_heads": config.get("n_head"),
            "max_position_embeddings": config.get("n_positions") or config.get("n_ctx"),
            "hidden_act": "gelu", "tie_word_embeddings": config.get("tie_word_embeddings", True)}


def check_config(config):
    """ValueError, in words for the visitor, unless this config.json describes a model the engine can run."""
    def refuse(reason):
        raise ValueError(f"This model cannot be converted: {reason}.")

    # qwen2 is a Llama with a bias on q, k and v: the converter writes those three vectors per layer, the engine
    # adds them after the projections (T64). Everything else about it is the same. qwen3 is a Llama that normalizes
    # every head of q and k (T124): two vectors per layer, the same way.
    # qwen3_5 (T229) is a Qwen3 most of whose layers are linear-attention ones (normalize() lifted its text_config).
    # granite (T253) is a Llama whose scores are scaled otherwise, which the conversion puts into q (query_scale()).
    # lfm2 (T260) is a Qwen3 some of whose layers are convolution layers (normalize() gave it the Llama's names).
    # smollm3 (T255) is a Llama some of whose layers RoPE leaves alone (unturned_layers()).
    if config.get("model_type") not in ("llama", "qwen2", "qwen3", "gpt2", "gpt_neox", "qwen3_5_text", "granite", "lfm2",
                                        "smollm3"):
        refuse(f"it is a {config.get('model_type', 'model of unknown type')}, and only Llama, Mistral, Granite, SmolLM3, "
               f"Qwen2, Qwen3, Qwen3.5, LFM2, GPT-2 and GPT-NeoX models are supported")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
                "max_position_embeddings"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            refuse(f"its config.json has no usable {key}")
    dim, n_heads = config["hidden_size"], config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    size = head_size(config)
    # a head of another size than dim / n_heads (T124) only where q and o are matrices of their own: a Llama's.
    # GPT-2's c_attn and GPT-NeoX's query_key_value are cut into heads of dim / n_heads
    divides = not dim % n_heads and size == dim // n_heads
    if not isinstance(size, int) or size <= 0 or size % 2 or n_heads % n_kv_heads \
            or not (divides or (config.get("head_dim") and architecture(config) in ("llama", "qwen35"))):
        refuse("its attention heads do not divide the hidden size the way llama2.c expects")
    scaling = config.get("rope_scaling")
    if scaling and (architecture(config) != "llama"
                    or scaling.get("rope_type", scaling.get("type")) not in ("llama3", "linear", "yarn")):
        # Llama 3's, the linear one and yarn are the kinds the RoPE tables know (llama2_numpy.rope_frequencies)
        refuse(f"it uses RoPE scaling of the {scaling.get('rope_type', scaling.get('type'))} kind")
    said = yarn(config)
    if said is not None:
        # T235: yarn as Ternary-Bonsai's config.json says it, a factor and the original context. What else transformers
        # reads of a yarn (attention_factor, mscale, mscale_all_dim, beta_fast, beta_slow, truncate) changes the angles
        # or how much the turned values are scaled, and the tables know none of it
        for key in sorted(set(said) - {"factor", "original_max_position_embeddings"}):
            refuse(f"its yarn RoPE scaling sets {key}, which the engine does not read")
        if not all(isinstance(said.get(key), (int, float)) and said[key] > 0 for key in ("factor", "original_max_position_embeddings")):
            refuse("its yarn RoPE scaling names no factor or no original context")
    if architecture(config) == "neox":
        if config.get("hidden_act", "gelu") not in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
            refuse(f"its activation is {config['hidden_act']}, and only GELU is supported")
        if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
            refuse("it has grouped-query attention, which GPT-NeoX models do not")
        if rotary_dim(config) < 2:
            refuse("it rotates none of each head")
        return
    if architecture(config) == "gpt2":
        # GPT-2 has one kind of everything; only the activation could be something the GELU kernel is not
        # gelu_fast (T126: rinna/japanese-gpt-1b) is gelu_new's tanh approximation written another way
        if config.get("activation_function", "gelu_new") not in ("gelu_new", "gelu", "gelu_pytorch_tanh", "gelu_fast"):
            refuse(f"its activation is {config['activation_function']}, and only GELU is supported")
        if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
            refuse("it has grouped-query attention, which GPT-2 models do not")
        return
    if config.get("hidden_act", "silu") != "silu":
        refuse(f"its activation is {config['hidden_act']}, not silu")
    if config.get("mlp_bias") or (config.get("attention_bias") and config.get("model_type") != "qwen2"):
        refuse("its layers have biases")
    if config.get("use_sliding_window"):
        refuse("it uses a sliding window of attention")
    if config.get("model_type") == "smollm3":
        # T255: a list of another length than the layers, or an interval that is no number, names no layers
        said, interval = config.get("no_rope_layers"), config.get("no_rope_layer_interval", 4)
        if said is None and (not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0):
            refuse("its config.json has no usable no_rope_layer_interval")
        if said is not None and (not isinstance(said, list) or len(said) != config["num_hidden_layers"]):
            refuse("its no_rope_layers does not name every layer")
    if config.get("model_type") == "granite":
        # T253: what the engine has not (see GRANITE_ONES), and a multiplier of the scores that is no number to scale
        # q by. transformers' Granite has heads of dim / heads only (its config has no head_dim)
        number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for key in GRANITE_ONES:
            if not number(config.get(key, 1.0)) or config.get(key, 1.0) != 1.0:
                refuse(f"its {key} is {config[key]}, and of a Granite's multipliers the engine has the attention's only")
        if not number(config.get("attention_multiplier", 1.0)) or config.get("attention_multiplier", 1.0) <= 0:
            refuse("its config.json has no usable attention_multiplier")
        if not divides:
            refuse("its attention heads do not divide the hidden size the way a Granite's do")
    if architecture(config) == "lfm2":
        # T260: what transformers' Lfm2 has and the engine has not (a bias in the convolution layers: conv_bias, which
        # no published model sets), and what conversion_plan() cannot tell apart (a single layer of a kind is one
        # tensor to it, as the only layer of a Llama is: the review of T229)
        layers = convolution_layers(config)["layers"]
        if config.get("conv_bias"):
            refuse("its convolution layers have biases")
        if min(layers.count("c"), layers.count("a")) < 2:
            refuse("it has fewer than two convolution layers or fewer than two attention layers")
    if architecture(config) == "qwen35":
        linear = linear_form(linear_layers(config))
        if config["num_hidden_layers"] < linear["every"]:
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


def checkpoint_header(config, source, max_seq_len):
    """The 7 ints of the legacy header. A negative vocabulary size signals a classifier of its own (llama2.c)."""
    config = normalize(config)
    # GPT-NeoX calls its classifier embed_out, everyone else lm_head
    classifier = "embed_out.weight" if architecture(config) == "neox" else "lm_head.weight"
    shared_classifier = config.get("tie_word_embeddings", False) or classifier not in source
    if architecture(config) == "gpt2":
        # the learned positions are a table of exactly n_positions rows: the context cannot be cut short
        max_seq_len = config["max_position_embeddings"]
    vocab_size = config["vocab_size"]
    # the KV cache grows with seq_len, so a long context can be cut down for the browser
    return (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
            config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size if shared_classifier else -vocab_size, min(config["max_position_embeddings"], max_seq_len))
