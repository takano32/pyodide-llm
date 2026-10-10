# config.json as the converter reads it: the flow every family shares. What a family reads and refuses of its own is
# in convert/families/ (T359), looked up here by the config's model_type.
from convert.families import FAMILIES, family_of, named
from convert.families.family import head_size, refuse, rotary_dim, yarn  # noqa: F401


def architecture(config):
    """Which set of tensors and which forward: "llama" (Qwen2 is a Llama with biases), "gpt2", or "neox"
    (GPT-NeoX: a GPT-2 with RoPE over part of each head, and optionally the two branches in parallel), or "qwen35"
    (T229: Qwen3.5 and Qwen3.8, a Qwen3 most of whose layers are linear-attention ones), or "lfm2" (T260: Liquid AI's
    LFM2 and LFM2.5, a Qwen3 some of whose layers are convolution layers)."""
    return family_of(config).arch


# What a family says of a config.json, asked by the names these had as functions of every config: each is the
# family's answer, and nothing of another family's.
def query_scale(config):
    """What the conversion multiplies q by (T253), 1.0 for every model but a Granite (families/llama.py)."""
    return family_of(config).scale(config)


def unturned_layers(config):
    """T255: the layers of a SmolLM3 whose q and k RoPE does not turn, in order (families/llama.py). None of another
    model's."""
    return family_of(config).options(config).get("unturned", [])


def made(config, key):
    make = family_of(config).form.get(key)
    return make(config) if make else None


def linear_layers(config):
    """FORM's "linear" from a Qwen3.5's (normalized) config.json (families/qwen35.py). None for the other
    architectures."""
    return made(config, "linear")


def convolution_layers(config):
    """FORM's "convolution" from an LFM2's (normalized) config.json (families/lfm2.py). None for the other
    architectures."""
    return made(config, "convolution")


def normalize(config):
    """config.json by the names the rest of the converter reads, whatever its family calls them (GPT-2 spells its
    config.json differently). Twice is once."""
    config = family_of(config).before(config)
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
    # (the family again: before() may have given the config the model_type of what it holds, a Qwen3.5's text_config)
    return family_of(config).after(config)


def check_config(config):
    """ValueError, in words for the visitor, unless this config.json describes a model the engine can run."""
    model_type = config.get("model_type")
    family = FAMILIES.get(model_type) if isinstance(model_type, str) else None  # (a list or a dict is no name)
    if family is None or family.renamed:
        refuse(f"it is a {config.get('model_type', 'model of unknown type')}, and only {named(FAMILIES.values())} "
               f"models are supported")
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
            or not (divides or (config.get("head_dim") and family.free_heads)):
        refuse("its attention heads do not divide the hidden size the way llama2.c expects")
    scaling = config.get("rope_scaling")
    if scaling and (not family.scaled or scaling.get("rope_type", scaling.get("type")) not in ("llama3", "linear", "yarn")):
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
    family.check(config)


def checkpoint_header(config, source, max_seq_len):
    """The 7 ints of the legacy header. A negative vocabulary size signals a classifier of its own (llama2.c)."""
    config = normalize(config)
    family = family_of(config)
    # (GPT-NeoX calls its classifier embed_out, everyone else lm_head)
    shared_classifier = config.get("tie_word_embeddings", False) or family.classifier not in source
    if family.fixed_context:
        # the learned positions are a table of exactly n_positions rows: the context cannot be cut short
        max_seq_len = config["max_position_embeddings"]
    vocab_size = config["vocab_size"]
    # the KV cache grows with seq_len, so a long context can be cut down for the browser
    return (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
            config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size if shared_classifier else -vocab_size, min(config["max_position_embeddings"], max_seq_len))
