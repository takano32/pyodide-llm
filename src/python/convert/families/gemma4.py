# convert/families/gemma4.py (T331): Google's Gemma 4, the language model of it (config.json's text_config, model_type
# gemma4_text), as engine/layout.py's "gemma4". The formulas are taken from transformers' modeling_gemma4.py and
# configuration_gemma4.py at 2126f65b (Apache-2.0; no line of them): Gemma4TextAttention, Gemma4TextDecoderLayer,
# Gemma4TextModel, Gemma4ForCausalLM, Gemma4RMSNorm, and modeling_rope_utils.py's proportional RoPE.
#
# A Gemma 3's layers (norms behind the attention and the FFN, norms of the heads of q and k, a gated FFN whose
# activation is a GELU, layers that attend a window of the last positions and turn by a base of their own), but: the
# norms are of their weight (x times w, where a Gemma 3's are of 1 + w); v is normalized head by head too, without a
# weight; the scores are not divided by anything (scaling 1.0); every layer ends by multiplying the stream by a number
# of its own (layer_scalar); the full-attention layers have heads of their own size (global_head_dim) and turn the first
# quarter of each by the angles of the whole head (proportional RoPE); the logits are capped by a tanh.
#
# What the engine has not yet (T330, T329) it refuses here: layers that read another layer's keys and values, and an
# embedding of every layer's own. Every published Gemma 4 of the sizes a page can hold has both.
import math

import numpy as np

from convert.families.family import Family, refuse
from convert.families.gemma3 import FULL, LOCAL_THETA, SLIDING, THETA, local
from engine.layers import rope_frequencies

# transformers' Gemma4TextConfig, where config.json says nothing (configuration_gemma4.py)
HEAD, WIDE_HEAD, PROPORTION = 256, 512, 0.25


def language_model(config):
    """config.json of the whole model (gemma4: its language model, and its vision and audio towers): the language
    model's alone, which is one level down. A config without a text_config stays what it is, and is refused by its
    name."""
    text = config.get("text_config")
    return {**text, "model_type": "gemma4_text"} if isinstance(text, dict) else config


def gemma4_config(config):
    """A Gemma 4's config.json by the names the rest of the converter reads, as transformers' Gemma4TextConfig reads
    it: the kinds of the layers listed (every sixth a full-attention one where none are, and the last one always), the
    two kinds of RoPE out of rope_parameters (normalize() would read it as one model's, and take the quarter the
    full-attention layers turn for every layer's) as rope_theta and the quarter (rope_proportion) of the full-attention
    layers and rope_local_base_freq of the others, the size of the full-attention layers' heads, and what transformers
    takes where config.json says nothing. A kind of RoPE that is neither the plain one nor the proportional one stays
    as rope_scaling, which check_config() refuses. Twice is once."""
    layers, kinds = config.get("num_hidden_layers"), config.get("layer_types")
    if kinds is None and isinstance(layers, int):
        kinds = [SLIDING if (layer + 1) % 6 else FULL for layer in range(layers)]
    if isinstance(kinds, list) and kinds:
        kinds = [*kinds[:-1], FULL]  # (transformers makes the last layer a full-attention one, whatever it is said to be)
    rope = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    of = lambda kind: rope.get(kind) if isinstance(rope.get(kind), dict) else {}
    if not rope and "rope_theta" not in config:
        rope = {SLIDING: {"rope_type": "default", "rope_theta": LOCAL_THETA},
                FULL: {"rope_type": "proportional", "partial_rotary_factor": PROPORTION, "rope_theta": THETA}}
    full, sliding = of(FULL), of(SLIDING)
    kind = lambda said: said.get("rope_type", said.get("type", "default"))
    scaling = next(({**said} for said, plain in ((full, ("default", "proportional")), (sliding, ("default",)))
                    if said and kind(said) not in plain), config.get("rope_scaling"))
    proportion = full.get("partial_rotary_factor", PROPORTION) if kind(full) == "proportional" else config.get("rope_proportion", 1.0)
    rest = {key: value for key, value in config.items() if key not in ("rope_parameters", "rope_scaling")}
    return {**rest, "layer_types": kinds, **({"rope_scaling": scaling} if scaling else {}),
            "rope_theta": full.get("rope_theta", config.get("rope_theta", THETA)),
            "rope_local_base_freq": sliding.get("rope_theta", config.get("rope_local_base_freq", LOCAL_THETA)),
            "rope_proportion": proportion,
            "tie_word_embeddings": config.get("tie_word_embeddings", True),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-6), "head_dim": config.get("head_dim") or HEAD,
            "global_head_dim": config.get("global_head_dim") or WIDE_HEAD,
            "hidden_size_per_layer_input": config.get("hidden_size_per_layer_input", 256),
            "num_kv_shared_layers": config.get("num_kv_shared_layers") or 0,
            "use_double_wide_mlp": bool(config.get("use_double_wide_mlp"))}


def full_layers(config):
    config = gemma4_config(config)  # (what transformers takes where config.json says nothing: twice is once)
    return [layer for layer, kind in enumerate(config["layer_types"]) if kind == FULL]


def wide(config):
    """FORM's "wide": the full-attention layers, where their heads are of another size than the others'."""
    layers, config = full_layers(config), gemma4_config(config)
    return {"layers": layers, "head_dim": config["global_head_dim"]} if layers and config["global_head_dim"] != config["head_dim"] else None


def shared_layers(config):
    """The last num_kv_shared_layers layers, which read the keys and values of a layer before them (T330)."""
    config = gemma4_config(config)
    layers = config["num_hidden_layers"]
    return list(range(layers - config["num_kv_shared_layers"], layers)) if config["num_kv_shared_layers"] else []


def double(config):
    """FORM's "double": the layers whose FFN is twice as wide, the ones that read another's keys and values where the
    config says use_double_wide_mlp (Gemma4TextMLP)."""
    return (shared_layers(config) or None) if gemma4_config(config)["use_double_wide_mlp"] else None


def rotary(config):
    """How many values of a full-attention layer's head RoPE turns (an even number): its quarter, of its own size."""
    config = gemma4_config(config)
    size = config["global_head_dim"] if full_layers(config) else config["head_dim"]
    return int(float(config["rope_proportion"]) * size) // 2 * 2


def gemma4_options(config):
    """What the file cannot say: the local layers (as a Gemma 3's), how much of a full-attention layer's head RoPE turns,
    and the cap of the logits."""
    config = gemma4_config(config)
    said = {"rotary": rotary(config)}
    if local(config):
        said["local"] = local(config)
    if config.get("final_logit_softcapping"):
        said["softcap"] = float(config["final_logit_softcapping"])
    return said


def gemma4_check(config):
    """What transformers' Gemma 4 computes and the engine has not, and a config.json that does not say its layers."""
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    if config.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        refuse(f"its activation is {config['hidden_activation']}, not gelu_pytorch_tanh")
    if config.get("attention_bias"):
        refuse("its layers have biases")
    if config.get("use_bidirectional_attention"):
        refuse("it attends in both directions")
    if config.get("attention_k_eq_v") or config.get("enable_moe_block") or config.get("num_global_key_value_heads"):
        refuse("its layers are not the ones of a Gemma 4 E2B or E4B")
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"] or set(kinds) - {SLIDING, FULL}:
        refuse("its layers are not sliding-attention layers and full-attention layers, one kind for every layer")
    window = config.get("sliding_window")
    if SLIDING in kinds and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
        refuse("its config.json has no usable sliding_window")
    if not number(config.get("rope_theta")) or not number(config.get("rope_local_base_freq")) \
            or not number(config.get("rope_proportion")) or config["rope_proportion"] > 1 or rotary(config) < 2:
        refuse("its config.json has no usable base of RoPE")
    size, wide_size = config["head_dim"], config["global_head_dim"]
    if not isinstance(wide_size, int) or isinstance(wide_size, bool) or wide_size <= 0 or wide_size % 2:
        refuse("its attention heads do not divide the hidden size the way llama2.c expects")
    full = full_layers(config)
    if wide(config) and 1 in (len(full), len(kinds) - len(full)):
        # (a stack of one layer is read as no stack: the converter has no file of it, as no LFM2 of one such layer)
        refuse("it has fewer than two layers of each size of its heads")
    if config["num_kv_shared_layers"]:
        refuse("its layers read the keys and values of other layers")
    if config["hidden_size_per_layer_input"]:
        refuse("it has an embedding for every layer")


def gemma4_sources(d, prefix, rotary):
    """By transformers' names of a Gemma4ForCausalLM (prefix "model.") or of the whole model ("model.language_model.").
    q and k, and the norms of their heads, are turned as a Qwen3's (heads of their own size in the wide layers).
    The model multiplies the embedding's row by sqrt(dim), which goes into the table, and out of the last norm where the
    table is the classifier too (as a Gemma 3's: families/gemma3.py). It does not divide the scores by the root of a
    head's size, which the engine does: that root goes into the weight of the norm of q's heads, which q is multiplied
    by last before RoPE, a rotation (and 16 of a head of 256 multiplies exactly)."""
    scale = math.sqrt(d.dim)
    of = lambda name, transform=None: (prefix + "layers.{}." + name + ".weight", transform)
    sources = {"token_embedding_table": (prefix + "embed_tokens.weight", ("scale", scale)),
               "rms_att_weight": of("input_layernorm"),
               "rms_att_post_weight": of("post_attention_layernorm"), "rms_ffn_weight": of("pre_feedforward_layernorm"),
               "rms_ffn_post_weight": of("post_feedforward_layernorm"),
               "layer_scale": (prefix + "layers.{}.layer_scalar", None),
               "rms_final_weight": (prefix + "norm.weight", ("scale", 1.0 / scale) if d.shared else None),
               "wcls": ("lm_head.weight", None)}
    for suffix, size in (("", d.head_size), ("_wide", d.wide_head or d.head_size)):
        head = ("permute", 1, size)
        sources.update({"wq" + suffix: of("self_attn.q_proj", ("permute", d.n_heads, size)),
                        "wk" + suffix: of("self_attn.k_proj", ("permute", d.n_kv_heads, size)),
                        "wv" + suffix: of("self_attn.v_proj"), "wo" + suffix: of("self_attn.o_proj"),
                        "q_norm" + suffix: of("self_attn.q_norm", (head, ("scale", math.sqrt(size)))),
                        "k_norm" + suffix: of("self_attn.k_norm", head)})
    for suffix in ("", "_double"):
        sources.update({"w1" + suffix: of("mlp.gate_proj"), "w2" + suffix: of("mlp.down_proj"),
                        "w3" + suffix: of("mlp.up_proj")})
    return sources


def gemma4_table(config, header, which):
    """The cos (which 0) or sin (1) RoPE table of a float32 or float16 file: of the full-attention layers, whose heads
    are the wide ones where there are such, the first rotary values of a head by the angles of its whole width (the
    rest of the table, which is never read, 0)."""
    size, seq_len = config["global_head_dim"] if full_layers(config) else config["head_dim"], header[6]
    angles = np.arange(seq_len, dtype=np.float64)[:, None] * rope_frequencies(size, config["rope_theta"])[:rotary(config) // 2]
    table = np.zeros((seq_len, size // 2), dtype=np.float64)
    table[:, :rotary(config) // 2] = (np.cos if which == 0 else np.sin)(angles)
    return table


def gemma4_prefix(source):
    """"model.language_model." (the whole model's checkpoint) or "model." (the language model saved alone)."""
    return "model.language_model." if "model.language_model.embed_tokens.weight" in source else "model."


GEMMA4 = Family("Gemma 4", "gemma4", gemma4_sources, gemma4_check, before=gemma4_config, options=gemma4_options,
                form={"wide": wide, "double": double}, prefix=gemma4_prefix, table=gemma4_table,
                free_heads=True, rms_norm=True)
# config.json of the whole model as it is published (gemma4: the language model and its towers of sight and sound): its
# text_config is the Gemma 4 above
GEMMA4_WHOLE = GEMMA4._replace(renamed=True, before=lambda config: gemma4_config(language_model(config)))
