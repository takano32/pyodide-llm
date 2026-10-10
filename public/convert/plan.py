# The plan of a conversion: which tensor of the source goes where in the checkpoint, and what is done to it on the way.
import numpy as np

from engine.layout import TABLE, Dims, form_of, linear_form, tensor_rows
from engine.layers import rope_frequencies, rope_magnitude, rotated_form, rotated_widths
from convert.config import (PARTLY_TURNED, architecture, convolution_layers, head_size, linear_layers, normalize,
                            rotary_dim)


def transformed(values, transform, head_size):
    """What a tensor of the Hugging Face checkpoint becomes in the checkpoint this engine reads.

    None: nothing (the rows go straight through, as they arrive). ("permute", heads): the head interleaving of
    wq and wk. ("transpose",): GPT-2 stores its matrices the other way round (Conv1D). ("part", i, n): one of
    the n stacked matrices of GPT-2's c_attn, transposed with it; ("row", i, n) the same for its bias.
    ("heads", parts, i, heads, rot): the i-th of the parts each head's rows are stacked in, with the halves of its
    first rot rows interleaved (GPT-NeoX's query_key_value: 3 parts; Qwen3.5's q_proj: q and its gate; 1 part: a k
    or a norm of which RoPE turns part). ("scale", c): every value times c (T253: a Granite's q). T229, Qwen3.5: ("one",) is a norm's weight stored around zero, ("decay",)
    A_log as the engine multiplies it, ("taps",) the convolution's taps, a row for each. A tuple of transforms is
    one after the other.
    """
    if transform is None:
        return values
    if isinstance(transform[0], tuple):
        for step in transform:
            values = transformed(values, step, head_size)
        return values
    if transform[0] == "permute":
        return permute_heads(values, transform[1], head_size)
    if transform[0] == "transpose":
        return values.T
    if transform[0] == "heads":
        # GPT-NeoX's query_key_value holds (heads, 3, head_size, dim) or (heads, 3, head_size): take one of the
        # three, and interleave the halves of the part that RoPE rotates (Hugging Face stores it as rotate_half does)
        parts, index, heads, rot = transform[1:]
        if values.shape[0] % (heads * parts) or rot > values.shape[0] // heads // parts:
            raise ValueError(f"{values.shape[0]} rows are not {heads} heads of {parts} parts that turn {rot} rows.")
        taken = values.reshape(heads, parts, values.shape[0] // heads // parts, -1)[:, index]
        if rot:
            rotated = taken[:, :rot].reshape(heads, 2, rot // 2, -1).transpose(0, 2, 1, 3).reshape(heads, rot, -1)
            taken = np.concatenate([rotated, taken[:, rot:]], axis=1)
        return taken.reshape(-1, values.shape[-1]) if values.ndim > 1 else taken.reshape(-1)
    if transform[0] == "part":
        index, parts = transform[1], transform[2]
        width = values.shape[1] // parts
        return values[:, index * width:(index + 1) * width].T
    if transform[0] == "row":
        index, parts = transform[1], transform[2]
        length = values.shape[0] // parts
        return values[index * length:(index + 1) * length]
    if transform[0] == "scale":
        # T253: a Granite's q by query_scale(), in float32 whatever the tensor was stored as
        return np.asarray(values, dtype=np.float32) * np.float32(transform[1])
    if transform[0] == "one":
        # Qwen3.5's RMSNorm multiplies by 1 + weight: the file holds what the engine's rmsnorm multiplies by
        return np.asarray(values, dtype=np.float32) + np.float32(1.0)
    if transform[0] == "decay":
        # g = -exp(A_log) * softplus(...): the factor, as llama.cpp's GGUF holds it too (ssm_a)
        return -np.exp(np.asarray(values, dtype=np.float32))
    if transform[0] == "taps":
        # conv1d.weight is (channels, 1, taps): a row of all the channels for each tap, the oldest token's first
        return np.asarray(values).reshape(values.shape[0], -1).T
    # a name nobody wrote must not quietly take a slice of rows (T77)
    raise ValueError(f"there is no transform called {transform[0]!r}")


def source_shape(shape, transform):
    """The shape the Hugging Face tensor must have to become a tensor of this shape."""
    if transform is None or transform[0] in ("permute", "one", "decay", "scale"):
        return tuple(shape)
    if isinstance(transform[0], tuple):
        for step in reversed(transform):
            shape = source_shape(shape, step)
        return tuple(shape)
    if transform[0] == "heads":  # one of the stacked parts, and the rows of all of them are one tensor
        return (shape[0] * transform[1], *shape[1:])
    if transform[0] == "taps":
        return (shape[1], 1, shape[0])
    if transform[0] == "transpose":
        return tuple(reversed(shape))
    if transform[0] == "part":
        return (shape[1], shape[0] * transform[2])
    return (shape[0] * transform[2],)


def permute_heads(w, heads, head_size):
    # Hugging Face stores each head of wq/wk as [first halves, second halves] (rotate_half);
    # llama2.c rotates adjacent pairs, so interleave the two halves again. A bias is a vector of the same rows,
    # and -1 as the last dimension lets one line do both.
    if w.shape[0] != heads * head_size:
        # a head of another size reshapes without complaint and turns rows across heads (the review of T124 found
        # half the rows of a Qwen3 0.6B's wq moved so)
        raise ValueError(f"{w.shape[0]} rows are not {heads} heads of {head_size}.")
    return w.reshape(heads, 2, head_size // 2, -1).transpose(0, 2, 1, 3).reshape(w.shape)


def gpt2_prefix(source):
    """openai-community/gpt2 publishes its tensors as wte.weight and h.0...., other GPT-2 models put
    transformer. in front of them. Both are the same model."""
    return "" if "wte.weight" in source else "transformer."


def name_prefix(source, arch):
    """What stands in front of the tensors' names: a GPT-2's "transformer." or nothing, and a Qwen3.5's (T229)
    "model.language_model." (the vision-language checkpoint) or "model." (the language model saved alone)."""
    if arch == "qwen35":
        return "model.language_model." if "model.language_model.embed_tokens.weight" in source else "model."
    return gpt2_prefix(source)


def has_bias(source):
    """Whether this checkpoint has the q, k and v biases of Qwen2 (o and the FFN never have one)."""
    return "model.layers.0.self_attn.q_proj.bias" in source


def has_qk_norm(source):
    """Whether this checkpoint normalizes every head of q and k before RoPE (Qwen3, T124)."""
    return "model.layers.0.self_attn.q_norm.weight" in source


def checkpoint_form(config, source):
    """The form of the checkpoint converted from this config.json and source (llama2_numpy.FORM): what its file will
    not say. head_dim is 0 where the heads fill dim exactly, the way the engine reads a form without one (T144: not
    where dim // heads is the head's size, which a dim that heads do not divide would pass with narrower heads)."""
    config = normalize(config)
    size = head_size(config)
    form = {"bias": has_bias(source), "arch": architecture(config), "qk_norm": has_qk_norm(source),
            "head_dim": 0 if size * config["num_attention_heads"] == config["hidden_size"] else size,
            "linear": linear_layers(config), "rotated": getattr(source, "rotated", None),
            "convolution": convolution_layers(config)}
    if form["rotated"] is not None:
        # T237: a rotated basis is no tensor and no number of config.json: the source says it (a GGUF's metadata).
        # Held to the model here: signs for every width its matrices read, in whole blocks, and no GPT-2's
        if form["arch"] in ("gpt2", "neox", "lfm2"):
            raise ValueError("This model cannot be converted: a GPT-2, a GPT-NeoX or an LFM2 in a rotated basis.")
        rotated_form(form["rotated"], rotated_widths(config["hidden_size"], size * config["num_attention_heads"],
                                                     config["intermediate_size"], linear_form(form["linear"])))
    return form


def conversion_plan(header, form=None, prefix="transformer.", rotary=0, scale=1.0):
    """For every row of the checkpoint (engine/layout.py's tensor_rows()), in file order: the tensors of the Hugging
    Face checkpoint it is made of, in order, as (name, transform); None instead of a list stands for a RoPE table.
    And the shapes of the rows. scale: what q is multiplied by (query_scale(), T253), for a Llama without biases and
    without norms of its heads."""
    form = form_of(form)
    arch, rows, d = form["arch"], tensor_rows(header, form), Dims(header, form)
    if scale != 1.0 and (arch != "llama" or form["bias"] or form["qk_norm"]):
        # a norm of q's heads undoes whatever q was multiplied by, and a bias of q would have to be multiplied too:
        # no Granite has either, and one that had would go through as another model without a word
        raise ValueError("This model cannot be converted: it scales its attention's scores, and has a bias or a "
                         "norm on its queries.")
    sources = SOURCES.get(arch, llama_sources)(d, prefix, rotary, scale)
    plan = []
    for row in rows:
        if row.role == TABLE:
            plan.append(None)
            continue
        # {} in a name: the layer, for a row that stacks the tensors of the layers its "per" says
        if row.name not in sources:
            raise ValueError(f"The converter has no source for the row {row.name} of a {arch}'s checkpoint.")
        name, transform = sources[row.name]
        plan.append([(name.format(layer), transform) for layer in d.layers(row.per)] if row.per else [(name, transform)])
    return plan, [row.shape for row in rows]


# What each row of a layout is made of, by the row's name (so that a row moved in the file takes its source with it):
# (the Hugging Face tensor's name, with {} for the layer where the row is a stack; the transform). A source for a row
# the model has not (a classifier of its own, a Qwen2's biases, a Qwen3's norms of the heads) is asked for by nothing.
def llama_sources(d, prefix, rotary, scale):
    layer = "model.layers.{}."
    turn_q = ("permute", d.n_heads) if scale == 1.0 else (("permute", d.n_heads), ("scale", scale))
    return {"token_embedding_table": ("model.embed_tokens.weight", None),
            "rms_att_weight": (layer + "input_layernorm.weight", None),
            "wq": (layer + "self_attn.q_proj.weight", turn_q),
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


def qwen35_sources(d, prefix, rotary, scale):
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


def lfm2_sources(d, prefix, rotary, scale):
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


def neox_sources(d, prefix, rotary, scale):
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


def gpt2_sources(d, prefix, rotary, scale):
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


SOURCES = {"llama": llama_sources, "qwen35": qwen35_sources, "lfm2": lfm2_sources, "neox": neox_sources,
           "gpt2": gpt2_sources}


def rope_table(config, header, which):
    """The cos (which = 0) or sin (1) table of the legacy format, for float32 and float16 checkpoints.

    GPT-NeoX rotates only rotary_pct of each head (and Qwen3.5), and the angles follow that width. The table keeps the shape
    the layout gives it (head_size // 2 columns); the columns past the rotated part are never read.
    """
    size, seq_len = head_size(config), header[6]
    width = rotary_dim(config) if architecture(config) in PARTLY_TURNED else size
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    frequencies = rope_frequencies(width, config.get("rope_theta", 10000.0), config.get("rope_scaling"))
    table = (np.cos if which == 0 else np.sin)(positions * frequencies) * rope_magnitude(config.get("rope_scaling"))
    if width == size:
        return table
    full = np.zeros((seq_len, size // 2), dtype=np.float64)
    full[:, :width // 2] = table
    return full
