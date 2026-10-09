# The plan of a conversion: which tensor of the source goes where in the checkpoint, and what is done to it on the way.
import numpy as np

from llama2_numpy import (convolution_form, form_of, layer_slots, linear_form, rope_frequencies, rope_magnitude,
                          rotated_form, rotated_widths)
from convert.checkpoint import layout
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
    """For every tensor of layout(): the tensors of the Hugging Face checkpoint it is made of, in order, as
    (name, transform); None instead of a list stands for a RoPE table. And the shapes of layout(). scale: what q is
    multiplied by (query_scale(), T253), for a Llama without biases and without norms of its heads."""
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    form = form_of(form)
    arch, shapes = form["arch"], [shape for shape, _ in layout(*header, **form)]
    if scale != 1.0 and (arch != "llama" or form["bias"] or form["qk_norm"]):
        # a norm of q's heads undoes whatever q was multiplied by, and a bias of q would have to be multiplied too:
        # no Granite has either, and one that had would go through as another model without a word
        raise ValueError("This model cannot be converted: it scales its attention's scores, and has a bias or a "
                         "norm on its queries.")

    if arch == "qwen35":
        # T229: the stacks of layout(), each from the layers of its kind. RoPE turns the first rotary rows of each
        # head of q and k (and so of the norms of their heads); the gate's rows are taken as they are
        slots = layer_slots(n_layers, linear_form(form["linear"]))
        every = range(n_layers)
        full, lines = ([layer for layer in every if slots[layer][0] == kind] for kind in (False, True))
        of = lambda which, name, transform=None: [(f"{prefix}layers.{layer}.{name}", transform) for layer in which]
        one, head_norm = ("one",), (("heads", 1, 0, 1, rotary), ("one",))
        plan = [[(prefix + "embed_tokens.weight", None)], of(every, "input_layernorm.weight", one),
                of(full, "self_attn.q_proj.weight", ("heads", 2, 0, n_heads, rotary)),
                of(full, "self_attn.q_proj.weight", ("heads", 2, 1, n_heads, 0)),
                of(full, "self_attn.k_proj.weight", ("heads", 1, 0, n_kv_heads, rotary)),
                of(full, "self_attn.v_proj.weight"), of(full, "self_attn.o_proj.weight"),
                of(full, "self_attn.q_norm.weight", head_norm), of(full, "self_attn.k_norm.weight", head_norm),
                of(lines, "linear_attn.in_proj_qkv.weight"), of(lines, "linear_attn.in_proj_z.weight"),
                of(lines, "linear_attn.in_proj_b.weight"), of(lines, "linear_attn.in_proj_a.weight"),
                of(lines, "linear_attn.conv1d.weight", ("taps",)), of(lines, "linear_attn.dt_bias"),
                of(lines, "linear_attn.A_log", ("decay",)), of(lines, "linear_attn.norm.weight"),
                of(lines, "linear_attn.out_proj.weight"),
                of(every, "post_attention_layernorm.weight", one),
                of(every, "mlp.gate_proj.weight"), of(every, "mlp.down_proj.weight"), of(every, "mlp.up_proj.weight"),
                [(prefix + "norm.weight", one)], None, None]
        if vocab_size < 0:
            plan.append([("lm_head.weight", None)])
        return plan, shapes

    if arch == "lfm2":
        # T260: the stacks of layout(), each from the layers of its kind, by transformers' names of an Lfm2. q and k
        # (and the norms of their heads) are turned as a Qwen3's; the taps as a Qwen3.5's convolution's
        slots = layer_slots(n_layers, None, convolution_form(form["convolution"], n_layers))
        every = range(n_layers)
        full, short = ([layer for layer in every if slots[layer][0] == kind] for kind in (False, True))
        of = lambda which, name, transform=None: [(f"model.layers.{layer}.{name}.weight", transform) for layer in which]
        plan = [[("model.embed_tokens.weight", None)], of(every, "operator_norm"),
                of(full, "self_attn.q_proj", ("permute", n_heads)), of(full, "self_attn.k_proj", ("permute", n_kv_heads)),
                of(full, "self_attn.v_proj"), of(full, "self_attn.out_proj"),
                of(full, "self_attn.q_layernorm", ("permute", 1)), of(full, "self_attn.k_layernorm", ("permute", 1)),
                of(short, "conv.in_proj"), of(short, "conv.conv", ("taps",)), of(short, "conv.out_proj"),
                of(every, "ffn_norm"),
                of(every, "feed_forward.w1"), of(every, "feed_forward.w2"), of(every, "feed_forward.w3"),
                [("model.embedding_norm.weight", None)], None, None]
        if vocab_size < 0:
            plan.append([("lm_head.weight", None)])
        return plan, shapes

    if arch == "neox":
        rot = rotary  # how many of each head RoPE turns, from the config
        def h(name, transform=None):
            return [(f"gpt_neox.layers.{layer}.{name}", transform) for layer in range(n_layers)]

        # only q and k are rotated, so only they are interleaved; v is taken as it is
        fused = lambda i: [(f"gpt_neox.layers.{layer}.attention.query_key_value.weight",
                            ("heads", 3, i, n_heads, rot if i < 2 else 0)) for layer in range(n_layers)]
        fused_bias = lambda i: [(f"gpt_neox.layers.{layer}.attention.query_key_value.bias",
                                 ("heads", 3, i, n_heads, rot if i < 2 else 0)) for layer in range(n_layers)]
        plan = [[("gpt_neox.embed_in.weight", None)], None, None,
                h("input_layernorm.weight"), h("input_layernorm.bias"),
                fused(0), fused(1), fused(2),
                fused_bias(0), fused_bias(1), fused_bias(2),
                h("attention.dense.weight"), h("attention.dense.bias"),
                h("post_attention_layernorm.weight"), h("post_attention_layernorm.bias"),
                h("mlp.dense_h_to_4h.weight"), h("mlp.dense_h_to_4h.bias"),
                h("mlp.dense_4h_to_h.weight"), h("mlp.dense_4h_to_h.bias"),
                [("gpt_neox.final_layer_norm.weight", None)], [("gpt_neox.final_layer_norm.bias", None)]]
        if vocab_size < 0:
            plan.append([("embed_out.weight", None)])
        return plan, shapes

    if arch == "gpt2":
        def h(name, transform=None):
            return [(f"{prefix}h.{layer}.{name}", transform) for layer in range(n_layers)]

        third = lambda i: ("part", i, 3)
        plan = [[(prefix + "wte.weight", None)], [(prefix + "wpe.weight", None)],
                h("ln_1.weight"), h("ln_1.bias"),
                h("attn.c_attn.weight", third(0)), h("attn.c_attn.weight", third(1)), h("attn.c_attn.weight", third(2)),
                h("attn.c_attn.bias", ("row", 0, 3)), h("attn.c_attn.bias", ("row", 1, 3)), h("attn.c_attn.bias", ("row", 2, 3)),
                h("attn.c_proj.weight", ("transpose",)), h("attn.c_proj.bias"),
                h("ln_2.weight"), h("ln_2.bias"),
                h("mlp.c_fc.weight", ("transpose",)), h("mlp.c_fc.bias"),
                h("mlp.c_proj.weight", ("transpose",)), h("mlp.c_proj.bias"),
                [(prefix + "ln_f.weight", None)], [(prefix + "ln_f.bias", None)]]
        if vocab_size < 0:
            plan.append([("lm_head.weight", None)])
        return plan, shapes

    def layers(name, transform=None, what="weight"):
        return [(f"model.layers.{layer}.{name}.{what}", transform) for layer in range(n_layers)]

    turn_q = ("permute", n_heads) if scale == 1.0 else (("permute", n_heads), ("scale", scale))
    plan = [[("model.embed_tokens.weight", None)], layers("input_layernorm"),
            layers("self_attn.q_proj", turn_q), layers("self_attn.k_proj", ("permute", n_kv_heads)),
            layers("self_attn.v_proj"),
            layers("self_attn.o_proj"), layers("post_attention_layernorm"),
            layers("mlp.gate_proj"), layers("mlp.down_proj"), layers("mlp.up_proj"), [("model.norm.weight", None)],
            None, None]
    if vocab_size < 0:
        plan.append([("lm_head.weight", None)])
    if form["bias"]:
        plan += [layers("self_attn.q_proj", ("permute", n_heads), "bias"),
                 layers("self_attn.k_proj", ("permute", n_kv_heads), "bias"), layers("self_attn.v_proj", None, "bias")]
    if form["qk_norm"]:
        # one weight for every head, over the rows of a head: interleaved like the rows it multiplies
        plan += [layers("self_attn.q_norm", ("permute", 1)), layers("self_attn.k_norm", ("permute", 1))]
    return plan, shapes


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
