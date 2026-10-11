# The plan of a conversion: which tensor of the source goes where in the checkpoint, and what is done to it on the way.
import numpy as np

from engine.layout import TABLE, Dims, form_of, linear_form
from engine.layers import rope_frequencies, rope_magnitude, rotated_form, rotated_widths
from convert.families import family_of, of_layout
from convert.config import head_size, normalize, rotary_dim


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


def name_prefix(source, arch):
    """What stands in front of the tensors' names in this source, for the layouts whose sources ask (their family's
    prefix()): a GPT-2's "transformer." or nothing, and a Qwen3.5's (T229) "model.language_model." or "model."."""
    return of_layout(arch).prefix(source)


def checkpoint_form(config, source):
    """The form of the checkpoint converted from this config.json and source (llama2_numpy.FORM): what its file will
    not say. head_dim is 0 where the heads fill dim exactly, the way the engine reads a form without one (T144: not
    where dim // heads is the head's size, which a dim that heads do not divide would pass with narrower heads)."""
    config = normalize(config)
    family, size = family_of(config), head_size(config)
    form = {"bias": False, "arch": family.arch, "qk_norm": False,
            "head_dim": 0 if size * config["num_attention_heads"] == config["hidden_size"] else size,
            "linear": None, "rotated": getattr(source, "rotated", None), "convolution": None}
    # what its tensors say (a Llama's biases and norms of its heads), and what its config.json says
    form.update(family.found(source))
    form.update({key: make(config) for key, make in family.form.items()})
    if form["rotated"] is not None:
        # T237: a rotated basis is no tensor and no number of config.json: the source says it (a GGUF's metadata).
        # Held to the model here: signs for every width its matrices read, in whole blocks, and no GPT-2's
        if not family.rotatable:
            raise ValueError("This model cannot be converted: a GPT-2, a GPT-NeoX or an LFM2 in a rotated basis.")
        rotated_form(form["rotated"], rotated_widths(config["hidden_size"], size * config["num_attention_heads"],
                                                     config["intermediate_size"], linear_form(form["linear"])))
    return form


def conversion_plan(header, form=None, prefix="transformer.", rotary=0, scale=1.0):
    """For every row of the checkpoint (engine/layout.py's tensor_rows()), in file order: the tensors of the Hugging
    Face checkpoint it is made of, in order, as (name, transform); None instead of a list stands for a RoPE table.
    And the shapes of the rows. The sources are the layout's family's, by the row's name (convert/families/). scale:
    what q is multiplied by (query_scale(), T253), for a layout without biases and without norms of its heads."""
    form = form_of(form)
    arch, d = form["arch"], Dims(header, form)
    rows = d.rows()
    sources = of_layout(arch).sources(d, prefix, rotary)
    if scale != 1.0:
        if {"bq", "q_norm"} & {row.name for row in rows}:
            # a norm of q's heads undoes whatever q was multiplied by, and a bias of q would have to be multiplied too:
            # no Granite has either, and one that had would go through as another model without a word
            raise ValueError("This model cannot be converted: it scales its attention's scores, and has a bias or a "
                             "norm on its queries.")
        name, turn = sources["wq"]
        sources["wq"] = (name, (turn, ("scale", scale)))
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


def model_plan(config, source, header, form):
    """conversion_plan() of a model: what stands in front of its source's names, how much of a head RoPE turns and
    what q is multiplied by are its family's to say."""
    family = family_of(config)
    return conversion_plan(header, form, name_prefix(source, form["arch"]),
                           rotary_dim(config) if family.partly else 0, family.scale(config))


def rope_table(config, header, which):
    """The cos (which = 0) or sin (1) table of the legacy format, for float32 and float16 checkpoints.

    GPT-NeoX rotates only rotary_pct of each head (and Qwen3.5), and the angles follow that width. The table keeps the shape
    the layout gives it (head_size // 2 columns); the columns past the rotated part are never read.
    """
    size, seq_len = head_size(config), header[6]
    width = rotary_dim(config) if family_of(config).partly else size
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    frequencies = rope_frequencies(width, config.get("rope_theta", 10000.0), config.get("rope_scaling"))
    table = (np.cos if which == 0 else np.sin)(positions * frequencies) * rope_magnitude(config.get("rope_scaling"))
    if width == size:
        return table
    full = np.zeros((seq_len, size // 2), dtype=np.float64)
    full[:, :width // 2] = table
    return full
