# What a family of models is to the converter (T359): one record of all that differs between one model_type of
# config.json and another, which convert/config.py, plan.py, gguf.py and conversion.py look up and ask, instead of
# comparing names themselves. And what of a config.json every family reads the same way.
from types import MappingProxyType
from typing import Any, Callable, NamedTuple, Optional

import numpy as np


def refuse(reason):
    """ValueError, in words for the visitor: this config.json describes no model the engine can run."""
    raise ValueError(f"This model cannot be converted: {reason}.")


def head_size(config):
    """The size of an attention head: config.json's head_dim where it says one (Qwen3 0.6B: 128 in a dim of 1024,
    T124), else dim / heads. "head_dim": null says as much as no head_dim at all (cyberagent/CAT-Translate-7b)."""
    return config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]


def rotary_dim(config):
    """How many of each head's values GPT-NeoX rotates (rotary_pct of them, an even number), and Qwen3.5 (T229: its
    partial_rotary_factor, which normalize() gives the same name)."""
    return int(head_size(config) * float(config.get("rotary_pct", 1.0))) // 2 * 2


def yarn(config):
    """What a config.json whose RoPE scaling is yarn says of it besides its kind (T235), else None."""
    scaling = config.get("rope_scaling") or {}
    if scaling.get("rope_type", scaling.get("type")) != "yarn":
        return None
    return {key: value for key, value in scaling.items() if key not in ("rope_type", "type") and value is not None}


def f32(value):
    """A number as a GGUF holds it, for a comparison with config.json's."""
    return float(np.float32(value))


def same(config):
    return config


def nothing(config):
    return {}


def as_stored(config):
    """A family whose tensors a GGUF holds as Hugging Face does: nothing to put back."""
    return lambda entry, info, target, parts: None


class Family(NamedTuple):
    """A family of models, by what the shared flow asks of it. A family that is another but for a few of these is
    that one's record with those replaced (a Granite is LLAMA._replace(...)): no class, and nothing to write for what
    it shares. Where a function is given a config, it is the normalize()d one unless said otherwise."""
    # how a refusal names it
    title: str
    # the layout of its file and its forward pass (engine/layout.py's FORM: "arch")
    arch: str
    # (d, prefix, rotary) -> what each row of the layout is made of, by the row's name: (the Hugging Face tensor's
    # name, with {} for the layer where the row is a stack; the transform, convert/plan.py's transformed()). d: the
    # layout's Dims, prefix: what prefix() said, rotary: how many values of a head RoPE turns (0 unless partly)
    sources: Callable
    # (config): refuses what the engine has not for this family, after the checks every family has (check_config())
    check: Callable
    # config.json says this model_type and normalize() gives it another (a Mistral is a "llama" from there on):
    # check_config() takes no config under this name
    renamed: bool = False
    # normalize(): before and after the names of RoPE that every family shares are read (rope_parameters). before
    # may give the config another model_type, whose family's after is then the one that runs
    before: Callable = same
    after: Callable = same
    # the keys of the checkpoint's form that only this family says (FORM's "linear", "convolution"), each with the
    # function of the config that makes it
    form: Any = MappingProxyType({})
    # (source) -> the keys of the checkpoint's form that this family's file says by the tensors it has (T369: a Llama's
    # "bias" and "qk_norm", rows its layout has or has not). A family whose layout always has them, or never, says
    # nothing: its form keeps FORM's value, whatever names its source happens to hold
    found: Callable = nothing
    # (config) -> the engine's options that only this family says, where it says any
    options: Callable = nothing
    # (source) -> what stands in front of its tensors' names, where that depends on the file
    prefix: Callable = lambda source: ""
    # (config) -> what the conversion multiplies q by (T253: a Granite's scores)
    scale: Callable = lambda config: 1.0
    # the classifier's name in a source, where it has one of its own
    classifier: str = "lm_head.weight"
    # the context is the config's whatever is asked for (a table of learned positions has exactly that many rows)
    fixed_context: bool = False
    # a head may be of another size than dim / heads (q and o are matrices of their own, T124)
    free_heads: bool = False
    # its RoPE may be scaled (the kinds the tables know: Llama 3's, linear, yarn)
    scaled: bool = False
    # its matrices may be in a rotated basis (T237)
    rotatable: bool = False
    # its norms are RMSNorms, whose epsilon the options say where it is not the engine's (T124)
    rms_norm: bool = False
    # RoPE turns part of each head only: the options say how much ("rotary")
    partly: bool = False
    # ---- a GGUF of it (None: llama.cpp has no architecture of this name; a Mistral's says "llama")
    gguf: Optional[str] = None
    # llama.cpp's names by Hugging Face's: (the tensors outside the layers, where a layer's go, the layer's names)
    gguf_names: Any = None
    # (key, tensors) -> config.json as the GGUF's metadata says it, by the names normalize() reads. key(name,
    # default): the value of "<architecture>.<name>"
    gguf_config: Optional[Callable] = None
    # (config, as gguf_config made it) -> a function (entry, info, target, parts) that marks in a tensor's entry of
    # the header what llama.cpp stores otherwise than Hugging Face (convert/gguf.py's gguf_model()). info: the GGUF's
    # own entry, target: Hugging Face's name, parts: the GGUF's name cut at its dots
    gguf_stored: Callable = as_stored
    # (own, config) -> what a GGUF and the original's config.json must agree on besides what every family's must
    # (gguf_agrees()), as (what, the GGUF's, the original's): the numbers of this family that no tensor shows
    agrees: Callable = lambda own, config: []
