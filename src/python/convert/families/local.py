# convert/families/local.py (T640.4): what the families whose layers are of two kinds, some attending a window of the
# last positions and turning by a base of RoPE of their own (engine/layout.py's local_layers()), read of a config.json
# the same way: a Gemma 3, a Gemma 4 and a Spark-X2.5. Each family's own reading (how its config.json lists the kinds,
# what it takes where nothing is said) stays in its module.
from convert.families.family import f32, positive, refuse

# the two kinds of layers, by config.json's names: one that attends a window of the last positions, and one that
# attends them all
SLIDING, FULL = "sliding_attention", "full_attention"


def by_kind(config, unsaid=None):
    """config.json's rope_parameters as a dict for each kind of layer (transformers 5 writes one so; normalize() would
    read it as one model's, and find no theta there): a function of the kind, {} for a kind it says nothing of.
    unsaid: what to take where the config.json has neither a rope_parameters nor a rope_theta."""
    rope = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    if not rope and unsaid and "rope_theta" not in config:
        rope = unsaid
    return lambda kind: rope.get(kind) if isinstance(rope.get(kind), dict) else {}


def rope_type(said):
    return said.get("rope_type", said.get("type", "default"))


def scaling_of(config, of, plain=None):
    """The first kind's RoPE (the full-attention layers', then the others') that is not a plain one (plain: the kinds of
    RoPE taken as plain for each kind of layer, "default" where not said), else config.json's rope_scaling: what
    check_config() refuses where it is anything."""
    plain = plain or {}
    return next(({**of(kind)} for kind in (FULL, SLIDING) if rope_type(of(kind)) not in plain.get(kind, ("default",))),
                config.get("rope_scaling"))


def without_rope(config):
    """config.json but for the names of RoPE that a family's reading replaces."""
    return {key: value for key, value in config.items() if key not in ("rope_parameters", "rope_scaling")}


def local(config):
    """The engine's option "local" from a (normalized) config.json: the sliding-attention layers, their window and the
    base RoPE turns them by (engine/layout.py's local_layers()). None for a model without any."""
    layers = [layer for layer, kind in enumerate(config["layer_types"]) if kind == SLIDING]
    return {"layers": layers, "window": config["sliding_window"], "rope_theta": float(config["rope_local_base_freq"])} \
        if layers else None


def local_options(config):
    # the file does not say which layers are local ones (only where there are any)
    said = local(config)
    return {"local": said} if said else {}


def local_check(config, rope=lambda config: True):
    """A config.json that does not say its layers, their window or their bases of RoPE (rope: what else of its RoPE
    must hold, asked after the bases are numbers)."""
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"] or set(kinds) - {SLIDING, FULL}:
        refuse("its layers are not sliding-attention layers and full-attention layers, one kind for every layer")
    window = config.get("sliding_window")
    if SLIDING in kinds and (not isinstance(window, int) or isinstance(window, bool) or window <= 0):
        refuse("its config.json has no usable sliding_window")
    if not positive(config.get("rope_theta")) or not positive(config.get("rope_local_base_freq")) or not rope(config):
        refuse("its config.json has no usable base of RoPE")


def local_agrees(own, config):
    """What a GGUF and the original's config.json must agree on that no tensor shows: which layers are local ones,
    their window and their base (as the GGUF holds a number)."""
    said = lambda c: local(c) and {**local(c), "rope_theta": f32(local(c)["rope_theta"])}
    return ("sliding-attention layers", said(own), said(config))
