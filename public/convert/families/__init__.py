# The families of models the converter takes (T359), by config.json's model_type: the one table the shared flow
# (convert/config.py, plan.py, gguf.py, conversion.py) looks a model up in. A new family is a record here (Family, in
# family.py) and nothing in those files: one that is a Llama but for a number or two is LLAMA._replace(...) in
# llama.py, as a Granite and a SmolLM3 are.
from convert.families.gpt2 import GPT2, NEOX
from convert.families.lfm2 import LFM2
from convert.families.llama import GRANITE, LLAMA, MISTRAL, QWEN2, QWEN3, SMOLLM3
from convert.families.qwen35 import QWEN35, QWEN35_WHOLE

# (in the order the refusals name them)
FAMILIES = {"llama": LLAMA, "mistral": MISTRAL, "granite": GRANITE, "smollm3": SMOLLM3, "qwen2": QWEN2, "qwen3": QWEN3,
            "qwen3_5_text": QWEN35, "qwen3_5": QWEN35_WHOLE, "lfm2": LFM2, "gpt2": GPT2, "gpt_neox": NEOX}


def by_gguf(families):
    """A GGUF's general.architecture: the model_type its config.json has, and its family. A family that is another's
    _replace() has that one's GGUF name until it says its own (gguf=None where it has none): two with one name would
    leave the later in this table and the earlier unreachable, so that is refused here and not left to be found."""
    table = {}
    for model_type, family in families.items():
        if family.gguf:
            if family.gguf in table:
                raise ValueError(f"{model_type} and {table[family.gguf][0]} are both the GGUF architecture {family.gguf}")
            table[family.gguf] = (model_type, family)
    return table


GGUF = by_gguf(FAMILIES)
# A form says its layout (FORM's "arch"), not its model: the family whose sources and prefix are the layout's
# (the first of a layout's families in the table)
LAYOUTS = {family.arch: family for family in reversed(FAMILIES.values())}


def family_of(config):
    """The family of a config.json, by its model_type. One of no family the converter knows is read as a Llama here
    (as its layout and the forward pass read an unknown arch) and refused by check_config()."""
    model_type = config.get("model_type")
    return FAMILIES.get(model_type, LLAMA) if isinstance(model_type, str) else LLAMA


def of_layout(arch):
    """The family that stands for a layout, for what is asked of a form alone (conversion_plan())."""
    return LAYOUTS.get(arch, LLAMA)


def named(families):
    """Their titles as a refusal lists them: "A, B and C"."""
    titles = list(dict.fromkeys(family.title for family in families))
    return f"{', '.join(titles[:-1])} and {titles[-1]}"
