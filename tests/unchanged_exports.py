# unchanged_exports.py (T346 review)
# The public names of the two Python windows (llama2_convert, llama2_numpy), as one JSON object: for each name its kind and
# its signature (functions, classes' __init__ and public methods, with whether it is a static or class method), or a hash
# of a constant's repr. The tests reach only the names they use; a name a split leaves behind (or a method that falls off
# a class) breaks the page, not the tests. tests/unchanged.mjs runs this on both trees.
#   python tests/unchanged_exports.py <the root of a tree>
import hashlib
import importlib
import inspect
import json
import re
import sys

root = sys.argv[1]
sys.path.insert(0, root + "/public")
found = {}


def short(value):
    return hashlib.sha256(re.sub(r" at 0x[0-9a-f]+", "", repr(value)).encode()).hexdigest()[:12]


def signature(function):
    try:
        return str(inspect.signature(function))
    except (TypeError, ValueError):
        return "(no signature)"


for module_name in ("llama2_convert", "llama2_numpy"):
    module = importlib.import_module(module_name)
    for name in sorted(vars(module)):
        value = vars(module)[name]
        if inspect.ismodule(value) or (name.startswith("__") and name.endswith("__")):
            continue
        key = f"{module_name}.{name}"
        if inspect.isclass(value) and value.__module__ == module_name:
            found[key] = "class" + signature(value)
            for member, raw in sorted(vars(value).items()):
                if member.startswith("__") and member != "__init__":
                    continue
                kind = type(raw).__name__
                target = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
                found[f"{key}.{member}"] = f"{kind}{signature(target)}" if callable(target) else f"{kind} {short(raw)}"
        elif inspect.isfunction(value) and value.__module__ == module_name:
            found[key] = "function" + signature(value)
        elif not (inspect.isfunction(value) or inspect.isclass(value) or inspect.isbuiltin(value)) and not hasattr(value, "__call__"):
            found[key] = "value " + short(value if not hasattr(value, "tobytes") else value.tobytes())
        else:
            found[key] = "imported " + type(value).__name__ + " " + getattr(value, "__name__", "")
print(json.dumps(found))
