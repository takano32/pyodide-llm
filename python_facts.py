# python_facts.py (T.628.1)
# The facts of the site's Python that its TypeScript reads before any Python runs (forward/memory.ts's footprint(), the
# worker as a conversion begins, /benchmark/), written once from Python's own tables into src/runtime/built/facts.ts (a
# module of one constant, FACTS, which the runtime imports as it imports a neighbour): the copy that was written by hand twice, and held to Python by a check, is Python's now
# (the owner, 2026-10-31: Python first, then TypeScript). Made by `make kernels` (before any check runs: the deploy's
# light suite runs before the build), and by python_archive.py (a build's and the development server's start).
# The same Python gives the same bytes.
#
#   python3 python_facts.py [the folder: src/runtime/built]
#
#   architectures  (forward/architectures.ts) of each of engine/layout.py's LAYOUTS: what sizes a model before it is
#                  loaded: a learned table of positions, a gate of the attention's output (the rows wg or head_gate),
#                  RoPE, a gated FFN, LayerNorms, the FFN's activation, local layers
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "python"))
from engine import layout as L  # noqa: E402

# (a small model of each kind of the layers that keep a state, whose rows the facts are read from: an architecture of
# such layers has no rows without them)
KINDS = {"linear": {"every": 2, "key_heads": 1, "value_heads": 1, "key_dim": 32, "value_dim": 32, "conv": 4},
         "convolution": {"layers": "ca", "taps": 3}}


def architectures():
    out = {}
    for arch, layout in L.LAYOUTS.items():
        form = {**L.FORM, "arch": arch}
        if layout.stateful is not None:
            form[layout.stateful.name] = KINDS[layout.stateful.name]
        rows = {row[0] for row in layout.rows(L.Dims([64, 128, 2, 4, 4, 100, 32], form))}
        out[arch] = {"positions": "positions" in rows, "gate": "wg" in rows or "head_gate" in rows, "rope": layout.rope is not None,
                     "gatedFfn": layout.gated_ffn, "layerNorm": layout.layer_norm, "activation": layout.activation, "local": layout.local}
    return out


def facts():
    return {"architectures": architectures()}


HEAD = """// src/runtime/built/facts.ts: written by python_facts.py from the site's Python (T.628.1). Not to be edited, and not
// committed (src/runtime/built/ is made): the facts the TypeScript reads before any Python runs.
"""


def write(folder=HERE / "src" / "runtime" / "built"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text = HEAD + "export const FACTS = " + json.dumps(facts(), indent=1, sort_keys=True) + " as const;\n"
    target = folder / "facts.ts"
    # (written only where it changes: a file its build reads is not made new for nothing)
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    return target


if __name__ == "__main__":
    print(write(*sys.argv[1:2]))
