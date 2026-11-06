# python_facts.py (T628.1)
# The facts of the site's Python that its TypeScript reads before any Python runs (forward/memory.ts's footprint(), the
# worker as a conversion begins, /benchmark/), written once from Python's own tables into src/runtime/built/facts/, a
# module a section (<section>.ts, of one constant PYTHON_<SECTION>, which the runtime imports as it imports a neighbour:
# T628.3, a module of its own each, since one that the worker's entry and the forward chunk both read is bundled
# twice, AGENTS.md's 「木と束」): the copy that was written by hand twice, and held to Python by a check, is Python's now
# (the owner, 2026-10-31: Python first, then TypeScript). Made by `make kernels` (before any check runs: the deploy's
# light suite runs before the build), and by python_archive.py (a build's and the development server's start).
# The same Python gives the same bytes. It reads the engine's and the converter's modules, which import NumPy: the
# python3 that runs it has NumPy (as `make models` needs it anyway).
#
#   python3 python_facts.py [the folder: src/runtime/built]
#
#   architectures  (forward/architectures.ts) of each of engine/layout.py's LAYOUTS: what sizes a model before it is
#                  loaded: a learned table of positions, a gate of the attention's output (the rows wg or head_gate),
#                  RoPE, a gated FFN, LayerNorms, the FFN's activation, local layers
#   tokenizers     (src/page/folder.ts, T628.2) convert/conduct.py's TOKENIZERS: the tokenizers a conversion tries, in
#                  its order, which the page looks for in a folder before Pyodide is there
#   blocks         (jobs.ts's BLOCK_TYPES, T628.3) of convert/superblocks.py's SUPERBLOCKS, in its order: llama.cpp's
#                  number of each type of block the kernels multiply, and a block's bytes (engine/dtypes.py's BLOCKS)
#   converter      (kept.ts's CONVERTER, T628.4) convert/version.py's CONVERTER: the version of what the converter makes
#   madeup         (benchmark/madeup.ts, T628.5) the plan engine/plan.py's forward_plan() makes for /benchmark/'s
#                  made-up model, an int8 Llama of MADEUP's header (the CPU section runs it without Pyodide): its
#                  header, its file's bytes and the plan, the bytes of its RoPE tables in base64, the classifier under
#                  the embedding's name alone (the engine reads that one of a shared classifier)
#   quantized      (kept.ts, T628.3) engine/dtypes.py's QUANTIZED: the dtypes a kept conversion may be of
#   bytes          (src/models/memory.ts's ratios, T628.3) the bytes of each of those of a row of WEIGHTS weights,
#                  its values and its scales: a ratio of two is a ratio of whole numbers
import base64
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" / "python"))
from convert import conduct  # noqa: E402
from convert.version import CONVERTER  # noqa: E402
from convert.superblocks import SUPERBLOCKS  # noqa: E402
from engine import dtypes as D  # noqa: E402
from engine import layout as L  # noqa: E402
from engine.layers import rope_frequencies, rope_magnitude  # noqa: E402
from engine.model import KV_START, OWN  # noqa: E402
from engine.plan import forward_plan  # noqa: E402
from engine.tensors import rope_tables  # noqa: E402
import llama2_convert  # noqa: E402

# (a small model of each kind of the layers that keep a state, whose rows the facts are read from: an architecture of
# such layers has no rows without them; a new kind of them is a line here too, or this fails by its name)
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
    return {"architectures": architectures(), "tokenizers": list(conduct.TOKENIZERS),
            "blocks": {name: [ggml, size] for name, (ggml, size, _) in SUPERBLOCKS.items()},
            "converter": CONVERTER, "madeup": madeup(), "quantized": list(D.QUANTIZED), "bytes": {name: bytes_of(name) for name in D.QUANTIZED}}


# /benchmark/'s made-up model: Llama 3.2 1B's width (dim 2048, FFN 8192, 32 heads of 64, 8 of them for keys and
# values) with two layers, a vocabulary of 32000 and a context of 64
MADEUP = [2048, 8192, 2, 32, 8, 32000, 64]


def madeup():
    """The plan of MADEUP as Llama(external=) makes it of an int8 file (T640.9: engine/plan.py's forward_plan(), called
    as it is): the matrices stay int8, room for KV_START positions, RoPE over the whole head by the default theta (an
    int8 file has no tables), and a final norm of ones (read(), which forward_plan() looks at for outlier channels:
    none)."""
    form = dict(L.FORM)
    dims = L.Dims(MADEUP, form)
    tables = rope_tables(dims.arch, dims.seq_len, dims.head_size, dims.head_size, rope_magnitude(OWN["rope_scaling"]),
                         "int8", lambda width: rope_frequencies(width, OWN["rope_theta"], OWN["rope_scaling"]))
    ones = lambda offset, length: struct.pack(f"<{length // 4}f", *[1.0] * (length // 4))
    plan = forward_plan(dims, "int8", ones, int8=True, kv_start=KV_START, rotary=dims.head_size, tables=tables)
    plan["tensors"] = {name: tensor for name, tensor in plan["tensors"].items() if name != "wcls"}
    plan["derived"] = {name: base64.b64encode(bytes(value)).decode() for name, value in plan["derived"].items()}
    return {"header": MADEUP, "size": llama2_convert.checkpoint_size(MADEUP, "int8", form), "plan": plan}


# (a row that is whole groups of every dtype's)
WEIGHTS = 1152


def bytes_of(name):
    dtype = D.DTYPES[name]
    return sum(dtype.stored_bytes(WEIGHTS, dtype.group(WEIGHTS)))


HEAD = """// src/runtime/built/facts/{name}.ts: written by python_facts.py from the site's Python (T628). Not to be edited, and
// not committed (src/runtime/built/ is made): facts the TypeScript reads before any Python runs. One file of the runtime
// imports it: a second one, of another bundle's graph, has the bundler write the chunk twice (python_facts.py's head).
"""


def write(folder=HERE / "src" / "runtime" / "built"):
    folder = Path(folder) / "facts"
    folder.mkdir(parents=True, exist_ok=True)
    # (the one module of before T628.3, gone)
    (folder.parent / "facts.ts").unlink(missing_ok=True)
    for name, value in facts().items():
        text = HEAD.format(name=name) + f"export const PYTHON_{name.upper()} = " + json.dumps(value, indent=1) + " as const;\n"
        target = folder / f"{name}.ts"
        # (written only where it changes: a file its build reads is not made new for nothing)
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
    return folder


if __name__ == "__main__":
    print(write(*sys.argv[1:2]))
