# ternary_rows.py
# T230: the converter's readers of the two ternary GGUF types (llama2_convert.pq2_0 and ptq1_0) held to a real model.
# No small model is published as PTQ1_0 (only Prism ML's Ternary Bonsai 2 27B, 5.9 GB), so nothing is downloaded whole:
# the heads of the model's three GGUF files (PTQ1_0, PQ2_0 and F16, the same weights) are fetched with Range requests,
# and stretches of rows of some of their tensors, a few megabytes each. For every stretch:
#   - the reference readers of tests/gguf_check.py (written apart from the converter's) and the converter's readers give
#     the same float32 values, to the bit;
#   - PTQ1_0, PQ2_0 and F16 hold the same values, to the bit (the F16 file is the same weights in float16);
#   - the ternary dtype's bytes and scales (llama2_numpy.ternary) are the same from either packed file, and widen back
#     to the F16 file's values.
# The survey that read the layouts did this by hand on three tensors (docs/notes/t228-bonsai-2-2026-10-01.md, 3).
#
#   python3 tests/ternary_rows.py [<repo>@<revision>] [--rows 32] [--tensors token_embd.weight,blk.0.attn_gate.weight]
#
# For CI (tests.yml's extra): about 60 MB of requests. Exit 1 where anything differs.
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import gguf_check  # noqa: E402
import llama2_convert  # noqa: E402
from llama2_numpy import ternary, unpack_ternary  # noqa: E402
from fetching import ranged  # noqa: E402

REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf@b072e1d3b35a0a630cece372c2127528e0994386"
FILES = {"PTQ1_0": "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "PQ2_0": "Ternary-Bonsai-2-27B-PQ2_0.gguf", "F16": "Ternary-Bonsai-2-27B-F16.gguf"}
# of every kind of ternary matrix of the model, and of every width of a row (5120, 6144, 17408)
TENSORS = ["token_embd.weight", "output.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
           "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight", "blk.3.attn_output.weight",
           "blk.31.ffn_gate.weight", "blk.63.ffn_up.weight", "blk.63.ffn_down.weight"]
BLOCK_BYTES = {gguf_check.PTQ1_0: 28, gguf_check.PQ2_0: 34}


def fetch(url, start, length):
    return ranged(url, start, length, said=lambda error: print(f"  again ({error})", flush=True))  # (tests/fetching.py, T357)


def head(url):
    """(metadata, tensors, base) of a GGUF behind a URL: its beginning fetched in growing pieces until it parses."""
    size = 1 << 24
    while True:
        data = fetch(url, 0, size)
        try:
            reader = gguf_check.Reader(data)
            assert data[:4] == b"GGUF"
            reader.at = 4
            reader.take("<I")
            tensors, entries = reader.take("<Q"), reader.take("<Q")
            metadata = {}
            for _ in range(entries):
                key = reader.string()
                metadata[key] = reader.value(reader.take("<I"))
            infos = {}
            for _ in range(tensors):
                name = reader.string()
                dims = [reader.take("<Q") for _ in range(reader.take("<I"))]
                infos[name] = {"shape": tuple(reversed(dims)), "type": reader.take("<I"), "offset": reader.take("<Q")}
            if reader.at > len(data):
                raise struct.error("past the end")
            alignment = metadata.get("general.alignment", 32)
            return metadata, infos, (reader.at + alignment - 1) // alignment * alignment
        except (struct.error, IndexError, UnicodeDecodeError):
            size *= 2


def main():
    args = sys.argv[1:]
    option = lambda name, default: args[args.index(name) + 1] if name in args else default
    repo, revision = next((a for a in args if "@" in a), REPO).split("@")
    rows, names = int(option("--rows", 32)), option("--tensors", ",".join(TENSORS)).split(",")
    urls = {kind: f"https://huggingface.co/{repo}/resolve/{revision}/{file}" for kind, file in FILES.items()}
    heads = {kind: head(url) for kind, url in urls.items()}
    for kind, (metadata, infos, base) in heads.items():
        kinds = sorted({gguf_check.TYPE_NAMES.get(info["type"], info["type"]) for info in infos.values()})
        print(f"{kind}: {len(infos)} tensors ({', '.join(map(str, kinds))}), the data from byte {base}", flush=True)
    failed, stretches, values_seen, fetched = False, 0, 0, 0
    for name in names:
        shape = heads["F16"][1][name]["shape"]
        total, width = shape[0], math.prod(shape[1:])
        for first in sorted({0, max(0, total // 2 - rows // 2), max(0, total - rows)}):
            count = min(rows, total - first)
            got = {}
            for kind, (_, infos, base) in heads.items():
                info = infos[name]
                assert info["shape"] == shape, f"{name}: {kind} has the shape {info['shape']}, not {shape}"
                if info["type"] in BLOCK_BYTES:
                    row = width // 128 * BLOCK_BYTES[info["type"]]
                    raw = fetch(urls[kind], base + info["offset"] + first * row, count * row)
                    reference = gguf_check.WIDEN[info["type"]][2](np.frombuffer(raw, np.uint8)).reshape(count, width)
                    ours = llama2_convert.SOURCES[gguf_check.TYPE_NAMES[info["type"]]].read(raw).reshape(count, width)
                    if not np.array_equal(reference.view(np.uint32), ours.view(np.uint32)):
                        failed = True
                        print(f"{name} rows {first}..{first + count}: the converter's {kind} reader differs from the reference's — FAILED")
                    got[kind] = ours
                else:
                    assert info["type"] == gguf_check.F16, f"{name} is of type {info['type']} in the {kind} file"
                    raw = fetch(urls[kind], base + info["offset"] + first * width * 2, count * width * 2)
                    got[kind] = np.frombuffer(raw, np.float16).astype(np.float32).reshape(count, width)
                fetched += len(raw)
            same = all(np.array_equal(got[kind], got["F16"]) for kind in ("PTQ1_0", "PQ2_0"))
            packs = [ternary(got[kind]) for kind in ("PTQ1_0", "PQ2_0", "F16")]
            packed = all(np.array_equal(packs[0][0], p[0]) and np.array_equal(packs[0][1], p[1]) for p in packs[1:])
            back = (unpack_ternary(packs[0][0]).reshape(-1, 128) * packs[0][1][:, None]).reshape(count, width)
            exact = np.array_equal(back, got["F16"])
            shares = [float(np.mean(np.sign(got["F16"]) == sign)) for sign in (-1, 0, 1)]
            ok = same and packed and exact
            failed |= not ok
            stretches += 1
            values_seen += count * width
            print(f"{name} {shape} rows {first}..{first + count}: PTQ1_0, PQ2_0 and F16 {'the same values' if same else 'DIFFER'}; "
                  f"the ternary dtype of each {'the same bytes' if packed else 'DIFFERS'}, {'the F16 values again' if exact else 'NOT the F16 values'}; "
                  f"-1, 0, +1: {shares[0]:.3f}, {shares[1]:.3f}, {shares[2]:.3f}; scales {packs[0][1].min():.3g} to {packs[0][1].max():.3g}"
                  f"{'' if ok else ' — FAILED'}", flush=True)
    print(f"ternary-rows: {stretches} stretches of {len(names)} tensors, {values_seen} values, {fetched / 1e6:.1f} MB of rows fetched: "
          f"{'FAILED' if failed else 'the two readers, the three files and the ternary dtype agree to the bit'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
