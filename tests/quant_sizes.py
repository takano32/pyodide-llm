# quant_sizes.py (T.513): what a model of the list would weigh in the forms the page can hold it in, from the heads of
# its files alone (no weights fetched): int8 and 6 bits (the entry's own file, quantized in the browser), and llama.cpp's
# blocks kept as they are (T.363's Conversion(blocks=True)) for each GGUF named, a file of the same repository as the
# entry's GGUF (a Q4_K_M, an IQ4_XS ...). For each: the bytes fetched (the weights' file), the checkpoint the memory
# holds, and for the blocks which rows stay in which type and which are widened to the dtype (a stack with one layer of
# a type the page has no kernel for is widened whole: Q6_K, Q5_K, Q8_0 and the blocks of 32 are read, not kept).
#
#   python tests/quant_sizes.py <entry id> [<a GGUF of the entry's repository> ...] [--work <directory = .tmp/quant-sizes>]
#
# The heads are fetched by Range and kept in the work directory (tests/conducting.py's Heads): a second run asks
# the network nothing. For a development machine (NumPy and the network).
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conducting  # noqa: E402
import format_check  # noqa: E402


class Sized(conducting.Nothing):
    """The sink that keeps what the converter opened it with: the checkpoint's size and form."""

    def open(self, size, header, dtype, form):
        self.size, self.header, self.dtype, self.form = size, list(header), dtype, form


def made(hf, work, dtype, blocks):
    sink = Sized()
    conversion = conducting.converted(conducting.Heads.of(hf, work, between="__", remember=True), hf, weights=False,
                                      dtype=dtype, sink=sink, **({"blocks": True} if blocks else {}))
    return sink, conversion


def kinds_of(form):
    """{row of the checkpoint: its type, or its layers' types joined} of a form's kinds: the rows that stay blocks"""
    return {row: kind if isinstance(kind, str) else ",".join(f"{one}x{count}" for one, count in collections.Counter(kind).items())
            for row, kind in (form or {}).get("kinds", {}).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entry")
    parser.add_argument("ggufs", nargs="*")
    parser.add_argument("--work", default=str(HERE.parent / ".tmp" / "quant-sizes"))
    asked = parser.parse_args()
    entry = next(entry for entry in format_check.entries() if entry["id"] == asked.entry)
    hf = entry["hf"]
    rows = []
    for dtype in ("int8", "int6"):
        sink, _ = made(hf, asked.work, dtype, False)
        rows.append({"form": dtype, "file": hf["weights"], "fetched": entry.get("download"), "checkpoint": sink.size})
    for name in asked.ggufs:
        other = {**hf, "weights": name}
        sizes = conducting.Heads.of(other, asked.work, between="__", remember=True)
        fetched = sizes.size("weights", name)
        try:
            sink, conversion = made(other, asked.work, "int8", True)
        except Exception as error:  # (a refusal is a row of the table: what it says)
            rows.append({"form": "blocks", "file": name, "fetched": fetched, "refused": str(error)})
            continue
        # the file's tensors by their role (the name without the layer) and type, F32 left out
        types = collections.Counter(f"{re.sub(r'\.(\d+)\.', '.N.', name)}:{info['dtype']}"
                                    for name, info in conversion.stream.tensors.items() if info["dtype"] != "F32")
        rows.append({"form": "blocks", "file": name, "fetched": fetched, "checkpoint": sink.size,
                     "kept": kinds_of(sink.form), "tensors": dict(types), "tiled": bool((sink.form or {}).get("tiled"))})
    for row in rows:
        print(f"quant_sizes {asked.entry}: {json.dumps(row)}")


main()
