# quantize.py
# Turn a llama2.c "legacy" float32 checkpoint into this project's int8 variant, about 3.5x smaller:
# the matrices become int8 with one float32 scale per group of 32 values (as llama2.c's export.py does for
# runq.c), the norm weights stay float32, and the RoPE tables are dropped because llama2_numpy.py computes them.
# Measured on stories15M: perplexity +0.04%. It runs when the site is deployed.
#
#   python3 quantize.py <in: float32 checkpoint> <out: int8 checkpoint> [int8 | int6]
#
# int6 (T98): six bits a value, 32 values packed into 24 bytes (llama2_numpy.pack6), the same float32 scales.
# (T640.8) The converter's Writer writes the file: it asks every row how it is stored in the dtype (engine/layout.py's
# kind_of(): a matrix packed, a vector float32, a RoPE table left out) and packs it as the converter does, so the
# file is the converter's to the byte.
import math
import struct
import sys
from pathlib import Path

import numpy as np

# the format and the arithmetic are shared with the converter, which also writes int8 directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))
from llama2_convert import Writer  # noqa: E402
from llama2_numpy import dtype_of, file_size, placed, tensor_rows  # noqa: E402


class File:
    """The Writer's sink: the checkpoint straight to a file, a tensor at a time. The file is made when the Writer
    opens it, after the dtype was found to suit the rows (int6 refuses rows that are not whole groups of 32)."""

    def __init__(self, path):
        self.path, self.f = path, None

    def open(self, size, header, dtype, form):
        self.f = open(self.path, "wb")
        self.f.truncate(size)

    def write(self, offset, raw):
        self.f.seek(offset)
        self.f.write(raw.tobytes())


if __name__ == "__main__":
    data = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")  # one tensor in memory at a time
    dtype = "int6" if sys.argv[3:] == ["int6"] else "int8"
    header = struct.unpack_from("<7i", data[:28].tobytes(), 0)
    rows = tensor_rows(header)
    assert file_size(rows, dtype_of("float32")) == len(data), "not a float32 legacy checkpoint"
    out = File(sys.argv[2])
    writer = Writer(None, header, dtype, sink=out)
    with out.f:
        for index, place in enumerate(placed(rows, dtype_of("float32"))):
            tensor = np.frombuffer(data, dtype=np.float32, count=math.prod(place.row.shape), offset=place.offset)
            writer.write(index, 0, tensor)
