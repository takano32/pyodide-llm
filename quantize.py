# quantize.py
# Turn a llama2.c "legacy" float32 checkpoint into this project's int8 variant, about 3.5x smaller:
# the matrices become int8 with one float32 scale per group of 32 values (as llama2.c's export.py does for
# runq.c), the norm weights stay float32, and the RoPE tables are dropped because llama2_numpy.py computes them.
# Measured on stories15M: perplexity +0.04%. It runs when the site is deployed.
#
#   python3 quantize.py <in: float32 checkpoint> <out: int8 checkpoint> [int8 | int6]
#
# int6 (T98): six bits a value, 32 values packed into 24 bytes (llama2_numpy.pack6), the same float32 scales.
import struct
import sys
from pathlib import Path

import numpy as np

# the format and the arithmetic are shared with the converter, which also writes int8 directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "public"))
from llama2_convert import group_size, layout, quantize  # noqa: E402, F401
from llama2_numpy import pack6, quantize6  # noqa: E402


if __name__ == "__main__":
    data = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")  # one tensor in memory at a time
    six = sys.argv[3:] == ["int6"]
    header = struct.unpack_from("<7i", data[:28].tobytes(), 0)
    offset = 28
    with open(sys.argv[2], "wb") as f:
        f.write(data[:28].tobytes())
        for shape, is_matrix in layout(*header):
            tensor = np.frombuffer(data, dtype=np.float32, count=int(np.prod(shape)), offset=offset)
            offset += tensor.nbytes
            if is_matrix is None:
                continue  # RoPE table
            if not is_matrix:
                f.write(tensor.tobytes())
                continue
            values, scales = (quantize6 if six else quantize)(tensor.reshape(-1, shape[-1]))
            f.write((pack6(values) if six else values).tobytes())
            f.write(scales.tobytes())
    assert offset == len(data), "not a float32 legacy checkpoint"
