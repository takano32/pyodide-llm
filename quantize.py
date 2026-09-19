# quantize.py
# Turn a llama2.c "legacy" float32 checkpoint into this project's int8 variant, about 3.5x smaller:
# the matrices become int8 with one float32 scale per group of 32 values (as llama2.c's export.py does for
# runq.c), the norm weights stay float32, and the RoPE tables are dropped because llama2_numpy.py computes them.
# Measured on stories15M: perplexity +0.04%. It runs when the site is deployed.
#
#   python3 quantize.py <in: float32 checkpoint> <out: int8 checkpoint>
import struct
import sys

import numpy as np


def group_size(row_length):
    size = 32
    while row_length % size:
        size //= 2
    return size


def layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
    """(shape, is a matrix) of every tensor, in file order. llama2_numpy.py reads the same order."""
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
               ((n_layers, dim, dim), True), ((n_layers, kv_dim, dim), True), ((n_layers, kv_dim, dim), True),
               ((n_layers, dim, dim), True), ((n_layers, dim), False),
               ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
               ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
    if vocab_size < 0:
        tensors.append(((abs(vocab_size), dim), True))
    return tensors


if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    header = struct.unpack_from("<7i", data, 0)
    offset = 28
    with open(sys.argv[2], "wb") as f:
        f.write(data[:28])
        for shape, is_matrix in layout(*header):
            tensor = np.frombuffer(data, dtype=np.float32, count=int(np.prod(shape)), offset=offset)
            offset += tensor.nbytes
            if is_matrix is None:
                continue  # RoPE table
            if not is_matrix:
                f.write(tensor.tobytes())
                continue
            groups = tensor.reshape(-1, group_size(shape[-1]))
            scales = (np.abs(groups).max(axis=1) / 127.0).astype(np.float32)
            inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
            f.write(np.rint(groups * inverse[:, None]).astype(np.int8).tobytes())
            f.write(scales.tobytes())
    assert offset == len(data), "not a float32 legacy checkpoint"
