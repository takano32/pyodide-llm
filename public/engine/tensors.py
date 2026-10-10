# The tensors of a model as the engine holds them: each row of the file (engine/layout.py) read into an array, or
# placed where it is outside Python, and the RoPE tables a file leaves out.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import numpy as np

from engine.checkpoint import outside
from engine.dtypes import DTYPES, QUANTIZED
from engine.layout import EMBEDDING, TABLE, check_suited, placed


def read_tensor(checkpoint, place, whole=True, copy=False):
    """A row of the file as the NumPy forward pass holds it: float32, whatever the file's kind. whole=False: as it is
    stored (a quantized row as (values in groups, a scale for each), a float16 one as float16), for a table that is
    only ever read a row at a time: a quarter or half of the memory, and whoever reads a row widens it.
    A float32 row is a view into the checkpoint buffer unless copy says otherwise (the float32 vectors of a quantized
    file: everything else of such a file is a copy already, and the buffer can then be freed)."""
    shape, count, kind = place.row.shape, math.prod(place.row.shape), DTYPES[place.kind]
    if kind.bits:
        # quantize.py: int8 values (or their packing), then one float32 scale per group
        values = kind.unpack(np.frombuffer(checkpoint, dtype=np.uint8, count=place.scales - place.offset, offset=place.offset))
        scales = np.frombuffer(checkpoint, dtype=np.float32, count=count // place.group, offset=place.scales)
        if not whole:
            return values.reshape(*shape[:-1], -1, place.group).copy(), scales.reshape(*shape[:-1], -1, 1).copy()
        return (values.reshape(-1, place.group).astype(np.float32) * scales[:, None]).reshape(shape)
    array = np.frombuffer(checkpoint, dtype=kind.name, count=count, offset=place.offset)
    if kind.name != "float32" and not whole:
        return array.reshape(shape).copy()
    return array.astype(np.float32, copy=copy).reshape(shape)


def read_rows(checkpoint, rows, dtype, shared_weights, external):
    """Every row of a file of this dtype, by its name: where it is (a Tensor) when the weights are outside Python
    (external), an array read from checkpoint otherwise. The classifier of a model that has no other is its
    embedding, under both names."""
    if external:
        held = dict(outside(rows, dtype))
    else:
        check_suited(rows, dtype)
        # the RoPE tables are the file's in float32 alone (rope_tables() makes the others).
        # With a separate classifier the embedding table is only ever read one row at a time, so an int8
        # or float16 table stays as it is and Llama.embedding() widens the row it needs.
        held = {place.row.name: read_tensor(checkpoint, place, shared_weights or place.row.role != EMBEDDING,
                                            copy=dtype in QUANTIZED)
                for place in placed(rows, dtype)
                if place.kind is not None and (place.row.role != TABLE or dtype == "float32")}
    if shared_weights:
        held["wcls"] = held["token_embedding_table"]
    return held


def rope_tables(arch, seq_len, head_size, rotary, magnitude, dtype, frequencies):
    """The RoPE tables (cos, sin) of a file that has none of its own: half precision is too coarse for the rotation
    angles, and a quantized file leaves the tables out. None where the file has them. frequencies(width): the angle
    per position of each pair; rotary: how many values of a head are turned; magnitude: yarn's (rope_magnitude())."""
    if arch == "gpt2":
        # no rotation: the position is a row of a learned table, added to the embedding
        zeros = np.zeros((seq_len, head_size // 2), dtype=np.float32)
        return zeros, zeros
    if dtype == "float32":
        return None
    if arch in ("neox", "qwen35"):
        return partial_tables(seq_len, head_size, rotary, frequencies)
    # (yarn's magnitude is in a Llama's tables alone, as it always was)
    magnitude = 1.0 if arch == "lfm2" else magnitude
    angles = np.arange(seq_len)[:, None] * frequencies(head_size)
    return tuple((turn(angles) * magnitude).astype(np.float32) for turn in (np.cos, np.sin))


def partial_tables(seq_len, head_size, rotary, frequencies):
    """The RoPE tables of a model that turns the first rotary values of a head only: the angles of that part, in
    tables of the shape the file has (the rest is never read)."""
    angles = np.arange(seq_len)[:, None] * frequencies(rotary)
    tables = [np.zeros((seq_len, head_size // 2), dtype=np.float32) for _ in range(2)]
    for table, values in zip(tables, (np.cos(angles), np.sin(angles))):
        table[:, :rotary // 2] = values
    return tables
