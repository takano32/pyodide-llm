# The dtypes of a checkpoint: how a file holds a tensor, one entry for each (DTYPES). What a kind takes in the file, the
# group of a quantized row, whether a row suits it, how its values are packed and read back, the kernel that packs
# them: everything that depends on the kind is asked here, by its name (T359). A new one is an entry here (and its
# packing in engine/packing.py; a row's kind that is no file's dtype is an entry of BLOCKS, T363); what the forward pass
# multiplies it with is forward.js's (jobs.js) and not said here.
# This file reads engine/packing.py alone: engine/layout.py and everything above read it.
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
from typing import Callable, NamedTuple

import numpy as np

from engine.packing import NOT_TERNARY, TERNARY_GROUP, group32, quantize, six, ternary, unpack_ternary, unpacked6


class Dtype(NamedTuple):
    """How a file holds a tensor. One that is not quantized is its values, itemsize bytes each. A quantized one is its
    values in groups along the row, bits each, then one scale for every group: a float32, or scale_type's.
    short: what forward.js calls the kind (Tensor's kind). group(length): the values of a group of a row of that
    length. pack(float32 rows, whole ones) -> (the bytes of their values, in rows of a group's bytes; their float32
    scales): NumPy's. unpack(the bytes of the values) -> the int8 values, one after the other; a value is that times
    the scale of its group. packer: the kernel that packs (its export's name: packer(bytes out, scales out, float32
    in, how many values), anything but 0 or nothing back for values the kind cannot hold; kernel_quantizer() calls
    it). rounds: any model's weights are rounded into it (False: only the weights that are its values already, and
    refusal are the words for the others; nobody is given this kind without asking). unsuited: the words for a model
    with rows that are no whole groups.
    T363, llama.cpp's blocks (BLOCKS): block, the bytes of a group in place of bits; scale, the bytes of the scale
    behind each group (0: it is inside); read(the bytes of whole groups) -> float32 values, in place of unpack.
    T533: scale, the bytes of a group's scale, says its type (scale_type): 4 a float32, 2 a float16 (ternary's, as
    PQ2_0 has it)."""
    name: str
    short: str
    itemsize: int = 0
    bits: int = 0
    group: Callable = None
    pack: Callable = None
    unpack: Callable = None
    packer: str = None
    rounds: bool = True
    refusal: str = None
    unsuited: str = None
    block: int = 0
    scale: int = 4
    read: Callable = None

    @property
    def scale_type(self):
        """The NumPy type of a group's scale in a checkpoint (T533: ternary's is a float16, PQ2_0's own)."""
        return np.float16 if self.scale == 2 else np.float32

    def stored_bytes(self, count, group):
        """(the bytes of the values, the bytes of the scales) of count values in groups of group."""
        return count // group * (self.block or group * self.bits // 8), self.scale * (count // group)

    def suits(self, length):
        """Whether a row of this length can be held: a kind in groups needs whole groups."""
        return not self.group or length % self.group(length) == 0

    def floats(self, raw, scales, group):
        """The float32 values of a tensor in groups, from the bytes of its values and its scales."""
        if self.read is not None:
            return self.read(raw)
        return (self.unpack(raw).reshape(-1, group).astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(-1)

    def stored(self, raw, scales, shape, group):
        """A table in groups as it is stored, for whoever widens a row at a time: (the int8 values in groups, a scale
        for each), or the rows' blocks."""
        if self.read is not None:
            return Blocks(raw.reshape(shape[0], -1).copy(), self.read, shape[1:])
        return self.unpack(raw).reshape(*shape[:-1], -1, group).copy(), scales.astype(np.float32).reshape(*shape[:-1], -1, 1)


class Blocks:
    """A table held as its rows' blocks: table[row] is the row, float32."""

    def __init__(self, raw, read, shape):
        self.raw, self.read, self.shape = raw, read, (len(raw), *shape)

    def __getitem__(self, row):
        return self.read(self.raw[row]).reshape(self.shape[1:])


def as_int8(raw):
    return raw.view(np.int8)


# int8: quantize.py's, groups of 32 or of the largest power of two below it that divides the row.
# int6 (T98): six bits a value, 24 bytes a group of 32 (engine/packing.py's pack6).
# ternary (T230): two bits a value, 32 bytes a group of 128 (pack_ternary); the weights of a ternary model as they are.
# T533: its scale a float16, as PQ2_0 and PTQ1_0 have it (34 bytes a group of 128, not 36).
DTYPES = {dtype.name: dtype for dtype in (
    Dtype("float32", "f32", itemsize=4),
    Dtype("float16", "f16", itemsize=2),
    Dtype("int8", "int8", bits=8, group=group32, pack=quantize, unpack=as_int8, packer="quantize_x"),
    Dtype("int6", "int6", bits=6, group=lambda length: 32, pack=six, unpack=unpacked6, packer="quantize6_x",
          unsuited="Six bits a weight needs rows of whole groups of 32, and this model has other rows."),
    Dtype("ternary", "ternary", bits=2, scale=2, group=lambda length: TERNARY_GROUP, pack=ternary, unpack=unpack_ternary,
          packer="ternary_x", rounds=False, refusal=NOT_TERNARY,
          unsuited="Ternary weights need rows of whole groups of 128, and this model has other rows."),
)}
# ---- llama.cpp's blocks (T363)
# A GGUF's K- and i-quantized types, kept as the file has them: 256 weights to a block of the row, its scales inside
# (docs/notes/t361-saluki-2026-10-13.md). A row's kinds alone (the form's "kinds"), never a file's dtype.
# NumPy's readers are the converter's (convert/superblocks.py and its codebooks, 15 KB zipped that the engine's
# archive, which every visit fetches, goes without): asked for when such a row is read in NumPy, which the reference
# does and the page does not (it multiplies these rows on the kernels).
BLOCK = 256
NOT_BLOCKS = "llama.cpp's blocks hold 256 weights of a row, and this model has rows that are no whole blocks."


def block_reader(name):
    def read(raw):
        from convert.superblocks import SUPERBLOCKS
        return SUPERBLOCKS[name][2](raw)
    return read


# the type's name in a GGUF, and the bytes of a block
BLOCKS = {name: Dtype(name, name, group=lambda length: BLOCK, block=size, scale=0, read=block_reader(name), rounds=False,
                      unsuited=NOT_BLOCKS)
          for name, size in (("Q2_K", 84), ("Q4_K", 144), ("IQ2_XXS", 66), ("IQ2_XS", 74), ("IQ3_XXS", 98), ("IQ1_S", 50),
                             ("IQ3_S", 110), ("IQ2_S", 82), ("IQ4_XS", 136), ("IQ1_M", 56))}
# the codebooks a file with such rows carries for the kernels (layout.py's row "codebooks"): these many bytes, which
# are convert/grids.py's codebooks()
BYTES = Dtype("uint8", "u8", itemsize=1)
CODEBOOKS = 33808
# every kind a row may be stored in, by its name: a file's dtypes, the blocks, and bytes
KINDS = {**DTYPES, **BLOCKS, BYTES.name: BYTES}

# the dtypes with groups and scales
QUANTIZED = tuple(name for name, dtype in DTYPES.items() if dtype.bits)
# those whose values are int8 in another packing: the engine widens them where it reads them, and is int8 from there on
PACKED = tuple(name for name in QUANTIZED if DTYPES[name].bits < 8)
# the ones a model converted with no dtype asked for may get (Stream's callable dtype, T115)
EITHER = tuple(name for name in QUANTIZED if DTYPES[name].rounds)


def kind_named(kind):
    """The kind a form gives a row, by its name: a block's or a dtype's (whose ValueError is for any other)."""
    return str(kind) if str(kind) in BLOCKS else dtype_of(kind)


def dtype_of(dtype):
    """A checkpoint's dtype by its name, from a name or a NumPy dtype (NumPy has neither int6 nor ternary). ValueError
    for anything that is none of DTYPES."""
    name = str(dtype)
    if name not in DTYPES:
        try:
            name = np.dtype(dtype).name
        except TypeError:
            name = None
    if name not in DTYPES:
        names = list(DTYPES)
        raise ValueError(f"dtype must be {', '.join(names[:-1])} or {names[-1]}, not {dtype}.")
    return name
