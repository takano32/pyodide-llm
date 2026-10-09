# The weights smaller than int8: six bits and ternary, packed and unpacked, and how many bytes a tensor takes.
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
import numpy as np

# T98: int6, six bits a weight. An int6 group is an int8 group whose values are multiples of 4 (-128..124: six
# significant bits) and whose scale is a quarter: the same products as six bits and a whole scale, to the bit (a
# power of two), and everything past the packing is int8. So the kernels widen a group straight into the int8 their
# dot products take, with no offset to take out. A group of 32 takes 24 bytes: the four low bits of the six of
# value j and of value j + 16 share byte j (0..15), and the top two bits of values k, k + 8, k + 16 and k + 24 share
# byte 16 + k (0..7), at bits 0, 2, 4 and 6. Masks and shifts by constants only (kernels/six.ts).


def pack6(values):
    """int8 values that are multiples of 4, groups of 32 (rows of them) -> 24 bytes a group (uint8)."""
    b = (np.asarray(values, dtype=np.int8).reshape(-1, 32).view(np.uint8) >> 2) & 63  # the six bits
    low = (b[:, :16] & 15) | ((b[:, 16:] & 15) << 4)
    top = b >> 4
    high = top[:, 0:8] | (top[:, 8:16] << 2) | (top[:, 16:24] << 4) | (top[:, 24:32] << 6)
    return np.concatenate([low, high], axis=1)


def unpack6(packed):
    """24 bytes a group -> the 32 int8 values of it (rows of them), multiples of 4."""
    b = np.asarray(packed, dtype=np.uint8).reshape(-1, 24)
    low = np.concatenate([b[:, :16] & 15, b[:, :16] >> 4], axis=1)
    top = np.concatenate([(b[:, 16:] >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1)
    return ((low | (top << 4)) << 2).astype(np.uint8).view(np.int8)


def quantize6(values):
    """float32 values, whole rows of groups of 32 -> (int8 values, float32 scales) in six bits: v = round(x / s) in
    -32..31 with s = the largest |x| of the group over 31, given as 4 v and s / 4 (see pack6)."""
    groups = np.asarray(values, dtype=np.float32).reshape(-1, 32)
    scales = (np.abs(groups).max(axis=1) / 31.0).astype(np.float32)
    inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
    six = np.clip(np.rint(groups * inverse[:, None]), -32, 31).astype(np.int8)
    return (six * 4).astype(np.int8), scales / np.float32(4)


# T230: ternary, two bits a weight. Every weight is -1, 0 or +1 times the scale of its group of 128 along the row (a
# float32): what Prism ML's Ternary Bonsai models are, and how their GGUFs hold them (PQ2_0: two bits a weight;
# PTQ1_0: five weights a byte in base 3). A group takes 32 bytes, in PQ2_0's own order: weight j is the code (weight +
# 1: 0, 1 or 2) in byte j // 4 at bits 2 (j % 4). The kernels multiply the codes as they are (kernels/ternary.ts): a
# shift of sixteen bytes and a mask give every fourth weight of 64, so nothing is widened, and nothing is kept for a
# row besides its weights and scales. The values are exactly the file's (int8 holds 127 times float32(d / 127)).
TERNARY_GROUP = 128
# the four weights of every byte, the lowest two bits first
TERNARY_VALUES = ((np.arange(256)[:, None] >> (0, 2, 4, 6) & 3) - 1).astype(np.int8)
NOT_TERNARY = "These weights are not ternary: a value is neither 0 nor the largest of its group of 128, or its negative."


def pack_ternary(values):
    """int8 values of -1, 0 and 1 -> a byte for every four of them (uint8), the first in the lowest two bits."""
    codes = (np.asarray(values, dtype=np.int8).reshape(-1, 4) + 1).astype(np.uint8)
    return codes[:, 0] | codes[:, 1] << 2 | codes[:, 2] << 4 | codes[:, 3] << 6


def unpack_ternary(packed):
    """A byte for every four values -> the int8 values, -1, 0 and 1 (2 for the code a ternary file never has)."""
    return TERNARY_VALUES[np.asarray(packed, dtype=np.uint8).reshape(-1)].reshape(-1)


def ternary(values):
    """float32 values, whole rows of groups of 128 -> (32 bytes a group, float32 scales): each value's sign, and the
    largest |value| of its group. Nothing is rounded: a value that is neither 0 nor plus or minus that scale is a
    ValueError (the weights of a model that is not ternary, which int8 is for)."""
    groups = np.asarray(values, dtype=np.float32).reshape(-1, TERNARY_GROUP)
    scales = np.abs(groups).max(axis=1)
    signs = np.sign(groups)
    if not np.array_equal(signs * scales[:, None], groups):
        raise ValueError(NOT_TERNARY)
    return pack_ternary(signs.astype(np.int8)).reshape(-1, TERNARY_GROUP // 4), scales


# the dtypes whose matrices are int8 values in another packing; what a packing does to a matrix: the bytes its count
# values take, and the group of a row of that length (int8 and int6: 32, or for int8 the largest power of two below
# it that divides the row)
PACKED = ("int6", "ternary")


def stored_bytes(count, packing=None):
    return count // 4 if packing == "ternary" else count * 3 // 4 if packing == "int6" else count


def group_of(length, packing=None):
    if packing == "ternary":
        return TERNARY_GROUP
    group = 32
    while length % group:
        group //= 2
    return group
