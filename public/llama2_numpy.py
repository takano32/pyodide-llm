# llama2_numpy.py
# Llama 2 inference with NumPy. A port of tairov/llama2.py (itself a port of karpathy/llama2.c) in which
# every loop over vector elements became a NumPy call, so the interpreter only sequences the layers.
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
#
# This file is the window (T348): the engine is the package engine/ beside it, and every name it had as one file is
# imported here, so that `import llama2_numpy` gives what it gave. A name set here is not the one the parts read:
# to change what the engine does (a test's KV_START, a tool's Tokenizer), set it in the part that reads it
# (engine.model).

import codecs  # noqa: F401
import heapq  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
import struct  # noqa: F401
import time  # noqa: F401
import unicodedata  # noqa: F401
import numpy as np  # noqa: F401

from engine.tokenizer import (BOS, BYTE_CHARS, CHARSMAP, CHAR_BYTES, CHAR_CLASSES, CONTRACTED,  # noqa: F401
                              CharClasses, Charsmap, DIGITS, PATTERNS, SPACES, STAGED, THREES,
                              Tokenizer, byte_chars, letter, mark, number, pretokenize)
from engine.packing import (NOT_TERNARY, TERNARY_GROUP, TERNARY_VALUES, group32, pack6,  # noqa: F401
                            pack_ternary, quantize, quantize6, six, ternary, unpack6, unpack_ternary,
                            unpacked6)
from engine.dtypes import DTYPES, EITHER, PACKED, QUANTIZED, Dtype, dtype_of  # noqa: F401
from engine.layout import (ATTENDING, CLASSIFIER, CONVOLUTION, EMBEDDING, EVERY, FORM, LAYOUTS, LINEAR,  # noqa: F401
                           MATRIX, POSITIONS, STATEFUL, TABLE, VECTOR, Dims, Place, Row, after,
                           convolution_form, file_size, form_of, kind_of, kinds_form, layer_slots,
                           linear_form, linear_widths, placed, suited, tensor_rows)
from engine.layers import (RMS_EPS, delta_rule, gelu, hadamard, head_norm, l2_heads, layernorm,  # noqa: F401
                           partial_rope, rmsnorm, rope, rope_frequencies, rope_magnitude, rotate,
                           rotated_form, rotated_widths, sign_bits, silu, softplus, unrotate)
from engine.kernels import kernel_quantizer, kernel_wideners, load_kernels  # noqa: F401
from engine.checkpoint import (OUTLIER_CHANNELS, OUTLIER_RATIO, SEVERAL_KINDS, Tensor,  # noqa: F401
                               check_tokenizer, checkpoint_dtype, external_tensors, outlier_channels,
                               outlier_columns)
from engine.sampler import NOT_FINITE, REPETITION_WINDOW, KernelSampler, NumpySampler, greedy  # noqa: F401
from engine.external import ExternalForward  # noqa: F401
from engine.plan import forward_plan, layer_facts, plan_widths  # noqa: F401
from engine.model import KV_START, Llama, PROMPT_BLOCK, SWITCHES  # noqa: F401
