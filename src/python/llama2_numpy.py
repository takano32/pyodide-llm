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
# This file is the window (T348): the engine is the package engine/ beside it, and the names that something outside
# src/python/ reads (the worker, the converter, the tests, the tools) are imported here, so that `import llama2_numpy`
# gives them what it gave. (T391.5: the names nothing outside read are the parts' own now; to give one out, import
# it here.) A name set here is not the one the parts read: to change what the engine does (a test's KV_START, a
# tool's Tokenizer), set it in the part that reads it (engine.model).

import math  # noqa: F401
import re  # noqa: F401
import struct  # noqa: F401
import time  # noqa: F401
import unicodedata  # noqa: F401
import numpy as np  # noqa: F401

from engine.tokenizer import (BOS, BYTE_CHARS, CharClasses, Charsmap, SPACES, STAGED, Tokenizer,  # noqa: F401
                              letter, mark, number, pretokenize)
from engine.packing import (NOT_TERNARY, TERNARY_GROUP, group32, pack6, pack_ternary, quantize,  # noqa: F401
                            quantize6, six, ternary, unpack6, unpack_ternary)
from engine.dtypes import DTYPES, EITHER, PACKED, QUANTIZED, Dtype, dtype_of  # noqa: F401
from engine.layout import (ATTENDING, CONVOLUTION, EVERY, FORM, LAYOUTS, LINEAR, MATRIX, PARTLY,  # noqa: F401
                           POSITIONS, STATEFUL, TABLE, VECTOR, WHOLE, Dims, Layout, Place, Row, Stateful,
                           after, convolution_form, file_size, form_of, kind_of, layout_of, linear_form,
                           linear_widths, placed, slots_of, stateful_form, stateful_kinds, suited,
                           tensor_rows, unturned_layers)
from engine.layers import (delta_rule, gelu, hadamard, head_norm, layernorm, rmsnorm, rope,  # noqa: F401
                           rope_frequencies, rope_magnitude, rotate, rotated_form, rotated_widths,
                           sign_bits, silu, softplus, unrotate)
from engine.kernels import kernel_quantizer, load_kernels  # noqa: F401
from engine.checkpoint import (OUTLIER_CHANNELS, OUTLIER_RATIO, SEVERAL_KINDS, Tensor,  # noqa: F401
                               check_tokenizer, checkpoint_dtype, external_tensors, outlier_channels,
                               outlier_columns)
from engine.sampler import NOT_FINITE, REPETITION_WINDOW, KernelSampler, NumpySampler, Sampling, greedy  # noqa: F401
from engine.external import ExternalForward  # noqa: F401
from engine.plan import Settings, forward_plan, layer_facts, plan_widths  # noqa: F401
from engine.model import KV_START, Llama, PROMPT_BLOCK, SWITCHES  # noqa: F401
