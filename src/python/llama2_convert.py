# llama2_convert.py
# Hugging Face Llama checkpoint -> what llama2_numpy.py loads, with nothing but NumPy, one piece of a tensor at a
# time: the weights are read through read(offset, length) and written into a buffer that has its final size from
# the start, float32, float16 or int8. So it never holds more than the output and a few megabytes, which is what
# lets the same code run when the site is built (convert_hf.py, quantize.py) and inside the browser, where the
# WebAssembly memory has 32 bits and never shrinks.
#
# This file is the window (T347): the converter is the package convert/ beside it, and the names that something
# outside src/python/ reads (the worker, the build's scripts, the tests, the tools) are imported here, so that
# `import llama2_convert` gives them what it gave. (T391.5: the names nothing outside read are the parts' own now; to
# give one out, import it here.) A name set here is not the one the parts read: to change what the converter does
# (a test's PIECE), set it in the part that reads it.

import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
import struct  # noqa: F401
import time  # noqa: F401
import numpy as np  # noqa: F401

# the RoPE angles are the engine's, which computes them itself when a file leaves the tables out (int8)
from llama2_numpy import (DTYPES, EITHER, FORM, QUANTIZED, TERNARY_GROUP, convolution_form,  # noqa: F401
                          dtype_of, form_of, linear_form, linear_widths, pack6, quantize, quantize6,
                          rope_frequencies, rope_magnitude, rotated_form, rotated_widths, sign_bits,
                          ternary)
from convert.checkpoint import IS_MATRIX, Writer, checkpoint_size, layout, tensor_bytes  # noqa: F401
from convert.template import (DAY, Loop, Namespace, Undefined, Unsupported, arithmetic, compare,  # noqa: F401
                              evaluate, jinja_environment, matching, one_turn, one_turn_template,
                              render, rendered, run)
from convert.readers import (GGUF_TENSORS, SOURCES, Source, base3, bfloat16, kernel_readers,  # noqa: F401
                             pq2_0, ptq1_0, q8_0, read_types, source_of)
from convert.sources import Arrays, ROTATED, Safetensors, Shards, joined_shards  # noqa: F401
from convert.families import FAMILIES, family_of  # noqa: F401
from convert.families.family import Family  # noqa: F401
from convert.families.qwen35 import QWEN35_TILED  # noqa: F401
from convert.config import (architecture, check_config, checkpoint_header, convolution_layers,  # noqa: F401
                            head_size, linear_layers, normalize, query_scale, rotary_dim,
                            unturned_layers, yarn)
from convert.plan import checkpoint_form, conversion_plan, permute_heads, rope_table, transformed  # noqa: F401
from convert.families.llama import has_bias, has_qk_norm  # noqa: F401
from convert.stream import (PIECE, Kept, Stream, convert_weights, left_to_do, unsplit,  # noqa: F401
                            untiled, unturned)
from convert.gguf import (Incomplete, gguf_agrees, gguf_model, gguf_read, gguf_rotated,  # noqa: F401
                          gguf_tokenizer, gguf_weights)
from convert.tokenizer import (UNMATCHABLE, prefixed_texts, pretokenizer_name,  # noqa: F401
                               sentencepiece_charsmap, sentencepiece_options, sentencepiece_pieces,
                               sentencepiece_specials, tokenizer_bin, tokenizer_json_charsmap,
                               tokenizer_json_options, tokenizer_json_pieces)
from convert.conversion import Conversion  # noqa: F401
from convert.conversion import described  # noqa: F401
