# llama2_convert.py
# Hugging Face Llama checkpoint -> what llama2_numpy.py loads, with nothing but NumPy, one piece of a tensor at a
# time: the weights are read through read(offset, length) and written into a buffer that has its final size from
# the start, float32, float16 or int8. So it never holds more than the output and a few megabytes, which is what
# lets the same code run when the site is built (convert_hf.py, quantize.py) and inside the browser, where the
# WebAssembly memory has 32 bits and never shrinks.
#
# This file is the window (T347): the converter is the package convert/ beside it, and every name it had as one file is
# imported here, so that `import llama2_convert` gives what it gave. A name set here is not the one the parts read:
# to change what the converter does (a test's PIECE), set it in the part that reads it.

import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
import struct  # noqa: F401
import time  # noqa: F401
import numpy as np  # noqa: F401

# the RoPE angles are the engine's, which computes them itself when a file leaves the tables out (int8)
from llama2_numpy import (CHARSMAP, DTYPES, EITHER, FORM, QUANTIZED, RMS_EPS, TERNARY_GROUP,  # noqa: F401
                          TERNARY_VALUES, convolution_form, dtype_of, form_of, layer_slots, linear_form,
                          linear_widths, pack6, quantize, quantize6, rope_frequencies, rope_magnitude,
                          rotated_form, rotated_widths, sign_bits, ternary)
from convert.checkpoint import IS_MATRIX, Writer, checkpoint_size, layout, tensor_bytes  # noqa: F401
from convert.template import (CHECK_DAYS, DAY, Loop, METHODS, MISSING, Namespace, STRFTIME, Undefined,  # noqa: F401
                              Unsupported, apply_filter, arithmetic, as_text, balanced, call_arguments,
                              closing_bracket, compare, config_token, evaluate, find_outside_quotes,
                              is_test, matching, next_branch, one_turn, one_turn_template, render, run,
                              split_operators, split_outside_quotes, string_end, tokenize_template,
                              truthy, unescape, value_of)
from convert.template import date_format, jinja_environment, rendered  # noqa: F401
from convert.readers import (GGUF_TENSORS, PQ2_0_CODES, SOURCES, Source, base3, bfloat16,  # noqa: F401
                             kernel_readers, pq2_0, ptq1_0, q8_0, read_types, source_of)
from convert.sources import Arrays, ROTATED, Safetensors, Shards, header_rotated, joined_shards  # noqa: F401
from convert.families import FAMILIES, family_of  # noqa: F401
from convert.families.family import Family  # noqa: F401
from convert.families.llama import GGUF_LAYER, GGUF_NAMES, GRANITE_ONES  # noqa: F401
from convert.families.gpt2 import gpt2_prefix  # noqa: F401
from convert.families.qwen35 import LINEAR_DEFAULTS, QWEN35_TILED  # noqa: F401
from convert.families.lfm2 import lfm2_config  # noqa: F401
from convert.config import (architecture, check_config, checkpoint_header, convolution_layers,  # noqa: F401
                            head_size, linear_layers, normalize, query_scale, rotary_dim,
                            unturned_layers, yarn)
from convert.plan import (checkpoint_form, conversion_plan, name_prefix,  # noqa: F401
                          permute_heads, rope_table, source_shape, transformed)
from convert.families.llama import has_bias, has_qk_norm  # noqa: F401
from convert.stream import (PIECE, Stream, convert_pieces, convert_weights, left_to_do, unsplit,  # noqa: F401
                            untiled, unturned)
from convert.gguf import (GGUF_NFC, GGUF_PRETOKENIZERS, GGUF_ROTATED, GGUF_VALUES, Incomplete,  # noqa: F401
                          gguf_agrees, gguf_model, gguf_read, gguf_rotated, gguf_tokenizer,
                          gguf_weights, rope_freqs_agree)
from convert.tokenizer import (PRETOKENIZERS, STAGED_PRETOKENIZERS, UNMATCHABLE, pretokenizer_name,  # noqa: F401
                               protobuf_fields, sentencepiece_charsmap, sentencepiece_options,
                               sentencepiece_pieces, sentencepiece_specials, tokenizer_bin,
                               tokenizer_json_bpe_pieces, tokenizer_json_charsmap,
                               tokenizer_json_options, tokenizer_json_pieces, tokenizer_kind_of)
from convert.tokenizer import described_options  # noqa: F401
from convert.tokenizer import prefixed_texts  # noqa: F401
from convert.conversion import Conversion  # noqa: F401
from convert.conversion import described  # noqa: F401
