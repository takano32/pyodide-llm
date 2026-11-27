"""T326: the context a conversion is made at changes nothing of a quantized checkpoint but its header's seq_len (the
engine makes the RoPE tables of the header), and so the worker keeps one conversion of it whatever the context and
sets the header to the context a load asks for (src/runtime/kept.ts's contextRead()), up to Conversion.longest: the
model's own. A float32 or float16 file holds RoPE tables as long as the context, and a GPT-2's file a table of learned
positions: those are read at the header's own context alone."""
import json
import struct

import numpy as np
import pytest
from made_up import hugging_face, safetensors_file, synthetic_weights
from test_gpt2 import POSITIONS, gpt2_model

from llama2_convert import Conversion
from llama2_numpy import Llama

SEQ_LEN = 64  # the made-up Llama's max_position_embeddings
UNIGRAM = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0, "vocab":
                      [["<unk>", 0.0], ["<s>", 0.0], ["</s>", 0.0]] + [[f"▁w{i}", -1.0 - i / 100] for i in range(317)]}}).encode()


def llama():
    config, weights = synthetic_weights(dim=128, hidden_dim=256, seq_len=SEQ_LEN)
    tensors, published = hugging_face(config, weights, True)
    return safetensors_file(tensors), published


def gpt2():
    tensors, config = gpt2_model()
    return safetensors_file(tensors), config


def converted(model, dtype, context):
    file, config = model()
    size = struct.unpack("<Q", file[:8])[0]
    made = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), UNIGRAM, "tokenizer.json", dtype=dtype,
                      max_seq_len=context, start=8 + size)
    made.feed(file[8 + size:])
    made.finish()
    return made


def seq_len(checkpoint):
    return struct.unpack_from("<i", checkpoint, 24)[0]


@pytest.mark.parametrize("dtype", ["int8", "int6"])
def test_a_quantized_checkpoint_is_the_same_at_every_context_but_for_its_header(dtype):
    short, long = converted(llama, dtype, 16), converted(llama, dtype, 32)
    assert (seq_len(short.checkpoint), seq_len(long.checkpoint)) == (16, 32)
    assert len(short.checkpoint) == len(long.checkpoint)
    assert short.checkpoint[:24] == long.checkpoint[:24] and short.checkpoint[28:] == long.checkpoint[28:]
    assert short.options == long.options and bytes(short.tokenizer) == bytes(long.tokenizer)
    # the longest it may be read at is the model's own, whatever the context it was made at
    assert short.longest == long.longest == SEQ_LEN
    assert converted(llama, dtype, 4096).longest == SEQ_LEN and seq_len(converted(llama, dtype, 4096).checkpoint) == SEQ_LEN


def test_a_checkpoint_kept_at_one_context_runs_past_it_at_another():
    """What a visit does with a conversion kept at 16 positions that asks for 64 (kept.ts's contextRead(): the header's
    four bytes): the file the converter makes at 64 to the byte, which the engine runs to its end, past the 16"""
    kept, wanted = converted(llama, "int8", 16), converted(llama, "int8", 64)
    read = bytearray(kept.checkpoint)
    struct.pack_into("<i", read, 24, 64)
    assert read == wanted.checkpoint
    engine = Llama(bytes(read), bytes(kept.tokenizer), **kept.options)
    assert engine.seq_len == 64
    logits = [engine.forward(token, pos) for pos, token in enumerate(range(3, 3 + 40))]
    assert all(np.isfinite(row).all() for row in logits[16:])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_a_float_checkpoint_holds_tables_as_long_as_its_context(dtype):
    short, long = converted(llama, dtype, 16), converted(llama, dtype, 32)
    assert len(short.checkpoint) < len(long.checkpoint), "its RoPE tables are as long as the context"
    assert (short.longest, long.longest) == (16, 32), "read at its header's context alone"


def test_a_gpt2s_learned_positions_are_its_context_at_every_dtype():
    made = converted(gpt2, "int8", 8)
    assert seq_len(made.checkpoint) == POSITIONS and made.longest == POSITIONS
