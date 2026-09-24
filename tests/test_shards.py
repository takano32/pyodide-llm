# A model split over several .safetensors files (T105): the page joins the shards' headers into the header of one
# file and feeds each shard's tensor data after the other (llama2_convert.joined_shards); the build reads them as one
# source (llama2_convert.Shards). Either way the checkpoint is the one the same tensors make in one file.
import json
import struct

import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gpt2 import gpt2_model

import llama2_convert
from llama2_convert import Safetensors, Shards, joined_shards


def split(tensors, pieces, order=None):
    """The tensors in pieces shards, the way transformers cuts a model: consecutive tensors, and every tensor whole.
    order: the order of the names (a shard may begin with lm_head and end with the embedding)."""
    names = order or list(tensors)
    size = -(-len(names) // pieces)
    return [safetensors_file({name: tensors[name] for name in names[i:i + size]}) for i in range(0, len(names), size)]


def fed(files, published, dtype, chunk):
    """What the page does: the headers joined, then each shard from its base, chunk bytes at a time."""
    headers, bases = [], []
    for file in files:
        (size,) = struct.unpack("<Q", file[:8])
        headers.append(file[8:8 + size].decode())
        bases.append(8 + size)
    header, lengths = joined_shards(headers)
    assert lengths == [len(file) - base for file, base in zip(files, bases)]
    stream = llama2_convert.Stream(json.loads(header), 0, published, dtype, 1 << 20)
    for file, base, length in zip(files, bases, lengths):
        data = file[base:base + length]
        for start in range(0, len(data), chunk):
            stream.feed(data[start:start + chunk])
    stream.finish()
    return bytes(stream.out)


def models():
    config, weights = synthetic_weights(n_kv_heads=2)
    yield "llama", *hugging_face(config, weights, False)
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    yield "qwen2", *qwen2(config, weights, False)
    yield "gpt2", *gpt2_model()


@pytest.mark.parametrize("pieces", [2, 3])
@pytest.mark.parametrize("dtype, chunk", [("float32", 1000), ("int8", 777)])
@pytest.mark.parametrize("name, tensors, published", list(models()), ids=[m[0] for m in models()])
def test_shards_give_the_checkpoint_of_one_file(name, tensors, published, pieces, dtype, chunk, monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 700)  # several pieces per tensor, and not a multiple of a row
    expected = converted(Safetensors(reader(safetensors_file(tensors))), published, dtype)
    # in the order the tensors come, and backwards (the classifier and the last layers first)
    for order in (list(tensors), list(reversed(tensors))):
        files = split(tensors, pieces, order)
        assert len(files) == pieces
        assert fed(files, published, dtype, chunk) == expected
        assert converted(Shards([Safetensors(reader(file)) for file in files]), published, dtype) == expected


def test_one_shard_is_one_file():
    """A model published as one shard with an index (T78) goes the same way and makes the same bytes."""
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    assert fed([file], published, "int8", 4096) == converted(Safetensors(reader(file)), published, "int8")


def test_a_tensor_in_two_shards_is_refused():
    config, weights = synthetic_weights()
    tensors, _ = hugging_face(config, weights, True)
    files = split(tensors, 2)
    twice = files + [safetensors_file({"model.norm.weight": tensors["model.norm.weight"]})]
    headers = [file[8:8 + struct.unpack("<Q", file[:8])[0]].decode() for file in twice]
    with pytest.raises(ValueError, match="two shards"):
        joined_shards(headers)
    with pytest.raises(ValueError, match="two shards"):
        Shards([Safetensors(reader(file)) for file in twice])


def test_a_shard_that_is_missing_is_found_missing():
    """A shard left out: the tensors it held are missing, as with one file that lacks them."""
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    files = split(tensors, 3)
    headers = [file[8:8 + struct.unpack("<Q", file[:8])[0]].decode() for file in files[:2]]
    header, _ = joined_shards(headers)
    with pytest.raises(ValueError, match="missing"):
        llama2_convert.Stream(json.loads(header), 0, published, "int8", 1 << 20)
