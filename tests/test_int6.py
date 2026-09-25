# int6 (T98): six bits a weight, groups of 32 with a float32 scale each, the values packed 32 to 24 bytes. The
# NumPy engine widens them to exactly what a float32 checkpoint of the same rounded values holds, so the logits of the
# two are the same to the bit. The kernels (forward.js) are held to NumPy by tests/forward-check.mjs.
import struct

import numpy as np
import pytest
from conftest import pack_tokenizer, synthetic_weights, tiny_vocab
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file, streamed

from llama2_convert import Arrays, Safetensors, checkpoint_size, checkpoint_header
from llama2_numpy import Llama, checkpoint_dtype, pack6, quantize6, unpack6


def test_packing_is_lossless_and_24_bytes_a_group():
    values = (np.random.default_rng(1).integers(-32, 32, size=(5, 64)) * 4).astype(np.int8)
    packed = pack6(values)
    assert packed.shape == (10, 24) and packed.dtype == np.uint8
    assert np.array_equal(unpack6(packed).reshape(values.shape), values)
    # the layout the kernels widen: the low four of the six bits of j and j + 16 in byte j, the top two in 16..23
    six = np.arange(-32, 0, dtype=np.int8)
    bits = six.view(np.uint8) & 63
    assert pack6(six * 4)[0, 0] == (bits[0] & 15) | ((bits[16] & 15) << 4)
    assert pack6(six * 4)[0, 16] == sum(((bits[k] >> 4) & 3) << s for k, s in ((0, 0), (8, 2), (16, 4), (24, 6)))


def test_quantize6_is_six_bits_given_as_int8():
    values = np.random.default_rng(2).standard_normal((4, 64)).astype(np.float32)
    ints, scales = quantize6(values)
    assert np.all(ints % 4 == 0) and ints.min() >= -128 and ints.max() <= 124
    assert np.abs(ints).max(axis=1).tolist() == [124] * 8, "the whole range of six bits"
    error = np.abs(ints.astype(np.float32) * scales[:, None] - values.reshape(-1, 32)).max(axis=1)
    assert np.all(error <= 2 * scales + 1e-7), "round to nearest: half of a step of 4"


def rounded(tensors):
    """The Hugging Face tensors with every matrix rounded the way int6 rounds it (row by row, groups of 32)."""
    out = {}
    for name, tensor in tensors.items():
        if tensor.ndim == 2 and "norm" not in name:
            ints, scales = quantize6(tensor.reshape(-1, tensor.shape[-1]))
            tensor = (ints.astype(np.float32) * scales[:, None]).reshape(tensor.shape)
        out[name] = tensor
    return out


@pytest.mark.parametrize("model", ["llama", "llama-own-classifier", "qwen2"])
def test_the_engine_reads_int6_as_the_float32_of_the_same_values(model):
    config, weights = synthetic_weights(n_kv_heads=2, shared=model != "llama-own-classifier")
    tensors, published = (qwen2 if model == "qwen2" else hugging_face)(config, weights, model != "llama-own-classifier")
    six = converted(Arrays(tensors), published, "int6")
    header = struct.unpack_from("<7i", six, 0)
    bias = model == "qwen2"
    assert checkpoint_dtype(header, len(six), bias=bias) == "int6"
    assert len(six) == checkpoint_size(checkpoint_header(published, Arrays(tensors), 1 << 20), "int6", bias)
    assert len(six) < len(converted(Arrays(tensors), published, "int8")), "smaller than int8"
    reference = converted(Arrays(rounded(tensors)), published, "float32")
    vocabulary = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    a = Llama(six, vocabulary, dtype="int6", bias=bias)
    b = Llama(reference, vocabulary, dtype="float32", bias=bias)
    for pos, token in enumerate([1, 5, 9, 3]):
        assert np.array_equal(a.forward(token, pos), b.forward(token, pos))


def test_the_file_in_its_own_order_gives_the_same_int6():
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    got, progress = streamed(file, published, "int6", 4096)
    assert got == converted(Safetensors(reader(file)), published, "int6") and progress[-1][0] == progress[-1][1]


def test_rows_that_are_not_groups_of_32_are_refused():
    config, weights = synthetic_weights(dim=48, hidden_dim=64, n_heads=2, n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    with pytest.raises(ValueError, match="groups of 32"):
        converted(Arrays(tensors), published, "int6")
