"""checkpoint_dtype(): what a local file holds, told from its header and its size alone."""
import struct
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from llama2_numpy import Llama, check_tokenizer, checkpoint_dtype
from quantize import layout

TOKENIZER = pack_tokenizer(tiny_vocab(320))
CONFIGS = [dict(n_kv_heads=4), dict(n_kv_heads=2), dict(n_kv_heads=4, shared=False), dict(n_kv_heads=4, hidden_dim=48)]


def build(tmp_path, **config):
    config, weights = synthetic_weights(**config)
    checkpoint = pack_checkpoint(config, weights)
    source, target = tmp_path / "model.f32", tmp_path / "model.bin"
    source.write_bytes(checkpoint)
    subprocess.run([sys.executable, str(ROOT / "quantize.py"), str(source), str(target)], check=True)
    return checkpoint, target.read_bytes()


@pytest.mark.parametrize("config", CONFIGS)
def test_the_three_variants_are_told_apart(tmp_path, config):
    float32, int8 = build(tmp_path, **config)
    header = struct.unpack_from("<7i", float32, 0)
    float16 = float32[:28] + np.frombuffer(float32, dtype=np.float32, offset=28).astype(np.float16).tobytes()
    for name, data in (("float32", float32), ("float16", float16), ("int8", int8)):
        assert checkpoint_dtype(header, len(data)) == name
        Llama(data, TOKENIZER, dtype=name)  # and the engine reads exactly that many bytes


def test_the_sizes_follow_the_layout_of_quantize():
    header = (64, 172, 3, 8, 2, -300, 128)
    floats = sum(int(np.prod(shape)) for shape, _ in layout(*header))
    assert checkpoint_dtype(header, 28 + 4 * floats) == "float32"
    assert checkpoint_dtype(list(header), 28 + 2 * floats) == "float16"


@pytest.mark.parametrize("bias, arch", [(True, "llama"), (False, "gpt2"), (False, "neox")])
@pytest.mark.parametrize("shared", [True, False])
def test_the_sizes_follow_the_layout_of_every_architecture(bias, arch, shared):
    """A local Qwen2, GPT-2 or GPT-NeoX file has other tensors than a Llama: the size check must know (T77)."""
    from llama2_convert import checkpoint_size
    header = (64, 172, 3, 8, 8, 300 if shared else -300, 128)
    for dtype in ("float32", "float16", "int8"):
        assert checkpoint_dtype(header, checkpoint_size(header, dtype, bias, arch), bias, arch) == dtype
    # and told apart from the same header as a plain Llama
    assert checkpoint_size(header, "float32", bias, arch) != checkpoint_size(header, "float32")


@pytest.mark.parametrize("size", [0, 27, 1000, 123456789])
def test_a_file_of_another_size_is_refused(tmp_path, size):
    float32, _ = build(tmp_path, n_kv_heads=4)
    with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
        checkpoint_dtype(struct.unpack_from("<7i", float32, 0), size)


@pytest.mark.parametrize("header", [(0, 0, 0, 0, 0, 0, 0), (32, 64, 2, 5, 5, 100, 16), (32, 64, 2, 4, 3, 100, 16),
                                    (-32, 64, 2, 4, 4, 100, 16), (1885434739, 1, 1, 1, 1, 1, 1)])
def test_a_header_that_makes_no_sense_is_refused(header):
    with pytest.raises(ValueError, match="not a llama2.c checkpoint"):
        checkpoint_dtype(header, 1 << 20)


def test_a_tokenizer_of_another_vocabulary_is_refused():
    header = (32, 64, 2, 4, 4, -320, 24)
    check_tokenizer(TOKENIZER, header)
    check_tokenizer(memoryview(TOKENIZER), list(header))
    with pytest.raises(ValueError, match="do not belong together"):
        check_tokenizer(pack_tokenizer(tiny_vocab(400)), header)
    for broken in (b"", TOKENIZER[:-3], TOKENIZER + b"x", b"\x00" * 64):
        with pytest.raises(ValueError, match="not a llama2.c tokenizer.bin"):
            check_tokenizer(broken, header)
