# The build's converter (convert_hf.py) on every architecture: it sized the output as a Llama, so a GPT-2 or a
# GPT-NeoX stopped on "the buffer has not the size of the checkpoint" (found in T85). The page's path was fine.
import json

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_qwen3 import qwen3
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gpt2 import gpt2_model
from test_neox import neox_model

import convert_hf
from llama2_convert import Safetensors


def llama_model():
    config, weights = synthetic_weights()
    return hugging_face(config, weights, True)


def qwen2_model():
    config, weights = synthetic_weights()
    return qwen2(config, weights, True)


def qwen3_model():
    # heads of 16 in a dim of 32 with 4 heads: q twice as wide as dim, as in Qwen3 0.6B (T124)
    config, weights = synthetic_weights(n_kv_heads=2, head_size=16)
    return qwen3(config, weights, True)


def qwen35_model():
    from conftest import qwen35_model as hybrid
    return hybrid()


# all four architectures (Fable's review: the Llama path was the one that worked, and a change to it should say so)
def lfm2_model():
    # T260: the FFN's inside is not config.json's intermediate_size but what normalize() makes of it (96 becomes 64)
    from conftest import lfm2_model as mixed
    return mixed()


@pytest.mark.parametrize("model", [llama_model, qwen2_model, qwen3_model, gpt2_model, neox_model, qwen35_model, lfm2_model],
                         ids=["llama", "qwen2", "qwen3", "gpt2", "neox", "qwen35", "lfm2"])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_convert_hf_writes_what_the_page_writes(tmp_path, model, dtype):
    tensors, config = model()
    file = safetensors_file(tensors)
    (tmp_path / "model.safetensors").write_bytes(file)
    (tmp_path / "config.json").write_text(json.dumps(config))
    convert_hf.convert(tmp_path, tmp_path / "out.bin", np.dtype(dtype), 2048)
    assert (tmp_path / "out.bin").read_bytes() == converted(Safetensors(reader(file)), config, dtype, 2048)


def test_a_smollm3_directory_is_told_to_carry_the_layers_in_its_options(tmp_path):
    """T255 review: the file of a SmolLM3 is a Llama's; only options name the layers RoPE leaves alone (convert_hf prints them)."""
    settings, weights = synthetic_weights(n_layers=4)
    _, config = hugging_face(settings, weights, True)
    (tmp_path / "config.json").write_text(json.dumps({**config, "model_type": "smollm3", "no_rope_layers": [1, 1, 1, 0]}))
    assert "unturned [3]" in convert_hf.options_note(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert convert_hf.options_note(tmp_path) is None


# ---- T374.2.3: the build tries the tokenizers the page tries, in its order, by the one list (convert.conduct's TOKENIZERS)
def tokenizers(vocab_size=64):
    from test_gguf import sentencepiece, unigram
    json_file, model = unigram(vocab_size), sentencepiece(vocab_size)
    return (json_file.encode() if isinstance(json_file, str) else json_file), model, vocab_size


def test_the_builds_candidates_are_the_conducts():
    from convert import conduct
    assert convert_hf.TOKENIZERS is conduct.TOKENIZERS


def test_the_build_takes_the_first_candidate_that_is_there(tmp_path):
    json_file, model, vocab_size = tokenizers()
    of = {name: json_file if name.endswith(".json") else model for name in convert_hf.TOKENIZERS}
    assert len({convert_hf.tokenizer_of(name, data, vocab_size) for name, data in of.items()}) == 2  # (two kinds, told apart)
    for at, name in enumerate(convert_hf.TOKENIZERS):
        # every candidate from this one on is there: this one is read
        for later in convert_hf.TOKENIZERS[at:]:
            (tmp_path / later).write_bytes(of[later])
        convert_hf.convert_tokenizer(tmp_path, tmp_path / "out", vocab_size)
        assert (tmp_path / "out").read_bytes() == convert_hf.tokenizer_of(name, of[name], vocab_size), name
        for later in convert_hf.TOKENIZERS[at:]:
            (tmp_path / later).unlink()


def test_the_build_goes_on_past_a_tokenizer_it_cannot_read_and_says_why_where_none_will_do(tmp_path):
    _, model, vocab_size = tokenizers()
    (tmp_path / "tokenizer.json").write_text("unreadable")
    (tmp_path / "spiece.model").write_bytes(model)
    convert_hf.convert_tokenizer(tmp_path, tmp_path / "out", vocab_size)
    assert (tmp_path / "out").read_bytes() == convert_hf.tokenizer_of("spiece.model", model, vocab_size)
    (tmp_path / "spiece.model").unlink()
    with pytest.raises(ValueError):  # (the refusal of the one that is there: tokenizer.json is not JSON)
        convert_hf.convert_tokenizer(tmp_path, tmp_path / "none", vocab_size)
    (tmp_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json, tokenizer.model, spiece.model"):
        convert_hf.convert_tokenizer(tmp_path, tmp_path / "none", vocab_size)
    assert not (tmp_path / "none").exists()


def test_a_candidate_added_to_the_list_is_one_the_build_reads(tmp_path, monkeypatch):
    _, model, vocab_size = tokenizers()
    monkeypatch.setattr(convert_hf, "TOKENIZERS", (*convert_hf.TOKENIZERS, "vocab.model"))
    (tmp_path / "vocab.model").write_bytes(model)
    convert_hf.convert_tokenizer(tmp_path, tmp_path / "out", vocab_size)
    assert (tmp_path / "out").read_bytes() == convert_hf.tokenizer_of("vocab.model", model, vocab_size)
