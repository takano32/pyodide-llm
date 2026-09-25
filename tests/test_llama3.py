# Llama 3 (T106): the llama3 kind of RoPE scaling, its pre-tokenizer (digits up to three at a time), ignore_merges,
# the BOS its chat template writes, and the special tokens of a chat template read from tokenizer.json.
import json
import struct

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bytebpe import CORPUS, TEXTS
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gguf import gguf_file

import llama2_convert
from llama2_convert import Arrays, Conversion, check_config, gguf_model, gguf_read, rope_table, tokenizer_json_options
from llama2_numpy import Llama, Tokenizer, pretokenize, rope_frequencies

LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+")
# config.json of Llama 3.2 1B Instruct (unsloth/Llama-3.2-1B-Instruct@5a8abab4)
SCALING = {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
           "rope_type": "llama3"}
# what llama.cpp's convert wrote as rope_freqs into bartowski/Llama-3.2-1B-Instruct-GGUF@067b946c (Q8_0): for
# head_size 64 and rope_theta 500000, the plain frequency over the scaled one, pair by pair. An implementation of
# the same formula independent of this one (transformers needs PyTorch, which this machine has not).
LLAMA_CPP_DIVISORS = [1.0] * 15 + [1.6513293, 3.2922628, 9.666731] + [32.0] * 14


def test_the_llama3_frequencies_are_llama_cpps():
    divisors = rope_frequencies(64, 500000.0) / rope_frequencies(64, 500000.0, SCALING)
    assert np.allclose(divisors, LLAMA_CPP_DIVISORS, rtol=1e-6)


def test_without_scaling_the_frequencies_are_what_they_were():
    assert np.array_equal(rope_frequencies(64, 10000.0), 1.0 / 10000.0 ** (np.arange(0, 64, 2) / 64))
    with pytest.raises(ValueError, match="yarn"):
        rope_frequencies(64, 10000.0, {"rope_type": "yarn", "factor": 4.0})


def test_only_the_llama3_kind_of_scaling_is_let_through():
    config, weights = synthetic_weights()
    _, published = hugging_face(config, weights, True)
    check_config({**published, "rope_scaling": SCALING})
    for kind in ("linear", "dynamic", "yarn"):
        with pytest.raises(ValueError, match="RoPE scaling"):
            check_config({**published, "rope_scaling": {"rope_type": kind, "factor": 2.0}})


def llama3(max_seq_len=64):
    """A small Llama with Llama 3's scaling (a short original context, so that all three bands are there)."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published = {**published, "rope_theta": 500000.0, "rope_scaling": {**SCALING, "original_max_position_embeddings": 32}}
    return config, tensors, published


def test_the_file_and_the_engine_make_the_same_scaled_tables():
    """float32 files hold the tables (the converter makes them), int8 files do not (the engine does): the two must
    agree, and the engine needs rope_scaling from the options for that (T72's lesson: test the options' path)."""
    config, tensors, published = llama3()
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(config["vocab_size"])]}}).encode()
    tables = {}
    for dtype in ("float32", "int8"):
        conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                                dtype=dtype, max_seq_len=config["seq_len"])
        conversion.feed(file)
        conversion.finish()
        options = conversion.options
        assert options["rope_scaling"] == published["rope_scaling"]
        llama = Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), dtype=options["dtype"],
                      rope_theta=options["rope_theta"], rope_scaling=options["rope_scaling"], tokenizer_kind="unigram")
        tables[dtype] = llama.freq_cis_real, llama.freq_cis_imag
    header = struct.unpack_from("<7i", converted(Arrays(tensors), published, "float32"), 0)
    plain = rope_table({**published, "rope_scaling": None}, header, 0)
    assert not np.allclose(tables["float32"][0], plain), "the scaling changes the table"
    for which in (0, 1):
        assert np.allclose(tables["int8"][which], tables["float32"][which], atol=1e-6)


def test_llama3_pretokenizer_follows_the_pattern():
    regex = pytest.importorskip("regex", reason="pip install regex to check the pattern itself")
    for text in TEXTS + ["12345678", " 1234 ", "x9999y", "٣٤٥٦"]:
        assert pretokenize(text, "llama3") == regex.findall(LLAMA3_PATTERN, text)


def real_bpe(ignore_merges):
    """A byte-level BPE with Llama 3's pattern, trained here, with ignore_merges on or off."""
    tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers to check against the real one")
    from tokenizers import Tokenizer as Real, decoders, models, pre_tokenizers, trainers
    real = Real(models.BPE(ignore_merges=ignore_merges))
    real.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(tokenizers.Regex(LLAMA3_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    real.decoder = decoders.ByteLevel()
    real.train_from_iterator([CORPUS] * 8, trainers.BpeTrainer(
        vocab_size=900, special_tokens=["<|begin_of_text|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    spec = json.loads(real.to_str())
    # a word the merges cannot make: in the vocabulary, but its merge is gone (588 of Llama 3's words are so)
    word = "".join(llama2_convert_chars("Pyodide"))
    spec["model"]["merges"] = [merge for merge in spec["model"]["merges"]
                               if (merge if isinstance(merge, str) else " ".join(merge)).replace(" ", "") != word]
    spec["model"]["ignore_merges"] = ignore_merges
    return Real.from_str(json.dumps(spec)), spec


def llama2_convert_chars(text):
    from llama2_numpy import BYTE_CHARS
    return [BYTE_CHARS[byte] for byte in text.encode("utf-8")]


@pytest.mark.parametrize("ignore_merges", [False, True])
@pytest.mark.parametrize("text", TEXTS + ["Pyodide", " Pyodide", "12345"])
def test_matches_the_real_tokenizer(ignore_merges, text):
    real, spec = real_bpe(ignore_merges)
    options = tokenizer_json_options(spec)
    assert options["pretokenizer"] == "llama3" and options["ignore_merges"] is ignore_merges
    vocab_size = real.get_vocab_size()
    from llama2_convert import tokenizer_bin, tokenizer_json_pieces
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size, kind="bytebpe",
                     pretokenizer="llama3", ignore_merges=ignore_merges)
    assert mine.encode(text) == real.encode(text, add_special_tokens=False).ids


def test_ignore_merges_makes_a_difference_here():
    """So that the test above tests something: the word the merges cannot make is one token only with the flag."""
    assert len(real_bpe(False)[0].encode("Pyodide", add_special_tokens=False).ids) > 1
    assert len(real_bpe(True)[0].encode("Pyodide", add_special_tokens=False).ids) == 1


CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def test_the_special_tokens_of_the_template_go_to_the_engine():
    """T73's leftover: with ?hf= the template's <|im_start|> was spelled out letter by letter."""
    config, weights = synthetic_weights(vocab_size=300)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocab = {"w": 0, "x": 1, "wx": 2, **{f"w{i}": i + 2 for i in range(1, 295)}}
    specials = {"<|im_start|>": 297, "<|im_end|>": 298, "<|unused|>": 299}
    vocabulary = json.dumps({"added_tokens": [{"id": id, "content": text, "special": True} for text, id in specials.items()],
                             "model": {"type": "BPE", "vocab": {**vocab, **specials}, "merges": ["w x"]},
                             "pre_tokenizer": {"type": "ByteLevel"}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="int8", max_seq_len=config["seq_len"],
                            tokenizer_config=json.dumps({"chat_template": CHATML, "bos_token": "<|im_start|>"}))
    assert conversion.options["specials"] == ["<|im_start|>", "<|im_end|>"]
    tokenizer = Tokenizer(bytes(conversion.tokenizer), config["vocab_size"], kind="bytebpe")
    assert tokenizer.encode("<|im_start|>wx<|im_end|>", conversion.options["specials"]) == [297, 2, 298]


def test_a_llama3_gguf_splits_like_llama3_and_refuses_the_rope_freqs_table():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"], pre="llama-bpe")
    conversion = Conversion.from_gguf(file, dtype="int8", max_seq_len=1 << 20)
    assert conversion.options["pretokenizer"] == "llama3" and conversion.options["ignore_merges"] is True
    metadata, found, base = gguf_read(file)
    with pytest.raises(ValueError, match="rope_freqs"):
        gguf_model(metadata, {**found, "rope_freqs.weight": {"type": 0, "shape": [4], "offset": 0}}, base)
