# T505.2: a GGUF opened alone (Conversion.from_gguf: its own vocabulary and settings, no original's files), where it
# was refused before: one that names no BOS and whose chat template begins with a token of its own; a Llama 3's
# rope_freqs table, read back into the rope_scaling that makes it; a sentencepiece BPE vocabulary (tokenizer.ggml.model
# "llama"); and two more of llama.cpp's names of a pre-tokenizer. Each on a small made-up GGUF against what the
# original's files give, and each with what is still refused, in the words it was refused in. The real files these were
# held to are in docs/notes/t505-gguf-alone-2026-10-13.md.
import json
import struct

import numpy as np
import pytest
from conducting import Directory, converted
from made_up import hugging_face, safetensors_file, synthetic_weights
from make_hf_fixture import field
from test_gguf import EPS, fed, gguf_file, with_original

import llama2_convert
from convert import conduct
from convert.gguf import (GGUF_IGNORE_MERGES, GGUF_NFC, GGUF_PRETOKENIZERS, NO_TABLE, gguf_read, gguf_tokenizer,
                          llama3_scaling, rope_freqs_values, sentencepiece_bpe)
from llama2_convert import Conversion, Incomplete
from llama2_numpy import Tokenizer, rope_frequencies

CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
START, END = 10, 11


def small(**shape):
    config, weights = synthetic_weights(n_kv_heads=2, **shape)
    tensors, published = hugging_face(config, weights, True)
    published["rms_norm_eps"] = 1e-5
    return config, tensors, published


def chat_vocabulary(size):
    """A byte-level BPE vocabulary whose tokens 10 and 11 are ChatML's two marks, control tokens as llama.cpp types them."""
    tokens, kinds = [f"w{i}" for i in range(size)], [1] * size
    tokens[START], tokens[END], kinds[START], kinds[END] = "<|im_start|>", "<|im_end|>", 3, 3
    return {"tokens": tokens, "kinds": kinds}


def chat_gguf(template=CHATML, bos=None, eos=END):
    config, tensors, published = small()
    more = [EPS, *([("tokenizer.chat_template", 8, template)] if template else [])]
    return gguf_file(tensors, published, config["vocab_size"], more=more, bos=bos, eos=eos,
                     vocabulary=chat_vocabulary(config["vocab_size"]))[0]


# ---- a GGUF that names no BOS
def test_a_gguf_that_names_no_bos_begins_with_the_token_its_template_begins_with():
    """unsloth's Qwen3.5 GGUFs: no bos_token_id, a template whose every text begins with <|im_start|>. That token is
    what the engine begins with, the format is what follows it, and the answer stops at the EOS alone: not at token 1,
    which nothing named (the review of T203: '"' of a byte-level vocabulary)."""
    got = fed(chat_gguf(), "int8")
    assert got.options["bos"] == START and got.options["stop_tokens"] == [END]
    assert got.options["template"] == "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    assert got.options["specials"] == ["<|im_start|>", "<|im_end|>"]
    # and the weights are those of the same file with a BOS named: the checkpoint to the byte
    named = fed(chat_gguf(bos=5), "int8")
    assert bytes(got.checkpoint) == bytes(named.checkpoint)
    # a BOS that is named stays what it was: the template's token begins the text, and the named one is stopped at
    assert named.options["bos"] == START and named.options["stop_tokens"] == [5, END]
    assert {**named.options, "stop_tokens": None} == {**got.options, "stop_tokens": None}


@pytest.mark.parametrize("template", [None, "{{ 'User: ' + messages[0]['content'] + '\nAnswer:' }}",
                                      "{{ messages[0]['content'] }}{{ '<|im_end|>' }}",
                                      "{% macro say(text) %}{{ text }}{% endmacro %}{{ say(messages | tojson | wordcount) }}{% if x is divisibleby 3 %}{% endif %}"],
                         ids=["no template", "a template that begins with text", "one that begins with what was typed",
                              "one that writes no turn"])
def test_a_gguf_that_names_no_bos_and_begins_with_no_token_is_refused_as_it_was(template):
    """Nothing says what such a model's texts begin with: refused in the words of before, and before the weights'
    memory is asked for (the sink)."""
    class Sink:
        def open(self, *args):
            raise AssertionError("the sink was opened for a GGUF that is refused")

    with pytest.raises(ValueError, match="This GGUF names no BOS token, which the engine needs: open it with its "
                                         "original's vocabulary and config.json."):
        Conversion.from_gguf(chat_gguf(template), sink=Sink())


def test_a_gguf_that_names_no_eos_is_refused_as_it_was():
    with pytest.raises(ValueError, match="This GGUF names no EOS token, which the engine needs"):
        Conversion.from_gguf(chat_gguf(bos=5, eos=None))
    with pytest.raises(ValueError, match="This GGUF names no EOS token, which the engine needs"):
        Conversion.from_gguf(chat_gguf(eos=None))


# ---- llama.cpp's table of a Llama 3's RoPE scaling
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def table_of(width, theta, scaling):
    """rope_freqs as llama.cpp's converter writes it: float32, the divisor of each pair's angle."""
    return (rope_frequencies(width, theta) / rope_frequencies(width, theta, scaling)).astype(np.float32)


def llama3_gguf(table=None, where="first", scaling=LLAMA3):
    config, tensors, published = small(dim=64, hidden_dim=128)
    published["rope_scaling"] = scaling
    table = table_of(16, 10000.0, scaling) if table is None else table
    file, same = gguf_file(tensors, published, config["vocab_size"], more=[EPS], **{where: {"rope_freqs.weight": table}})
    return config, published, file, same


@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_a_rope_freqs_table_alone_is_read_back_into_the_scaling_that_makes_it(dtype):
    """bartowski's Llama 3.1 and 3.2 GGUFs: the table is all a GGUF says of the scaling. Alone, the options say the
    rope_scaling the original's config.json says, and the checkpoint (a float32 one has the RoPE tables in it) is the
    one made with the original's files."""
    config, published, file, same = llama3_gguf()
    got = fed(file, dtype)
    assert got.options["rope_scaling"] == LLAMA3
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(config["vocab_size"])]}}).encode()
    staged = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    assert bytes(got.checkpoint) == bytes(staged.checkpoint) and staged.options["rope_scaling"] == LLAMA3
    # and it is not the checkpoint of the same weights turned by a plain RoPE, where the file has the tables
    plain = fed(gguf_file(*small(dim=64, hidden_dim=128)[1:], 320, more=[EPS])[0], dtype)
    assert "rope_scaling" not in plain.options
    assert (bytes(plain.checkpoint) != bytes(got.checkpoint)) is (dtype == "float32")


def test_the_table_is_read_off_the_head_and_more_is_asked_for_until_it_is_there(tmp_path, monkeypatch):
    """The table is the first tensor, right after the head: a head that ends before it is Incomplete (and nothing was
    opened), and the conduct of a conversion fetches on."""
    _, _, file, _ = llama3_gguf()
    _, tensors, base = gguf_read(file)
    assert tensors["rope_freqs.weight"]["offset"] == 0
    for size in (base, base + 31):
        with pytest.raises(Incomplete):
            Conversion.from_gguf(file[:size])
    assert Conversion.from_gguf(file[:base + 32]).options["rope_scaling"] == LLAMA3
    (tmp_path / "model.gguf").write_bytes(file)
    monkeypatch.setattr(conduct, "GGUF_HEAD", base + 8)  # the first piece ends inside the table

    class Asked(Directory):
        ranges = []

        def range(self, where, name, begin, end):
            self.ranges.append((begin, end))
            return super().range(where, name, begin, end)

    conversion = converted(Asked(tmp_path), {"weights": "model.gguf"}, dtype="int8")
    assert Asked.ranges == [(0, base + 8), (base + 8, 4 * (base + 8))]  # (no byte asked for twice, T374.3)
    assert conversion.options["rope_scaling"] == LLAMA3 and bytes(conversion.checkpoint) == bytes(fed(file, "int8").checkpoint)


@pytest.mark.parametrize("change, why", [
    (lambda table: table * np.float32(1.01), "no llama3 scaling makes it"),
    (lambda table: table[::-1].copy(), "the fast pairs slowed and the slow ones not"),
    (lambda table: np.where(np.arange(table.size) == 3, np.float32(2.5), table), "one pair off the line"),
    (lambda table: table[:-1], "a pair too few"),
    (lambda table: table * np.float32(0.5), "divisors below 1"),
    (lambda table: np.where(np.arange(table.size) == 1, np.float32("nan"), table), "a NaN"),
], ids=lambda value: value if isinstance(value, str) else "")
def test_a_table_no_llama3_scaling_makes_is_refused_as_it_was(change, why):
    """The engine makes its RoPE tables from a rope_scaling: a table it cannot make would be dropped without a word."""
    _, _, file, _ = llama3_gguf(change(table_of(16, 10000.0, LLAMA3)))
    with pytest.raises(ValueError, match="This GGUF scales its RoPE with a rope_freqs table, which the engine does not read."):
        fed(file, "int8")


def test_a_table_that_is_not_the_first_tensor_or_not_float32_is_refused_as_it_was():
    """Swallow 8B's GGUF has it after every other tensor, gigabytes in: a float32 checkpoint's RoPE tables are made
    before the first tensor is read. (With its original's config.json it is checked as it streams past, as before.)"""
    _, _, file, _ = llama3_gguf(where="extra")
    with pytest.raises(ValueError, match="rope_freqs table, which the engine does not read"):
        Conversion.from_gguf(file)
    # (and at once, with the head alone in hand: more of such a file is never asked for, to its last gigabyte)
    with pytest.raises(ValueError, match="rope_freqs table, which the engine does not read"):
        Conversion.from_gguf(file[:gguf_read(file)[2]])
    _, _, file, _ = llama3_gguf()
    metadata, tensors, base = gguf_read(file)
    tensors["rope_freqs.weight"]["type"] = 1
    with pytest.raises(ValueError, match="rope_freqs table, which the engine does not read"):
        rope_freqs_values(file, tensors, base)
    assert rope_freqs_values(file, {}, base) is None and NO_TABLE.startswith("This GGUF scales its RoPE")


def test_a_table_of_ones_is_no_scaling():
    _, _, file, _ = llama3_gguf(np.ones(8, dtype=np.float32))
    assert "rope_scaling" not in fed(file, "int8").options


@pytest.mark.parametrize("width, theta, scaling, found", [
    # Llama 3.1 8B and Llama 3.2 1B: what their real GGUFs' tables gave (docs/notes/t505-gguf-alone-2026-10-13.md)
    (128, 500000.0, dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192), None),
    (64, 500000.0, dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192), None),
    (64, 10000.0, dict(factor=2.5, low_freq_factor=1.0, high_freq_factor=8.0, original_max_position_embeddings=4096), None),
    # one pair between the fast and the slow: no line to read, and a Llama 3's 4 and 8192 make the table
    (16, 10000.0, dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192), None),
    # only original / low and original / high show in a table: the same table by the scaling with low = 1
    (64, 10000.0, dict(factor=3.0, low_freq_factor=2.0, high_freq_factor=6.0, original_max_position_embeddings=4096),
     dict(factor=3.0, low_freq_factor=1.0, high_freq_factor=3.0, original_max_position_embeddings=2048)),
])
def test_the_scaling_read_off_a_table_makes_the_table(width, theta, scaling, found):
    scaling = {"rope_type": "llama3", **scaling}
    config = {"hidden_size": 4 * width, "num_attention_heads": 4, "rope_theta": theta}
    got = llama3_scaling(table_of(width, theta, scaling), config)
    assert got == {"rope_type": "llama3", **(found or scaling)}
    assert isinstance(got["original_max_position_embeddings"], int)
    made, wanted = rope_frequencies(width, theta, got), rope_frequencies(width, theta, scaling)
    assert float(np.max(np.abs(made - wanted) / wanted)) < 1e-6


def test_a_table_with_no_slow_pair_says_no_factor_and_is_refused():
    """Four pairs of which none turns slowly enough to be slowed all the way: the largest divisor is not the factor."""
    scaling = {"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    with pytest.raises(ValueError, match="rope_freqs table"):
        llama3_scaling(table_of(8, 10000.0, scaling), {"hidden_size": 32, "num_attention_heads": 4, "rope_theta": 10000.0})


# ---- a sentencepiece vocabulary
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE = 1, 2, 3, 4, 6


def pieces_of(size, bytes_too=True):
    """A sentencepiece BPE model's pieces, as a Llama's and a Mistral's are laid out: <unk>, <s>, </s>, the bytes, and
    pieces scored by minus the number of their merge. One user-defined piece among them."""
    pieces = [("<unk>", 0.0, UNKNOWN), ("<s>", 0.0, CONTROL), ("</s>", 0.0, CONTROL)]
    pieces += [(f"<0x{byte:02X}>", 0.0, BYTE) for byte in range(256 if bytes_too else 0)]
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["▁", *letters, *[f"▁{a}" for a in letters], *[a + b for a in letters for b in letters]]
    pieces += [(text, -float(rank), NORMAL) for rank, text in enumerate(words[:size - len(pieces) - 1])]
    return pieces + [("[MARK]", 0.0, USER_DEFINED)]


def model_of(pieces, collapse=False):
    """The sentencepiece model file of the pieces: BPE (trainer_spec.model_type 2), the identity normalizer."""
    model = b"".join(field(1, field(1, text.encode()) + field(2, score) + field(3, kind)) for text, score, kind in pieces)
    return model + field(2, field(3, 2)) + field(3, field(1, b"identity") + field(4, int(collapse)))


def spm_gguf(pieces, more=(), scores=True, **shape):
    config, tensors, published = small(vocab_size=len(pieces), **shape)
    vocabulary = {"model": "llama", "tokens": [text for text, _, _ in pieces], "kinds": [kind for _, _, kind in pieces],
                  "scores": [score for _, score, _ in pieces] if scores else None}
    file, _ = gguf_file(tensors, published, len(pieces), more=[EPS, *more], pre=None, vocabulary=vocabulary)
    return published, file


@pytest.mark.parametrize("bytes_too", [True, False], ids=["with byte pieces", "without (an unknown piece)"])
@pytest.mark.parametrize("collapse", [False, True])
def test_a_sentencepiece_bpe_vocabulary_alone_is_the_models_own_files(bytes_too, collapse):
    """TheBloke's Mistral 7B: tokenizer.ggml.model "llama", the model's pieces, scores and types as they are. The
    tokenizer.bin and the tokenizer's options are those the model's own tokenizer.model gives, and so are the ids of a
    text."""
    pieces = pieces_of(1000, bytes_too)
    published, file = spm_gguf(pieces, more=[("tokenizer.ggml.remove_extra_whitespaces", 7, collapse)])
    got = fed(file, "int8")
    staged = with_original(file, published, model_of(pieces, collapse), "tokenizer.model", "int8")
    assert bytes(got.tokenizer) == bytes(staged.tokenizer) and bytes(got.checkpoint) == bytes(staged.checkpoint)
    assert got.options == staged.options
    assert got.options["tokenizer_kind"] == "bpe" and ("unknown" in got.options) is (not bytes_too)
    assert got.options.get("collapse", False) is collapse and "prefixed" not in got.options and "pretokenizer" not in got.options
    assert got.options["bos"] == 1 and got.options["stop_tokens"] == [1, 2]
    tokenizer = Tokenizer(got.tokenizer, len(pieces), kind="bpe", **{key: got.options[key] for key in ("collapse", "unknown") if key in got.options})
    ids = tokenizer.encode("hello  world é", ())
    # (merged by the scores: "▁h" before "he", "el" before "ll"; a letter no piece has is its bytes or the unknown piece)
    texts = [pieces[id][0] for id in ids]
    assert texts[:3] == ["▁h", "el", "lo"] and ("<0xC3>" in texts) is bytes_too and ("<unk>" in texts) is (not bytes_too)


@pytest.mark.parametrize("what", ["every score 0 (made from a tokenizer.json)", "every score the same", "a Unigram model's logarithms",
                                  "no scores", "no space before the text"])
def test_a_sentencepiece_vocabulary_that_is_no_bpe_with_its_scores_is_refused_as_it_was(what):
    """TheBloke's TinyLlama (every score 0), llm-jp's (every score -1000), a Unigram model (merged as a BPE's it would
    be another tokenizer, and a GGUF does not say which it was), and add_space_prefix false."""
    pieces = pieces_of(400)
    rescored = {"every score 0 (made from a tokenizer.json)": lambda score, rank: 0.0, "every score the same": lambda score, rank: -1000.0,
                "a Unigram model's logarithms": lambda score, rank: -3.25 - rank / 7}.get(what, lambda score, rank: score)
    pieces = [(text, rescored(score, rank) if kind == NORMAL else score, kind) for rank, (text, score, kind) in enumerate(pieces)]
    more = [("tokenizer.ggml.add_space_prefix", 7, False)] if what == "no space before the text" else []
    published, file = spm_gguf(pieces, more=more, scores=what != "no scores")
    assert not sentencepiece_bpe(gguf_read(file)[0])
    with pytest.raises(ValueError, match=r"This GGUF has a llama vocabulary: only byte-level BPE ones \(gpt2\) are supported."):
        Conversion.from_gguf(file)


def test_a_real_models_scores_are_read_as_a_bpes():
    """A Mistral 7B's: two pieces at -1e9 among whole numbers that are otherwise all different."""
    pieces = pieces_of(400)
    pieces[259] = (pieces[259][0], -1000000000.0, NORMAL)
    pieces[260] = (pieces[260][0], -1000000000.0, NORMAL)
    assert sentencepiece_bpe(gguf_read(spm_gguf(pieces)[1])[0])


# ---- llama.cpp's names of a pre-tokenizer
# the name -> (the engine's pattern, whether the text is normalized to NFC, whether a piece in the vocabulary is taken
# whole): written out a second time
NAMES = {"gpt-2": ("gpt2", False, False), "gpt2": ("gpt2", False, False), "smollm": ("gpt2-digits", False, False),
         "qwen2": ("qwen", True, False), "llama-bpe": ("llama3", False, True), "qwen35": ("qwen35", True, False),
         "granite-docling": ("gpt2", False, False), "minicpm5": ("minicpm5", False, False), "lfm2": ("llama3", False, False),
         # T505.2: each held to a real GGUF and its original's tokenizer.json
         "smaug-bpe": ("llama3", False, True), "deepseek-r1-qwen": ("qwen", True, False)}


def test_the_names_are_the_table():
    assert {name: (pattern, name in GGUF_NFC, name in GGUF_IGNORE_MERGES) for name, pattern in GGUF_PRETOKENIZERS.items()} == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_a_gguf_says_how_its_text_is_split_by_llama_cpps_name(name):
    config, tensors, published = small()
    file, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS], pre=name)
    options = fed(file, "int8").options
    assert (options["pretokenizer"], options["nfc"], options["ignore_merges"]) == NAMES[name]
    assert options["tokenizer_kind"] == "bytebpe" and options["nfkc"] is False


@pytest.mark.parametrize("name", ["dbrx", "tekken", "default", "llama3", "gpt-4o", "falcon3"])
def test_a_name_no_real_file_was_held_to_is_refused_as_it_was(name):
    """llama.cpp splits dbrx's and falcon3's text by Llama 3's pattern and knows llama3 as a name of it; no real GGUF
    of the name was held to its original here (HyperCLOVAX's original is gated), and tekken, gpt-4o and llama.cpp's
    default are other patterns than the engine's six."""
    config, tensors, published = small()
    file, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS], pre=name)
    with pytest.raises(ValueError, match=f"This GGUF splits text as {name}, which the engine does not know."):
        Conversion.from_gguf(file)


def test_a_gguf_alone_that_does_not_say_how_its_text_is_split_is_refused():
    """T505.3 (the owner, 2026-10-13): a byte-level BPE GGUF without tokenizer.ggml.pre was read as a GPT-2's, which is
    another split than a Llama 3's or a Qwen's own (three of the 500 most downloaded were such files). llama.cpp reads
    it as "default", and that is the name the refusal says. A sentencepiece vocabulary has no such name to say, and is
    opened as before (spm_gguf() writes none)."""
    config, tensors, published = small()
    file, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS], pre=None)
    assert "tokenizer.ggml.pre" not in gguf_read(file)[0]
    with pytest.raises(ValueError, match="^This GGUF splits text as default, which the engine does not know.$"):
        Conversion.from_gguf(file)
    assert fed(spm_gguf(pieces_of(400))[1], "int8").options["tokenizer_kind"] == "bpe"


def test_a_sentencepiece_vocabulary_that_comes_with_its_normalizer_is_refused():
    """T505.3: tokenizer.ggml.precompiled_charsmap is a sentencepiece model's normalizer, which nothing here reads from
    a GGUF: a vocabulary with one (not an empty one) is not read as if its normalizer were identity."""
    metadata = gguf_read(spm_gguf(pieces_of(400))[1])[0]
    key = "tokenizer.ggml.precompiled_charsmap"
    assert sentencepiece_bpe(metadata) and sentencepiece_bpe({**metadata, key: []}) and sentencepiece_bpe({**metadata, key: ""})
    for charsmap in ([0, 16, 0, 0, 1], "\x00\x10"):
        assert not sentencepiece_bpe({**metadata, key: charsmap})
        with pytest.raises(ValueError, match=r"^This GGUF has a llama vocabulary: only byte-level BPE ones \(gpt2\) are supported.$"):
            gguf_tokenizer({**metadata, key: charsmap}, len(metadata["tokenizer.ggml.tokens"]))
    assert gguf_tokenizer(metadata, len(metadata["tokenizer.ggml.tokens"]))[1]["tokenizer_kind"] == "bpe"


def test_a_gguf_alone_of_before_gives_the_tokenizer_it_gave():
    """What gguf_tokenizer() gives of a byte-level BPE GGUF is what it gave before T505.2: the pieces by their merges'
    ranks, the padding empty, the control tokens special and the user-defined ones added."""
    tokens, kinds = ["a", "b", "ab", "<s>", "<x>", "[PAD5]"], [1, 1, 1, 3, 4, 5]
    metadata = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": tokens,
                "tokenizer.ggml.token_type": kinds, "tokenizer.ggml.merges": ["a b"], "tokenizer.ggml.bos_token_id": 3,
                "tokenizer.ggml.eos_token_id": 3, "tokenizer.chat_template": "T"}
    blob, options, config, controls, added = gguf_tokenizer(metadata, 6)
    assert options == {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": True, "pretokenizer": "qwen", "ignore_merges": False}
    assert config == {"chat_template": "T", "bos_token": "<s>", "eos_token": "<s>"} and (controls, added) == (["<s>"], ["<x>"])
    rows, at = [], 4
    for _ in tokens:
        score, length = struct.unpack_from("<fi", blob, at)
        rows.append((blob[at + 8:at + 8 + length].decode(), score))
        at += 8 + length
    assert rows == [("a", -1e9), ("b", -1e9), ("ab", 0.0), ("<s>", -1e9), ("<x>", -1e9), ("", -1e9)]
