# Encode -> decode round trips, on a synthetic vocabulary and on the real ones `make models` fetches.
import unicodedata

import pytest

from conftest import checkpoint_vocab_size, detokenize, model_file, tiny_tokenizer
from llama2_numpy import Tokenizer

TEXTS = [
    "hello world",
    "これからの流行りは、猫です",   # Japanese
    "sushi \U0001f363 and \U0001f363\U0001f363",                 # emoji, outside the vocabulary
    "\U00030EDE ಠ_ಠ",                                  # rare characters, byte fallback only
    "tabs\tand\nnewlines  and   spaces",
    "a",
    "éèê",
]


def roundtrip(tokenizer, text):
    tokens = tokenizer.encode(text)
    assert tokens, "encode produced nothing"
    return detokenize(tokenizer, tokens)


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
@pytest.mark.parametrize("text", TEXTS)
def test_synthetic_roundtrip(kind, text):
    tokenizer = tiny_tokenizer(kind=kind)
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
def test_nfkc_normalizes(kind):
    tokenizer = tiny_tokenizer(kind=kind, nfkc=True)
    assert roundtrip(tokenizer, "Ａ") == unicodedata.normalize("NFKC", "Ａ") == "A"


def test_byte_fallback_spells_out_unknown_characters():
    tokenizer = tiny_tokenizer(kind="bpe")
    tokens = tokenizer.encode("\U0001f363")
    # four bytes, none of which is a real piece
    assert [tokenizer.vocab[token] for token in tokens[1:]] == [b"<0xF0>", b"<0x9F>", b"<0x8D>", b"<0xA3>"]


def test_unigram_prefers_the_longest_well_scored_piece():
    tokenizer = tiny_tokenizer(kind="unigram")
    # " 流行り" scores better than " " + "流" + "行" + "り"
    assert [tokenizer.vocab[token] for token in tokenizer.encode("流行り")] == [" 流行り".encode("utf-8")]


def test_scores_below_the_unmatchable_threshold_are_never_used():
    tokenizer = tiny_tokenizer(kind="unigram")
    unmatchable = {i for i, score in enumerate(tokenizer.scores) if score <= Tokenizer.UNMATCHABLE}
    tokens = tokenizer.encode("hello world")
    assert not unmatchable.intersection(tokens)


# --------------------------------------------------------------------------- the real vocabularies

def real_tokenizer(name, vocab_size, kind="bpe", nfkc=False):
    return Tokenizer(model_file(name).read_bytes(), vocab_size, kind=kind, nfkc=nfkc)


@pytest.mark.parametrize("text", TEXTS)
def test_llama2_bpe_roundtrip(text):
    tokenizer = real_tokenizer("tokenizer.bin", checkpoint_vocab_size("stories15M.bin"))
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("text", TEXTS)
def test_tok512_bpe_roundtrip(text):
    tokenizer = real_tokenizer("tok512.bin", checkpoint_vocab_size("stories260K.bin"))
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("text", TEXTS)
def test_tiny_lm_unigram_roundtrip(text):
    tokenizer = real_tokenizer("tiny-lm.tokenizer.bin", checkpoint_vocab_size("tiny-lm.bin"),
                               kind="unigram", nfkc=True)
    assert roundtrip(tokenizer, text) == unicodedata.normalize("NFKC", text)


def test_llama2_bpe_matches_sentencepiece_on_a_known_sentence():
    tokenizer = real_tokenizer("tokenizer.bin", checkpoint_vocab_size("stories15M.bin"))
    # the ids llama2.c prints for this prompt (vocabulary of Llama 2)
    assert tokenizer.encode("Once upon a time") == [9038, 2501, 263, 931]


def test_special_tokens_inside_a_prompt_become_their_token():
    tokenizer = tiny_tokenizer("bpe")
    end = tokenizer.index[b"</s>"]
    tokens = tokenizer.encode("hello</s>\nworld", ("</s>",))
    assert tokens.count(end) == 1
    before, after = tokens[:tokens.index(end)], tokens[tokens.index(end) + 1:]
    assert before == tokenizer.encode("hello")
    # no dummy space after a special token: "\nworld", not " \nworld"
    assert after == tokenizer.encode("x\nworld")[len(tokenizer.encode("x")):]
    # without being told, the engine spells the same characters out
    assert end not in tokenizer.encode("hello</s>\nworld")
    assert tokenizer.encode("hello", ("</s>",)) == tokenizer.encode("hello")


# ------------------------------------------------------------------ a sentencepiece model's normalizer (T126)

def sentencepiece(normalizer, spelled=False):
    """A sentencepiece unigram model with no byte pieces, as rinna's are, and piece 13 a word: byte + 3 for a
    newline, which is what the engine spelled one with before it read the normalizer (rinna's 13 is った).
    spelled: with a byte piece, as Llama's, Mistral's and tiny-lm's have them (256 of them there)."""
    from make_hf_fixture import field
    NORMAL, UNKNOWN, CONTROL, BYTE = 1, 2, 3, 6
    words = ["▁", "▁a", "▁b", "a", "b", "▁x", "x", "y", "z", "c", "った"]
    pieces = [("[UNK]", UNKNOWN), ("<s>", CONTROL), ("</s>", CONTROL)] + [(word, NORMAL) for word in words]
    pieces += [("<0x0A>", BYTE)] if spelled else []
    assert pieces[13][0] == "った"
    model = b"".join(field(1, field(1, text.encode()) + field(2, -1.0 - i / 10) + field(3, kind))
                     for i, (text, kind) in enumerate(pieces))
    return model + field(2, field(3, 1)) + field(3, normalizer), len(pieces)


def test_an_nmt_normalizer_makes_newlines_and_tabs_spaces_and_one_space_of_many():
    """The review of T126: rinna's three models are nmt_nfkc with remove_extra_whitespaces, and a newline was
    written as token 13 (った in japanese-gpt-1b) where sentencepiece writes ▁. The converter says so to the engine."""
    from make_hf_fixture import field
    from llama2_convert import sentencepiece_options, sentencepiece_pieces, tokenizer_bin
    model, size = sentencepiece(field(1, b"nmt_nfkc"))  # remove_extra_whitespaces unset: true, sentencepiece's default
    options = sentencepiece_options(model)
    assert options == {"tokenizer_kind": "unigram", "nfkc": True, "nmt": True, "collapse": True, "unknown": 0}
    data = tokenizer_bin(sentencepiece_pieces(model), size)
    tokenizer = Tokenizer(data, size, kind="unigram", nfkc=True, nmt=True, collapse=True, unknown=0)
    pieces = lambda text: [tokenizer.vocab[token].decode() for token in tokenizer.encode(text)]
    for text in ["a\nb", "a\tb", "a  b", " a\r\n\n b ", "a​b", "a\x01\nb"]:
        assert pieces(text) == [" a", " b"], text
    assert pieces("\n") == [] and pieces("x　y") == [" x", " ", "y"]
    # characters the vocabulary lacks: the unknown piece, one for a run of them, as sentencepiece writes them
    assert pieces("a\U00020BB7\U00020BB7b") == [" a", "[UNK]", "b"] and pieces("a \U00020BB7 b") == [" a", " ", "[UNK]", " b"]
    bpe = Tokenizer(data, size, kind="bpe", nfkc=True, nmt=True, collapse=True, unknown=0)  # rinna's 1B is BPE
    assert [bpe.vocab[token].decode() for token in bpe.encode("a\U00020BB7\U00020BB7\nb")] == [" a", "[UNK]", " b"]
    before = Tokenizer(data, size, kind="unigram", nfkc=True)
    assert 13 in before.encode("a\nb"), "without the normalizer's settings: byte + 3"


def test_an_identity_normalizer_without_collapsing_changes_nothing():
    """Llama's and Mistral's tokenizer.model: identity, remove_extra_whitespaces off; tiny-lm's: nfkc, off."""
    from make_hf_fixture import field
    from llama2_convert import sentencepiece_options
    for name, nfkc in ((b"identity", False), (b"nfkc", True)):
        model, _ = sentencepiece(field(1, name) + field(4, 0), spelled=True)
        assert sentencepiece_options(model) == {"tokenizer_kind": "unigram", "nfkc": nfkc}


def test_the_conversion_hands_the_normalizer_to_the_engine():
    """T72's lesson: what the file does not say reaches the engine through the conversion's options"""
    import json
    import struct
    from conftest import synthetic_weights
    from make_hf_fixture import field
    from test_convert import hugging_face, safetensors_file
    import llama2_convert
    from llama2_numpy import Llama

    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    model, pieces = sentencepiece(field(1, b"nmt_nfkc"))
    model += b"".join(field(1, field(1, f"▁w{i}".encode()) + field(2, -9.0) + field(3, 1)) for i in range(settings["vocab_size"] - pieces))
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), model,
                                           "tokenizer.model", dtype="float32", max_seq_len=settings["seq_len"], start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    assert conversion.options["nmt"] and conversion.options["collapse"] and conversion.options["unknown"] == 0
    options = {key: value for key, value in conversion.options.items() if key != "dtype"}
    llama = Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), **options)
    assert llama.tokenizer.encode("a\n\tb") == llama.tokenizer.encode("a b")
