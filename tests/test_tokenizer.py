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
