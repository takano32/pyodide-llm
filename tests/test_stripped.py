"""T258: the added tokens that take in the whitespace beside them (a tokenizer.json's rstrip and lstrip: Phi-3's and
Phi-4 mini's turn marks eat the newline their templates write after them), what the converter says of them and what the
engine does with it, against the tokenizers library itself. A tokenizer without such tokens says nothing new."""
import json

import pytest

import llama2_convert
from convert.vocabulary import stripping
from engine.tokenizer import WHITE_SPACE
from llama2_numpy import Tokenizer
from made_up_tokenizers import byte_level, sentencepiece

# a byte-level vocabulary whose <|end|> strips on its right, <|pad|> on its left, <|both|> on both sides, <|none|> on
# neither, and a word added (not special) that strips on its right as Phi-4 mini's <|tool|> does
MARKS = {"<|end|>": (True, False), "<|pad|>": (False, True), "<|both|>": (True, True), "<|none|>": (False, False),
         "<|tool|>": (True, False)}


def marked():
    """(the tokenizer.json, the ids of its added tokens) of MARKS, every key a real tokenizer.json has."""
    spec, ids = byte_level(["<|end|>", "<|pad|>", "<|both|>", "<|none|>"], ["<|tool|>"])
    spec["added_tokens"] = [{**token, "single_word": False, "lstrip": MARKS[token["content"]][1],
                             "rstrip": MARKS[token["content"]][0], "normalized": False} for token in spec["added_tokens"]]
    return spec, ids


def real_of(spec):
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_str(json.dumps({"version": "1.0", "truncation": None, "padding": None, "normalizer": None,
                                                     "post_processor": None, "decoder": None, **spec,
                                                     "pre_tokenizer": {**spec["pre_tokenizer"], "trim_offsets": True},
                                                     "model": {**spec["model"], "byte_fallback": False}}))


def engine_of(spec, stripped):
    pieces = list(llama2_convert.tokenizer_json_pieces(spec))
    return Tokenizer(llama2_convert.tokenizer_bin(pieces, len(pieces), spaces=False), len(pieces), kind="bytebpe",
                     pretokenizer="gpt2", stripped=stripped)


def test_white_space_is_the_characters_python_calls_spaces_less_four():
    """Unicode's White_Space, which the real tokenizers' char::is_whitespace is: str.isspace less \\x1c to \\x1f (the
    file and group separators, which Python calls spaces and Unicode does not)."""
    assert set(WHITE_SPACE) == {chr(code) for code in range(0x110000) if chr(code).isspace() and not 0x1c <= code <= 0x1f}


def test_the_whitespace_a_mark_takes_in_is_the_real_tokenizers():
    """Every character Python calls a space, the four it should not and a few that are no spaces at all, after a mark
    that strips on its right and before one that strips on its left: taken in where the real tokenizers take it."""
    spec, ids = marked()
    real = real_of(spec)
    candidates = [chr(code) for code in range(0x110000) if chr(code).isspace()] + ["​", "᠎", "﻿", "x"]
    for char in candidates:
        after = real.encode(f"<|end|>{char}a", add_special_tokens=False).ids
        before = real.encode(f"a{char}<|pad|>", add_special_tokens=False).ids
        assert (after == [ids["<|end|>"], *real.encode("a", add_special_tokens=False).ids]) == (char in WHITE_SPACE), repr(char)
        assert (before == [*real.encode("a", add_special_tokens=False).ids, ids["<|pad|>"]]) == (char in WHITE_SPACE), repr(char)


def test_the_converter_says_which_marks_strip_and_nothing_where_none_does():
    spec, _ = marked()
    assert stripping(spec["added_tokens"]) == {"stripped": {"right": ["<|both|>", "<|end|>", "<|tool|>"], "left": ["<|both|>", "<|pad|>"]}}
    plain, _ = byte_level(["<|end|>"])
    plain["added_tokens"] = [{**token, "lstrip": False, "rstrip": False} for token in plain["added_tokens"]]
    assert stripping(plain["added_tokens"]) == {} and stripping(byte_level(["<|end|>"])[0]["added_tokens"]) == {}
    # (and the options of a conversion carry it: the vocabulary's reading, convert/vocabulary.py)
    options = llama2_convert.vocabulary_of(json.dumps(spec).encode(), "tokenizer.json", 320).options
    assert options["stripped"] == stripping(spec["added_tokens"])["stripped"]
    assert "stripped" not in llama2_convert.vocabulary_of(json.dumps(plain).encode(), "tokenizer.json", 320).options


TEXTS = ["<|end|>\nuser", "user <|end|>  \n\n assistant", "a <|pad|> b", "a \n<|pad|>b", " <|both|> ", "x<|none|> y \n",
         "<|end|><|end|> a", "<|end|>   ", "   <|pad|>", "a<|tool|>\n b", "a\u3000<|both|>\u2028\u0085b", "q\x1c<|pad|>\x1cq",
         "<|end|>\n<|pad|>\n<|end|>", "user\n\n"]


def test_the_engine_strips_where_the_real_tokenizers_do():
    spec, ids = marked()
    real, stripped = real_of(spec), stripping(spec["added_tokens"])["stripped"]
    engine = engine_of(spec, stripped)
    specials = tuple(sorted(MARKS, key=lambda token: (-len(token), token)))
    for text in TEXTS:
        assert engine.encode(text, specials) == real.encode(text, add_special_tokens=False).ids, repr(text)
    # without it, the newline after a mark is a token of its own (what the engine did before T258)
    assert engine_of(spec, None).encode("<|end|>\nuser", specials) != real.encode("<|end|>\nuser", add_special_tokens=False).ids


def test_a_text_after_a_mark_that_strips_begins_without_its_whitespace():
    """A format that begins with such a mark (Phi-3's <|user|>) has it for its BOS (convert/beginning.py), and the text
    after it is encoded alone: the engine strips it where the token it follows does (encode()'s after)."""
    spec, ids = marked()
    real, engine = real_of(spec), engine_of(spec, stripping(spec["added_tokens"])["stripped"])
    specials = tuple(sorted(MARKS, key=lambda token: (-len(token), token)))
    for text in ("\n  hello<|end|>\n", "hello", " \t<|pad|>x"):
        assert [ids["<|end|>"], *engine.encode(text, specials, ids["<|end|>"])] == \
            real.encode("<|end|>" + text, add_special_tokens=False).ids, repr(text)
        assert [ids["<|none|>"], *engine.encode(text, specials, ids["<|none|>"])] == \
            real.encode("<|none|>" + text, add_special_tokens=False).ids, repr(text)


def test_a_sentencepiece_vocabulary_strips_after_its_bos_too():
    """A sentencepiece vocabulary spells its specials' pieces with spaces (cut()): the BOS is found by its piece."""
    data, ids = sentencepiece(controls=("<s>", "</s>", "<|user|>", "<|assistant|>", "<▁mark>"))
    pieces = list(llama2_convert.sentencepiece_pieces(data))
    tokenizer = Tokenizer(llama2_convert.tokenizer_bin(pieces, len(pieces)), len(pieces), kind="unigram",
                          stripped={"right": ["<|user|>", "<▁mark>"]})
    for bos in ("<|user|>", "<▁mark>"):
        assert tokenizer.encode("\nhello", ("<|user|>",), ids[bos]) == tokenizer.encode("hello", ("<|user|>",)), bos
    assert tokenizer.encode("\nhello", ("<|user|>",), ids["<s>"]) != tokenizer.encode("hello", ("<|user|>",))


@pytest.mark.parametrize("stripped", [{"rigth": ["<|end|>"]}, {"right": "<|end|>"}, {"right": [""]}, {"left": [3]}, ["<|end|>"]])
def test_the_engine_refuses_a_stripped_it_cannot_read(stripped):
    spec, _ = marked()
    with pytest.raises(ValueError, match="Unsupported stripped"):
        engine_of(spec, stripped)


def test_generate_encodes_the_prompt_after_the_bos():
    """Whether the BOS takes in the whitespace its prompt begins with is decided where the prompt is encoded, in
    generate(): it hands encode() the BOS (a Phi-3's prompt is the rest of a format whose first mark is the BOS). The
    tokenizer's own part is held above; this holds the call (T258's review: generate() passing no BOS broke none)."""
    from made_up import phi3_model
    from test_phi3 import engine
    llama, _ = engine(*phi3_model())
    seen, encode = [], llama.tokenizer.encode
    llama.tokenizer.encode = lambda text, specials=(), after=None: seen.append(after) or encode(text, specials, after)
    list(llama.generate("w1 w2", steps=12))
    assert llama.bos is not None and seen[:1] == [llama.bos]  # (the prompt's is the first call: others come after it)
