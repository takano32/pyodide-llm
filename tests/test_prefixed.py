"""The dummy prefix before the text after a special token (T308, T369): what the converter reads of it in a
tokenizer.json and in a sentencepiece model's tokenizer_config.json, against the tokenizers library itself.

A sentencepiece tokenizer puts "▁" before the first text. Some put it before the text after a special token too:
before every such stretch ("every"), or before every one that does not begin with a space already ("wanting"). The
list had a space written after each special token of three formats (zephyr's, EuroLLM's, llm-jp-4's) to say so.
"""
import json

import pytest

import llama2_convert
from engine.model import OPTIONS
from llama2_numpy import Llama, Tokenizer
from made_up_tokenizers import byte_level, converted, sentencepiece


def test_a_legacy_llama_tokenizer_prefixes_the_text_after_a_special_token():
    """zephyr's and EuroLLM's (tokenizer_class LlamaTokenizer, legacy true or not said) put the dummy prefix before the
    text after </s> too, unless that text begins with a space (what transformers makes of one: tests/format_check.py
    holds the list's two to it); TinyLlama's and Mistral's (legacy false) before the first text only."""
    data, ids = sentencepiece()
    for described in ({"tokenizer_class": "LlamaTokenizer", "legacy": True}, {"tokenizer_class": "LlamaTokenizerFast"},
                      {"tokenizer_class": "LlamaTokenizer", "legacy": None}):
        options, engine = converted(data, "tokenizer.model", **described)
        assert options["prefixed"] == "wanting", described
        assert engine.encode("hello</s>hello", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["▁hello"]]
        # a text that begins with a space has its "▁" already: what a visitor types after a token he typed
        assert engine.encode("hello</s> hello", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["▁hello"]]
        assert engine.encode("hello</s>  hello", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["▁"], ids["▁hello"]]
    for described in ({"tokenizer_class": "LlamaTokenizer", "legacy": False}, {"tokenizer_class": "T5Tokenizer"}, {}):
        options, engine = converted(data, "tokenizer.model", **described)
        assert "prefixed" not in options, described
        assert engine.encode("hello</s>hello", ("</s>",)) == [ids["▁hello"], ids["</s>"], ids["hello"]]


def test_a_model_with_no_dummy_prefix_says_never_over_a_legacy_tokenizer_class():
    """T400 (sarashina2.2's, CAT-Translate): the model puts no dummy prefix anywhere, so its legacy LlamaTokenizer's "wanting" gives
    way to "never" (convert/vocabulary.py's sentencepiece(); T640.1's review: no test held this before)."""
    data, ids = sentencepiece(dummy=False)
    for described in ({"tokenizer_class": "LlamaTokenizer", "legacy": True}, {"tokenizer_class": "LlamaTokenizerFast"}):
        options, engine = converted(data, "tokenizer.model", **described)
        assert options["prefixed"] == "never", described


UNIGRAM = {"added_tokens": [{"id": 1, "content": "<sp>", "special": True}],
           "model": {"type": "Unigram", "unk_id": 0,
                     "vocab": [["<unk>", 0.0], ["<sp>", 0.0], ["▁", -3.0], ["▁hello", -1.0], ["hello", -2.0], ["▁world", -1.5],
                               ["world", -2.5], ["h", -6.0], ["e", -6.0], ["l", -6.0], ["o", -6.0], ["w", -6.0], ["r", -6.0], ["d", -6.0]]}}
SPACES = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
DESCRIBED = [  # (what tokenizer.json says, which stretches of text are prefixed besides the first)
    ({"normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, SPACES]}}, "every"),
    ({"normalizer": {"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"Regex": "(?<!\\n)^"}, "content": "▁"},
                                                         {"type": "Replace", "pattern": {"Regex": " "}, "content": "▁"}]}}, "every"),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False}}, "wanting"),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": False}}, False),
    ({"normalizer": {"type": "Sequence", "normalizers": [SPACES]}}, False),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": False}}, False),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}}, "wanting"),  # (before prepend_scheme)
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": False}}, False),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}}, "wanting"),  # (neither key: the tokenizers library's default is "always")
    ({"normalizer": {"type": "Prepend", "prepend": "_"}}, False),
    ({"normalizer": {"type": "Replace", "pattern": {"Regex": "^x"}, "content": "▁"}}, False),
    ({"normalizer": {"type": "Replace", "pattern": {"String": "^"}, "content": "▁"}}, False),
    ({}, False),
]


@pytest.mark.parametrize("said, prefixed", DESCRIBED)
def test_a_tokenizer_json_says_which_texts_are_prefixed(said, prefixed):
    assert llama2_convert.prefixed_texts({**UNIGRAM, **said}) == prefixed
    options = llama2_convert.tokenizer_json_options({**UNIGRAM, **said})
    assert options.get("prefixed", False) == prefixed and ("prefixed" in options) == bool(prefixed)
    # a byte-level BPE has no dummy prefix at all: never said there
    tokenizer, _ = byte_level(["<|endoftext|>"])
    assert "prefixed" not in llama2_convert.tokenizer_json_options({**tokenizer, **{key: value for key, value in said.items() if key == "normalizer"}})


@pytest.mark.parametrize("said, prefixed", DESCRIBED[:5])
def test_the_prefix_after_a_special_token_is_the_real_tokenizers(said, prefixed):
    """The same descriptions through the tokenizers library itself: from the first special token on (what stands before
    the first text is not T308's), the engine's ids with what the converter says are the real ones, also where the
    text after the token begins with a space or two, or with a newline."""
    tokenizers = pytest.importorskip("tokenizers")
    real = tokenizers.Tokenizer.from_str(json.dumps({"version": "1.0", "truncation": None, "padding": None, "normalizer": None,
                                                     "pre_tokenizer": None, "post_processor": None, "decoder": None,
                                                     **UNIGRAM, **said,
                                                     "added_tokens": [{**UNIGRAM["added_tokens"][0], "single_word": False, "lstrip": False,
                                                                       "rstrip": False, "normalized": False}],
                                                     "model": {**UNIGRAM["model"], "byte_fallback": False}}))
    pieces = list(llama2_convert.tokenizer_json_pieces({**UNIGRAM, **said}))
    options = llama2_convert.tokenizer_json_options({**UNIGRAM, **said})
    assert options.get("prefixed", False) == prefixed
    engine = Tokenizer(llama2_convert.tokenizer_bin(pieces, len(pieces)), len(pieces), kind="unigram", prefixed=prefixed)
    for text in ("<sp>hello<sp>world", "<sp>hello world<sp>hello", "hello<sp>world hello<sp>", "<sp> hello<sp>  world",
                 "<sp>hello<sp> world<sp>hello"):
        ids = real.encode(text, add_special_tokens=False).ids
        after = ids.index(1)
        assert engine.encode(text, ("<sp>",))[-(len(ids) - after):] == ids[after:], (text, said)
    assert real.encode("<sp>hello", add_special_tokens=False).ids == ([1, 3] if prefixed else [1, 4])
    assert real.encode("<sp> hello", add_special_tokens=False).ids == ([1, 2, 3] if prefixed == "every" else [1, 3])


def test_the_engine_knows_the_two_ways_and_no_other():
    pieces = list(llama2_convert.tokenizer_json_pieces(UNIGRAM))
    data = llama2_convert.tokenizer_bin(pieces, len(pieces))
    for way in (True, "always", "first", 1):
        with pytest.raises(ValueError, match="Unsupported prefixed"):
            Tokenizer(data, len(pieces), kind="unigram", prefixed=way)
    # the first text has the dummy prefix whichever way, as it always had
    for way in (False, "every", "wanting"):
        assert Tokenizer(data, len(pieces), kind="unigram", prefixed=way).encode("hello world") == [3, 5]


def test_the_engine_takes_the_new_options_and_refuses_none_of_the_converters():
    """What the converter says goes to Llama() as keywords (the worker spreads the options): each is one it takes
    (T639.1: engine/model.py's OPTIONS, the names of its tables)."""
    taken = set(OPTIONS)
    assert {"lowercase", "prefixed"} <= taken
    data, ids = sentencepiece()
    options, _ = converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"]}, do_lower_case=True, tokenizer_class="LlamaTokenizer",
                           chat_template="<|user|>{{ messages[0].content }}")
    assert {"lowercase", "prefixed", "template"} <= set(options) and set(options) - {"template"} <= taken


FIRST_SPACED = [  # T400 (3): (what tokenizer.json says, whether the first text that begins with a space goes without "▁")
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False}}, True),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": False}}, True),
    ({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}}, True),  # (before prepend_scheme)
    ({"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}]}}, True),
    ({"normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, SPACES]}}, False),
]
# (and two that say no "▁" at all, where the engine still puts one before the first text: not T400 (3)'s, see TASKS.md)
NOT_SPACED = [{"normalizer": {"type": "Sequence", "normalizers": [SPACES]}}, {}]


@pytest.mark.parametrize("said", NOT_SPACED)
def test_no_metaspace_is_not_spaced(said):
    from convert.tokenizer import metaspaced
    assert not metaspaced({**UNIGRAM, **said}) and "spaced" not in llama2_convert.tokenizer_json_options({**UNIGRAM, **said})


@pytest.mark.parametrize("said, spaced", FIRST_SPACED)
def test_a_metaspace_puts_no_prefix_before_a_first_text_that_begins_with_a_space(said, spaced):
    """T400 (3): the tokenizers library's Metaspace puts "▁" before a stretch of text only where it does not begin with
    one already, the first stretch too; a Prepend normalizer puts it before every one. The converter says the first as
    spaced, and the engine's ids are the real ones for the whole text, a first text with a space, two, or none."""
    from convert.tokenizer import metaspaced
    assert metaspaced({**UNIGRAM, **said}) == spaced
    options = llama2_convert.tokenizer_json_options({**UNIGRAM, **said})
    assert options.get("spaced", False) == spaced and ("spaced" in options) == spaced
    tokenizers = pytest.importorskip("tokenizers")
    real = tokenizers.Tokenizer.from_str(json.dumps({"version": "1.0", "truncation": None, "padding": None, "normalizer": None,
                                                     "pre_tokenizer": None, "post_processor": None, "decoder": None,
                                                     **UNIGRAM, **said,
                                                     "added_tokens": [{**UNIGRAM["added_tokens"][0], "single_word": False, "lstrip": False,
                                                                       "rstrip": False, "normalized": False}],
                                                     "model": {**UNIGRAM["model"], "byte_fallback": False}}))
    pieces = list(llama2_convert.tokenizer_json_pieces({**UNIGRAM, **said}))
    engine = Tokenizer(llama2_convert.tokenizer_bin(pieces, len(pieces)), len(pieces), kind="unigram",
                       **{key: options[key] for key in ("prefixed", "spaced") if key in options})
    for text in (" hello world", "  hello", " hello<sp>world", "hello world", "hello<sp> world", " hello<sp> world"):
        assert engine.encode(text, ("<sp>",)) == real.encode(text, add_special_tokens=False).ids, (text, said)
    # and a first text with a space has its "▁" from the space alone (the Prepend's two: "▁", "▁hello")
    assert real.encode(" hello", add_special_tokens=False).ids == ([3] if spaced else [2, 3])


def test_sentencepiece_itself_puts_its_prefix_before_a_first_text_with_a_space():
    """T400 (3): a tokenizer.model is read as sentencepiece reads it (spaced not said): "▁" before the first text
    whatever it begins with, also for a legacy Llama tokenizer ("wanting" after a special token only)."""
    spm = pytest.importorskip("sentencepiece")
    data, ids = sentencepiece()
    real = spm.SentencePieceProcessor(model_proto=data)
    for described in ({"tokenizer_class": "LlamaTokenizer", "legacy": True}, {"tokenizer_class": "LlamaTokenizer", "legacy": False}, {}):
        options, engine = converted(data, "tokenizer.model", **described)
        assert "spaced" not in options, described
        for text in (" hello", "hello", " hello world"):
            assert engine.encode(text) == real.encode(text), (text, described)
