"""T400: what a tokenizer says of how its text is read and begins, which the converter did not read until then.
(1) a tokenizer.json's Lowercase normalizer, (2) a sentencepiece model whose normalizer adds no dummy prefix
(sarashina2.2's and CAT-Translate's: add_dummy_prefix false), (5) the BOS 1 that nothing names is no stop token.
Each against the real library where there is one here (tokenizers, sentencepiece)."""
import json

import pytest

from made_up_tokenizers import byte_level, converted, sent, sentencepiece

UNIGRAM = {"added_tokens": [{"id": 1, "content": "<sp>", "special": True}],
           "model": {"type": "Unigram", "unk_id": 0,
                     "vocab": [["<unk>", 0.0], ["<sp>", 0.0], ["▁", -3.0], ["▁hello", -1.0], ["hello", -2.0], ["▁world", -1.5],
                               ["world", -2.5], ["h", -6.0], ["e", -6.0], ["l", -6.0], ["o", -6.0], ["w", -6.0], ["r", -6.0], ["d", -6.0],
                               ["H", -6.0], ["W", -6.0], *([f"▁w{i}", -20.0] for i in range(304))]},  # (the made-up model's 320)
           "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": False}}
LOWER = {"type": "Lowercase"}


def real_tokenizer(tokenizer):
    tokenizers = pytest.importorskip("tokenizers")
    model = {**tokenizer["model"], **({"byte_fallback": False} if tokenizer["model"]["type"] == "Unigram" else {})}
    pre = tokenizer.get("pre_tokenizer")
    tokenizer = {**tokenizer, **({"pre_tokenizer": {**pre, "trim_offsets": True}} if pre and pre["type"] == "ByteLevel" else {})}
    return tokenizers.Tokenizer.from_str(json.dumps({
        "version": "1.0", "truncation": None, "padding": None, "normalizer": None, "pre_tokenizer": None, "post_processor": None,
        "decoder": None, **tokenizer, "model": model,
        "added_tokens": [{**token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
                         for token in tokenizer["added_tokens"]]}))


@pytest.mark.parametrize("normalizer", [LOWER, {"type": "Sequence", "normalizers": [{"type": "NFKC"}, LOWER]},
                                        {"type": "Sequence", "normalizers": [LOWER, {"type": "NFC"}]}])
def test_a_tokenizer_json_that_lowers_its_text(normalizer):
    """(1) The text is lowered before anything else reads it, a special token is not: as the tokenizers library does."""
    tokenizer = {**UNIGRAM, "normalizer": normalizer}
    options, engine = converted(tokenizer, config={"bos_token_id": 1})
    assert options["lowercase"] is True
    real = real_tokenizer(tokenizer)
    for text in ("Hello World", "HELLO<sp>WORLD", "hello", "hELLo<sp> World"):
        assert engine.encode(text, ("<sp>",)) == real.encode(text, add_special_tokens=False).ids, (text, normalizer)
    # a byte-level BPE: the same
    byte, ids = byte_level(["<|endoftext|>"])
    options, engine = converted({**byte, "normalizer": normalizer}, config={"bos_token_id": ids["<|endoftext|>"]})
    assert options["lowercase"] is True
    real = real_tokenizer({**byte, "normalizer": normalizer})
    for text in ("USER Assistant", "<|endoftext|>UsEr"):
        assert engine.encode(text, ("<|endoftext|>",)) == real.encode(text, add_special_tokens=False).ids


def test_no_lowercase_where_the_tokenizer_json_does_not_lower():
    for normalizer in (None, {"type": "NFKC"}, {"type": "Sequence", "normalizers": [{"type": "NFC"}]}):
        options, engine = converted({**UNIGRAM, "normalizer": normalizer}, config={"bos_token_id": 1})
        assert "lowercase" not in options
        assert engine.encode("Hello") != engine.encode("hello")


def test_a_sentencepiece_model_without_the_dummy_prefix():
    """(2) sarashina2.2's and CAT-Translate's tokenizer.model: no "▁" before the first text either ("never", T257's), as
    the sentencepiece library reads it; and decode() takes no space off what follows the BOS."""
    spm = pytest.importorskip("sentencepiece")
    data, ids = sentencepiece(dummy=False)
    options, engine = converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]},
                                tokenizer_class="LlamaTokenizer", legacy=False, add_prefix_space=False)
    assert options["prefixed"] == "never"
    real = spm.SentencePieceProcessor(model_proto=data)
    for text in ("hello world", "helloworld", " hello", "world hello"):
        assert engine.encode(text) == real.encode(text), text
    # a model that adds it, or says nothing (sentencepiece's default): as it was
    for said in (sentencepiece()[0],):
        options, engine = converted(said, "tokenizer.model", {"bos_token_id": ids["<s>"]}, tokenizer_class="LlamaTokenizer", legacy=False)
        assert "prefixed" not in options and engine.encode("hello") == spm.SentencePieceProcessor(model_proto=said).encode("hello")
    # a legacy Llama tokenizer's "wanting" (T308) is for a model that has the dummy prefix: none here is none anywhere
    options, _ = converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"]}, tokenizer_class="LlamaTokenizer", legacy=True)
    assert options["prefixed"] == "never"


def test_without_the_dummy_prefix_the_format_begins_with_its_own_first_token():
    """(2) sarashina2.2's format begins with <|user|> and the text typed follows it: with no dummy prefix anywhere,
    taking <|user|> off reads the rest the same, so it is the BOS and the ids are apply_chat_template's (T264), with
    no <s> in front."""
    data, ids = sentencepiece(dummy=False)
    template = "{% for m in messages %}{{ '<|user|>' if m.role == 'user' else '<|assistant|>' }}{{ m.content }}</s>{% endfor %}" \
               "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    options, engine = converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]},
                                chat_template=template, bos_token="<s>", tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["bos"] == ids["<|user|>"] and options["template"] == "{prompt}</s><|assistant|>"
    assert sent(options, engine, "hello world") == [ids["<|user|>"], ids["hello"], ids["▁world"], ids["</s>"], ids["<|assistant|>"]]
    # with the dummy prefix (as before T400 for these two): the BOS stays in front
    data, ids = sentencepiece()
    options, engine = converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]},
                                chat_template=template, bos_token="<s>", tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["bos"] == ids["<s>"] and options["template"] == "<|user|>{prompt}</s><|assistant|>"


def test_the_bos_nobody_names_is_no_stop_token():
    """(5) A model whose config.json has no bos_token_id, whose tokenizer names none and whose template is none this
    can read begins with the guess 1 (the engine begins with a token), and does not stop at it: token 1 of a
    byte-level vocabulary is '"'. The BOS that is named still stops the answer."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_end|>"])
    options, _ = converted(tokenizer, config={"eos_token_id": ids["<|im_end|>"]})
    assert options["bos"] == 1 and options["stop_tokens"] == [ids["<|im_end|>"]]
    # named by the tokenizer, and not by config.json: that one, and no guessed 1 beside it
    options, _ = converted(tokenizer, config={"eos_token_id": ids["<|im_end|>"]}, bos_token="<|endoftext|>")
    assert options["bos"] == ids["<|endoftext|>"] and options["stop_tokens"] == [ids["<|endoftext|>"], ids["<|im_end|>"]]
    # named by config.json: as it was
    options, _ = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]})
    assert options["stop_tokens"] == [ids["<|endoftext|>"], ids["<|im_end|>"]]
