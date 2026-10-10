"""The token a text begins with, where the converter read the model's own chat template (T264, T369).

The rule: a model whose template the converter read sends the ids of transformers' apply_chat_template, which puts
nothing in front of what the template writes. The engine begins every text with its BOS, so that is the token the
template writes first, and the format is what follows it.
"""
import json

import llama2_convert
from made_up_tokenizers import byte_level, converted, model, sent, sentencepiece

CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{{ '<|im_start|>assistant\\n' }}"


def test_a_template_that_begins_with_another_token_than_the_bos_begins_the_text():
    """A Qwen's, a Hermes 3's: the tokenizer names a BOS (or config.json does), the template writes <|im_start|> first
    and no BOS. The engine's BOS is <|im_start|> then, the format what follows it, and the ids are those of the
    template's text, none in front. The answer stops at the mark of a new turn and still at the BOS and the EOS."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    for named in ({"bos_token": "<|endoftext|>"}, {"bos_token": {"content": "<|endoftext|>"}}, {"bos_token": None}):
        options, engine = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]},
                                    chat_template=CHATML, **named)
        assert options["bos"] == ids["<|im_start|>"]
        assert options["template"] == "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        assert options["stop_tokens"] == [ids["<|im_start|>"], ids["<|endoftext|>"], ids["<|im_end|>"]]
        assert options["specials"] == ["<|im_start|>", "<|im_end|>"]
        whole = engine.encode("<|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\n", ("<|im_start|>", "<|im_end|>"))
        assert sent(options, engine, "hi there") == whole and whole[0] == ids["<|im_start|>"]


def test_a_template_that_writes_the_bos_first_is_as_it_was():
    """Llama 3's, LFM2's: the template writes bos_token itself. The engine's BOS is that one, and the format begins
    after it (T106): one BOS, not two."""
    tokenizer, ids = byte_level(["<|begin|>", "<|im_start|>", "<|im_end|>"])
    options, engine = converted(tokenizer, config={"bos_token_id": ids["<|begin|>"], "eos_token_id": ids["<|im_end|>"]},
                                chat_template="{{ bos_token }}" + CHATML, bos_token="<|begin|>")
    assert options["bos"] == ids["<|begin|>"] and options["stop_tokens"] == [ids["<|begin|>"], ids["<|im_end|>"]]
    assert options["template"] == "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    assert sent(options, engine, "x")[:2] == [ids["<|begin|>"], ids["<|im_start|>"]]


def test_a_bos_the_template_spells_out_is_not_sent_twice():
    """The tokenizer names no BOS and the template writes config.json's own by its text: it is the token the template
    begins with, which is the engine's BOS already. Before T264 the page sent it twice."""
    tokenizer, ids = byte_level(["<s>", "<|im_end|>"])
    options, engine = converted(tokenizer, config={"bos_token_id": ids["<s>"], "eos_token_id": ids["<|im_end|>"]},
                                chat_template="<s>[INST]{{ messages[0].content }}[/INST]")
    assert options["bos"] == ids["<s>"] and options["template"] == "[INST]{prompt}[/INST]"
    assert options["stop_tokens"] == [ids["<s>"], ids["<s>"], ids["<|im_end|>"]]
    assert sent(options, engine, "x").count(ids["<s>"]) == 1


def test_a_template_that_begins_with_text_keeps_the_bos_in_front():
    """RakutenAI's, an Alpaca's: no token to begin with but the first one of a text, which is no fixed one. As before."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_end|>"])
    options, engine = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]},
                                chat_template="USER: {{ messages[0].content }} ASSISTANT:")
    assert options["bos"] == ids["<|endoftext|>"] and options["template"] == "USER: {prompt} ASSISTANT:"
    assert options["stop_tokens"] == [ids["<|endoftext|>"], ids["<|im_end|>"]]
    assert "specials" not in options


def test_without_a_template_nothing_changes():
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    options, _ = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]},
                           bos_token="<|endoftext|>")
    assert options["bos"] == ids["<|endoftext|>"] and "template" not in options
    assert options["stop_tokens"] == [ids["<|endoftext|>"], ids["<|im_end|>"]]


def test_the_token_a_template_begins_with_may_be_an_added_one_and_is_the_longest():
    """The added tokens tokenizer.json does not call special are one token wherever they are (T143), at the head too.
    Of two that begin the template, the engine reads the longer."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_end|>"], added=["<turn>", "<turn>>"])
    options, engine = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"]},
                                chat_template="<turn>>user {{ messages[0].content }}<|im_end|>")
    assert options["bos"] == ids["<turn>>"] and options["template"] == "user {prompt}<|im_end|>"
    assert sent(options, engine, "a") == engine.encode("<turn>>user a<|im_end|>", tuple(options["specials"]))


def test_a_token_at_the_head_only_is_still_one_token_where_it_is_typed():
    """<|user|> is written once, at the head: it is the BOS now and no more in the format, but it stays among the
    specials (as it was before T264), so a visitor who types it writes the token and not its letters."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|user|>", "<|bot|>"])
    options, engine = converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"]},
                                chat_template="<|user|>{{ messages[0].content }}<|bot|>")
    assert options["bos"] == ids["<|user|>"] and options["template"] == "{prompt}<|bot|>"
    assert options["specials"] == ["<|user|>", "<|bot|>"]


def test_a_bos_that_nothing_names_is_no_stop_token_once_the_template_says_where_a_text_begins():
    """NeoHorse-1's config.json has no bos_token_id and its tokenizer names no BOS: the engine's was 1, a guess, which
    is '"' in a byte-level vocabulary, and the answer stopped at it. With the template's own first token for a BOS
    the guess is no token of the model's any more. A BOS that config.json or the tokenizer names still stops the answer."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    settings, published, header, base = model()
    config = {key: value for key, value in published.items() if key != "bos_token_id"}
    made = lambda **more: llama2_convert.Conversion(header, base, json.dumps({**config, "eos_token_id": ids["<|im_end|>"]}),
                                                    json.dumps(tokenizer).encode(), "tokenizer.json", dtype="float32",
                                                    max_seq_len=settings["seq_len"], tokenizer_config=json.dumps(more)).options
    options = made(chat_template=CHATML)
    assert options["bos"] == ids["<|im_start|>"] and options["stop_tokens"] == [ids["<|im_start|>"], ids["<|im_end|>"]]
    assert made(chat_template=CHATML, bos_token="<|endoftext|>")["stop_tokens"] == [ids["<|im_start|>"], ids["<|endoftext|>"], ids["<|im_end|>"]]
    # without a template the guess is the BOS, as it was
    assert made()["bos"] == 1 and made()["stop_tokens"] == [1, ids["<|im_end|>"]]


def test_a_sentencepiece_template_keeps_its_bos_where_the_rest_would_be_read_otherwise():
    """T264 takes the token a template begins with off the format only where the rest is read the same without it. A
    sentencepiece tokenizer prefixes the first text and not the text after a special token: sarashina's <|user|>{prompt}
    would have its prompt read as the first text. There the BOS stays in front and the format whole, as before. Where
    the text after a token is prefixed too (a legacy Llama tokenizer: EuroLLM's) and it is the template's own, or where
    a token follows the first, the rule holds as for a byte-level vocabulary."""
    data, ids = sentencepiece()
    config = {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]}
    marks = ("<|user|>", "</s>", "<|assistant|>")
    sarashina = "{% for m in messages %}<|user|>{{ m.content }}</s>{% endfor %}<|assistant|>"
    for described in ({"legacy": False}, {}):  # (legacy: what was typed may begin with a space, and would be read otherwise)
        options, engine = converted(data, "tokenizer.model", config, chat_template=sarashina, tokenizer_class="LlamaTokenizer", **described)
        assert options["bos"] == ids["<s>"] and options["template"] == "<|user|>{prompt}</s><|assistant|>"
        assert options["stop_tokens"] == [ids["<s>"], ids["</s>"]]
        assert options["specials"] == ["<|assistant|>", "<|user|>", "</s>"]
    # legacy, and the template's own text follows the token: prefixed as the first text is
    eurollm = "<|user|>hello\n{{ messages[0].content }}</s><|assistant|>"
    options, engine = converted(data, "tokenizer.model", config, chat_template=eurollm, tokenizer_class="LlamaTokenizer")
    assert options["bos"] == ids["<|user|>"] and options["template"] == "hello\n{prompt}</s><|assistant|>"
    assert options["stop_tokens"] == [ids["<|user|>"], ids["<s>"], ids["</s>"]]
    assert sent(options, engine, "world") == engine.encode("<|user|>hello\nworld</s><|assistant|>", marks)
    assert sent(options, engine, "world")[:2] == [ids["<|user|>"], ids["▁hello"]]
    # the same template, not legacy: "hello" after the token has no prefix, and would have one as the first text
    options, engine = converted(data, "tokenizer.model", config, chat_template=eurollm, tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["bos"] == ids["<s>"] and options["template"] == "<|user|>hello\n{prompt}</s><|assistant|>"
    # not legacy, and a token follows the first: no text is read otherwise
    two = "<|user|><|assistant|>{{ messages[0].content }}</s>"
    options, engine = converted(data, "tokenizer.model", config, chat_template=two, tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["bos"] == ids["<|user|>"] and options["template"] == "<|assistant|>{prompt}</s>"
    assert sent(options, engine, "hello") == engine.encode("<|user|><|assistant|>hello</s>", marks)


def test_a_unigram_tokenizer_json_that_prefixes_every_text_begins_with_the_templates_token():
    """llm-jp-4's: every stretch of text has the "▁" (its normalizer), so the text after the template's first token is
    read as the first text is, what was typed too."""
    vocab = [["<unk>", 0.0], ["<s>", 0.0], ["<|start|>", 0.0], ["<|end|>", 0.0], ["▁", -3.0], ["▁hello", -1.0], ["hello", -2.0]]
    vocab += [[f"▁w{i}", -9.0] for i in range(320 - len(vocab))]
    tokenizer = {"added_tokens": [{"id": id, "content": text, "special": True} for id, text in ((1, "<s>"), (2, "<|start|>"), (3, "<|end|>"))],
                 "normalizer": {"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"Regex": "(?<!\\n)^"}, "content": "▁"},
                                                                    {"type": "Replace", "pattern": {"Regex": " "}, "content": "▁"}]},
                 "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab}}
    template = "<|start|>{{ messages[0].content }}<|end|><|start|>"
    options, engine = converted(tokenizer, config={"bos_token_id": 1, "eos_token_id": 3}, chat_template=template, bos_token="<s>")
    assert options["prefixed"] == "every" and options["bos"] == 2 and options["template"] == "{prompt}<|end|><|start|>"
    assert options["stop_tokens"] == [2, 1, 3]
    for typed in ("hello", " hello"):
        assert sent(options, engine, typed) == engine.encode(f"<|start|>{typed}<|end|><|start|>", ("<|start|>", "<|end|>"))
    # the same without that normalizer: the BOS stays in front
    del tokenizer["normalizer"]
    options, _ = converted(tokenizer, config={"bos_token_id": 1, "eos_token_id": 3}, chat_template=template, bos_token="<s>")
    assert "prefixed" not in options and options["bos"] == 1 and options["template"] == "<|start|>{prompt}<|end|><|start|>"


def test_the_answer_stops_at_the_eos_the_tokenizer_names_too():
    """T369: config.json's eos_token_id and the tokenizer's eos_token may be two tokens (a Qwen3.5's: <|endoftext|> and
    <|im_end|>, which ends a turn of its template), and transformers' generate() stops at either. The tokenizer's is a
    stop token where it names one that is a piece of its own and not one already."""
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    config = {"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|endoftext|>"]}
    stops = lambda **described: converted(tokenizer, config=config, **described)[0]["stop_tokens"]
    as_it_was = [ids["<|endoftext|>"], ids["<|endoftext|>"]]
    # named and another, as text or as {"content": ...}; with a template and without
    assert stops(eos_token="<|im_end|>") == [*as_it_was, ids["<|im_end|>"]]
    assert stops(eos_token={"content": "<|im_end|>"}) == [*as_it_was, ids["<|im_end|>"]]
    assert stops(eos_token="<|im_end|>", chat_template=CHATML) == [ids["<|im_start|>"], *as_it_was, ids["<|im_end|>"]]
    # the same as config.json's, or one the template's first token already is: said once
    assert stops(eos_token="<|endoftext|>") == as_it_was
    assert stops(eos_token="<|im_start|>", chat_template=CHATML) == [ids["<|im_start|>"], *as_it_was]
    # named, but no piece of the vocabulary (its letters are): no token to stop at
    assert stops(eos_token="<|end|>") == as_it_was
    assert stops(eos_token="user<|im_end|>") == as_it_was
    # none named: nothing, and not the empty pieces a vocabulary is padded with
    for nothing in ({}, {"eos_token": None}, {"eos_token": ""}, {"eos_token": {"content": None}}, {"eos_token": 7}):
        assert stops(bos_token="<|endoftext|>", **nothing) == as_it_was, nothing
    short, at = byte_level(["<|endoftext|>", "<|im_end|>"], size=316)  # (four pieces of padding, each empty: ids 316 to 319)
    for nothing in ({}, {"eos_token": ""}):
        assert converted(short, config={"bos_token_id": at["<|endoftext|>"], "eos_token_id": at["<|endoftext|>"]},
                         **nothing)[0]["stop_tokens"] == [at["<|endoftext|>"]] * 2
    # a sentencepiece model's control piece
    data, pieces = sentencepiece()
    options, _ = converted(data, "tokenizer.model", {"bos_token_id": pieces["<s>"], "eos_token_id": pieces["</s>"]}, eos_token="<|assistant|>")
    assert options["stop_tokens"] == [pieces["<s>"], pieces["</s>"], pieces["<|assistant|>"]]
