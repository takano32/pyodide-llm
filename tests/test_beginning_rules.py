"""T640.1: the rules of how a text begins, where an answer stops and which tokens a format writes as one
(convert/beginning.py), and the options the engine gets of them (options_of()), each from values alone: a
config.json, a Vocabulary and a template, with no weights and no header. The rules' history and the real models they
hold to are in test_beginning.py, test_answer_end.py and test_tokenizer_says.py, which take them through the whole
conversion."""
import pytest

from convert.beginning import Beginning, beginning_and_stops, options_of, turn_head
from convert.tokenizer import tokenizer_bin
from convert.vocabulary import VOCABULARIES, Vocabulary, vocabulary_kind
from engine.layout import FORM

CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{{ '<|im_start|>assistant\\n' }}"
BYTE_LEVEL = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": False, "pretokenizer": "gpt2", "ignore_merges": False}
SIZE = 16


def vocabulary(*tokens, options=BYTE_LEVEL, described=None, specials=(), added=(), bos=None, dropped=None):
    """A Vocabulary of SIZE pieces whose first ones are tokens (ids 0, 1, ...), the rest made-up words."""
    pieces = [(text, 0.0, False) for text in tokens] + [(f"w{i}", -1.0, True) for i in range(len(tokens), SIZE)]
    return Vocabulary(tokenizer_bin(pieces, SIZE, spaces=False), options, described or {}, list(specials), list(added),
                      bos, dropped)


MARKS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
CONFIG = {"vocab_size": SIZE, "bos_token_id": 0, "eos_token_id": 2}


def test_a_template_that_begins_with_a_token_of_its_own():
    """A Qwen's: <|im_start|> is the BOS, the format follows it, and the answer stops at the BOS and the EOSes."""
    said = vocabulary(*MARKS, described={"chat_template": CHATML, "eos_token": "<|im_end|>"}, specials=MARKS, bos=0)
    assert beginning_and_stops(CONFIG, said) == Beginning(1, [0, 2], "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                                                          ["<|im_start|>", "<|im_end|>"])
    # the template of chat_template.jinja is read before the tokenizer_config's
    plain = vocabulary(*MARKS, specials=MARKS, bos=0)
    assert beginning_and_stops(CONFIG, plain, CHATML) == beginning_and_stops(CONFIG, said)
    # the token after an answer stops it (T524), unless it is the one every text begins with
    restart = "{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}<|im_start|>"
    assert beginning_and_stops(CONFIG, plain, restart) == Beginning(1, [0, 2], "{prompt}<|im_start|>", ["<|im_start|>"])
    assert beginning_and_stops(CONFIG, plain, restart.replace("{% endfor %}", "<|im_end|>{% endfor %}")).stop_tokens == [0, 2]
    assert beginning_and_stops(CONFIG, plain, "{% for m in messages %}{{ m.content }}<|endoftext|>{% endfor %}").stop_tokens == [0, 2]


def test_no_template_and_a_bos_nothing_names():
    """No template: the BOS stays, no format. A config.json with no bos_token_id: the 1 is a guess and no stop."""
    plain = vocabulary(*MARKS, specials=MARKS)
    assert beginning_and_stops(CONFIG, plain) == Beginning(0, [0, 2], None, [])
    unnamed = {"vocab_size": SIZE, "eos_token_id": 2}
    assert beginning_and_stops(unnamed, plain) == Beginning(1, [2], None, [])
    # and a list of EOSes, said once each
    assert beginning_and_stops({**CONFIG, "eos_token_id": [2, 0]}, plain).stop_tokens == [0, 2, 0]


def test_the_rules_refuse():
    plain = vocabulary(*MARKS, specials=MARKS)
    with pytest.raises(ValueError, match="^no head$"):
        beginning_and_stops(CONFIG, plain, "USER: {{ messages[0].content }}", headless="no head")
    dropped = vocabulary(*MARKS, specials=MARKS, dropped={SIZE: "<|tool|>"})
    assert beginning_and_stops(CONFIG, dropped, CHATML).bos == 1
    with pytest.raises(ValueError, match=r"^This tokenizer.json has <\|tool\|> past the model's vocabulary of 16, and "
                                         r"the chat format writes it\.$"):
        beginning_and_stops(CONFIG, dropped, CHATML.replace("<|im_end|>", "<|tool|>"))


def test_the_head_of_a_turn():
    ids = {b"<s>": 0, b"<|user|>": 3}
    sentencepiece = {"tokenizer_kind": "unigram"}
    assert turn_head("<s>{prompt}", "<s>", ["<s>"], sentencepiece, ids) == ("{prompt}", None, None)
    assert turn_head("<|user|>{prompt}", "<s>", ["<|user|>"], BYTE_LEVEL, ids) == ("{prompt}", 3, "<|user|>")
    # a sentencepiece tokenizer would read the prompt as the first text: the BOS stays in front
    assert turn_head("<|user|>{prompt}", "<s>", ["<|user|>"], sentencepiece, ids) == ("<|user|>{prompt}", None, None)
    for prefixed in ("every", "never"):
        assert turn_head("<|user|>{prompt}", "<s>", ["<|user|>"], {**sentencepiece, "prefixed": prefixed}, ids)[1] == 3
    assert turn_head("hello {prompt}", "<s>", ["<|user|>"], BYTE_LEVEL, ids) == ("hello {prompt}", None, None)
    assert turn_head(None, "<s>", [], BYTE_LEVEL, ids) == (None, None, None)


def test_the_options():
    """The vocabulary's options first, then the dtype, the RoPE's base, the BOS and the stops, the form where it is not
    the default (bias and arch always), the format, its tokens and what config.json says beyond."""
    form = {**FORM, "bias": False, "arch": "llama"}
    begun = Beginning(1, [0, 2], "x{prompt}", ["<|im_start|>"])
    options = options_of(form, "int8", {"rms_norm_eps": 1e-6, "rope_theta": 5e5, "model_type": "llama"},
                         vocabulary(options={"tokenizer_kind": "bpe"}), begun)
    assert list(options.items()) == [("tokenizer_kind", "bpe"), ("dtype", "int8"), ("rope_theta", 500000.0), ("bos", 1),
                                     ("stop_tokens", [0, 2]), ("bias", False), ("arch", "llama"), ("template", "x{prompt}"),
                                     ("specials", ["<|im_start|>"]), ("rms_norm_eps", 1e-6)]
    options = options_of(form, "float32", {"model_type": "llama"}, vocabulary(options={}), Beginning(1, [2], None, []))
    assert options == {"dtype": "float32", "rope_theta": 10000.0, "bos": 1, "stop_tokens": [2], "bias": False, "arch": "llama"}


def test_the_kinds_of_tokenizer_files():
    assert vocabulary_kind("tokenizer.json") == "tokenizer.json" and vocabulary_kind("Tokenizer.JSON") == "tokenizer.json"
    assert vocabulary_kind("tokenizer.model") == "sentencepiece" and vocabulary_kind("spiece.model") == "sentencepiece"
    assert set(VOCABULARIES) == {"tokenizer.json", "sentencepiece"}
