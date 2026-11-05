"""T640.1: what the converter gives of a few made-up models, written as values: the whole of the options (their keys in
their order: the worker keeps them as JSON) and the tokenizer.bin by its hash. The rules of the beginning, the stops
and the format each have their tests (test_beginning.py, test_answer_end.py, test_tokenizer_says.py), which look at
one key or two; these hold the rest, so that a change to how Conversion is put together moves nothing it gives.
The values are those of the converter before T640.1 (commit ee334f33)."""
import hashlib
import json

import pytest

import llama2_convert
from made_up_tokenizers import byte_level, model, sentencepiece
from test_answer_end import GEMMA4
from test_beginning import CHATML
from test_gguf import fed
from test_gguf_alone import chat_gguf

SARASHINA = "{% for m in messages %}<|user|>{{ m.content }}</s>{% endfor %}<|assistant|>"
EUROLLM = "<|user|>hello\n{{ messages[0].content }}</s><|assistant|>"


def converted(tokenizer, name="tokenizer.json", config=None, **described):
    settings, published, header, base = model()
    data = json.dumps(tokenizer).encode() if isinstance(tokenizer, dict) else tokenizer
    return llama2_convert.Conversion(header, base, json.dumps({**published, **(config or {})}), data, name, dtype="float32",
                                     max_seq_len=settings["seq_len"],
                                     tokenizer_config=json.dumps(described) if described else None)


def chatml():
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]},
                     chat_template=CHATML, bos_token="<|endoftext|>", eos_token="<|im_end|>")


def gemma4():
    tokenizer, ids = byte_level(["<bos>", "<eos>", "<|turn>", "<turn|>"])
    return converted(tokenizer, config={"bos_token_id": ids["<bos>"], "eos_token_id": ids["<eos>"]}, chat_template=GEMMA4,
                     bos_token="<bos>", eos_token="<eos>")


def added():
    tokenizer, ids = byte_level(["<|endoftext|>", "<|im_end|>"], added=["<turn>", "<turn>>"])
    return converted(tokenizer, config={"bos_token_id": ids["<|endoftext|>"]},
                     chat_template="<turn>>user {{ messages[0].content }}<|im_end|>")


def spm(template=None, dummy=True, **described):
    data, ids = sentencepiece(dummy=dummy)
    return converted(data, "tokenizer.model", {"bos_token_id": ids["<s>"], "eos_token_id": ids["</s>"]},
                     **({"chat_template": template} if template else {}), **described)


CASES = {
    "a byte-level ChatML": chatml,
    "a Gemma 4's turns": gemma4,
    "an added token at the head": added,
    "sarashina2.2's": lambda: spm(SARASHINA, tokenizer_class="LlamaTokenizer", legacy=False),
    "EuroLLM's": lambda: spm(EUROLLM, tokenizer_class="LlamaTokenizer"),
    "no dummy prefix": lambda: spm(dummy=False, eos_token="<|assistant|>"),
    "a GGUF that names no BOS": lambda: fed(chat_gguf(), "int8"),
    "a GGUF that names one": lambda: fed(chat_gguf(bos=5), "int8"),
}

BYTE_LEVEL = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": False, "pretokenizer": "gpt2", "ignore_merges": False}
WANT = {
    "a byte-level ChatML": ("3c1675920994a9fa", {
        **BYTE_LEVEL, "dtype": "float32", "rope_theta": 10000.0, "bos": 318, "stop_tokens": [317, 319], "bias": False,
        "arch": "llama", "template": "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "specials": ["<|im_start|>", "<|im_end|>"]}),
    "a Gemma 4's turns": ("910dbd4ab00d3697", {
        **BYTE_LEVEL, "dtype": "float32", "rope_theta": 10000.0, "bos": 316, "stop_tokens": [316, 317, 319], "bias": False,
        "arch": "llama", "template": "<|turn>user\n{prompt:trim}<turn|>\n<|turn>model\n", "specials": ["<turn|>", "<|turn>"]}),
    "an added token at the head": ("f0c40a4c66847da7", {
        **BYTE_LEVEL, "dtype": "float32", "rope_theta": 10000.0, "bos": 319, "stop_tokens": [316, 2], "bias": False,
        "arch": "llama", "template": "user {prompt}<|im_end|>", "specials": ["<|im_end|>", "<turn>>", "<turn>"]}),
    "sarashina2.2's": ("128e5aaaad76ec5e", {
        "tokenizer_kind": "unigram", "unknown": 0, "dtype": "float32", "rope_theta": 10000.0, "bos": 1, "stop_tokens": [1, 2],
        "bias": False, "arch": "llama", "template": "<|user|>{prompt}</s><|assistant|>",
        "specials": ["<|assistant|>", "<|user|>", "</s>"]}),
    "EuroLLM's": ("128e5aaaad76ec5e", {
        "tokenizer_kind": "unigram", "unknown": 0, "prefixed": "wanting", "dtype": "float32", "rope_theta": 10000.0, "bos": 3,
        "stop_tokens": [1, 2], "bias": False, "arch": "llama", "template": "hello\n{prompt}</s><|assistant|>",
        "specials": ["<|assistant|>", "<|user|>", "</s>"]}),
    "no dummy prefix": ("128e5aaaad76ec5e", {
        "tokenizer_kind": "unigram", "unknown": 0, "prefixed": "never", "dtype": "float32", "rope_theta": 10000.0, "bos": 1,
        "stop_tokens": [1, 2, 4], "bias": False, "arch": "llama"}),
    "a GGUF that names no BOS": ("908c57a4c20182bb", {
        **BYTE_LEVEL, "dtype": "int8", "rope_theta": 10000.0, "bos": 10, "stop_tokens": [11], "bias": False, "arch": "llama",
        "template": "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", "specials": ["<|im_start|>", "<|im_end|>"]}),
    "a GGUF that names one": ("908c57a4c20182bb", {
        **BYTE_LEVEL, "dtype": "int8", "rope_theta": 10000.0, "bos": 10, "stop_tokens": [5, 11], "bias": False, "arch": "llama",
        "template": "user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", "specials": ["<|im_start|>", "<|im_end|>"]}),
}


@pytest.mark.parametrize("case", CASES)
def test_what_the_converter_gives(case):
    conversion = CASES[case]()
    tokenizer, options = WANT[case]
    assert list(conversion.options.items()) == list(options.items())
    assert hashlib.sha256(bytes(conversion.tokenizer)).hexdigest()[:16] == tokenizer
