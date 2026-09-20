# Byte-level BPE (GPT-2, SmolLM2, Qwen): the pre-tokenizers and the whole path tokenizer.json -> converter ->
# tokenizer.bin -> engine, against Hugging Face's own tokenizers. The vocabularies are trained here in a second,
# so nothing is downloaded and no file is checked in.
import json

import pytest

from llama2_convert import tokenizer_bin, tokenizer_json_options, tokenizer_json_pieces
from llama2_numpy import Tokenizer, pretokenize

tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers to check against the real one")

GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")

CORPUS = (
    "The quick brown fox jumps over the lazy dog. 日本語の文章も混ぜる。"
    "Don't stop; it's theirs, they'll go. I'D LIKE 'IT'.\n"
    "Pyodide は WebAssembly 版の Python で、ブラウザの中で NumPy が動く。"
    "def forward(x, w):\n\treturn w @ x  # matmul\n\n\n"
    "価格は1,234,567円（税込）です。2026-09-21T00:00:00Z\r\n"
    "絵文字 \U0001f600\U0001f389 と外字 \U00029E3D、全角ＡＢＣ１２３、半角ｶﾅ。"
    "https://example.com/a/b?c=1&d=2#frag  'single' \"double\" `tick`\n"
    "   spaces\tand\ttabs\n\n\nnewlines   \nTHE END. the end. The End?!  "
)
# every kind of boundary the patterns care about
TEXTS = [CORPUS, " ", "  ", "\n", "\r\n", " \n ", "0123", " 42 ", "a", " a", "  a", "\ta", "(abc", "、あ",
         "a 1b", " 1,234", "1a2", "v1.2.3", "第1章 2節", "it's a dog's life", "IT'S", "end.  ", "x\n\n\ny"]


def trained(pattern, digits):
    """A small byte-level BPE, and the tokenizer.json it saves."""
    from tokenizers import Tokenizer as Real, decoders, models, pre_tokenizers, trainers
    real = Real(models.BPE())
    steps = ([pre_tokenizers.Digits(individual_digits=True)] if digits else [])
    steps += ([pre_tokenizers.Split(tokenizers.Regex(pattern), behavior="isolated")] if pattern else [])
    steps += [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=not pattern)]
    real.pre_tokenizer = pre_tokenizers.Sequence(steps)
    real.decoder = decoders.ByteLevel()
    real.train_from_iterator([CORPUS] * 8, trainers.BpeTrainer(
        vocab_size=900, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return real, json.loads(real.to_str())


@pytest.mark.parametrize("name, pattern, digits", [
    ("gpt2", None, False), ("gpt2-digits", None, True), ("qwen", QWEN_PATTERN, False)])
@pytest.mark.parametrize("text", TEXTS)
def test_matches_the_real_tokenizer(name, pattern, digits, text):
    real, spec = trained(pattern, digits)
    options = tokenizer_json_options(spec)
    assert options["tokenizer_kind"] == "bytebpe" and options["pretokenizer"] == name
    vocab_size = real.get_vocab_size()
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size,
                     kind="bytebpe", pretokenizer=options["pretokenizer"])
    assert mine.encode(text) == real.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("name, pattern, digits", [
    ("gpt2", None, False), ("gpt2-digits", None, True), ("qwen", QWEN_PATTERN, False)])
def test_decodes_every_piece(name, pattern, digits):
    real, spec = trained(pattern, digits)
    vocab_size = real.get_vocab_size()
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size,
                     kind="bytebpe", pretokenizer=name)
    for id in range(vocab_size):
        # skip_special_tokens=False: the engine hands the page every piece, and stop tokens end the run instead
        assert mine.decode(0, id).decode("utf-8", "replace") == real.decode([id], skip_special_tokens=False)


@pytest.mark.parametrize("text", TEXTS)
def test_pretokenizers_follow_the_patterns(text):
    """The engine spells the patterns out by hand, because re has no \\p{L}."""
    regex = pytest.importorskip("regex", reason="pip install regex to check the patterns themselves")
    assert pretokenize(text, "gpt2") == regex.findall(GPT2_PATTERN, text)
    assert pretokenize(text, "qwen") == regex.findall(QWEN_PATTERN, text)
    digits = [part for chunk in regex.findall(r"\d|\D+", text) for part in regex.findall(GPT2_PATTERN, chunk)]
    assert pretokenize(text, "gpt2-digits") == digits


def test_refuses_what_the_engine_cannot_split():
    with pytest.raises(ValueError, match="does not know"):
        tokenizer_json_options({"model": {"type": "BPE"}, "pre_tokenizer": {"type": "Whitespace"}})
    with pytest.raises(ValueError, match="adds a space"):
        tokenizer_json_options({"model": {"type": "BPE"},
                                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True}})
