"""Made-up models and tokenizers for the tests of what the converter says of how a model's text begins and is read
(T369: test_beginning.py, test_lowercase.py, test_prefixed.py): a tiny Llama, a byte-level BPE tokenizer.json, a
sentencepiece model, and the conversion of the one with the other as the worker makes it."""
import json
import struct

import llama2_convert
from llama2_numpy import Tokenizer
from conftest import synthetic_weights
from make_hf_fixture import field
from test_convert import hugging_face, safetensors_file

NORMAL, UNKNOWN, CONTROL = 1, 2, 3


def model():
    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    return settings, published, file[8:8 + size].decode(), 8 + size


def byte_level(specials, added=(), size=320):
    """A byte-level BPE tokenizer.json of size pieces: the 256 bytes, a few words, then the special and the added
    tokens (as Qwen's has them at the end), each by its text."""
    from llama2_numpy import BYTE_CHARS
    vocab = {BYTE_CHARS[byte]: byte for byte in range(256)}
    merges = []
    for word in ("us", "er", "user", "as", "ass", "is", "ist", "ant", "assist", "assistant"):
        for cut in range(1, len(word)):
            if word[:cut] in vocab and word[cut:] in vocab and word not in vocab:
                vocab[word] = len(vocab)
                merges.append(f"{word[:cut]} {word[cut:]}")
    tokens = [*specials, *added]
    first = size - len(tokens)
    for filler in range(len(vocab), first):
        vocab[f"filler{filler}"] = filler
    return {"added_tokens": [{"id": first + at, "content": token, "special": token in specials} for at, token in enumerate(tokens)],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
            "model": {"type": "BPE", "vocab": vocab, "merges": merges}}, {token: first + at for at, token in enumerate(tokens)}


def sentencepiece(controls=("<s>", "</s>", "<|user|>", "<|assistant|>"), words=("hello", "world", "a", "b", "h", "e", "l", "o")):
    """A sentencepiece model (unigram, identity normalizer) of the tiny model's size: <unk>, the control pieces, then
    the words with and without the space in front."""
    size = synthetic_weights()[0]["vocab_size"]
    pieces = [("<unk>", UNKNOWN), *((text, CONTROL) for text in controls), ("▁", NORMAL)]
    pieces += [(text, NORMAL) for word in words for text in (f"▁{word}", word)]
    pieces += [(f"▁w{i}", NORMAL) for i in range(size - len(pieces))]
    data = b"".join(field(1, field(1, text.encode()) + field(2, -float(i + 1)) + field(3, kind)) for i, (text, kind) in enumerate(pieces))
    return data + field(2, field(3, 1)) + field(3, field(1, b"identity") + field(4, 0)), {text: i for i, (text, _) in enumerate(pieces)}


def converted(tokenizer, name="tokenizer.json", config=None, **described):
    settings, published, header, base = model()
    data = json.dumps(tokenizer).encode() if isinstance(tokenizer, dict) else tokenizer
    conversion = llama2_convert.Conversion(header, base, json.dumps({**published, **(config or {})}), data, name,
                                           dtype="float32", max_seq_len=settings["seq_len"],
                                           tokenizer_config=json.dumps(described) if described else None)
    options = conversion.options
    engine = Tokenizer(conversion.tokenizer, settings["vocab_size"], kind=options["tokenizer_kind"],
                       **{key: options[key] for key in ("nfkc", "nfc", "pretokenizer", "ignore_merges", "collapse", "unknown",
                                                        "lowercase", "prefixed") if key in options})
    return options, engine


def sent(options, engine, prompt):
    """The ids the page sends: the engine's BOS, then the format filled with what was typed"""
    return [options["bos"]] + engine.encode(options["template"].replace("{prompt}", prompt), tuple(options.get("specials", ())))
