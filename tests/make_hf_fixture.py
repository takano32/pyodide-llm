# make_hf_fixture.py
# The files Hugging Face would publish for stories260K, made from the llama2.c checkpoint and tokenizer of this
# directory: model.safetensors, config.json, tokenizer.model (sentencepiece) and settings.json. tests/e2e.mjs opens
# them through the folder button, and the page has to convert them back and write the same story.
#
#   python3 tests/make_hf_fixture.py <directory> [F32|BF16]
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from tree import python_folder
sys.path.insert(0, python_folder(ROOT))
from made_up import hugging_face, safetensors_file, settings_of, tensors_of  # noqa: E402


def varint(value):
    out = b""
    while True:
        out += bytes([value & 0x7F | (0x80 if value > 0x7F else 0)])
        value >>= 7
        if not value:
            return out


def field(number, payload):
    """One protobuf field: bytes are length-delimited, an int is a varint, a float is 32 bits."""
    if isinstance(payload, bytes):
        return varint(number << 3 | 2) + varint(len(payload)) + payload
    if isinstance(payload, float):
        return varint(number << 3 | 5) + struct.pack("<f", payload)
    return varint(number << 3) + varint(payload)


def sentencepiece_model(tokenizer, vocab_size):
    NORMAL, UNKNOWN, CONTROL, BYTE, BPE = 1, 2, 3, 6, 2
    model, offset = b"", 4
    for id in range(vocab_size):
        score, length = struct.unpack_from("<fi", tokenizer, offset)
        text = tokenizer[offset + 8:offset + 8 + length].decode("utf-8")
        offset += 8 + length
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        kind = UNKNOWN if id == 0 else CONTROL if id in (1, 2) else BYTE if is_byte else NORMAL
        model += field(1, field(1, text.replace(" ", "▁").encode("utf-8")) + field(2, float(score)) + field(3, kind))
    return model + field(2, field(3, BPE)) + field(3, field(1, b"identity"))


if __name__ == "__main__":
    out, stored = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "F32"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = (ROOT / "stories260K.bin").read_bytes()
    # the checkpoint's tensors by the engine's names (read by its rows), and what Hugging Face would publish for them
    settings = settings_of(struct.unpack_from("<7i", checkpoint, 0))
    tensors, published = hugging_face(settings, tensors_of(checkpoint), settings["shared"])
    (out / "model.safetensors").write_bytes(safetensors_file(tensors, stored))
    (out / "config.json").write_text(json.dumps({**published, "hidden_act": "silu", "bos_token_id": 1, "eos_token_id": 2}, indent=2))
    (out / "tokenizer.model").write_bytes(sentencepiece_model((ROOT / "tok512.bin").read_bytes(), settings["vocab_size"]))
    # (T.401: a chat template, as a published model has, which writes the prompt as it is: the converter reads it by
    # jinja2 in the browser (the worker asks Pyodide for the package), and the text is the one without a template)
    (out / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"}))
    (out / "settings.json").write_text(json.dumps(dict(
        name="stories260K from Hugging Face files", conversion=dict(dtype="float32"), options=dict(stop_tokens=[1]),
        generation=dict(steps=256, temperature=0.0), prompt="Once upon a time")))
    print(f"{out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
