"""CONVERTER (public/kept.js) goes up when what the converter makes of a model changes (T116, T369 review): a kept
conversion of the old version is used otherwise, with the old options.

tests/fixtures/converter-output.json holds the version and a digest of what the converter makes of three made-up models
(a byte-level BPE with a ChatML template, a legacy Llama sentencepiece model with a template, a lower-casing one).
When the digest changes and the version is the one in the file, this fails: raise CONVERTER, or, where the change is
not one a kept conversion would show (it is for the three models only), say so by writing the digest again
(`python3 tests/test_converter_version.py --write`, with the version it is for). This is a prompt, not a proof: it sees
only what the three models exercise (the net's `python` and tests/conversions_compare.py see the rest)."""
import hashlib
import json
import re
import sys
from pathlib import Path

import llama2_convert
from made_up_tokenizers import byte_level, model, sentencepiece
from tree import runtime_folder

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "converter-output.json"
CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{{ '<|im_start|>assistant\\n' }}"


def version():
    return int(re.search(r"export const CONVERTER = (\d+);", Path(runtime_folder(Path(__file__).resolve().parent.parent, "kept.js")).read_text()).group(1))


def digest():
    settings, published, header, base = model()
    byte, ids = byte_level(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    pieces, marks = sentencepiece()
    tokenizer_config = {"chat_template": CHATML, "tokenizer_class": "Qwen2Tokenizer", "bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}
    llama_config = {"chat_template": "<|user|>hello\n{{ messages[0].content }}</s><|assistant|>", "tokenizer_class": "LlamaTokenizer",
                    "bos_token": "<s>", "eos_token": "</s>"}
    cases = [(json.dumps(byte).encode(), "tokenizer.json", tokenizer_config, {"bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"]}),
             (pieces, "tokenizer.model", llama_config, {"bos_token_id": marks["<s>"], "eos_token_id": marks["</s>"]}),
             (pieces, "tokenizer.model", {**llama_config, "do_lower_case": True, "legacy": False}, {"bos_token_id": marks["<s>"]})]
    sha = hashlib.sha256()
    for data, name, described, config in cases:
        conversion = llama2_convert.Conversion(header, base, json.dumps({**published, **config}), data, name, dtype="int8",
                                               max_seq_len=settings["seq_len"], tokenizer_config=json.dumps(described))
        sha.update(json.dumps(conversion.options, sort_keys=True).encode())
        sha.update(bytes(conversion.tokenizer))
    return sha.hexdigest()[:16]


def test_a_change_of_what_the_converter_makes_comes_with_a_new_converter_version():
    recorded, now = json.loads(FIXTURE.read_text()), version()
    changed = digest() != recorded["digest"]
    assert not (changed and now == recorded["converter"]), (
        f"what the converter makes of the made-up models changed while CONVERTER is still {now}: raise CONVERTER in "
        "public/kept.js, or, if no kept conversion would differ, write the digest again with "
        "python3 tests/test_converter_version.py --write")
    assert now == recorded["converter"], (
        f"CONVERTER is {now} and tests/fixtures/converter-output.json is for {recorded['converter']}: write it again "
        "with --write (and say in the commit what the new version is for)")


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    FIXTURE.write_text(json.dumps({"converter": version(), "digest": digest()}, indent=1) + "\n")
