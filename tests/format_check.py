# format_check.py
# The format of every Hugging Face model of the list against the real one (T124, T127, T138): the IDs the page
# sends ([bos] + llama2_numpy.Tokenizer on the converter's tokenizer.bin, with the options the worker merges and
# filled()'s rules) against transformers' apply_chat_template on the files of the pinned revision. Only config.json,
# the tokenizer files and the safetensors headers (by Range) are fetched; no weights.
#
#   python3 tests/format_check.py <directory for the downloads> [model id ...]
#
# Needs the reference tools, which the page never uses: a venv with tests/requirements-reference.txt (docs/dev-setup.md).
# The first BOS may differ (the page always starts with it, T131). What a card always passes besides the prompt
# (Swallow-MS's system message) is in SYSTEM. Models with only a sentencepiece tokenizer.model are compared through
# transformers' slow tokenizer, which is not the real one for every model (sarashina's): read those diffs with care.
import inspect
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
import llama2_convert  # noqa: E402
import llama2_numpy  # noqa: E402

PROMPTS = ["これからの流行りを3つ挙げてください。", "What will be popular next? Name three things.",
           "  leading and trailing spaces  ", "trailing only \n", "改行\nを含む\n\n文", "A", "ＡＢＣ１２３ｶﾀｶﾅ①",
           "Hello  world\ttab", "<think> and <|im_end|> typed"]
# the system message a model card always passes with the prompt
SYSTEM = {"hf-swallow-ms-7b-instruct": "あなたは誠実で優秀な日本人のアシスタントです。"}


def entries():
    script = ("import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify("
              "MODELS.filter((m) => m.hf && !m.hf.weights.endsWith('.gguf')))))")
    return json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))


def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120).read()


def fetch(entry, directory):
    """The small files of the entry's repository at its revision, and the headers of its safetensors."""
    repo, revision = entry["hf"]["repo"], entry["hf"]["revision"]
    folder = directory / repo.replace("/", "__") / revision
    folder.mkdir(parents=True, exist_ok=True)
    at = lambda name: f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    tokenizers = [entry["hf"]["tokenizer"]] if isinstance(entry["hf"]["tokenizer"], str) else entry["hf"]["tokenizer"]
    # tokenizer.json and special_tokens_map.json too: transformers' reference reads them where they are
    for name in ["config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json",
                 "special_tokens_map.json", *tokenizers]:
        target = folder / name
        if target.exists() or (folder / f"{name}.missing").exists():
            continue
        try:
            data = get(at(name))
        except urllib.error.HTTPError:
            (folder / f"{name}.missing").touch()  # asked once; an empty file would look like an empty template
            continue
        target.write_bytes(data)
    try:
        get(at("model.safetensors"), {"Range": "bytes=0-7"})
        shards = ["model.safetensors"]
    except urllib.error.HTTPError:
        index = json.loads(get(at("model.safetensors.index.json")))
        shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        target = folder / f"{shard}.header.json"
        if not target.exists():
            (size,) = struct.unpack("<Q", get(at(shard), {"Range": "bytes=0-7"}))
            target.write_bytes(get(at(shard), {"Range": f"bytes=8-{7 + size}"}))
    return folder, shards, tokenizers


class Sink:
    """The converter writes no weights here: only its options and its tokenizer are wanted."""

    def open(self, *args):
        pass

    def write(self, *args):
        pass


def filled(template, prompt):
    """src/models.js's filled(), for today"""
    template = re.sub(r"\{date(?::([^}]*))?\}", lambda found: time.strftime(found.group(1) or "%Y-%m-%d"), template)
    return re.sub(r"\{prompt(:trim)?\}", lambda found: prompt.strip() if found.group(1) else prompt, template, count=1)


def conversion(folder, shards, tokenizers):
    """What the worker builds: the joined headers, config.json, the first tokenizer the converter takes, the template"""
    if shards == ["model.safetensors"]:
        header = (folder / "model.safetensors.header.json").read_text()
        base = 8 + len(header.encode())
    else:
        header, _ = llama2_convert.joined_shards([(folder / f"{shard}.header.json").read_text() for shard in shards])
        base = 0
    config = (folder / "tokenizer_config.json").read_text() if (folder / "tokenizer_config.json").exists() else ""
    has_template = bool(json.loads(config).get("chat_template")) if config else False
    jinja = folder / "chat_template.jinja"
    chat_template = jinja.read_text() if not has_template and jinja.exists() else None
    refusal = None
    for name in tokenizers:
        try:
            return llama2_convert.Conversion(header, base, (folder / "config.json").read_text(), (folder / name).read_bytes(),
                                             name, dtype="int8", start=base, tokenizer_config=config,
                                             chat_template=chat_template, sink=Sink())
        except (ValueError, FileNotFoundError) as error:
            refusal = error
    raise refusal


def main():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    directory, only = Path(sys.argv[1]), sys.argv[2:]
    accepted = set(inspect.signature(llama2_numpy.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    failed = []
    for entry in entries():
        if only and entry["id"] not in only:
            continue
        folder, shards, tokenizers = fetch(entry, directory)
        made = conversion(folder, shards, tokenizers)
        options = {**made.options, **entry.get("options", {})}
        template = entry.get("template") or made.options.get("template")
        if not template:
            continue  # no format: what was typed is continued as it is
        tokenizer = llama2_numpy.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                           **{key: value for key, value in options.items() if key in accepted})
        reference = AutoTokenizer.from_pretrained(folder)
        thinking = {"enable_thinking": False} if "(no thinking)" in entry["name"] else {}
        same, diffs = 0, []
        for prompt in PROMPTS:
            page = [options["bos"]] + tokenizer.encode(filled(template, prompt), tuple(options.get("specials", ())))
            messages = ([{"role": "system", "content": SYSTEM[entry["id"]]}] if entry["id"] in SYSTEM else []) \
                + [{"role": "user", "content": prompt}]
            real = reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **thinking)
            real = list(real["input_ids"] if hasattr(real, "keys") else real)
            if page == real or page[1:] == real:
                same += 1
            else:
                diffs.append(f"{prompt!r}\n    page {reference.convert_ids_to_tokens(page)[:60]}"
                             f"\n    real {reference.convert_ids_to_tokens(real)[:60]}")
        where = "the list" if entry.get("template") else "the converter"
        print(f"{'ok  ' if not diffs else 'DIFF'} {entry['id']}: {same}/{len(PROMPTS)} (format of {where})", flush=True)
        for diff in diffs[:3]:
            print("  ", diff)
        if diffs:
            failed.append(entry["id"])
    if failed:
        print(f"{len(failed)} differ: {' '.join(failed)} (some are known: see AGENTS.md, T131 and T138)")


if __name__ == "__main__":
    main()
