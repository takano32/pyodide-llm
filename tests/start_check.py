# start_check.py (T249's review): what the token in front of a text costs a model. The page begins every text with the
# BOS the converter says (src/models.js can set another), where the real tokenizer of a model may put another token in
# front or none (a T5Tokenizer puts nothing in front and </s> after; Qwen's puts nothing). Not something a format
# check says (tests/format_check.py compares what comes after) and not something to settle by reading what the model
# writes: this scores the same targets under every start, on the original in float32 with transformers.
#
#   python3 tests/start_check.py <model id of src/models.js> [--also <token id> ...] [--tokens N = 1500] [--window W = 512]
#                                [--text <file>] [--directory <where the downloads go> = .tmp/start-check] [--stored | --by-layer]
#
# The ids are the page's (the converter's tokenizer.bin through llama2_numpy.Tokenizer, which tests/format_check.py and
# its sentencepiece comparisons hold to the real one), the weights are the original's at the revision the list pins
# (tests/hf_fetch.py). The text, if none is given, is the beginning of three Japanese Wikipedia articles (T85's, fetched
# when this runs). Every window of W tokens is scored from its second token, so that each start is held to the same
# targets: the first token is context (or, with a start, the second), as in T131's measure. The starts are: none, the
# page's BOS, and every other token that config.json, tokenizer_config.json and special_tokens_map.json name (plus
# --also). Per start: perplexity, and nats per character (so that a model of another vocabulary can be set beside it:
# tests/start_check.mjs prints the same for the page's engine).
#
# Needs torch and transformers (a venv, or CI's tests.yml extra=: docs/notes/dev-setup.md). Memory: float32 weights.
# --stored: the weights are held as the original stores them (bfloat16) and every product is float32: Granite 4.2 3B is 14.6 GB as
# float32. --by-layer: a model that cannot be loaded even so (Granite 4.2 8B: 16.8 GB in bfloat16) a layer at a time
# (reference_llama.ByLayer: transformers' own layer with the weights of each layer read in turn, in float32).
import argparse
import inspect
import json
import math
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import format_check  # noqa: E402
import llama2_numpy  # noqa: E402

ARTICLES = ["富士山", "夏目漱石", "新幹線"]  # tests/wikipedia.mjs's ARTICLES.ja


def wikipedia(titles):
    text = ""
    for title in titles:
        url = ("https://ja.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain&format=json"
               f"&titles={urllib.parse.quote(title)}")
        request = urllib.request.Request(url, headers={"User-Agent": "pyodide-llm start measurement"})
        pages = json.loads(urllib.request.urlopen(request, timeout=60).read())["query"]["pages"]
        text += next(iter(pages.values()))["extract"][:6000] + "\n"
    return text


def named_tokens(folder, tokenizer):
    """{id: where a file names it} of the tokens the original's files name (bos, eos, pad, unk, cls, sep ...), by the
    page's vocabulary"""
    found = {}
    config = json.loads((folder / "config.json").read_text())
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        values = config.get(key)
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, int):
                found.setdefault(value, f"config.json {key.removesuffix('_id')}")
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        if (folder / name).exists():
            for key, value in json.loads((folder / name).read_text()).items():
                value = value.get("content") if isinstance(value, dict) else value
                if key.endswith("_token") and isinstance(value, str) and value.encode() in tokenizer.index:
                    found.setdefault(tokenizer.index[value.encode()], f"{name} {key}")
    return found


def score_by_layer(model, windows, starts):
    """score() for a reference_llama.ByLayer: every window under every start is one row of one batch through the layers
    (one reading of each layer's weights), and the targets are scored from the hidden states"""
    import torch
    rows, labels, firsts = [], [], []
    for targets in windows:
        for label, start in starts.items():
            rows.append(([] if start is None else [start]) + targets)
            labels.append(label)
            firsts.append(start)
    totals, scored = {label: 0.0 for label in starts}, sum(len(targets) - 1 for targets in windows)
    for label, row, hidden, start in zip(labels, rows, model.hidden(rows), firsts):
        picked = torch.log_softmax(model.logits(hidden[:-1]), -1)[torch.arange(len(row) - 1), torch.tensor(row[1:])]
        totals[label] -= (picked if start is None else picked[1:]).sum().item()
    return totals, scored


def score(model, windows, starts, stored=False):
    """({start: the sum of the negative log likelihoods of the targets}, how many targets): every window of ids scored
    from its second token under each start (a token or None), the model's logits through transformers. stored: the
    weights are held as stored and every product is float32 (--stored), the embeddings float32 before the first layer"""
    import torch
    totals, scored = {label: 0.0 for label in starts}, 0
    with torch.no_grad():
        for number, targets in enumerate(windows):
            scored += len(targets) - 1
            for label, start in starts.items():
                row = ([] if start is None else [start]) + targets
                given = ({"inputs_embeds": model.get_input_embeddings()(torch.tensor([row])).to(torch.float32)} if stored
                         else {"input_ids": torch.tensor([row])})
                logprobs = torch.log_softmax(model(**given).logits[0, :-1].float(), -1)
                picked = logprobs[torch.arange(len(row) - 1), torch.tensor(row[1:])]
                # row[1:] is every target but the first with no start; with one, the first target is row[1]: scored
                # from the second, the same targets
                totals[label] -= (picked if start is None else picked[1:]).sum().item()
            print(f"  window {number + 1}/{len(windows)} done", flush=True)
    return totals, scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--also", type=int, nargs="*", default=[])
    parser.add_argument("--tokens", type=int, default=1500)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--text")
    parser.add_argument("--directory", default=str(HERE.parent / ".tmp" / "start-check"))
    # T253's review: a model whose float32 is more than a runner has (Granite 4.2 3B: 14.6 GB) is held as the original
    # stores it (bfloat16), every product in float32 (reference_llama.float32_arithmetic: the same float32 arithmetic)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument("--by-layer", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    entry = next(entry for entry in format_check.entries() if entry["id"] == args.model)
    folder, shards, tokenizers = format_check.fetch(entry, directory)
    made = format_check.conversion(entry, folder, shards, tokenizers)
    options = {**made.options, **entry.get("options", {})}
    accepted = set(inspect.signature(llama2_numpy.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    tokenizer = llama2_numpy.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                       **{key: value for key, value in options.items() if key in accepted})
    text = Path(args.text).read_text() if args.text else wikipedia(ARTICLES)
    every = tokenizer.encode(text, tuple(options.get("specials", ())))
    ids = every[:args.tokens]
    characters = len(text) * len(ids) / len(every)  # of the text, as far as ids go

    # the original, at the revision the list pins: the vocabulary's repository for a GGUF, else the entry's own
    source = entry["hf"].get("vocabulary") or entry["hf"]
    original = Path(subprocess.check_output([sys.executable, str(HERE / "hf_fetch.py"), f"hf:{source['repo']}@{source['revision']}",
                                             str(directory / "weights")], text=True).splitlines()[-1])
    import torch
    from transformers import AutoModelForCausalLM
    held = torch.float32
    if args.stored:
        import reference_llama
        reference_llama.float32_arithmetic()
        said = json.loads((Path(original) / "config.json").read_text())
        held = getattr(torch, said.get("dtype") or said.get("torch_dtype") or "float32")
    if args.by_layer:
        import reference_llama
        model = reference_llama.ByLayer(original)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(original, dtype=held).eval()
        except TypeError:  # transformers before 4.56 calls it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(original, torch_dtype=held).eval()
    limit = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 2048)
    window = min(args.window, limit - 1)  # (a start takes one place of the model's)

    bos = options["bos"]
    starts = {"none": None, f"the page's BOS {bos}": bos}
    for token, why in {**named_tokens(folder, tokenizer), **{token: "--also" for token in args.also}}.items():
        if token not in starts.values():
            starts[f"{why} {token}"] = token
    piece = lambda token: tokenizer.vocab[token].decode("utf-8", "replace") if token is not None else ""
    print(f"start_check {args.model}: {len(ids)} tokens of about {characters:.0f} characters, windows of {window}, "
          f"{type(model).__name__} {'held as ' + str(held) + ', every product' if args.stored else 'a layer at a time,' if args.by_layer else ''} in float32 from "
          f"{source['repo']}@{source['revision'][:8]}; the page's BOS {bos} {piece(bos)!r}, "
          f"stops {options.get('stop_tokens')}", flush=True)
    windows = [window_ids for window_ids in (ids[i:i + window] for i in range(0, len(ids), window)) if len(window_ids) >= 64]
    totals, scored = score_by_layer(model, windows, starts) if args.by_layer else score(model, windows, starts, stored=args.stored)
    per_character = characters * scored / len(ids)  # the characters the scored targets stand for
    page_total = totals[f"the page's BOS {bos}"]
    print("\n| start | token | perplexity | against none | against the page's BOS | nats per character |\n|---|---|---:|---:|---:|---:|")
    for label, start in starts.items():
        print(f"| {label} | {piece(start)!r} | {math.exp(totals[label] / scored):.3f} | "
              f"{(math.exp((totals[label] - totals['none']) / scored) - 1) * 100:+.2f}% | "
              f"{(math.exp((totals[label] - page_total) / scored) - 1) * 100:+.2f}% | "
              f"{totals[label] / per_character:.4f} |")
    print(f"start_check {args.model}: scored {scored} targets in {len(windows)} windows")


if __name__ == "__main__":
    main()
