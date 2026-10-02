# layerwise_qwen35.py (the review of T247, a probe for CI, not for main): transformers' own Qwen3.5 forward pass, float32, a
# layer at a time, for a model whose float32 is more than a runner's memory (the 9B: 35.8 GB; its bfloat16 weights are
# 19 GB on the disk). The model is built on the meta device; each decoder layer gets its weights from the safetensors
# (bfloat16 widened to float32, exact) as it is called and gives them back after it, the embedding rows are gathered from the
# file and the classifier is multiplied a chunk of the vocabulary at a time. What a layer computes is transformers' code.
#
#   python tests/layerwise_qwen35.py <directory with the original's files> --out <prefix> [--text <file>] [--positions 96]
#   python tests/layerwise_qwen35.py <directory> --out <prefix> --check     (also the whole model's pass, for a model that fits)
#
# Saves <prefix>-sentence-logits.npy (the first --positions positions of tests/reference_qwen35.py's text, BOS 248044) and
# <prefix>-nll.npy (the negative log likelihood of each token of the windows tests/perplexity.py makes of --text). Every line of the
# log begins with "LAYERWISE".
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))

say = lambda *parts: print("LAYERWISE", *parts, flush=True)
# the originals the list's entries take their vocabulary and configuration from (src/models.js)
MODELS = {"0.8B": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
          "2B": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
          "4B": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
          "9B": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")}


class Weights:
    """The original's tensors by name, over its shards (the index says which)."""

    def __init__(self, directory):
        from safetensors import safe_open

        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = index
        self.files = {name: safe_open(str(directory / name), framework="pt") for name in sorted(set(index.values()))}

    def get(self, name):
        return self.files[self.where[name]].get_tensor(name)

    def __contains__(self, name):
        return name in self.where


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text")
    parser.add_argument("--positions", type=int, default=96)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=1500)
    parser.add_argument("--model", help="0.8B, 2B, 4B or 9B: fetch the original's files into the directory first")
    arguments = parser.parse_args()
    torch.set_num_threads(__import__("os").cpu_count() or 4)
    import tokenizers
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

    import reference_qwen35 as ref

    directory = Path(arguments.directory)
    if arguments.model:
        repo, revision = MODELS[arguments.model]
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
            ref.fetch(name, directory, repo, revision)
        for shard in sorted(set(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values())):
            ref.fetch(shard, directory, repo, revision)
        say(f"{repo}@{revision}: the files are here")
    config = json.loads((directory / "config.json").read_text())
    text_config = Qwen3_5TextConfig(**{k: v for k, v in config["text_config"].items() if k not in ("model_type", "dtype", "torch_dtype")})
    tied = bool(config["text_config"].get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    weights = Weights(directory)
    prefix = "model.language_model."
    say(f"{text_config.num_hidden_layers} layers, hidden {text_config.hidden_size}, vocabulary {text_config.vocab_size}, "
        f"{'a tied classifier' if tied else 'a classifier of its own'}, {len(weights.files)} shards")

    # the model on the meta device, the layers' weights coming in when a layer is called
    with torch.device("meta"):
        text = Qwen3_5TextModel(text_config)
    text.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_config)  # (a real one: its inverse frequencies are float32)
    text.norm.to_empty(device="cpu")
    with torch.no_grad():
        text.norm.weight.copy_(weights.get(prefix + "norm.weight").float())
    loading = {"seconds": 0.0}

    def load(layer, name):
        began = time.perf_counter()
        layer.to_empty(device="cpu")
        with torch.no_grad():
            for key, parameter in layer.named_parameters():
                parameter.copy_(weights.get(f"{prefix}layers.{name}.{key}").float())
        loading["seconds"] += time.perf_counter() - began

    def before(name):
        def hook(module, args, kwargs):  # (a hook that returns something replaces the call's arguments or its output)
            load(module, name)
        return hook

    def after(module, args, output):
        module.to("meta")

    for index, layer in enumerate(text.layers):
        layer.register_forward_pre_hook(before(index), with_kwargs=True)
        layer.register_forward_hook(after)

    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    bos = int(__import__("os").environ.get("BOS", ref.BOS))  # BOS=248045: the first token the list's entries begin with
    sentence = ([bos] + tokenizer.encode(ref.TEXT, add_special_tokens=False).ids)[:arguments.positions]
    rows = [sentence]
    Path(f"{arguments.out}-sentence-ids.json").write_text(json.dumps(sentence))
    if arguments.text:
        tokens = tokenizer.encode(Path(arguments.text).read_text(), add_special_tokens=False).ids[:arguments.tokens]
        rows += [[bos] + tokens[start:start + arguments.window - 1] for start in range(0, len(tokens), arguments.window - 1)]
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), bos, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for at, row in enumerate(rows):
        ids[at, :len(row)] = torch.tensor(row)
        mask[at, :len(row)] = 1

    # the embedding rows, gathered from the file (a table of 2 GB or so, widened a row at a time)
    table = weights.get(prefix + "embed_tokens.weight")
    embeds = table[ids].float()
    del table
    gc.collect()
    began = time.perf_counter()
    with torch.no_grad():
        hidden = text(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    say(f"{len(rows)} rows of up to {width} positions through {text_config.num_hidden_layers} layers in {time.perf_counter() - began:.0f} s "
        f"({loading['seconds']:.0f} s of it loading weights)")

    classifier = weights.get(prefix + "embed_tokens.weight" if tied else "lm_head.weight")
    vocabulary = classifier.shape[0]
    positions = len(sentence)
    first = np.zeros((positions, vocabulary), dtype=np.float32)
    lse = torch.full((len(rows), width), -math.inf, dtype=torch.float64)
    target_logit = torch.zeros((len(rows), width), dtype=torch.float64)
    targets = torch.zeros((len(rows), width), dtype=torch.long)
    for at, row in enumerate(rows):
        targets[at, :len(row) - 1] = torch.tensor(row[1:])
    with torch.no_grad():
        for start in range(0, vocabulary, 16384):
            piece = classifier[start:start + 16384].float()
            logits = hidden @ piece.T
            lse = torch.logaddexp(lse, torch.logsumexp(logits.double(), dim=-1))
            inside = (targets >= start) & (targets < start + logits.shape[-1])
            chosen = logits.gather(-1, (targets - start).clamp(0, logits.shape[-1] - 1)[..., None])[..., 0].double()
            target_logit = torch.where(inside, chosen, target_logit)
            first[:, start:start + logits.shape[-1]] = logits[0, :positions].numpy()
    nll = lse - target_logit
    np.save(f"{arguments.out}-sentence-logits.npy", first)
    if arguments.text:
        per_token = np.concatenate([nll[at, :len(row) - 1].numpy() for at, row in enumerate(rows) if at > 0])
        np.save(f"{arguments.out}-nll.npy", per_token)
        say(f"perplexity of the {len(per_token)} tokens of {Path(arguments.text).name}: {math.exp(float(per_token.mean())):.3f}; "
            f"of the first 192: {math.exp(float(per_token[:192].mean())):.3f}")
    say(f"logits of the sentence: min {first.min():.3f}, max {first.max():.3f}, mean {first.mean():.4f}, std {first.std():.4f}")

    if arguments.check:
        # the whole model's own pass over the same rows, where the model fits: the layerwise one must be the same
        from transformers import Qwen3_5ForConditionalGeneration

        del classifier, hidden
        gc.collect()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
        with torch.no_grad():
            theirs = model(input_ids=ids[:1, :positions], attention_mask=mask[:1, :positions]).logits[0].float().numpy()
        gap = float(np.abs(theirs - first).max())
        same = int((theirs.argmax(axis=1) == first.argmax(axis=1)).sum())
        say(f"check: the whole model's logits of the sentence against the layerwise ones: largest difference {gap:.2e}, the same most "
            f"likely token at {same} of {positions}")
        assert gap < 1e-3


if __name__ == "__main__":
    main()
