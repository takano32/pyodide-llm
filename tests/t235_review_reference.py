# t235_review_reference.py (T235's review, a probe for CI, not for main): what transformers itself makes of
# Ternary-Bonsai 1.7B (the original's float16 safetensors, in float32) under three readings of its RoPE, on several
# texts and lengths, with and without a BOS; and why generate() wrote a different fifth token than a forward pass ranks
# first. One download, one model load: a reading is set by replacing the rotary embedding's inv_freq and
# attention_scaling in place.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t235_review_reference.py <directory for the download>
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION, BOS = "prism-ml/Ternary-Bonsai-1.7B-unpacked", "3aca840085293d026ce6f6b80fafdae937fd2eeb", 151643
PROMPT = "<|im_start|>user\nこれからの流行りを3つ挙げてください。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
CHECKPOINTS = [256, 512, 1024, 1536, 2048, 3072, 4096]
started = time.time()


def say(*parts):
    print(f"[{time.time() - started:6.0f} s]", *parts, flush=True)


directory = Path(snapshot_download(REPO, revision=REVISION, local_dir=Path(sys.argv[1]) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
config = json.loads((directory / "config.json").read_text())


def wikipedia(language, titles):
    text = ""
    for title in titles:
        url = (f"https://{language}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain"
               f"&format=json&titles={urllib.parse.quote(title)}")
        request = urllib.request.Request(url, headers={"User-Agent": "pyodide-llm perplexity measurement"})
        pages = json.load(urllib.request.urlopen(request, timeout=60))["query"]["pages"]
        text += next(iter(pages.values()))["extract"][:6000] + "\n"
    return text


texts = {}  # name: (text, most tokens)
for name, source, count in (("arch-md", "tests/fixtures/t235/architecture.md", 1536), ("text-txt", "tests/fixtures/t235/text.txt", 256)):
    if Path(source).exists():
        texts[name] = (Path(source).read_text(), count)
for name, language, titles in (("ja-fuji", "ja", ["富士山"]), ("ja-soseki", "ja", ["夏目漱石"]),
                               ("en-3", "en", ["Mount Fuji", "Natsume Sōseki", "Shinkansen"])):
    try:
        texts[name] = (wikipedia(language, titles), 4096)
    except Exception as error:  # the network is not what this measures
        say(f"{name}: no text ({error!r})")

# ---- the tokenizers: transformers', the tokenizers library's and the engine's, on the same texts
sys.path.insert(0, "public")
import llama2_convert as C  # noqa: E402
import llama2_numpy as N  # noqa: E402
from tokenizers import Tokenizer as Rust  # noqa: E402

parsed = json.loads((directory / "tokenizer.json").read_text())
options = C.tokenizer_json_options(parsed)
pieces = list(C.tokenizer_json_pieces(parsed))
packed = C.tokenizer_bin(pieces, config["vocab_size"], spaces=options["tokenizer_kind"] != "bytebpe", charsmap=C.tokenizer_json_charsmap(parsed))
engine = N.Tokenizer(packed, config["vocab_size"], kind=options["tokenizer_kind"], nfkc=options.get("nfkc", False), nfc=options.get("nfc", False),
                     pretokenizer=options.get("pretokenizer", "gpt2"), ignore_merges=options.get("ignore_merges", False))
rust = Rust.from_file(str(directory / "tokenizer.json"))
ids = {}
for name, (text, count) in texts.items():
    by_transformers = tokenizer(text, add_special_tokens=False)["input_ids"]
    by_rust = rust.encode(text, add_special_tokens=False).ids
    by_engine = [int(i) for i in engine.encode(text)]
    ids[name] = by_transformers[:count]
    say(f"tokens {name}: {len(by_transformers)} in all, transformers == tokenizers {by_transformers == by_rust}, "
        f"transformers == engine {by_transformers == by_engine}" +
        ("" if by_transformers == by_engine else f" (first difference at {next(i for i, (a, b) in enumerate(zip(by_transformers, by_engine)) if a != b)})"))

# ---- the model, once
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
rotary = model.model.rotary_emb
say(f"loaded: rope_type {rotary.rope_type}, attention_scaling {rotary.attention_scaling:.6f}, pad_token_id {model.generation_config.pad_token_id}, "
    f"eos {model.generation_config.eos_token_id}, begin_suppress {getattr(model.generation_config, 'begin_suppress_tokens', None)}")
yarn_inv = rotary.inv_freq.clone()
yarn_scale = float(rotary.attention_scaling)
plain_inv = 1.0 / (config["rope_theta"] ** (torch.arange(0, config["head_dim"], 2, dtype=torch.float) / config["head_dim"]))
READINGS = {"yarn": (yarn_inv, yarn_scale), "angles": (yarn_inv, 1.0), "plain": (plain_inv, 1.0)}


def reading(name):
    inv, scale = READINGS[name]
    rotary.inv_freq = inv.clone()
    rotary.attention_scaling = scale


# ---- the engine's tables against transformers' (cos and sin with attention_scaling), at the positions of a 4096 context
reading("yarn")
position_ids = torch.arange(4096)[None]
with torch.no_grad():
    cos, sin = rotary(torch.zeros(1, 1, 1), position_ids)
scaling = config["rope_scaling"]
frequencies = N.rope_frequencies(config["head_dim"], config["rope_theta"], scaling)
magnitude = N.rope_magnitude(scaling)
angles = np.arange(4096)[:, None] * frequencies
mine_cos = (np.cos(angles) * magnitude).astype(np.float32)
mine_sin = (np.sin(angles) * magnitude).astype(np.float32)
theirs_cos, theirs_sin = cos[0, :, :64].numpy(), sin[0, :, :64].numpy()
say(f"tables over 4096 positions: attention_scaling {rotary.attention_scaling:.9f} against the engine's magnitude {magnitude:.9f}; "
    f"largest |cos| difference {np.abs(mine_cos - theirs_cos).max():.2e}, |sin| {np.abs(mine_sin - theirs_sin).max():.2e}; "
    f"largest cos^2 + sin^2 of the engine's {np.max(mine_cos.astype(np.float64) ** 2 + mine_sin.astype(np.float64) ** 2):.9f} "
    f"(transformers' {np.max(theirs_cos.astype(np.float64) ** 2 + theirs_sin.astype(np.float64) ** 2):.9f}, magnitude squared {magnitude ** 2:.9f})")


def losses(token_ids, with_bos):
    """The loss of every token of token_ids after the BOS (with_bos: the first from the BOS), one forward pass."""
    sequence = ([BOS] if with_bos else []) + token_ids
    inputs = torch.tensor([sequence])
    out = []
    with torch.no_grad():
        hidden = model.model(inputs, use_cache=False).last_hidden_state[0]
        for start in range(0, len(sequence) - 1, 256):
            logits = model.lm_head(hidden[start:min(start + 256, len(sequence) - 1)]).double()
            targets = inputs[0, start + 1:start + 1 + logits.shape[0]]
            out.append(torch.nn.functional.cross_entropy(logits, targets, reduction="none"))
    return torch.cat(out).numpy()


def perplexity(values, count):
    return math.exp(float(np.mean(values[:count])))


# the readings and texts to measure. with a BOS (as the page sends them): every reading on every text; without: yarn and
# plain on two
WITH = [("yarn", name) for name in ids] + [("plain", name) for name in ids] + [("angles", name) for name in ids if name in ("ja-fuji", "en-3", "arch-md", "text-txt")]
WITHOUT = [(r, name) for r in ("yarn", "plain") for name in ids if name in ("ja-fuji", "en-3")]
results = {}
for r, name in WITH:
    reading(r)
    began = time.time()
    results[(r, name, True)] = losses(ids[name], True)
    say(f"{r} {name} with BOS: {len(ids[name])} tokens in {time.time() - began:.0f} s")
for r, name in WITHOUT:
    reading(r)
    began = time.time()
    results[(r, name, False)] = losses(ids[name], False)
    say(f"{r} {name} without a BOS: {len(ids[name])} tokens in {time.time() - began:.0f} s")
for (r, name, with_bos), values in results.items():
    n = len(ids[name])
    for count in sorted({c for c in CHECKPOINTS if c <= n} | {n}):
        if with_bos:
            # all the targets t0.. after the BOS (the page's), and t1.. alone, the targets a run without a BOS also has
            print(f"PPL {r} {name} bos {count} {perplexity(values, count):.4f} t1.. {math.exp(float(np.mean(values[1:count]))):.4f}", flush=True)
        else:
            print(f"PPL {r} {name} nobos {count} - t1.. {perplexity(values, count - 1):.4f}", flush=True)

# ---- generate(): the fifth token
prompt = [BOS] + tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
say(f"the prompt: {len(prompt)} ids, pad {model.generation_config.pad_token_id} is the first of them: {prompt[0] == model.generation_config.pad_token_id}")
engine_tokens = [87752, 15322, 5373, 50230, 56833]


def written(text_ids):
    return tokenizer.decode(text_ids)


def top(logits, k=4):
    best = torch.topk(logits.double(), k)
    return [[int(i), round(float(v), 3)] for v, i in zip(best.values, best.indices)]


for r in ("yarn", "angles", "plain"):
    reading(r)
    with torch.no_grad():
        # (1) the forward logits at the fifth token's position, with the BOS and without it (positions from 0)
        for with_bos in (True, False):
            sequence = (prompt if with_bos else prompt[1:]) + engine_tokens[:4]
            logits = model(torch.tensor([sequence])).logits[0, -1]
            say(f"{r}: forward, {'with' if with_bos else 'without'} the BOS, the four most likely after the engine's first four: {top(logits)}")
        # (2) generate(): its default, with an attention mask of ones, and without the BOS
        default = model.generate(torch.tensor([prompt]), max_new_tokens=16, do_sample=False)[0, len(prompt):].tolist()
        ones = model.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long), max_new_tokens=16,
                              do_sample=False)[0, len(prompt):].tolist()
        without = model.generate(torch.tensor([prompt[1:]]), max_new_tokens=16, do_sample=False)[0, len(prompt) - 1:].tolist()
        # (3) a greedy loop by hand with the BOS: every step a whole forward pass, and with a cache
        sequence, uncached = list(prompt), []
        for _ in range(16):
            logits = model(torch.tensor([sequence]), use_cache=False).logits[0, -1]
            uncached.append(int(logits.argmax()))
            sequence.append(uncached[-1])
        out = model(torch.tensor([prompt]), use_cache=True)
        cache, token, cached, widest = out.past_key_values, int(out.logits[0, -1].argmax()), [], 0.0
        cached.append(token)
        steps = list(prompt) + [token]
        for _ in range(15):
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            full = model(torch.tensor([steps]), use_cache=False).logits[0, -1]
            widest = max(widest, float((out.logits[0, -1] - full).abs().max()))
            token = int(out.logits[0, -1].argmax())
            cached.append(token)
            steps.append(token)
        say(f"{r}: generate() default {written(default)!r} {default[:8]}")
        say(f"{r}: generate() with the attention mask of ones {written(ones)!r} {ones[:8]}")
        say(f"{r}: generate() on the prompt without its BOS {written(without)!r} {without[:8]}")
        say(f"{r}: by hand, no cache {written(uncached)!r} {uncached[:8]}")
        say(f"{r}: by hand, with a cache {written(cached)!r} {cached[:8]}; the cached step's logits against the whole forward pass: "
            f"at most {widest:.2e} apart")
say("done")
