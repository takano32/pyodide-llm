# t246_chat_reference.py (the review of T246, a probe for CI, not for main; after T246's own tests/t246_reference.py on
# the branch t246-probe): what transformers itself makes of a Ternary Bonsai's original (the -unpacked float16
# safetensors) under config.json's yarn and under a plain RoPE, on texts the first measurement did not use and on
# chat turns, for the owner's choice between the two (T235's review, T246).
#
# What T246's tool did and this keeps: the ids [BOS] + the tokenizer's for a text (BOS 151643, as the list's options
# say), no attention mask needed (one text a row, nothing padded before its end: a causal model does not look at the
# padding after it), the readings made by replacing the rotary embedding's inv_freq and attention_scaling (yarn:
# transformers' own function, float32; plain: theta only), float16 weights computed in float32 a layer at a time
# (4B; float16 to float32 is exact), and for the 8B, which cannot be loaded (16 GB), transformers' own modules with ONE
# layer whose weights are read for each of the model's layers in turn (--by-layer; T246 showed it to give the 4B's
# numbers to four decimals).
# What is new: (1) --chat <file>: lines {"prompt", "answer"}, a turn of the chat in the model's own template (the answer
# at once, as the list's entry asks) with the answer written by hand (so by no model of this family): the loss is that of
# the answer's tokens and of <|im_end|>, by language. (2) The texts are fetched by the caller from other titles than
# T246's. (3) Each text's loss at 256, 512 and 1024 tokens as before.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t246_chat_reference.py <directory> <1.7B | 4B | 8B> <revision of the -unpacked repo> [--by-layer]
#       [--chat <jsonl>] <text file>:<tokens> ...
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

sys.path.insert(0, "public")
import llama2_numpy as N  # noqa: E402

# T246_BOS: what stands before a text (the list's 8B has begun at <|im_start|>, 151644, since T250's review: the 8B is
# 70% worse on plain text with <|endoftext|> in front); T246_CHAT_BOS=0: a chat turn is the real template's ids alone
BOS, IM_END = int(os.environ.get("T246_BOS", 151643)), 151645
started = time.time()


def say(*parts):
    print(f"T246CHAT [{time.time() - started:6.0f} s]", *parts, flush=True)


arguments = sys.argv[1:]
chat_file = None
if "--chat" in arguments:
    at = arguments.index("--chat")
    chat_file = arguments[at + 1]
    del arguments[at:at + 2]
by_layer = "--by-layer" in arguments
arguments = [argument for argument in arguments if argument != "--by-layer"]
where, size, revision, *passages = arguments
repo = f"prism-ml/Ternary-Bonsai-{size}-unpacked"
directory = Path(snapshot_download(repo, revision=revision, local_dir=Path(where) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
config = json.loads((directory / "config.json").read_text())
say(f"{repo}@{revision}: fetched; transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
    f"{torch.get_num_threads()} threads")

# (name, ids, the index in ids of the first token that counts: 1 for a text, the answer's first for a chat turn)
items = []
for passage in passages:
    path, count = passage.rsplit(":", 1)
    content = Path(path).read_text()
    ids = [BOS] + tokenizer(content, add_special_tokens=False)["input_ids"][:int(count)]
    items.append((Path(path).name, ids, 1))
    say(f"{Path(path).name} (sha256 {hashlib.sha256(content.encode()).hexdigest()[:12]}): {len(ids) - 1} tokens after the BOS, "
        f"the first {ids[1:9]}, their sum {sum(ids)}")
chats = []
if chat_file:
    for number, line in enumerate(Path(chat_file).read_text().splitlines()):
        pair = json.loads(line)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True,
                                               tokenize=False)
        head = ([BOS] if os.environ.get("T246_CHAT_BOS", "1") == "1" else []) + tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer = tokenizer(pair["answer"], add_special_tokens=False)["input_ids"] + [IM_END]
        language = "ja" if any(ord(c) > 0x2E80 for c in pair["prompt"]) else "en"
        items.append((f"chat-{number}-{language}", head + answer, len(head)))
        chats.append(f"chat-{number}-{language}")
    say(f"{len(chats)} chat turns; the first one's ids {items[len(passages)][1][:24]}...")

if by_layer:
    one = AutoConfig.from_pretrained(directory)
    layers, one.num_hidden_layers = one.num_hidden_layers, 1
    if getattr(one, "layer_types", None):
        one.layer_types = one.layer_types[:1]
    model = AutoModelForCausalLM.from_config(one).float()
    model.eval()
    index = directory / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    files = [safe_open(directory / name, "pt") for name in names]
    where_is = {key: file for file in files for key in file.keys()}

    def put(module, prefix):
        state = {key: where_is[prefix + key].get_tensor(prefix + key).float() for key in module.state_dict()}
        module.load_state_dict(state, strict=True)

    put(model.model.embed_tokens, "model.embed_tokens.")
    put(model.model.norm, "model.norm.")
    tied = "lm_head.weight" not in where_is
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        put(model.lm_head, "lm_head.")
    say(f"by layer: {layers} layers from {len(names)} file(s), {len(where_is)} tensors; the classifier is {'the embedding' if tied else 'its own'}")
else:
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    model.model.norm.float()
    model.lm_head.float()
    model.model.embed_tokens.float()

    def widen(module, args, kwargs):
        module.halves = [parameter.data for parameter in module.parameters()]
        for parameter in module.parameters():
            parameter.data = parameter.data.float()

    def narrow(module, args, kwargs, output):  # the float16 tensors themselves come back: nothing is rounded
        for parameter, half in zip(module.parameters(), module.halves):
            parameter.data = half
        module.halves = None

    for layer in model.model.layers:
        layer.register_forward_pre_hook(widen, with_kwargs=True)
        layer.register_forward_hook(narrow, with_kwargs=True)
rotary = model.model.rotary_emb
yarn_inv, yarn_scale = ROPE_INIT_FUNCTIONS["yarn"](model.config, "cpu")
yarn_inv = yarn_inv.float()
head, theta, scaling = config["head_dim"], config["rope_theta"], config["rope_scaling"]
plain_inv = 1.0 / (theta ** (torch.arange(0, head, 2, dtype=torch.float) / head))
mine = N.rope_frequencies(head, theta, scaling)
say(f"loaded: rope_type {rotary.rope_type}; yarn: transformers' factor {float(yarn_scale):.9f} against the engine's magnitude "
    f"{N.rope_magnitude(scaling):.9f}, inv_freq against the engine's the largest relative difference "
    f"{float(np.max(np.abs(yarn_inv.double().numpy() / mine - 1))):.2e}")
READINGS = {"yarn (config.json)": (yarn_inv, float(yarn_scale)), "plain RoPE": (plain_inv, 1.0)}


def reading(name):
    inv, scale = READINGS[name]
    rotary.inv_freq = inv.clone()
    rotary.attention_scaling = scale


def hidden_by_layer(batch):
    ids = torch.tensor(batch)
    length = ids.shape[1]
    hidden = model.model.embed_tokens(ids)
    positions = torch.arange(length)[None]
    embeddings = rotary(hidden, positions)
    mask = torch.full((length, length), torch.finfo(hidden.dtype).min).triu(1)[None, None]
    layer = model.model.layers[0]
    for number in range(layers):
        put(layer, f"model.layers.{number}.")
        out = layer(hidden, attention_mask=mask, position_ids=positions, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return model.model.norm(hidden)


def loss_of(hidden, ids):
    """The loss of every token of ids after the first from the hidden states of its positions"""
    out = []
    for start in range(0, len(ids) - 1, 256):
        logits = model.lm_head(hidden[start:min(start + 256, len(ids) - 1)].float()).double()
        out.append(torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1 + start:1 + start + logits.shape[0]]), reduction="none"))
    return torch.cat(out)


results = {name: {} for name in READINGS}


def keep(name, item, loss, seconds):
    label, ids, first = item
    counted = loss[first - 1:]
    results[name][label] = (float(counted.sum()), len(counted))
    if label in chats:
        say(f"CHATREF {label} {name}: nll {float(counted.mean()):.4f} over {len(counted)} tokens")
    else:
        prefixes = ", ".join(f"{count}: {math.exp(float(loss[:count].mean())):.4f}" for count in (256, 512, 1024) if count < len(loss))
        say(f"PPLREF {label} {name}: {math.exp(float(loss.mean())):.4f} over {len(loss)} targets ({seconds:.0f} s); prefixes {prefixes}")


for name in READINGS:
    reading(name)
    with torch.no_grad():
        if by_layer:
            # the texts in one pass and the chat turns (short) in another: one batch of both, padded to the longest text,
            # would hold the attention scores and the FFN of 28 rows of 1025 tokens beside the 8B's two float32 tables
            for group in ([item for item in items if item[0] not in chats], [item for item in items if item[0] in chats]):
                if not group:
                    continue
                began = time.time()
                longest = max(len(ids) for _, ids, _ in group)
                hidden = hidden_by_layer([ids + [BOS] * (longest - len(ids)) for _, ids, _ in group])
                seconds = time.time() - began
                for row, item in enumerate(group):
                    keep(name, item, loss_of(hidden[row], item[1]), seconds)
                del hidden
        else:
            for item in items:
                began = time.time()
                hidden = model.model(input_ids=torch.tensor([item[1]])).last_hidden_state[0]
                keep(name, item, loss_of(hidden, item[1]), time.time() - began)
# the totals: the loss of all the tokens of the chat turns (by language) and of the texts, a reading against the other
for group, labels in (("chat ja", [c for c in chats if c.endswith("-ja")]), ("chat en", [c for c in chats if c.endswith("-en")]),
                      ("chat all", chats), ("texts", [label for label, _, _ in items if label not in chats])):
    if not labels:
        continue
    totals = {name: (sum(results[name][c][0] for c in labels), sum(results[name][c][1] for c in labels)) for name in READINGS}
    nll = {name: totals[name][0] / totals[name][1] for name in READINGS}
    wins = sum(1 for c in labels if results["plain RoPE"][c][0] / results["plain RoPE"][c][1] < results["yarn (config.json)"][c][0] / results["yarn (config.json)"][c][1])
    say(f"TOTAL {size} {group} ({len(labels)} items, {totals['plain RoPE'][1]} tokens): mean nll yarn {nll['yarn (config.json)']:.4f}, "
        f"plain {nll['plain RoPE']:.4f}; perplexity yarn {math.exp(nll['yarn (config.json)']):.3f}, plain {math.exp(nll['plain RoPE']):.3f}; "
        f"yarn / plain - 1 = {(math.exp(nll['yarn (config.json)'] - nll['plain RoPE']) - 1) * 100:+.2f}%; plain is lower on {wins} of {len(labels)}")
say("done")
