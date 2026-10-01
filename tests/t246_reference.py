# t246_reference.py (T246, a probe for CI, not for main): what transformers itself makes of a Ternary Bonsai's original
# (the -unpacked float16 safetensors) under config.json's yarn and under a plain RoPE, to hold the engine to
# (tests/t246_page.mjs): the perplexity of the same texts with the same ids ([BOS] + the tokenizer's), and greedy text
# from the same prompts in the model's own template.
#
# The 4B's weights are 8 GB in float16 and would be 16 GB in float32, more than a runner has. So the model is loaded
# in float16 and computed in float32 a layer at a time: a layer is widened as the forward pass enters it and put back
# as it leaves (float16 to float32 is exact, so this is the float32 model's arithmetic), the embedding (the
# classifier, tied) and the last norm are widened once.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t246_reference.py <directory for the download> <1.7B | 4B> <revision> [--greedy <tokens>] <text file>:<tokens> ...
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

sys.path.insert(0, "public")
import llama2_numpy as N  # noqa: E402

BOS = 151643
PROMPTS = ["これからの流行りを3つ挙げてください。", "日本で一番高い山はどこですか。理由も説明してください。",
           "What is the capital of Japan? Answer in two sentences.", "富士山について、三つの文で説明してください。",
           "次の文を英語に訳してください。「今日は天気がいいので、散歩に行きます。」"]
started = time.time()


def say(*parts):
    print(f"T246REF [{time.time() - started:6.0f} s]", *parts, flush=True)


arguments = sys.argv[1:]
greedy = 0
if "--greedy" in arguments:
    at = arguments.index("--greedy")
    greedy = int(arguments[at + 1])
    del arguments[at:at + 2]
where, size, revision, *passages = arguments
repo = f"prism-ml/Ternary-Bonsai-{size}-unpacked"
directory = Path(snapshot_download(repo, revision=revision, local_dir=Path(where) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
config = json.loads((directory / "config.json").read_text())
say(f"{repo}@{revision}: fetched")

texts = []
for passage in passages:
    path, count = passage.rsplit(":", 1)
    content = Path(path).read_text()
    ids = [BOS] + tokenizer(content, add_special_tokens=False)["input_ids"][:int(count)]
    texts.append((Path(path).name, ids))
    say(f"{Path(path).name} (sha256 {hashlib.sha256(content.encode()).hexdigest()[:12]}): {len(ids) - 1} tokens after the BOS, "
        f"the first {ids[1:9]}, their sum {sum(ids)}")

model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float16, low_cpu_mem_usage=True)
model.eval()
model.model.norm.float()
model.lm_head.float()  # (where the embedding is the classifier's weight, it is widened with it)
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
say(f"loaded in float16: rope_type {rotary.rope_type}, attention_scaling {float(rotary.attention_scaling):.9f}, inv_freq {rotary.inv_freq.dtype}, "
    f"the embedding {model.model.embed_tokens.weight.dtype}, tied {model.lm_head.weight is model.model.embed_tokens.weight}")

# the readings: transformers' own yarn, computed here in float32 whatever the model's dtype made of its buffer, and plain
yarn_inv, yarn_scale = ROPE_INIT_FUNCTIONS["yarn"](model.config, "cpu")
yarn_inv = yarn_inv.float()
head, theta, scaling = config["head_dim"], config["rope_theta"], config["rope_scaling"]
plain_inv = 1.0 / (theta ** (torch.arange(0, head, 2, dtype=torch.float) / head))
mine = N.rope_frequencies(head, theta, scaling)
say(f"yarn: transformers' factor {float(yarn_scale):.9f} against the engine's magnitude {N.rope_magnitude(scaling):.9f}; inv_freq against the "
    f"engine's rope_frequencies, the largest relative difference {float(np.max(np.abs(yarn_inv.double().numpy() / mine - 1))):.2e}; "
    f"against the buffer the model was loaded with {float((rotary.inv_freq.float() - yarn_inv).abs().max()):.2e}; "
    f"pairs that turn as the plain ones {int((yarn_inv == plain_inv).sum())}, the slowest's ratio {float(plain_inv[-1] / yarn_inv[-1]):.4f}")
READINGS = {"yarn (config.json)": (yarn_inv, float(yarn_scale)), "plain RoPE": (plain_inv, 1.0)}


def reading(name):
    inv, scale = READINGS[name]
    rotary.inv_freq = inv.clone()
    rotary.attention_scaling = scale


def losses(ids):
    """The loss of every token of ids after the first, one forward pass; the classifier a stretch at a time."""
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, :-1]
        out = []
        for start in range(0, hidden.shape[0], 256):
            logits = model.lm_head(hidden[start:start + 256].float()).double()
            out.append(torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1 + start:1 + start + logits.shape[0]]), reduction="none"))
    return torch.cat(out)


for name in READINGS:
    reading(name)
    for path, ids in texts:
        began = time.time()
        loss = losses(ids)
        prefixes = ", ".join(f"{count}: {math.exp(float(loss[:count].mean())):.4f}" for count in (256, 512, 1024, 1536) if count < len(loss))
        say(f"PPLREF {path} {name}: {math.exp(float(loss.mean())):.4f} over {len(loss)} targets ({time.time() - began:.0f} s){'; prefixes ' + prefixes if prefixes else ''}")

if greedy:
    for name in READINGS:
        reading(name)
        for prompt in PROMPTS[:3]:
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            ids = [BOS] + tokenizer(text, add_special_tokens=False)["input_ids"]
            inputs = torch.tensor([ids])
            began = time.time()
            with torch.no_grad():
                out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=greedy, do_sample=False)[0, len(ids):]
            # the engine stops at the BOS token and at <|im_end|>
            stops = [i for i, t in enumerate(out.tolist()) if t in (BOS, 151645)]
            written = tokenizer.decode(out[:stops[0]] if stops else out)
            say(f"GREEDYREF {name} ({time.time() - began:.0f} s, {len(ids)} ids) {json.dumps(prompt, ensure_ascii=False)}: {json.dumps(written, ensure_ascii=False)}")
say("done")
