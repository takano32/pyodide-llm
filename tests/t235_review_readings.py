# t235_review_readings.py (T235's review, a probe for CI, not for main): what transformers writes greedily (a BOS first,
# an attention mask of ones: as the engine writes) for 64 tokens from a few prompts in the model's template, under the
# three readings of Ternary-Bonsai's RoPE: config.json's yarn, yarn's angles with no longer cos and sin, and plain RoPE.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t235_review_readings.py <directory for the download>
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION, BOS = "prism-ml/Ternary-Bonsai-1.7B-unpacked", "3aca840085293d026ce6f6b80fafdae937fd2eeb", 151643
PROMPTS = ["これからの流行りを3つ挙げてください。", "日本で一番高い山はどこですか。理由も説明してください。",
           "What is the capital of Japan? Answer in two sentences.", "富士山について、三つの文で説明してください。",
           "日本の首都はどこですか。", "水の化学式を教えてください。", "夏目漱石の代表作を三つ挙げてください。",
           "1 + 1 はいくつですか。理由も書いてください。", "Who wrote the novel Botchan? Answer in one sentence."]
NEW = 64
started = time.time()
say = lambda *parts: print(f"[{time.time() - started:5.0f} s]", *parts, flush=True)

directory = Path(snapshot_download(REPO, revision=REVISION, local_dir=Path(sys.argv[1]) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
config = json.loads((directory / "config.json").read_text())
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
rotary = model.model.rotary_emb
yarn_inv, yarn_scale = rotary.inv_freq.clone(), float(rotary.attention_scaling)
plain_inv = 1.0 / (config["rope_theta"] ** (torch.arange(0, config["head_dim"], 2, dtype=torch.float) / config["head_dim"]))
READINGS = {"yarn": (yarn_inv, yarn_scale), "angles": (yarn_inv, 1.0), "plain": (plain_inv, 1.0)}
written = {}
for name, (inv, scale) in READINGS.items():
    rotary.inv_freq, rotary.attention_scaling = inv.clone(), scale
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        ids = [BOS] + tokenizer(text, add_special_tokens=False)["input_ids"]
        inputs = torch.tensor([ids])
        out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=NEW, do_sample=False)[0, len(ids):]
        stops = [i for i, t in enumerate(out.tolist()) if t in (BOS, 151645)]
        written[(name, prompt)] = tokenizer.decode(out[:stops[0]] if stops else out)
        say(f"{name} {prompt!r}: {json.dumps(written[(name, prompt)], ensure_ascii=False)}")
for prompt in PROMPTS:
    say(f"{prompt!r}: yarn == plain {written[('yarn', prompt)] == written[('plain', prompt)]}, yarn == angles {written[('yarn', prompt)] == written[('angles', prompt)]}")
say("done")
