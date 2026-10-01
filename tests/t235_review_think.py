# t235_review_think.py (T235's review, a probe for CI, not for main): does Ternary-Bonsai 1.7B think when its answer is begun
# with an open thought, as Qwen3's thinking form begins it (<|im_start|>assistant\n<think>\n)? The original's template always
# begins the answer with an empty thought, so this form is not what it ships with. Greedy, 300 tokens, a BOS first, an
# attention mask of ones, config.json's yarn.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t235_review_think.py <directory for the download>
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, REVISION, BOS = "prism-ml/Ternary-Bonsai-1.7B-unpacked", "3aca840085293d026ce6f6b80fafdae937fd2eeb", 151643
PROMPTS = ["日本で一番高い山はどこですか。理由も説明してください。", "1 + 1 はいくつですか。理由も書いてください。",
           "What is 17 times 3? Think it through."]
started = time.time()
say = lambda *parts: print(f"[{time.time() - started:5.0f} s]", *parts, flush=True)

directory = Path(snapshot_download(REPO, revision=REVISION, local_dir=Path(sys.argv[1]) / "original"))
tokenizer = AutoTokenizer.from_pretrained(directory)
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
for prompt in PROMPTS:
    shut = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    open_thought = shut[:-len("<think>\n\n</think>\n\n")] + "<think>\n"
    assert open_thought.endswith("<|im_start|>assistant\n<think>\n"), open_thought[-60:]
    ids = [BOS] + tokenizer(open_thought, add_special_tokens=False)["input_ids"]
    inputs = torch.tensor([ids])
    out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=300, do_sample=False)[0, len(ids):]
    text = tokenizer.decode(out)
    closed = "</think>" in text
    say(f"{prompt!r}: closes its thought {closed}, {len(out)} tokens: {json.dumps(text, ensure_ascii=False)}")
say("done")
