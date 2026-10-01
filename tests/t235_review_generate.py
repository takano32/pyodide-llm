# t235_review_generate.py (T235's review, a probe for CI, not for main): what the engine (the page's converter and engine,
# float32, NumPy, a BOS first, yarn's tables) and transformers (the original's safetensors in float32, the same ids with a
# BOS, an attention mask of ones) write greedily for 64 tokens from several prompts in the model's own template; where they
# part, with the logits at the first token that differs. One download of each, run in turn.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers==4.57.6
#   python3 tests/t235_review_generate.py <directory for the downloads>
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
sys.path.insert(0, "public")
import fixed_outputs  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

BOS = 151643
PROMPTS = ["これからの流行りを3つ挙げてください。", "日本で一番高い山はどこですか。理由も説明してください。",
           "What is the capital of Japan? Answer in two sentences.", "富士山について、三つの文で説明してください。",
           "次の文を英語に訳してください。「今日は天気がいいので、散歩に行きます。」"]
NEW = 64
started = time.time()
say = lambda *parts: print(f"[{time.time() - started:5.0f} s]", *parts, flush=True)

directory = Path(sys.argv[1])
entry = next(e for e in fixed_outputs.entries() if e["id"] == "hf-ternary-bonsai-1.7b")
conversion, checkpoint = fixed_outputs.converted(entry, directory)
options = {**conversion.options, **entry.get("options", {})}
template = entry.get("template") or options.pop("template", None)
options.pop("template", None)
say(f"the engine's options: { {k: v for k, v in options.items() if k not in ('specials',)} }")
llama = Llama(np.memmap(checkpoint, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None, **options)
engine = {}
for prompt in PROMPTS:
    text = template.replace("{prompt:trim}", "{prompt}").replace("{prompt}", prompt)
    steps = len(llama.tokenizer.encode(text, llama.specials)) + NEW
    began = time.time()
    engine[prompt] = "".join(llama.generate(text, steps=steps, temperature=0.0, echo=False))
    say(f"engine ({time.time() - began:.0f} s) {prompt!r}: {json.dumps(engine[prompt], ensure_ascii=False)}")
del llama
gc.collect()
checkpoint.unlink()

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO, REVISION = "prism-ml/Ternary-Bonsai-1.7B-unpacked", "3aca840085293d026ce6f6b80fafdae937fd2eeb"
original = Path(snapshot_download(REPO, revision=REVISION, local_dir=directory / "original"))
tokenizer = AutoTokenizer.from_pretrained(original)
model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.float32)
model.eval()
for prompt in PROMPTS:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = [BOS] + tokenizer(text, add_special_tokens=False)["input_ids"]
    inputs = torch.tensor([ids])
    began = time.time()
    out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=NEW, do_sample=False)[0, len(ids):]
    # the engine stops at the BOS token and at <|im_end|>; so does transformers (eos <|im_end|>), but not at <|endoftext|>
    stops = [i for i, t in enumerate(out.tolist()) if t in (BOS, 151645)]
    written = tokenizer.decode(out[:stops[0]] if stops else out)
    same = written == engine[prompt]
    say(f"transformers ({time.time() - began:.0f} s) {prompt!r}: {json.dumps(written, ensure_ascii=False)}  -> {'SAME' if same else 'DIFFERENT'}")
    if not same:
        at = next((i for i, (a, b) in enumerate(zip(written, engine[prompt])) if a != b), min(len(written), len(engine[prompt])))
        say(f"  first difference at character {at}: transformers {written[at:at + 12]!r}, engine {engine[prompt][at:at + 12]!r}")
        # the logits there: the text up to the difference, as ids, and the top four after it
        prefix = ids + tokenizer(written[:at], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = model(torch.tensor([prefix])).logits[0, -1].double()
        best = torch.topk(logits, 4)
        say(f"  transformers' four most likely there: {[[tokenizer.decode([int(i)]), round(float(v), 3)] for v, i in zip(best.values, best.indices)]}")
say("done")
