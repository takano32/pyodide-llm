# t236_klother.py (the review of T236, a probe for CI): what the page's BOS in front of the real chat ids does to the
# model's own answers, for a model that is not a Qwen3.5 (tests/t236_bos.py's chat part, without the engine): the real
# model writes greedily from the real template's ids (A); the same answer is scored with the BOS in front (B), and the
# distance of the next-token distributions along it is the Kullback-Leibler distance, and how often the most likely
# token is the same.
#
#   python tests/t236_klother.py <owner/repository@revision> <BOS> <end of a turn> [<tokens = 40>]
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

target, bos, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
count = int(sys.argv[4]) if len(sys.argv) > 4 else 40
repo, revision = target.split("@")
directory = Path(snapshot_download(repo, revision=revision, local_dir=Path("/mnt/t236/dl") / repo.replace("/", "--")))
started = time.time()
say = lambda *parts: print(f"T236KL [{time.time() - started:6.0f} s]", *parts, flush=True)
tokenizer = AutoTokenizer.from_pretrained(directory)
model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.float32).eval()
PROMPTS = [json.loads(line)["prompt"] for line in Path("tests/t246_chat.jsonl").read_text().splitlines()]
say(f"{target}: {type(model).__name__}, the BOS {bos}, {len(PROMPTS)} prompts, {count} tokens")


def log_softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def tail(prefix, cont):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prefix + cont]), use_cache=False, logits_to_keep=len(cont) + 1)
    return log_softmax(out.logits[0, :-1].double().numpy())


rows = []
for index, prompt in enumerate(PROMPTS):
    real = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    head = [int(t) for t in (real["input_ids"] if hasattr(real, "keys") else real)]
    inputs = torch.tensor([head])
    with torch.no_grad():
        out = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=count, do_sample=False,
                             eos_token_id=[bos, end], pad_token_id=bos)
    answer = out[0, len(head):].tolist()
    a, b = tail(head, answer), tail([bos] + head, answer)
    row = {"i": index, "tokens": len(answer), "nll_A": float(-a[np.arange(len(answer)), answer].mean()),
           "nll_B": float(-b[np.arange(len(answer)), answer].mean()),
           "kl": float((np.exp(a) * (a - b)).sum(axis=1).mean()), "top1": float((a.argmax(1) == b.argmax(1)).mean())}
    rows.append(row)
    say("ROW", json.dumps(row))
mean = lambda key: sum(r[key] for r in rows) / len(rows)
sd = lambda key: float(np.std([r[key] for r in rows], ddof=1) / math.sqrt(len(rows)))
say(f"TOTAL {target} ({len(rows)} prompts, the model's own greedy answers): nll A {mean('nll_A'):.4f}, B {mean('nll_B'):.4f} (the answer scored worse "
    f"under B on {sum(1 for r in rows if r['nll_B'] > r['nll_A'])} of {len(rows)}); KL(A || B) per token {mean('kl'):.4f} (se {sd('kl'):.4f}); "
    f"the most likely token the same at {mean('top1') * 100:.1f}% of the positions")
