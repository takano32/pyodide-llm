# t236_plainbos.py (the review of T236, a probe for CI): which first token costs a Qwen3.5 0.8B the least in front of plain
# text (what ?hf=Qwen/Qwen3.5-0.8B has, where the page cannot read a chat format and begins the text with the
# converter's BOS, <|endoftext|>). transformers, float32, the original's weights; 4 texts (2 English, 2 Japanese) of
# 512 tokens; the perplexity over the same targets (tokens 1 to 511 of the text) with each candidate in front, and by
# position.
#
#   python tests/t236_plainbos.py <the original's directory> <text file> ...
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers
import torch
from transformers import Qwen3_5ForConditionalGeneration

directory, *texts = sys.argv[1:]
directory = Path(directory)
started = time.time()
say = lambda *parts: print(f"T236PLAINBOS [{time.time() - started:6.0f} s]", *parts, flush=True)
tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
model = Qwen3_5ForConditionalGeneration.from_pretrained(str(directory), dtype=torch.float32).eval()
newline = tokenizer.encode("\n", add_special_tokens=False).ids
CANDIDATES = {"none": [], "<|endoftext|>": [248044], "<|im_start|>": [248045], "<|im_end|>": [248046], "newline": newline,
              "<think>": [248068], "</think>": [248069], "<|im_start|> + 'user' + newline": [248045] + tokenizer.encode("user\n", add_special_tokens=False).ids}
say("candidates", {name: ids for name, ids in CANDIDATES.items()})


def log_softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


for path in texts:
    text = Path(path).read_text()
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:512]
    targets = ids[1:]
    for name, prefix in CANDIDATES.items():
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prefix + ids[:-1]]), use_cache=False).logits[0].float().numpy()
        logits = logits[len(prefix):]
        losses = np.empty(len(targets))
        for start in range(0, len(targets), 128):
            lp = log_softmax(logits[start:start + 128].astype(np.float64))
            losses[start:start + 128] = -lp[np.arange(lp.shape[0]), targets[start:start + 128]]
        buckets = [(0, 15), (15, 63), (63, 255), (255, len(targets))]
        parts = ", ".join(f"[{a + 1}:{b + 1}] {math.exp(losses[a:b].mean()):.2f}" for a, b in buckets if b > a)
        say(f"{Path(path).name} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:10]}) first token {name}: perplexity {math.exp(losses.mean()):.3f}; by position {parts}")
        del logits
say("done")
