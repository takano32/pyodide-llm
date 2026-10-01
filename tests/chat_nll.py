# chat_nll.py (the review of T236): what the first token of the page's text does to a model, on chat turns that no model
# wrote. Torch and transformers are needed, which the page and the deploy never use: for CI (tests.yml's extra=), never
# on the development machine.
#
# The page begins every text with a BOS, the converter's, in front of the ids the model's own template makes (the
# real tokenizers of most models begin a chat with none, or with the very token the template writes). T131 measured
# plain text and found ±3%; T236 found a Qwen3.5 0.8B (linear-attention layers) 46% and 95% worse on 299 tokens of
# Wikipedia (English, Japanese) with <|endoftext|> in front, and begins its entries with the template's own first token
# instead. This scores 24 chat turns written by hand (tests/fixtures/chat-answers.jsonl: 12 Japanese, 12 English, a
# prompt and a short answer each, by no model, so that neither way of beginning has an advantage on them as it has on
# the real model's own greedy answers) as the mean negative log likelihood of the answer's tokens and of the end of its
# turn: A, the real template's ids (what a Qwen3.5 entry sends), and B, the BOS in front of them (what the page sent
# a Qwen3.5 before T236, and sends the others). The form that answers at once is the one scored (the template's empty
# thought; no hand-written answer has a thought).
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers safetensors huggingface_hub
#   python3 tests/chat_nll.py <the original's directory | owner/repository@revision> [<BOS> <end of a turn>]
#   (the defaults are Qwen3.5's: 248044 and 248046; a Qwen3: 151643 and 151645)
#
# Qwen3.5 0.8B (CI's x86-64 runner, 24 answers, 714 tokens): A 2.0572, B 2.0494 nats a token, B 0.77% lower in
# perplexity, worse on 12 of the 24 answers, the difference of a pair 0.0007 below zero with a standard error of 0.027:
# in chat form the first token is no measurable cost (TODO.md's T236).
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

PAIRS = Path(__file__).resolve().parent / "fixtures" / "chat-answers.jsonl"
target = sys.argv[1]
bos, end = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (248044, 248046)
if "@" in target:
    from huggingface_hub import snapshot_download
    repo, revision = target.split("@")
    directory = Path(snapshot_download(repo, revision=revision))
else:
    directory = Path(target)
started = time.time()
say = lambda *parts: print(f"chat_nll [{time.time() - started:5.0f} s]", *parts, flush=True)
tokenizer = AutoTokenizer.from_pretrained(directory)
hybrid = json.loads((directory / "config.json").read_text()).get("model_type", "").startswith("qwen3_5")
model = (Qwen3_5ForConditionalGeneration if hybrid else AutoModelForCausalLM).from_pretrained(str(directory), dtype=torch.float32).eval()
say(f"{target}: {type(model).__name__}, the BOS {bos}, the end of a turn {end}")


def nll(prefix, answer):
    """the summed negative log likelihood of the answer's tokens after the prefix"""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prefix + answer]), use_cache=False, logits_to_keep=len(answer) + 1).logits[0, :-1].double()
    return float(torch.nn.functional.cross_entropy(logits, torch.tensor(answer), reduction="sum"))


rows = []
for number, line in enumerate(PAIRS.read_text().splitlines()):
    pair = json.loads(line)
    real = tokenizer.apply_chat_template([{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True, tokenize=True,
                                         **({"enable_thinking": False} if hybrid else {}))
    head = [int(token) for token in (real["input_ids"] if hasattr(real, "keys") else real)]
    answer = tokenizer(pair["answer"], add_special_tokens=False)["input_ids"] + [end]
    rows.append({"i": number, "language": "ja" if any(ord(c) > 0x2E80 for c in pair["prompt"]) else "en", "tokens": len(answer),
                 "A": nll(head, answer), "B": nll([bos] + head, answer)})
for language in ("ja", "en", None):
    part = [row for row in rows if language is None or row["language"] == language]
    tokens = sum(row["tokens"] for row in part)
    a, b = sum(row["A"] for row in part) / tokens, sum(row["B"] for row in part) / tokens
    pairs = np.array([row["B"] / row["tokens"] - row["A"] / row["tokens"] for row in part])
    say(f"{language or 'all'} ({len(part)} answers, {tokens} tokens): nll A {a:.4f}, B {b:.4f} (B - A {b - a:+.4f}, perplexity "
        f"{(math.exp(b - a) - 1) * 100:+.2f}%); B is worse on {int((pairs > 0).sum())} of {len(part)}; a pair's B - A "
        f"{pairs.mean():+.4f}, standard error {pairs.std(ddof=1) / math.sqrt(len(part)):.4f}")
