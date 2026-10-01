# t236_handwritten.py (the review of T236, a probe for CI): the first token against answers that no model wrote. 24 chat
# turns of tests/t246_chat.jsonl (a prompt and an answer written by hand, 12 Japanese and 12 English), in the form that
# answers at once, scored by transformers (float32, the original's weights) as the mean negative log likelihood of the
# answer's tokens and of <|im_end|>, under A (the real template's ids, what the page sends since T236) and B
# (<|endoftext|> in front, what it sent before): a neutral text, which A cannot have an advantage on as the real model's
# own greedy answers (tests/t236_bos.py) have.
#
#   python tests/t236_handwritten.py <the original's directory> <the jsonl>
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

# <the original's directory | owner/repository@revision> <the jsonl> [<the page's BOS> <the end of a turn>]: the defaults are
# Qwen3.5's (248044, 248046); for a Ternary Bonsai (a Qwen3, which the list begins with <|endoftext|>) 151643 and 151645
target, pairs = sys.argv[1], Path(sys.argv[2])
ENDOFTEXT, IM_END = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (248044, 248046)
if "@" in target:
    from huggingface_hub import snapshot_download
    repo, revision = target.split("@")
    directory = Path(snapshot_download(repo, revision=revision, local_dir=Path("/mnt/t236/dl") / repo.replace("/", "--")))
else:
    directory = Path(target)
started = time.time()
say = lambda *parts: print(f"T236HAND [{time.time() - started:6.0f} s]", *parts, flush=True)
tokenizer = AutoTokenizer.from_pretrained(directory)
qwen35 = json.loads((directory / "config.json").read_text()).get("model_type", "").startswith("qwen3_5")
model = (Qwen3_5ForConditionalGeneration if qwen35 else AutoModelForCausalLM).from_pretrained(str(directory), dtype=torch.float32).eval()
say(f"{target}: {type(model).__name__}, the page's BOS {ENDOFTEXT}, the end of a turn {IM_END}")


def nll(prefix, answer):
    ids = torch.tensor([prefix + answer])
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(answer) + 1).logits[0, :-1].double()
    return float(torch.nn.functional.cross_entropy(logits, torch.tensor(answer), reduction="sum"))


rows = []
for number, line in enumerate(pairs.read_text().splitlines()):
    pair = json.loads(line)
    for form in ("at once", "thinking"):
        real = tokenizer.apply_chat_template([{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True, tokenize=True,
                                             enable_thinking=(form == "thinking"))
        head = [int(t) for t in (real["input_ids"] if hasattr(real, "keys") else real)]
        # (the thinking form: the answer follows an empty thought here, as nothing handwritten has a thought; so only the
        # form that answers at once is a fair reading, and the other is left out)
        if form == "thinking":
            continue
        answer = tokenizer(pair["answer"], add_special_tokens=False)["input_ids"] + [IM_END]
        a, b = nll(head, answer), nll([ENDOFTEXT] + head, answer)
        language = "ja" if any(ord(c) > 0x2E80 for c in pair["prompt"]) else "en"
        rows.append({"i": number, "language": language, "tokens": len(answer), "nll_A": a, "nll_B": b})
        say("ROW", json.dumps(rows[-1]))
for language in ("ja", "en", None):
    part = [r for r in rows if language is None or r["language"] == language]
    n = sum(r["tokens"] for r in part)
    a, b = sum(r["nll_A"] for r in part) / n, sum(r["nll_B"] for r in part) / n
    worse = sum(1 for r in part if r["nll_B"] / r["tokens"] > r["nll_A"] / r["tokens"])
    diffs = np.array([r["nll_B"] / r["tokens"] - r["nll_A"] / r["tokens"] for r in part])
    say(f"TOTAL {language or 'all'} ({len(part)} answers, {n} tokens): mean nll A {a:.4f}, B {b:.4f} (B - A = {b - a:+.4f}); perplexity A {math.exp(a):.3f}, "
        f"B {math.exp(b):.3f} ({(math.exp(b - a) - 1) * 100:+.2f}%); B is worse on {worse} of {len(part)}; per answer B - A: mean {diffs.mean():+.4f}, sd {diffs.std(ddof=1):.4f}")
say("done")
