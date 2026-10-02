# chat_nll.py (the review of T236): what the first token of the page's text does to a model in chat form. Torch and
# transformers are needed, which the page and the deploy never use: for CI (tests.yml's extra=), never on the development
# machine.
#
# The page begins every text with a BOS, the converter's, in front of the ids the model's own template makes (the real
# tokenizers of most models begin a chat with none, or with the very token the template writes). T131 measured plain
# text and found +-3%; T236 found a Qwen3.5 0.8B (linear-attention layers) 46% and 95% worse on 299 tokens of Wikipedia
# with <|endoftext|> in front, and begins its entries with the template's own first token instead, so that the page
# sends the ids the real template writes. A is that, B the BOS in front of them (what the page sent a Qwen3.5 before
# T236 and sends the others). Two readings, which answer two questions:
#
#   1. Fluency, on text no model wrote: 24 chat turns written by hand (tests/fixtures/chat-answers.jsonl: 12 Japanese,
#      12 English, a prompt and a short answer each), the mean negative log likelihood of the answer's tokens and of
#      the end of its turn under A and under B. Only the form that answers at once (no hand-written answer has a thought).
#   2. Fidelity, on the model's own answers: the model writes greedily from A (40 tokens), and the same answer is scored
#      under B: how far the next-token distributions of the two are along it (the Kullback-Leibler distance, nats a
#      token), how often their most likely token is the same, and under which the answer is the likelier. A Qwen3.5 also
#      in its thinking form. What B changes of what the model writes shows here and not in 1: a distribution moves while
#      the likelihood of an independent text does not.
#
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers safetensors huggingface_hub
#   python3 tests/chat_nll.py <the original's directory | owner/repository@revision> [<BOS> <end of a turn>] [--stored]
#                             [--thinking-arg] [--fidelity=N]
#   (the defaults are Qwen3.5's: 248044 and 248046; a Qwen3: 151643 and 151645; Granite 4.2's <s> and <|im_end|>: 100283 and 100257)
# --stored (T253's review): the weights are held as the original stores them (bfloat16) and every product is float32
# (reference_llama.float32_arithmetic, the embeddings widened before they go in): Granite 4.2 3B is 14.6 GB as float32,
# more than a runner of 16 GB has. --thinking-arg: the template is given enable_thinking (Qwen3.5 always is): False for the
# answers written by hand and for the form that answers at once, True for the thinking form (Granite 4.2, MiniCPM5).
# --fidelity=N: the model's own answers for the first N prompts only (a greedy token of a 3B is a second or two here).
#
# Qwen3.5 0.8B (CI's x86-64 runner, run 36935583881): 1. fluency A 2.0572, B 2.0494 nats a token over the 24 answers (B
# 0.77% lower in perplexity, worse on 12 of 24, the pair's difference -0.0007 with a standard error of 0.027); 2. fidelity
# KL(A || B) 0.115 a token at once (standard error 0.010) and 0.178 thinking (0.017), the most likely token the same at
# 87.0% and 91.0% of the positions, the answer likelier under A on 23 of 24 and 24 of 24 prompts. Ternary Bonsai 1.7B (a
# Qwen3, 151643 and 151645; run 36936367926): 1. B 4.4% better than A (worse on 3 of 24; standard error 0.010); 2. KL 0.0080
# (standard error 0.0010), the most likely token the same at 96.9%. TODO.md's T236 and T246 have the rest.
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

PAIRS = Path(__file__).resolve().parent / "fixtures" / "chat-answers.jsonl"
TOKENS = 40
flags = [argument for argument in sys.argv[1:] if argument.startswith("--")]
arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
target = arguments[0]
bos, end = (int(arguments[1]), int(arguments[2])) if len(arguments) > 2 else (248044, 248046)
stored = "--stored" in flags
fidelity = int(next((flag.split("=", 1)[1] for flag in flags if flag.startswith("--fidelity=")), 1 << 30))
if "@" in target:
    from huggingface_hub import snapshot_download
    repo, revision = target.split("@")
    directory = Path(snapshot_download(repo, revision=revision))
else:
    directory = Path(target)
started = time.time()
say = lambda *parts: print(f"chat_nll [{time.time() - started:5.0f} s]", *parts, flush=True)
tokenizer = AutoTokenizer.from_pretrained(directory)
configured = json.loads((directory / "config.json").read_text())
hybrid = configured.get("model_type", "").startswith("qwen3_5")
thinks = hybrid or "--thinking-arg" in flags
if stored:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reference_llama
    reference_llama.float32_arithmetic()
    kept = getattr(torch, configured.get("dtype") or configured.get("torch_dtype") or "float32")
else:
    kept = torch.float32
model = (Qwen3_5ForConditionalGeneration if hybrid else AutoModelForCausalLM).from_pretrained(str(directory), dtype=kept).eval()
say(f"{target}: {type(model).__name__}, the BOS {bos}, the end of a turn {end}, weights held as {kept}"
    f"{', every product in float32' if stored else ''}{', enable_thinking said' if thinks else ''}")
pairs = [json.loads(line) for line in PAIRS.read_text().splitlines()]


def head_of(prompt, thinking):
    real = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
                                         **({"enable_thinking": thinking} if thinks else {}))
    return [int(token) for token in (real["input_ids"] if hasattr(real, "keys") else real)]


def log_probs(prefix, answer):
    """the log probabilities (len(answer), vocabulary) of the next token at every position of the answer"""
    with torch.no_grad():
        ids = torch.tensor([prefix + answer])
        # (stored: the embeddings are float32 before the first layer, as reference_llama.py feeds them, or the norms of
        # a bfloat16 model round their output to bfloat16)
        given = {"inputs_embeds": model.get_input_embeddings()(ids).to(torch.float32)} if stored else {"input_ids": ids}
        logits = model(**given, use_cache=False, logits_to_keep=len(answer) + 1).logits[0, :-1].double().numpy()
    logits -= logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


# ---- 1. fluency, on the answers written by hand
rows = []
for number, pair in enumerate(pairs):
    head = head_of(pair["prompt"], False)
    answer = tokenizer(pair["answer"], add_special_tokens=False)["input_ids"] + [end]
    nll = lambda prefix: float(-log_probs(prefix, answer)[np.arange(len(answer)), answer].sum())
    rows.append({"language": "ja" if any(ord(c) > 0x2E80 for c in pair["prompt"]) else "en", "tokens": len(answer),
                 "A": nll(head), "B": nll([bos] + head)})
for language in ("ja", "en", None):
    part = [row for row in rows if language is None or row["language"] == language]
    tokens = sum(row["tokens"] for row in part)
    a, b = sum(row["A"] for row in part) / tokens, sum(row["B"] for row in part) / tokens
    differences = np.array([row["B"] / row["tokens"] - row["A"] / row["tokens"] for row in part])
    say(f"fluency, {language or 'all'} ({len(part)} answers written by hand, {tokens} tokens): nll A {a:.4f}, B {b:.4f} (B - A {b - a:+.4f}, "
        f"perplexity {(math.exp(b - a) - 1) * 100:+.2f}%); B is worse on {int((differences > 0).sum())} of {len(part)}; a pair's B - A "
        f"{differences.mean():+.4f}, standard error {differences.std(ddof=1) / math.sqrt(len(part)):.4f}")

# ---- 2. fidelity, on the model's own answers
for form, thinking in (("at once", False), ("thinking", True)) if thinks else (("at once", False),):
    own = []
    # (the prompts are the pairs' in order, 12 Japanese and then 12 English: --fidelity=N takes every other one of them
    # up to N, so that both languages are in it)
    for pair in (pairs if fidelity >= len(pairs) else pairs[::max(1, len(pairs) // fidelity)][:fidelity]):
        head = head_of(pair["prompt"], thinking)
        if stored:
            written = reference_llama.greedy(model, head, TOKENS, {bos, end})
        else:
            inputs = torch.tensor([head])
            with torch.no_grad():
                written = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=TOKENS, do_sample=False,
                                         eos_token_id=[bos, end], pad_token_id=bos)[0, len(head):].tolist()
        a, b = log_probs(head, written), log_probs([bos] + head, written)
        own.append({"A": float(-a[np.arange(len(written)), written].mean()), "B": float(-b[np.arange(len(written)), written].mean()),
                    "kl": float((np.exp(a) * (a - b)).sum(axis=1).mean()), "top1": float((a.argmax(1) == b.argmax(1)).mean())})
    kl = np.array([row["kl"] for row in own])
    say(f"fidelity, {form} ({len(own)} prompts, the model's own greedy answers of {TOKENS} tokens): nll of the answer under A "
        f"{np.mean([r['A'] for r in own]):.4f}, under B {np.mean([r['B'] for r in own]):.4f} (worse under B on "
        f"{sum(1 for r in own if r['B'] > r['A'])} of {len(own)}); KL(A || B) {kl.mean():.4f} nats a token (standard error "
        f"{kl.std(ddof=1) / math.sqrt(len(kl)):.4f}); the most likely token the same at {np.mean([r['top1'] for r in own]) * 100:.1f}% of the positions")
