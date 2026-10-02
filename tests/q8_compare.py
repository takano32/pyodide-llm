# q8_compare.py (the review of T245 and T247, a probe for CI, not for main): what tests/q8_lab.py and tests/q8_engine.py
# saved, set against each other: the engine on the 4B's GGUF (E1), transformers on the original (T0) and transformers
# on the original's weights rounded to Q8_0 in Hugging Face's order (T1), at the 96 positions of the sentence and over the
# first tokens of the text.
#
#   python tests/q8_compare.py <lab prefix> <engine prefix>
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q8_lab import compare_logits, say  # noqa: E402

lab, engine = sys.argv[1], sys.argv[2]
t0, t1 = (np.load(f"{lab}-{v}-sentence-logits.npy") for v in ("v0", "v1"))
e1 = np.load(f"{engine}-sentence-logits.npy")
n0, n1 = (np.load(f"{lab}-{v}-nll.npy") for v in ("v0", "v1"))
ne = np.load(f"{engine}-nll.npy")
say(f"{len(e1)} positions of the sentence; {len(ne)} tokens of the text")
compare_logits("E1 (the engine on the GGUF) against T0 (transformers, the original)", t0[:len(e1)], e1)
compare_logits("E1 (the engine on the GGUF) against T1 (transformers, the original rounded to Q8_0)", t1[:len(e1)], e1)
compare_logits("T1 against T0 (what Q8_0 alone does)", t0[:len(e1)], t1[:len(e1)])
n = min(len(n0), len(ne))
for label, a, b in (("E1 against T1", ne[:n], n1[:n]), ("E1 against T0", ne[:n], n0[:n]), ("T1 against T0", n1[:n], n0[:n])):
    d = a - b
    say(f"negative log likelihood of the first {n} tokens, {label}: the mean change {d.mean():+.4f} nats a token "
        f"(perplexity {100 * (np.exp(d.mean()) - 1):+.2f}%), the largest {np.abs(d).max():.3f}, the middle one's {np.median(np.abs(d)):.4f}")
say(f"perplexity of the first {n}: E1 {np.exp(ne[:n].mean()):.3f}, T1 {np.exp(n1[:n].mean()):.3f}, T0 {np.exp(n0[:n].mean()):.3f}")
