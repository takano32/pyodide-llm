# q8_page_compare.py (the review of T247, a probe for CI, not for main): transformers' float32 logits of the sentence (a layer at a
# time: tests/layerwise_qwen35.py) against the page's forward pass on the list's GGUF as int8 (tests/q8_page_logits.mjs), with
# 7-bit and 8-bit activations: how often the most likely token is the same, how far the logits are, KL.
#
#   python tests/q8_page_compare.py <layerwise prefix> <page prefix>
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q8_lab import compare_logits, say  # noqa: E402

layerwise, page = sys.argv[1], sys.argv[2]
theirs = np.load(f"{layerwise}-sentence-logits.npy")
for label in ("7-bit", "8-bit"):
    file = Path(f"{page}-{label}-logits.f32")
    if not file.exists():
        continue
    ours = np.fromfile(file, dtype=np.float32).reshape(-1, theirs.shape[1])
    compare_logits(f"the page's forward pass with {label} activations against transformers' (float32, the original)", theirs[:len(ours)], ours)
