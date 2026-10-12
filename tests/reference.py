# reference.py
# T357.4: what the tools that hold the engine to transformers itself share (tests/reference_llama.py,
# tests/reference_qwen35.py, tests/reference_lfm2.py: they run in CI, where torch is). Nothing here needs torch.
import numpy as np


def differences(ours, theirs):
    """(largest difference, mean difference, positions whose most likely token is the same, the largest gap between
    the two most likely tokens of theirs where the most likely token differs)"""
    ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    gap = np.abs(ours - theirs)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    top = np.sort(theirs, axis=1)[:, -2:]
    margins = (top[:, 1] - top[:, 0])[~same]
    return float(gap.max()), float(gap.mean()), int(same.sum()), float(margins.max()) if margins.size else 0.0


def relative(a, b):
    """how far a is from b, in b's own size: the norm of the difference over the norm of b"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
