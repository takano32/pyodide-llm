# t236_gates.py (T236's review, a probe for CI, not for main): what Q8_0 does to the two small matrices of a Qwen3.5's
# gates (in_proj_a and in_proj_b of every linear-attention layer: llama.cpp writes them Q8_0, 5.7e-3 off the original's
# bf16 values, where the page's own conversion of the safetensors keeps them in float32), measured on the logits.
# The NumPy engine, float32 everywhere (no quantized activations, no int8 kernels), the same text and BOS, five weights:
#   O  the original's (the safetensors, float32): the reference
#   G  the list's GGUF's (every matrix as Q8_0 rounds it, the gates too)
#   H  the original's with only the two gate matrices of G's: the gates' own share
#   H2 G's with the original's gate matrices: all Q8_0 but the gates (what the page's int8 conversion of the
#      original has, give or take the scales' float16 rounding)
# For G, H and H2 against O: the Kullback-Leibler distance of the next-token distributions (mean over the positions), how
# often the most likely token is O's, the perplexity of each. If the gates' share (H) is small beside what int8 matrices
# do anyway (H2), no row of the perplexity table is outside noise because of the gates.
#
#   python tests/t236_gates.py <o32 out> <g32 out> <tokens> <text file> ...
#   (the <out> of tests/perplexity_prepare.py, float32, of the original and of the list's GGUF)
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama  # noqa: E402

started = time.time()


def say(*parts):
    print(f"T236GATES [{time.time() - started:6.0f} s]", *parts, flush=True)


def load(out, mode="r"):
    options = json.loads(Path(f"{out}.json").read_text())
    return Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode=mode), Path(f"{out}.tokenizer.bin").read_bytes(), kernels=None, **options)


def log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    return logits - math.log(np.exp(logits).sum())


def main():
    o_out, g_out, count, *texts = sys.argv[1:]
    count = int(count)
    O, G = load(o_out), load(g_out)
    H, H2 = load(o_out), load(g_out)
    H.wa, H.wb = G.wa, G.wb  # the original's, with G's two gate matrices
    H2.wa, H2.wb = O.wa, O.wb  # G's, with the original's
    say(f"loaded: the gates differ from the original's by (relative, the norm of the difference over the norm) "
        f"in_proj_a {np.linalg.norm(G.wa - O.wa) / np.linalg.norm(O.wa):.3e}, in_proj_b {np.linalg.norm(G.wb - O.wb) / np.linalg.norm(O.wb):.3e}; "
        f"the first token of both: {O.bos} and {G.bos}")
    for path in texts:
        text = Path(path).read_text()
        ids = O.tokenizer.encode(text)[:count]
        assert ids == G.tokenizer.encode(text)[:count], "the two tokenizers differ"
        tokens = [O.bos] + ids
        began = time.time()
        reference = [log_softmax(O.forward(tokens[pos], pos)) for pos in range(len(tokens) - 1)]
        targets = tokens[1:]
        o_nll = -np.mean([lp[t] for lp, t in zip(reference, targets)])
        say(f"{Path(path).name}: {len(ids)} tokens, O (the original, float32) perplexity {math.exp(o_nll):.4f} ({time.time() - began:.0f} s)")
        for name, llama in (("G (all Q8_0, gates too)", G), ("H (the gates alone Q8_0)", H), ("H2 (all Q8_0 but the gates)", H2)):
            began = time.time()
            kl, same, nll, worst = [], 0, [], 0.0
            for pos in range(len(tokens) - 1):
                lp = log_softmax(llama.forward(tokens[pos], pos))
                p = np.exp(reference[pos])
                kl.append(float((p * (reference[pos] - lp)).sum()))
                same += int(lp.argmax() == reference[pos].argmax())
                nll.append(-lp[targets[pos]])
                worst = max(worst, float(np.abs(lp - reference[pos]).max()))
            say(f"{Path(path).name}: {name}: KL(O || it) mean {np.mean(kl):.3e} (largest of a position {max(kl):.3e}), most likely token the same at "
                f"{same} of {len(kl)}, perplexity {math.exp(np.mean(nll)):.4f} ({(math.exp(np.mean(nll)) / math.exp(o_nll) - 1) * 100:+.3f}%), "
                f"largest change of a log probability {worst:.3e} ({time.time() - began:.0f} s)")
        del reference
    say("done")


if __name__ == "__main__":
    main()
