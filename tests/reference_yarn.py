# reference_yarn.py
# T532: the RoPE tables of a yarn scaling (llama2_numpy.rope_frequencies(), rope_magnitude()) against transformers'
# own _compute_yarn_parameters, with every key a config.json may say of a yarn (attention_factor, mscale and
# mscale_all_dim, beta_fast, beta_slow, truncate), on a grid of heads, thetas, factors and original contexts.
# tests/test_llama3.py holds the same to a transcription of the file in float64 (without PyTorch); this holds the
# transcription's reading to the code itself. PyTorch is not on the development machine, so this runs in CI:
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="pip install --quiet numpy && pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu && pip install --quiet transformers==5.16.1 && python tests/reference_yarn.py" --ref <branch> --grep "yarn:"
#
# transformers computes the inverse frequencies in float32 (PyTorch), the engine in float64: the line is
# 1e-6 + 1.5e-7 factor of each frequency (float32's rounding of base ** (2i / dim), and of the ramp, whose error of
# about 1e-7 a pair in the ramp's middle carries over the factor between its two ends: 2.97e-6 at factor 32 with
# truncate off, CI's first run), and 1e-12 of the magnitude (Python's float in both). Every line of the log begins with
# "yarn:"; anything past the lines is exit 1.
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
from llama2_numpy import rope_frequencies, rope_magnitude  # noqa: E402

say = lambda *parts: print("yarn:", *parts, flush=True)
# what a config.json says besides the factor and the original context (tests/test_llama3.py's YARN_SETTINGS, and the
# keys gpt-oss's, Olmo 3's and Ministral 3's config.json write)
SETTINGS = {"none": {}, "attention_factor": dict(attention_factor=1.2), "mscale": dict(mscale=0.707, mscale_all_dim=1.0),
            "mscale_all_dim": dict(mscale=1.0, mscale_all_dim=0.707), "both mscales of 1": dict(mscale=1.0, mscale_all_dim=1.0),
            "mscale alone": dict(mscale=0.707), "beta_fast": dict(beta_fast=16), "beta_slow": dict(beta_slow=2),
            "both betas": dict(beta_fast=64, beta_slow=0.5), "truncate off": dict(truncate=False),
            "truncate off, betas": dict(truncate=False, beta_fast=16, beta_slow=2),
            "gpt-oss": dict(beta_fast=32.0, beta_slow=1.0, truncate=False),
            "olmo 3": dict(attention_factor=1.2079441541679836, beta_fast=32.0, beta_slow=1.0)}
LINE, MAGNITUDE_LINE = lambda factor: 1e-6 + 1.5e-7 * factor, 1e-12


def theirs(head, base, factor, original, said):
    """transformers' inverse frequencies (float64 of its float32) and attention factor for a Llama of this RoPE"""
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = LlamaConfig(hidden_size=head * 2, num_attention_heads=2, head_dim=head, max_position_embeddings=int(original * factor),
                         rope_parameters={"rope_type": "yarn", "rope_theta": base, "factor": factor,
                                          "original_max_position_embeddings": original, **said})
    frequencies, magnitude = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    return frequencies.double().numpy(), float(magnitude)


def main():
    import transformers
    import torch
    say(f"transformers {transformers.__version__}, torch {torch.__version__}")
    failed = False
    for name, said in SETTINGS.items():
        worst, past, worst_magnitude, count = 0.0, 0.0, 0.0, 0
        for head, base, factor, original in ((h, b, f, o) for h in (8, 64, 128, 256) for b in (1e4, 1e6, 5e6)
                                             for f in (1.5, 4.0, 32.0) for o in (64, 4096, 8192, 32768)):
            scaling = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": original, **said}
            frequencies, magnitude = theirs(head, base, factor, original, said)
            ours = rope_frequencies(head, base, scaling)
            off = float(np.max(np.abs(ours / frequencies - 1)))
            worst, past = max(worst, off), max(past, off / LINE(factor)) if off == off else float("inf")
            worst_magnitude = max(worst_magnitude, abs(rope_magnitude(scaling) / magnitude - 1))
            count += 1
        bad = not (past <= 1) or not (worst_magnitude <= MAGNITUDE_LINE)
        failed |= bad
        say(f"{name} ({count} heads, thetas, factors and contexts): the frequencies within {worst:.2e} of transformers' ({past:.2f} of the line), "
            f"the magnitude within {worst_magnitude:.2e}{' — FAILED' if bad else ''}")
    say("FAILED" if failed else "the engine's yarn tables are transformers'")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
