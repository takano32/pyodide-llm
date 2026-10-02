"""T233 review probe (a throwaway branch): a sign of the rotated basis flipped in forward.js alone (public/forward.js has a
hook, T233_FLIP=<width | all>:<place | mid | last>), the engine's NumPy forward pass left right: does tests/forward-check.mjs
on the made-up rotated models see it? The made-up models: a Qwen3.5 folded into a rotated basis (blocks of 16, float32 and
int8) and the ternary hybrid said to be rotated (blocks of 128, ternary)."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
(ROOT / ".tmp").mkdir(exist_ok=True)


def run(command, env=None):
    done = subprocess.run(command, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return done.returncode, done.stdout, done.stderr


for command in (["python", "tests/make_qwen35.py", ".tmp/made-up-rotated-float32", "float32", "small", "rotated"],
                ["python", "tests/make_qwen35.py", ".tmp/made-up-rotated-int8", "int8", "small", "rotated"],
                ["python", "tests/make_ternary.py", ".tmp/made-up-ternary-rotated", "ternary", "rotated"],
                ["python", "tests/make_ternary.py", ".tmp/made-up-ternary-rotated-tied", "ternary", "rotated-tied"]):
    code, out, err = run(command)
    print(f"probe: made {command[2]}: exit {code}", flush=True)
    if code:
        print(err[-2000:])
        sys.exit(1)

MODELS = [".tmp/made-up-rotated-float32", ".tmp/made-up-rotated-int8", ".tmp/made-up-ternary-rotated", ".tmp/made-up-ternary-rotated-tied"]
FIRST = ["node", "tests/forward-check.mjs", "--rounds", "1", "--positions", "128"]


def check(model, flip):
    code, out, err = run(FIRST[:2] + [model] + FIRST[2:], {"T233_FLIP": flip} if flip else None)
    line = next((l for l in out.splitlines() if "most likely token" in l or "FAILED" in l), out.strip().splitlines()[-1] if out.strip() else err.strip()[-300:])
    return code, line, err


widths = {}
for model in MODELS:
    code, line, err = check(model, None)
    print(f"probe: {model} right: exit {code}: {line[:230]}", flush=True)
    code, line, err = check(model, "all:0")
    widths[model] = sorted({int(n) for n in re.findall(r"of signs\.(\d+)", err)})
    print(f"probe: {model} widths {widths[model]}; all:0: exit {code}: {line[:230]}", flush=True)
    specs = ["all:mid", "all:last"] + [f"{n}:{place}" for n in widths[model] for place in ("0", "mid", "last")]
    for spec in specs:
        code, line, err = check(model, spec)
        print(f"probe: {model} flipped {spec}: {'SEEN (FAILED)' if code else 'not seen'}: {line[:230]}", flush=True)
