# t260_mutants.py (the review of T260; a throwaway branch's): the repository's own checks against LFM2 code broken one thing at
# a time. Each mutant is one replacement in one file; the checks the deploy runs for it (pytest, conv-check, memory-check,
# worker-sink-check, gpu-hybrid-check: the light suite's) and the ones only the full suite runs (forward-check on made-up
# LFM2s, threads-check). A line "mutants: <name>: CAUGHT by ..." or "NOT CAUGHT", for CI (tests.yml's extra=):
#
#   python tests/t260_mutants.py <py | js | kernel> [name fragment ...]
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
say = lambda *parts: print("mutants:", *parts, flush=True)

NUMPY, CONVERT, FORWARD, KERNEL = "public/llama2_numpy.py", "public/llama2_convert.py", "public/forward.js", "kernels/kernel.ts"
# (category, name, file, old, new, nth occurrence)
MUTANTS = [
    # --- the NumPy engine
    ("py", "taps in the other order", NUMPY, "out = (taps[:-1] * before).sum(axis=0) + taps[-1] * values", "out = (taps[1:] * before).sum(axis=0) + taps[0] * values", 1),
    ("py", "the rows shift the other way", NUMPY, "        before[:-1] = before[1:]\n        before[-1] = values", "        before[1:] = before[:-1].copy()\n        before[0] = values", 1),
    ("py", "the state is not cleared at position 0", NUMPY, "            self.conv_state.fill(0.0)", "            pass", 1),
    ("py", "the gates B and C in each other's place", NUMPY, "(mixed[dim:2 * dim] * self.convolved(a, mixed[:dim] * mixed[2 * dim:]))", "(mixed[:dim] * self.convolved(a, mixed[dim:2 * dim] * mixed[2 * dim:]))", 1),
    ("py", "layer kinds inverted in layer_slots", NUMPY, 'kinds = [kind == "c" for kind in convolution["layers"]]', 'kinds = [kind == "a" for kind in convolution["layers"]]', 1),
    ("py", "only a Qwen3.5 follows its state", NUMPY, "if self.linear is not None or self.convolution is not None:\n            self.follow(pos)", "if self.linear is not None:\n            self.follow(pos)", 1),
    ("py", "checkpoint_dtype leaves the taps out", NUMPY, 'short * convolution["taps"] * dim', "short * dim", 1),
    ("py", "the engine reads the taps before the matrix in", NUMPY, 'self.win = matrix(short, 3 * dim, dim)\n        self.conv = vector(short, self.convolution["taps"], dim)', 'self.conv = vector(short, self.convolution["taps"], dim)\n        self.win = matrix(short, 3 * dim, dim)', 1),
    # --- the converter
    ("py", "the FFN's size is not rounded up", CONVERT, "            hidden = multiple * ((hidden + multiple - 1) // multiple)", "            pass", 1),
    ("py", "the keys' head norm is not turned", CONVERT, 'of(full, "self_attn.k_layernorm", ("permute", 1))', 'of(full, "self_attn.k_layernorm")', 1),
    ("py", "the taps are not transposed", CONVERT, 'of(short, "conv.conv", ("taps",))', 'of(short, "conv.conv")', 1),
    ("py", "the GGUF's gate and up in each other's place", CONVERT, '"ffn_gate": "feed_forward.w1", "ffn_up": "feed_forward.w3",', '"ffn_gate": "feed_forward.w3", "ffn_up": "feed_forward.w1",', 1),
    ("py", "the GGUF's convolution keeps no axis of one", CONVERT, 'if arch == "lfm2" and target.endswith("conv.conv.weight") and len(info["shape"]) == 2:', "if False:", 1),
    ("py", "layout() writes the taps first", CONVERT, '((short, 3 * dim, dim), True), ((short, convolution["taps"], dim), False), ((short, dim, dim), True),', '((short, convolution["taps"], dim), False), ((short, 3 * dim, dim), True), ((short, dim, dim), True),', 1),
    ("py", "normalize() leaves an LFM2's config as it is", CONVERT, "        config = lfm2_config(config)  # T260", "        pass  # T260", 1),
    ("py", "conv_bias is accepted", CONVERT, '        if config.get("conv_bias"):\n            refuse("its convolution layers have biases")', '        if False:\n            refuse("its convolution layers have biases")', 1),
    ("py", "an LFM2's epsilon is not carried", CONVERT, '("llama", "qwen35", "lfm2") and eps != RMS_EPS', '("llama", "qwen35") and eps != RMS_EPS', 1),
    ("py", "the FFN's gate and up in each other's place (plan)", CONVERT, 'of(every, "feed_forward.w1"), of(every, "feed_forward.w2"), of(every, "feed_forward.w3"),', 'of(every, "feed_forward.w3"), of(every, "feed_forward.w2"), of(every, "feed_forward.w1"),', 1),
    ("py", "full_attn_idxs is not read", CONVERT, "attending = range(layers) if attending is None else attending", "attending = range(layers) if attending is None else []", 1),
    ("py", "theta defaults to 1e4 where a config says none", CONVERT, '"rope_theta": rope.get("rope_theta", config.get("rope_theta", 1000000.0)),', '"rope_theta": rope.get("rope_theta", config.get("rope_theta", 10000.0)),', 1),
    ("py", "taps default to 4 where a config says none", CONVERT, 'taps = config.get("conv_L_cache", 3)', 'taps = config.get("conv_L_cache", 4)', 1),
    # --- forward.js
    ("js", "the rows do not move up", FORWARD, "      F.copyWithin(rows / 4, (rows + D) / 4, (rows + convBytes) / 4);\n      k.short_conv(", "      k.short_conv(", 1),
    ("js", "the state is not cleared at position 0", FORWARD, "      F.fill(0, convRows / 4, (convRows + lineCount * convBytes) / 4);", "      // mutant", 1),
    ("js", "only a Qwen3.5 follows its state", FORWARD, "    if (convBytes) follow(pos0);", "    if (linear) follow(pos0);", 1),
    ("js", "the position that comes next is not kept", FORWARD, "    if (convBytes) stateAt = pos0 + count;", "    if (linear) stateAt = pos0 + count;", 1),
    ("js", "the rows are one short", FORWARD, "convolution ? convolution.taps * dim * 4 : 0", "convolution ? (convolution.taps - 1) * dim * 4 : 0", 1),
    ("js", "layer 0's taps for every layer", FORWARD, "k.short_conv(xb + t * S, taps + a * convBytes, rows, mixed + t * S, dim, convolution.taps);", "k.short_conv(xb + t * S, taps, rows, mixed + t * S, dim, convolution.taps);", 1),
    ("js", "layer 0's matrix in for every layer", FORWARD, "matmuls(xb, count, [[win, mixed, S, a]]);", "matmuls(xb, count, [[win, mixed, S, 0]]);", 1),
    ("js", "layer 0's matrix out for every layer", FORWARD, "    matmuls(xb, count, [[wout, xb2, S, a]]);", "    matmuls(xb, count, [[wout, xb2, S, 0]]);", 2),
    ("js", "the tokens of a block in reverse", FORWARD, "    for (let t = 0; t < count; t++) {\n      F.copyWithin(rows / 4", "    for (let t = count - 1; t >= 0; t--) {\n      F.copyWithin(rows / 4", 1),
    ("js", "the kernel is told one tap too few", FORWARD, "mixed + t * S, dim, convolution.taps);", "mixed + t * S, dim, convolution.taps - 1);", 1),
    ("js", "layer kinds inverted in layerSlots", FORWARD, 'convolution.layers[l] === "c"', 'convolution.layers[l] !== "c"', 1),
    ("js", "footprint() leaves the rows out", FORWARD, "  if (convolution) bytes += convolutionStateBytes(layers, dim, convolution);  // T260", "  // mutant", 1),
    ("js", "footprint() counts keys and values for every layer", FORWARD, "keys = direct ? 0 : seqLen * attendingLayers(layers, linear, convolution) * 2 * kvDim;", "keys = direct ? 0 : seqLen * attendingLayers(layers, linear) * 2 * kvDim;", 1),
    ("js", "the frame's matrix-in output is a third of its size", FORWARD, '    ...(convolution ? [["mixed", 3 * dim * 4]] : []),', '    ...(convolution ? [["mixed", dim * 4]] : []),', 1),
    ("js", "the GPU is not refused for convolution layers", FORWARD, '    if (convolution) return "convolution layers are not on the GPU yet";  // T260', "    // mutant", 1),
    # --- the kernel
    ("kernel", "h = B + z", KERNEL, "v128.store(newest + o, f32x4.mul(v128.load(b + o), v128.load(z + o)));", "v128.store(newest + o, f32x4.add(v128.load(b + o), v128.load(z + o)));", 1),
    ("kernel", "no gate C", KERNEL, "v128.store(out + o, f32x4.mul(v128.load(gate + o), acc));", "v128.store(out + o, acc);", 1),
    ("kernel", "B is the gate", KERNEL, "const newest = rows + <usize>(count - 1) * stride, gate = b + stride, z = gate + stride;", "const newest = rows + <usize>(count - 1) * stride, gate = b, z = b + 2 * stride;", 1),
    ("kernel", "the leftover channels have no gate", KERNEL, "store<f32>(out + o, load<f32>(gate + o) * acc);", "store<f32>(out + o, acc);", 1),
    ("kernel", "the leftover channels h = B + z", KERNEL, "store<f32>(newest + o, load<f32>(b + o) * load<f32>(z + o));", "store<f32>(newest + o, load<f32>(b + o) + load<f32>(z + o));", 1),
    ("kernel", "the newest row is the one before", KERNEL, "const newest = rows + <usize>(count - 1) * stride,", "const newest = rows + <usize>(count - 2) * stride,", 1),
    ("kernel", "the taps are subtracted", KERNEL, "acc = f32x4.add(acc, f32x4.mul(v128.load(taps + at), v128.load(rows + at)));", "acc = f32x4.sub(acc, f32x4.mul(v128.load(taps + at), v128.load(rows + at)));", 2),
]

CHECKS = {  # name: (command, runs for categories)
    "pytest": (["python", "-m", "pytest", "tests", "-q", "-x", "-p", "no:cacheprovider"], ("py",)),
    "conv-check": (["node", "tests/conv-check.mjs"], ("py", "js", "kernel")),
    "memory-check": (["node", "tests/memory-check.mjs"], ("py", "js", "kernel")),
    "worker-sink-check": (["node", "tests/worker-sink-check.mjs"], ("py", "js", "kernel")),
    "gpu-hybrid-check": (["node", "tests/gpu-hybrid-check.mjs", ".tmp/made-up-lfm2-int8"], ("js", "kernel")),
}
FULL = {  # only the full suite runs these
    "forward-check": ["node", "tests/forward-check.mjs", ".tmp/made-up-lfm2-float32", ".tmp/made-up-lfm2-int8",
                      ".tmp/made-up-lfm2-four-int8", "--rounds", "1", "--positions", "128"],
    "threads-check": ["node", "tests/threads-check.mjs", ".tmp/made-up-lfm2-float32", ".tmp/made-up-lfm2-int8", "--rounds", "1"],
}


def run(command, minutes=12):
    began = time.time()
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=minutes * 60,
                              env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
        ok, tail = done.returncode == 0, (done.stdout + done.stderr)[-300:]
    except subprocess.TimeoutExpired:
        ok, tail = False, "timed out"
    return ok, time.time() - began, tail.replace("\n", " | ")


def main():
    category, fragments = sys.argv[1], sys.argv[2:]
    subprocess.run(["bash", "-c", "mkdir -p .tmp; for dtype in float32 int8; do python tests/make_lfm2.py .tmp/made-up-lfm2-$dtype $dtype; done; "
                    "python tests/make_lfm2.py .tmp/made-up-lfm2-four-int8 int8 four"], cwd=ROOT, check=True, capture_output=True)
    # the right code first: every check must pass, or a mutant's "caught" means nothing
    for name, (command, categories) in CHECKS.items():
        if category in categories:
            ok, seconds, tail = run(command)
            say(f"clean: {name}: {'passes' if ok else 'FAILS ' + tail} ({seconds:.0f} s)")
    if category != "py":
        for name, command in FULL.items():
            ok, seconds, tail = run(command)
            say(f"clean: {name}: {'passes' if ok else 'FAILS ' + tail} ({seconds:.0f} s)")
    results = []
    for mutant_category, name, file, old, new, nth in MUTANTS:
        if mutant_category != category or (fragments and not any(f in name for f in fragments)):
            continue
        path = ROOT / file
        original = path.read_text()
        assert original.count(old) >= nth, f"{name}: the code to break is not there"
        at = -1
        for _ in range(nth):
            at = original.index(old, at + 1)
        path.write_text(original[:at] + new + original[at + len(old):])
        try:
            if category == "kernel":
                subprocess.run(["make", "kernels"], cwd=ROOT, check=True, capture_output=True)
            caught = []
            for check, (command, categories) in CHECKS.items():
                if category in categories:
                    ok, seconds, tail = run(command)
                    if not ok:
                        caught.append(check)
            if category != "py":
                for check, command in FULL.items():
                    ok, seconds, tail = run(command)
                    if not ok:
                        caught.append(check)
        finally:
            path.write_text(original)
            if category == "kernel":
                subprocess.run(["make", "kernels"], cwd=ROOT, check=True, capture_output=True)
        light = [check for check in caught if check not in FULL]
        verdict = (f"CAUGHT by the light suite ({', '.join(light)})" + (f" and the full one ({', '.join(c for c in caught if c in FULL)})" if len(caught) > len(light) else "")
                   if light else f"CAUGHT only by the full suite ({', '.join(caught)})" if caught else "NOT CAUGHT")
        say(f"{category}: {name}: {verdict}")
        results.append((name, verdict))
    say(f"{len(results)} mutants: {sum('NOT' in v for _, v in results)} not caught, {sum('only by the full' in v for _, v in results)} only by the full suite")


if __name__ == "__main__":
    main()
