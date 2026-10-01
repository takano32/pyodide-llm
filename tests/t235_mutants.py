# Break T235's code on purpose, one place at a time (or two together), and say which tests fail (a throwaway tool).
#
#   python3 .tmp/t235r/mutants.py [--only name,name] -- <pytest arguments>
#
# Each mutant is (name, [(file, old text, new text), ...]): every old text must occur exactly once. The files are put
# back after each run.
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONVERT, NUMPY = "public/llama2_convert.py", "public/llama2_numpy.py"
CHECK = "tests/gguf_check.py"
FILE_TABLE = ("(np.cos if which == 0 else np.sin)(positions * frequencies) * rope_magnitude(config.get(\"rope_scaling\"))")
ENGINE_TABLE = "((turn(angles) * self.rope_magnitude).astype(np.float32)"
CODES = "(0, 2, 4, 6) & 3) - 1)"
MUTANTS = [
    ("bit order of a byte reversed", [(CONVERT, CODES, "(6, 4, 2, 0) & 3) - 1)")]),
    ("scale read from the last two bytes", [(CONVERT,
     "scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)\n    values = PQ2_0_CODES[blocks[:, 2:]].view(np.int8)",
     "scales = np.ascontiguousarray(blocks[:, 32:]).view(np.float16).astype(np.float32)\n    values = PQ2_0_CODES[blocks[:, :32]].view(np.int8)")]),
    ("code 1 not 0 is zero (q - 2)", [(CONVERT, CODES, "(0, 2, 4, 6) & 3) - 2)")]),
    ("code 3 read as 0, not +2 d", [(CONVERT, "(0, 2, 4, 6) & 3) - 1).astype(np.int8)", "(0, 2, 4, 6) & 3) - 1).clip(-1, 1).astype(np.int8)")]),
    ("yarn low off by one (+1)", [(NUMPY, "low, high = max(math.floor(pair(32)), 0),", "low, high = max(math.floor(pair(32)) + 1, 0),")]),
    ("yarn high rounded down", [(NUMPY, "min(math.ceil(pair(1)), width - 1)", "min(math.floor(pair(1)), width - 1)")]),
    ("yarn low rounded up", [(NUMPY, "max(math.floor(pair(32)), 0)", "max(math.ceil(pair(32)), 0)")]),
    ("yarn ramp wrong way", [(NUMPY, "return frequencies * (1 - slowed) + frequencies / float(scaling[\"factor\"]) * slowed",
                              "return frequencies * slowed + frequencies / float(scaling[\"factor\"]) * (1 - slowed)")]),
    ("magnitude only on the cos table of the file", [(CONVERT, FILE_TABLE, FILE_TABLE.replace("* rope_magnitude(config.get(\"rope_scaling\"))", "* (rope_magnitude(config.get(\"rope_scaling\")) if which == 0 else 1.0)"))]),
    ("magnitude only on the sin table of the file", [(CONVERT, FILE_TABLE, FILE_TABLE.replace("* rope_magnitude(config.get(\"rope_scaling\"))", "* (rope_magnitude(config.get(\"rope_scaling\")) if which == 1 else 1.0)"))]),
    ("magnitude left out of the file's tables", [(CONVERT, FILE_TABLE, "(np.cos if which == 0 else np.sin)(positions * frequencies)")]),
    ("magnitude only on the cos table of the engine", [(NUMPY, ENGINE_TABLE, "((turn(angles) * (self.rope_magnitude if turn is np.cos else 1.0)).astype(np.float32)")]),
    ("magnitude only on the sin table of the engine", [(NUMPY, ENGINE_TABLE, "((turn(angles) * (self.rope_magnitude if turn is np.sin else 1.0)).astype(np.float32)")]),
    ("magnitude left out of the engine's tables (int8)", [(NUMPY, ENGINE_TABLE, "((turn(angles)).astype(np.float32)")]),
    ("magnitude applied twice in the engine", [(NUMPY, ENGINE_TABLE, "((turn(angles) * self.rope_magnitude ** 2).astype(np.float32)")]),
    # both builders at fault in the same way (as one shared helper written wrong would be)
    ("JOINT: magnitude only on the cos tables, in the file and the engine", [
        (CONVERT, FILE_TABLE, FILE_TABLE.replace("* rope_magnitude(config.get(\"rope_scaling\"))", "* (rope_magnitude(config.get(\"rope_scaling\")) if which == 0 else 1.0)")),
        (NUMPY, ENGINE_TABLE, "((turn(angles) * (self.rope_magnitude if turn is np.cos else 1.0)).astype(np.float32)")]),
    ("JOINT: magnitude only on the sin tables, in the file and the engine", [
        (CONVERT, FILE_TABLE, FILE_TABLE.replace("* rope_magnitude(config.get(\"rope_scaling\"))", "* (rope_magnitude(config.get(\"rope_scaling\")) if which == 1 else 1.0)")),
        (NUMPY, ENGINE_TABLE, "((turn(angles) * (self.rope_magnitude if turn is np.sin else 1.0)).astype(np.float32)")]),
    ("JOINT: magnitude left out of both", [
        (CONVERT, FILE_TABLE, "(np.cos if which == 0 else np.sin)(positions * frequencies)"),
        (NUMPY, ENGINE_TABLE, "((turn(angles)).astype(np.float32)")]),
    ("JOINT: magnitude 1.2 times too much in both (a wrong constant)", [
        (CONVERT, FILE_TABLE, FILE_TABLE.replace("* rope_magnitude(config.get(\"rope_scaling\"))", "* rope_magnitude(config.get(\"rope_scaling\")) * 1.2")),
        (NUMPY, ENGINE_TABLE, "((turn(angles) * self.rope_magnitude * 1.2).astype(np.float32)")]),
    ("yarn ignored by the engine's tables", [(NUMPY, "frequencies = lambda width: rope_frequencies(width, rope_theta, rope_scaling)",
                                              "frequencies = lambda width: rope_frequencies(width, rope_theta, None)")]),
    ("rope_magnitude is 1", [(NUMPY, "return 0.1 * math.log(max(float(scaling[\"factor\"]), 1.0)) + 1.0", "return 1.0")]),
    ("magnitude 0.1 log2", [(NUMPY, "0.1 * math.log(max(float(scaling[\"factor\"]), 1.0)) + 1.0", "0.1 * math.log2(max(float(scaling[\"factor\"]), 1.0)) + 1.0")]),
    ("rows not of blocks of 128 let through", [(CONVERT, "if info[\"shape\"][-1] % BLOCKS.get(dtype, 1):", "if False:")]),
    ("gguf_agrees leaves yarn out", [(CONVERT, "(\"yarn RoPE scaling\", scaled(own), scaled(config))]", "]")]),
    ("check_config lets other yarn keys through", [(CONVERT, "for key in sorted(set(said) - {\"factor\", \"original_max_position_embeddings\"}):", "for key in []:")]),
    ("yarn's original context from the context length, not the GGUF's", [(CONVERT, "key(\"rope.scaling.original_context_length\", key(\"context_length\"))", "key(\"context_length\")")]),
    ("the test reader's bit order reversed (gguf_check.py)", [(CHECK, "np.arange(0, 8, 2, dtype=np.uint8)) & 3", "np.arange(6, -1, -2, dtype=np.uint8)) & 3")]),
    ("the test reader's scale position (gguf_check.py)", [(CHECK, "return (codes.reshape(-1, 128).astype(np.float32) - 1) * half(blocks[:, :2])",
                                                           "return (codes.reshape(-1, 128).astype(np.float32) - 1) * half(blocks[:, 32:])")]),
    # a reader that looks at only the first byte's low 2 bits of every four values would pass a table of zeros
    ("converter reads PQ2_0 as 0 for every value (the model without weights)", [(CONVERT, "values = PQ2_0_CODES[blocks[:, 2:]].view(np.int8)", "values = PQ2_0_CODES[blocks[:, 2:]].view(np.int8) * 0")]),
]


def run(arguments):
    # no stale bytecode: a mutant of the same size as the source, written within a second of the last compile, would
    # otherwise run the old code
    for cache in list(Path("public").glob("__pycache__")) + list(Path("tests").glob("__pycache__")):
        for pyc in cache.glob("*.pyc"):
            pyc.unlink()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", *arguments],
                            capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    failed = re.findall(r"^FAILED (\S+)", result.stdout, flags=re.M)
    last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return failed, last


def main():
    args = sys.argv[1:]
    only = None
    if args and args[0] == "--only":
        only = set(args[1].split(","))
        args = args[2:]
    if args and args[0] == "--":
        args = args[1:]
    for name, edits in MUTANTS:
        if only and name not in only and not any(name.startswith(prefix[:-1]) for prefix in only if prefix.endswith("*")):
            continue
        saved = {}
        for file, old, new in edits:
            text = saved.setdefault(file, Path(file).read_text())
            assert text.count(old) == 1, f"{name}: {text.count(old)} occurrences of {old!r} in {file}"
        began = time.time()
        try:
            for file, old, new in edits:
                Path(file).write_text(Path(file).read_text().replace(old, new))
            failed, last = run(args)
        finally:
            for file, text in saved.items():
                Path(file).write_text(text)
        print(f"MUTANT {name}: {len(failed)} failing ({time.time() - began:.0f} s): {last}", flush=True)
        for test in failed[:14]:
            print(f"    FAILED {test}", flush=True)
        if len(failed) > 14:
            print(f"    ... and {len(failed) - 14} more", flush=True)
        if not failed:
            print("    **SURVIVED** (no test failed)", flush=True)


if __name__ == "__main__":
    main()
