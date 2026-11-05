"""T649.1: PLaMo's tokenizer.jsonl, read by the converter and the engine, against PLaMo's own tokenizer reading the same
folder (tokenization_plamo.py's Plamo3Tokenizer, which transformers loads from the folder as AutoTokenizer would):
the ids of many texts, one by one, and what the engine decodes. Run by hand (the files are PFN's and megabytes;
pytest has a made-up vocabulary of the same shape, tests/test_plamo_tokenizer.py):

    python tests/plamo_tokenizer_check.py <a folder with tokenizer.jsonl, tokenizer_config.json and
        tokenization_plamo.py> ... [--texts <a file> ...] [--planes 17]

The texts are tests/tokenizer_json_check.py's (the nine prompts, the lines and the first characters of the
repository's files, the boundaries, every character between two letters, alone and after a space, random texts of
the vocabulary's own pieces with the control pieces thrown in), and runs of spaces and of one character, which PLaMo
cuts around. The engine is handed the specials the converter names, as a chat template's text would be.

tokenization_plamo.py imports numba for its search, which the venv has not; it goes into a folder of .tmp, and
PYTHONPATH names it (numba 0.68.1 and llvmlite 0.50.0 with the venv's NumPy 2.3.5, 2026-11-05):

    pip install --target .tmp/numba-site --no-deps numba llvmlite
    PYTHONPATH=.tmp/numba-site python tests/plamo_tokenizer_check.py …

(Not a stand-in that runs its search as plain Python: there NumPy's booleans add up as booleans, True + True is True,
and a character of three bytes that no piece begins with is counted as two.) transformers is the one of the venv.
Any text whose ids differ fails (exit 1) and is printed.
"""
import importlib.util
import sys
import time
from pathlib import Path

from tree import python_folder

sys.path.insert(0, python_folder())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import detokenize  # noqa: E402
from convert.vocabulary import vocabulary_of  # noqa: E402
from engine.tokenizer import SETTINGS, Tokenizer  # noqa: E402
from tokenizer_json_check import DEFAULT_TEXTS, ROOT, texts_of  # noqa: E402


def real_of(folder):
    """PLaMo's own tokenizer of the folder, from its tokenization_plamo.py and tokenizer_config.json."""
    import json
    spec = importlib.util.spec_from_file_location("tokenization_plamo", folder / "tokenization_plamo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    said = {key: value for key, value in config.items() if key in ("break_around_consecutive_spaces_threshold",
                                                                    "break_around_repeated_chars_threshold")}
    return module.Plamo3Tokenizer(str(folder / "tokenizer.jsonl"), **said)


def read(folder):
    """(the engine's tokenizer, the specials to hand it, PLaMo's own, the control pieces, the pieces' texts)"""
    data = (folder / "tokenizer.jsonl").read_bytes()
    config = (folder / "tokenizer_config.json").read_text(encoding="utf-8")
    vocabulary = vocabulary_of(data, "tokenizer.jsonl", data.count(b"\n"), config)
    started = time.perf_counter()
    engine = Tokenizer(vocabulary.tokenizer, data.count(b"\n"),
                       **{SETTINGS[key]: value for key, value in vocabulary.options.items()})
    built = time.perf_counter() - started
    real = real_of(folder)
    rows = real.data
    controls = [text for text, _, kind, *_ in rows if kind in ("CONTROL", "UNKNOWN")]
    print(f"{folder}: {vocabulary.options}, {len(rows):,} pieces, {len(controls)} control ones "
          f"({len(vocabulary.specials)} of them specials), the engine's tokenizer made in {built * 1e3:.0f} ms")
    return engine, tuple(vocabulary.specials), real, controls, [text for text, _, kind, *_ in rows if kind == "NORMAL"]


def runs():
    """runs of spaces and of one character, which PLaMo cuts around, next to others and to text"""
    return [run * count + after for run in (" ", "a", "あ", "\n", "-", "=", "　", "<", "|")
            for count in range(1, 9) for after in ("", "b", " b", "aa", "  ")] + \
           ["a  b   c    d", "aaaa  bbbb", "x" * 40 + " " * 40, "    def f():\n        return 1\n", "ーーーー", "!!!!!!", "<|plamo:x|>   <|plamo:msg|>aaaa"]


def check(folder, files, planes):
    engine, specials, real, controls, pieces = read(folder)
    wrong, total = [], 0
    texts = texts_of(files, pieces, controls, planes)
    texts["runs"] = runs()
    for kind, group in texts.items():
        started, differ, back = time.perf_counter(), 0, 0
        for text in group:
            want = real.encode(text, add_special_tokens=False)
            got = engine.encode(text, specials)
            if got != want:
                differ += 1
                wrong.append((kind, text, got, want))
            else:
                # what the engine writes back is what PLaMo's decodes (U+EE00 is left out of the ids, and so of both)
                back += detokenize(engine, got, bos=-1) != real.decode(want, skip_special_tokens=False)
        total += len(group)
        print(f"  {kind}: {len(group):,} texts, {differ} with other ids than PLaMo's, {back} decoded otherwise "
              f"({time.perf_counter() - started:.0f} s)")
        wrong += [(kind, "decoded otherwise", back, 0)] if back else []
    for kind, text, got, want in wrong[:40]:
        print(f"  WRONG ({kind}) {text[:100]!r}\n    the engine {got[:30] if isinstance(got, list) else got}\n"
              f"    PLaMo's    {want[:30] if isinstance(want, list) else want}")
    print(f"  {folder}: {total:,} texts, {len(wrong)} wrong")
    return not wrong


def main(arguments):
    planes = int(arguments[arguments.index("--planes") + 1]) if "--planes" in arguments else 1
    files = arguments[arguments.index("--texts") + 1:] if "--texts" in arguments else [ROOT / name for name in DEFAULT_TEXTS]
    files = [file for file in files if not str(file).startswith("--") and str(file) != str(planes)]
    folders = [Path(argument) for argument in (arguments[:arguments.index("--texts")] if "--texts" in arguments else arguments)
               if not argument.startswith("--") and argument != str(planes)]
    if not folders:
        raise SystemExit(__doc__)
    return 0 if all([check(folder, files, planes) for folder in folders]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
