"""T648.2: engine/translation.py's translations against fxtranslate's, sentence by sentence (the owner: greedy, 100 of
100 the same, a near tie not let through), for en-ja 3.0, ja-en 3.1 (Android's) and ja-en 3.0 (the desktop's).

    .venv/bin/python tests/translation_check.py [--only <model>] [--limit <n>]
    .venv/bin/python tests/translation_check.py --write     # fxtranslate's translations, again, into the fixture
    node tests/translation-kernels.ts [--only <model>] [--limit <n>]   # T648.3: the same, its affines on the kernel

--kernels runs every affine on the SIMD kernel affine_q8 (engine/translation.py's KernelAffine), which loads in
Pyodide only: tests/translation-kernels.ts runs this check there with it.

The models are not in the repository (policy 1): the check looks for them in MARIAN_FILES (folders, os.pathsep
between), translations/ (make models fetches the three there: T648.5) and the owner's checkout of bergamot-vs-shisa
(its models/), by the end of the name and the size
(ja-en's 3.0 and 3.1 have one name), and leaves out a model it does not find (exit 0, saying so). The reference is
tests/fixtures/translations.json: every one of the 100 sentences of bergamot-vs-shisa's data/sentences.jsonl through
every model (the other direction's too: their nonsense is translated by the same steps), as fxtranslate's translate()
wrote it: one unit, no shortlist. (translate_long() cuts the text into sentences with ICU4X first, which writes 16 of
the 100 otherwise; the engine cuts none: the page will, T648.4.) --write makes it again, running fxtranslate where
FXTRANSLATE_PYTHON is (bergamot-vs-shisa's .venv by default), which nothing here installs.

A translation is compared as fxtranslate's text: fxtranslate writes a byte piece as its text ("<0xE2><0x98><0x95>"
for ☕: ja-en's sentence 97), and the engine's text() writes the byte. The check says where the two texts differ
besides. Where a translation differs, the first step whose token differs is said with the two tokens' logits and
the gap (fxtranslate gives no ids: theirs is the step's best token whose piece goes on with their text).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from tree import python_folder  # noqa: E402

sys.path.insert(0, python_folder(ROOT))
import numpy as np  # noqa: E402

from convert.tokenizer import sentencepiece_charsmap, sentencepiece_options, sentencepiece_pieces, tokenizer_bin  # noqa: E402
from engine.kernels import load_kernels  # noqa: E402
from engine.marian import read_marian  # noqa: E402
from engine.tokenizer import Tokenizer  # noqa: E402
from engine.translation import Translator  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "translations.json"
HOME = Path.home() / "GitHub" / "bergamot-vs-shisa"
FOLDERS = [Path(folder) for folder in os.environ.get("MARIAN_FILES", "").split(os.pathsep) if folder] + [ROOT / "translations", HOME / "models"]
SENTENCES = Path(os.environ.get("SENTENCES", HOME / "data" / "sentences.jsonl"))
FXTRANSLATE = os.environ.get("FXTRANSLATE_PYTHON", str(HOME / ".venv" / "bin" / "python"))
# a model -> (its file, the source's vocabulary, the target's): each a name's end and a size (tests/test_marian.py's)
JAEN = ("vocab.jaen.spm", 1_443_222)
MODELS = {
    "en-ja 3.0": (("model.enja.intgemm.alphas.bin", 43_849_787), ("srcvocab.enja.spm", 796_275), ("trgvocab.enja.spm", 827_144)),
    "ja-en 3.1": (("model.jaen.intgemm.alphas.bin", 43_977_787), JAEN, JAEN),
    "ja-en 3.0": (("model.jaen.intgemm.alphas.bin", 59_504_955), JAEN, JAEN),
}
# what runs in fxtranslate's Python: the files and the texts in on stdin, the translations out
FX = """import importlib.metadata, json, sys
from fxtranslate import Translator
asked = json.load(sys.stdin)
out = {"version": importlib.metadata.version("fxtranslate"), "models": {}}
for name, (model, source, target) in asked["models"].items():
    read = lambda path: open(path, "rb").read()
    t = Translator(model=read(model), src_vocab=read(source), trg_vocab=read(target))
    out["backend"] = t.backend()
    out["models"][name] = [t.translate(text) for text in asked["texts"]]
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def found(ending, size):
    for folder in FOLDERS:
        if folder.is_dir():
            for path in sorted(folder.rglob(f"*{ending}")):
                if path.is_file() and path.stat().st_size == size:
                    return path
    return None


def files(name):
    paths = [found(*file) for file in MODELS[name]]
    return paths if all(paths) else None


def spm_tokenizer(path):
    """The engine's Tokenizer of a sentencepiece model, as the converter reads one (convert/tokenizer.py)."""
    model = path.read_bytes()
    pieces = list(sentencepiece_pieces(model))
    options = sentencepiece_options(model)
    data = tokenizer_bin(pieces, len(pieces), charsmap=sentencepiece_charsmap(model))
    return Tokenizer(data, len(pieces), kind=options.pop("tokenizer_kind"), **options)


def fx_text(translator, ids):
    """fxtranslate's text of ids: the pieces one after another, a byte piece as its own text, </s> and <unk> left
    out, "▁" as a space and the first spaces taken off (its spm.rs decode())."""
    vocab, unknown = translator.target.vocab, translator.target.index.get(b"<unk>")
    return b"".join(vocab[t] for t in ids if t not in (translator.eos[1], unknown)).decode("utf-8", "replace").lstrip()


def first_difference(translator, ids, theirs):
    """Where the engine's greedy path leaves fxtranslate's text: the first step whose token does not go on with their
    text, the engine's token and its logit, and the step's best token that does (</s> where their text has ended),
    its logit and the gap. fxtranslate gives no ids, so its own token there is not known: the best that goes on is
    the nearest the engine came to writing their text."""
    vocab, eos, unknown = translator.target.vocab, translator.eos[1], translator.target.index.get(b"<unk>")
    wanted = theirs.encode("utf-8")
    written = b""

    def goes_on(token):
        if token == eos:
            return written.lstrip() == wanted
        piece = b"" if token == unknown else vocab[token]
        return wanted.startswith((written + piece).lstrip())

    def said(token):
        return f"{vocab[token].decode('utf-8', 'replace')!r} ({token})"

    for step, logits in enumerate(translator.steps(ids)):
        ours = int(np.argmax(logits))
        if not goes_on(ours):
            going = [token for token in range(len(logits)) if goes_on(token)]
            if not going:
                return f"step {step}: ours {said(ours)} {logits[ours]:.4f}, and no token goes on with their text"
            best = max(going, key=lambda token: logits[token])
            return (f"step {step}: ours {said(ours)} {logits[ours]:.4f}, the best of theirs {said(best)} "
                    f"{logits[best]:.4f}, gap {logits[ours] - logits[best]:.4f}")
        if ours == eos:
            break
        written += b"" if ours == unknown else vocab[ours]
    return "the same text, written by other tokens"


def write(names):
    sentences = [json.loads(line) for line in SENTENCES.read_text(encoding="utf-8").splitlines() if line.strip()]
    asked = {"texts": [s["src"] for s in sentences], "models": {name: [str(p) for p in files(name)] for name in names}}
    made = subprocess.run([FXTRANSLATE, "-I", "-c", FX], input=json.dumps(asked), capture_output=True, text=True,
                          check=True, timeout=3600)
    out = json.loads(made.stdout)
    FIXTURE.write_text(json.dumps({
        "made": f"fxtranslate {out['version']} (PyPI), backend {out['backend']}, Translator.translate(): greedy, "
                "no shortlist, one unit; by tests/translation_check.py --write",
        "sentences": [{key: s[key] for key in ("id", "src_lang", "src")} for s in sentences],
        "models": {name: {"file": MODELS[name][0][0], "bytes": MODELS[name][0][1], "translations": out["models"][name]}
                   for name in names}}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE.relative_to(ROOT)}: {', '.join(names)}, {len(sentences)} sentences each")


def check(names, limit, kernels=None):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sentences = fixture["sentences"][:limit]
    failed = False
    for name in names:
        model, source, target = files(name)
        translator = Translator(read_marian(model.read_bytes()), spm_tokenizer(source), spm_tokenizer(target), kernels=kernels)
        theirs = fixture["models"][name]["translations"]
        same, own, began = 0, [], time.perf_counter()
        for sentence, their in zip(sentences, theirs):
            ids = translator.source_ids(sentence["src"])
            written = translator.greedy(ids)
            if fx_text(translator, written) == their:
                same += 1
                if translator.text(written) != their:
                    own.append(sentence["id"])
                continue
            failed = True
            print(f"  {name} sentence {sentence['id']} ({sentence['src_lang']}): DIFFERENT\n    ours   {fx_text(translator, written)}\n"
                  f"    theirs {their}\n    {first_difference(translator, ids, their)}")
        seconds = (time.perf_counter() - began) / max(len(sentences), 1)
        print(f"{name}{' (the kernel)' if kernels else ''}: {same} of {len(sentences)} the same as fxtranslate's ({seconds:.2f} s a sentence)"
              + (f"; the engine's own text writes the bytes of sentences {own}" if own else ""), flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--only", action="append", choices=list(MODELS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kernels", action="store_true")
    arguments = parser.parse_args()
    kernels = None
    if arguments.kernels:
        kernels = load_kernels(os.environ.get("TRANSLATION_KERNELS", "simdkernel.so"))
        if not kernels:
            sys.exit("--kernels: the SIMD kernels load in Pyodide only (node tests/translation-kernels.ts)")
    names = list(MODELS) if arguments.write else arguments.only or list(MODELS)  # (--write makes the whole fixture)
    missing = [name for name in names if files(name) is None]
    for name in missing:
        print(f"{name}: left out, its files are not in {', '.join(map(str, FOLDERS))} (MARIAN_FILES)")
    names = [name for name in names if name not in missing]
    if arguments.write:
        if missing:
            sys.exit("--write makes every model's translations: give MARIAN_FILES the missing ones")
        write(names)
        return
    sys.exit(1 if check(names, arguments.limit, kernels) else 0)


if __name__ == "__main__":
    main()
