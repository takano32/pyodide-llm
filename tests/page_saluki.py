# page_saluki.py (T.364.2): Underdog Saluki 27B converted as the page will convert it, its blocks kept as they are
# (Conversion(..., blocks=True) by the conduct of a conversion, T.363.3), by the native Python of a CI runner, into a
# file: the checkpoint (7.96 GB) that tests/page-27b.ts runs on the page's forward pass. tests/page_saluki.sh runs it;
# the development machine does not (the GGUF is 7.9 GB). What is converted and how it is asked are the list's two
# entries' (T.364.3: listed_entries()), which tests/page-27b.ts takes with --as (a form more beside them, and the dry
# stage's without their options).
#
#   python tests/page_saluki.py folder <the directory with the GGUF> <a directory for the original's files>
#   python tests/page_saluki.py convert <folder> <out> [--context 4096] [--broken tiled]
#   python tests/page_saluki.py entries <out.json> [--made-up]
#
# folder: fetches the original's config.json, tokenizer and template (Qwen/Qwen3.8-27B's, as the entry names them:
#   what the conduct of the conversion asks for, tests/conducting.py's Fetched) and links the GGUF
#   beside them; prints the folder.
# convert: <out>.bin (a memory map), <out>.tokenizer.bin and <out>.json (the options the page would hand the engine:
#   the form's "kinds", a type to every matrix and to every layer of a stack, and "tiled"), with the seconds and the
#   peak of the process's own memory (a tensor whose rows move waits whole for its last byte: T.363.3).
#   --broken tiled: the value heads of every tensor left in the file's order (llama.cpp tiles them, T.245), for the
#   comparison to fail on: an error only a model with more value heads than key heads shows.
# entries: the entries the comparison and the writing take the page's options from, as JSON: the list's two ("at-once",
#   "thinking") and "sampled" (--made-up: with no options of their own, for the dry stage's small model, whose
#   vocabulary has none of the special tokens).
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from page_27b import own_memory  # noqa: E402 (puts the engine's Python on the path)
import convert.stream  # noqa: E402
from conducting import Fetched, Mapped, Nothing, converted, listed  # noqa: E402
from reference_saluki import FILE, REPOSITORY, REVISION  # noqa: E402

# T.364.3: the list's two entries of the model (src/models/hf-families.ts): what they fetch and convert (hf, conversion)
# and what they hand the engine and the page (the options, the two formats, each form's sampling). The file the
# references are made of (tests/reference_saluki.py's) must be the one the list names
IDS = {"at-once": "hf-underdog-saluki-27b", "thinking": "hf-underdog-saluki-27b-thinking"}


def listed_entries():
    script = "import('./src/models.ts').then(({ MODELS }) => console.log(JSON.stringify(MODELS.filter((m) => m.conversion?.blocks))))"
    found = {entry["id"]: entry for entry in json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))}
    entries = {name: found[id] for name, id in IDS.items()}
    for entry in entries.values():
        hf = entry["hf"]
        assert (hf["repo"], hf["revision"], hf["weights"]) == (REPOSITORY, REVISION, FILE), f"{entry['id']} names another file than the references' ({hf})"
        assert entry["conversion"] == {"dtype": "int8", "blocks": True}, entry["conversion"]
    # "sampled": the form that answers at once with the list's set for a Qwen3.5 that does (no entry: the owner chose the
    # card's greedy, T.364: with this set one answer of four fell apart), to write with beside it
    sampled = {"steps": 0, "temperature": 0.7, "topp": 0.8, "repetition_penalty": 1, "top_k": 20, "presence_penalty": 1.5}
    return {**entries, "sampled": {**entries["at-once"], "id": f"{IDS['at-once']} (a Qwen3.5's sampling)", "generation": sampled}}


def model():
    return listed_entries()["at-once"]["hf"]


def folder(work, directory):
    MODEL = model()
    answerer = Fetched.of(MODEL, directory)
    link = answerer.folder("weights") / FILE
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to((Path(work) / FILE).resolve())
    converted(answerer, MODEL, weights=False, sink=Nothing(), dtype="int8", blocks=True)
    original = answerer.folder("vocabulary")
    beside = original / FILE
    if not beside.exists():
        beside.symlink_to((Path(work) / FILE).resolve())
    print(original)


def convert_(folder, out, context, broken):
    peak = own_memory()

    def fed():
        nonlocal peak
        peak = max(peak, own_memory())

    hf, answerer = listed(folder, fed=fed)
    gguf = Path(folder) / hf["weights"]
    if broken == "tiled":
        # (the part's own name, as a test would set it: order() and Stream.convert() read it there)
        convert.stream.untiled = lambda w, *places, **more: w
        print("page: broken on purpose: the value heads of every tensor are left in the file's order")
    elif broken:
        sys.exit(f"no break called {broken}: tiled")
    sink = Mapped(f"{out}.bin")
    began, size = time.perf_counter(), gguf.stat().st_size
    conversion = converted(answerer, hf, dtype="int8", max_seq_len=context, sink=sink, blocks=True)
    sink.data.flush()
    seconds = time.perf_counter() - began
    Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    Path(f"{out}.json").write_text(json.dumps(options))
    kinds = options.get("kinds") or {}
    types = sorted({kind for value in kinds.values() for kind in ([value] if isinstance(value, str) else value)})
    shown = {**options, "kinds": f"{len(kinds)} rows of {len(types)} types ({', '.join(types)})"}
    print(f"page: {gguf.name} ({size:,} bytes) to {out}.bin: {sink.size:,} bytes ({sink.size / 2 ** 30:.3f} GiB) of {sink.dtype} with its blocks kept, "
          f"header {sink.header}, in {seconds:.0f} s ({size / 1e6 / seconds:.0f} MB/s of the file); the process's own memory {peak:.0f} MB at most "
          f"(with the pages of the checkpoint's file it touched {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB)")
    print(f"page: the options of the conversion: {json.dumps(shown, ensure_ascii=False)}")
    print(f"page: a template the converter read: {'yes' if 'template' in conversion.options else 'none'}; "
          f"tokenizer.bin {len(conversion.tokenizer):,} bytes; the options as JSON {len(json.dumps(options)):,} characters")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    named = lambda name, default: arguments[arguments.index(name) + 1] if name in arguments else default
    if arguments[:1] == ["folder"]:
        folder(arguments[1], arguments[2])
    elif arguments[:1] == ["convert"]:
        convert_(arguments[1], arguments[2], int(named("--context", 4096)), named("--broken", ""))
    elif arguments[:1] == ["entries"]:
        made_up = "--made-up" in arguments
        Path(arguments[1]).write_text(json.dumps({name: {**entry, **({"options": {}} if made_up else {})} for name, entry in listed_entries().items()}, ensure_ascii=False))
    else:
        sys.exit("usage: python tests/page_saluki.py folder <work> <directory> | convert <folder> <out> [--context N] [--broken tiled] | entries <out.json>")
