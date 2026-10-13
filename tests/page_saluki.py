# page_saluki.py (T364.2): Underdog Saluki 27B converted as the page will convert it, its blocks kept as they are
# (Conversion(..., blocks=True) by the conduct of a conversion, T363.3), by the native Python of a CI runner, into a
# file: the checkpoint (7.96 GB) that tests/page-27b.mjs runs on the page's forward pass. tests/page_saluki.sh runs it;
# the development machine does not (the GGUF is 7.9 GB). The list has no entry for the model yet (T364.3), so what an
# entry would say is here: MODEL (the files) and ENTRIES (the options, the two formats and their sampling), which
# tests/page-27b.mjs takes with --as.
#
#   python tests/page_saluki.py folder <the directory with the GGUF> <a directory for the original's files>
#   python tests/page_saluki.py convert <folder> <out> [--context 4096] [--broken tiled]
#   python tests/page_saluki.py entries <out.json> [--made-up]
#
# folder: fetches the original's config.json, tokenizer and template (Qwen/Qwen3.8-27B's, as Ternary Bonsai 2 27B's
#   entry names them: what the conduct of the conversion asks for, tests/conducting.py's Fetched) and links the GGUF
#   beside them; prints the folder.
# convert: <out>.bin (a memory map), <out>.tokenizer.bin and <out>.json (the options the page would hand the engine:
#   the form's "kinds", a type to every matrix and to every layer of a stack, and "tiled"), with the seconds and the
#   peak of the process's own memory (a tensor whose rows move waits whole for its last byte: T363.3).
#   --broken tiled: the value heads of every tensor left in the file's order (llama.cpp tiles them, T245), for the
#   comparison to fail on: an error only a model with more value heads than key heads shows.
# entries: the entries the comparison and the writing take the page's options from, as JSON (--made-up: with no options
#   of their own, for the dry stage's small model, whose vocabulary has none of the special tokens).
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from page_27b import own_memory  # noqa: E402 (puts the engine's Python on the path)
import convert.stream  # noqa: E402
from conducting import Fetched, Mapped, Nothing, converted, listed  # noqa: E402
from reference_saluki import FILE, REPOSITORY, REVISION  # noqa: E402

# the original the vocabulary, the config.json and the template come from: Ternary Bonsai 2 27B's (the same base model)
MODEL = {"repo": REPOSITORY, "revision": REVISION, "weights": FILE,
         "vocabulary": {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "tokenizer": "tokenizer.json"}}
# What an entry of the list would hand the engine and the page (docs/notes/t364-saluki-2026-10-13.md has the draft and
# what the owner is to choose). The options and the two formats are the list's for a Qwen3.5 (src/models/formats.js):
# the BOS is the format's first token, <|im_start|>. The sampling: the card's for the form that thinks; for the form
# that answers at once the card says temperature 0 (greedy), and "sampled" is the list's set for a Qwen3.5 that answers
# at once, to write with beside it.
OPTIONS = {"bos": 248045, "stop_tokens": [248044, 248045, 248046],
           "specials": ["</tool_response>", "<tool_response>", "<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|repo_name|>",
                        "</tool_call>", "<|file_sep|>", "<|im_start|>", "<tool_call>", "<|fim_pad|>", "<|im_end|>", "</think>", "<think>"]}
AT_ONCE = "user\n{prompt:trim}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINKING = "user\n{prompt:trim}<|im_end|>\n<|im_start|>assistant\n<think>\n"
ENTRIES = {
    "at-once": {"id": "underdog-saluki-27b (answers at once, greedy: the card's)", "options": OPTIONS, "template": AT_ONCE,
                "generation": {"temperature": 0, "topp": 1, "repetition_penalty": 1}},
    "sampled": {"id": "underdog-saluki-27b (answers at once, a Qwen3.5's sampling)", "options": OPTIONS, "template": AT_ONCE,
                "generation": {"temperature": 0.7, "topp": 0.8, "repetition_penalty": 1, "top_k": 20, "presence_penalty": 1.5}},
    "thinking": {"id": "underdog-saluki-27b (thinking, the card's sampling)", "options": OPTIONS, "template": THINKING,
                 "generation": {"temperature": 0.6, "topp": 0.95, "repetition_penalty": 1, "top_k": 20}},
}


def folder(work, directory):
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
        Path(arguments[1]).write_text(json.dumps({name: {**entry, **({"options": {}} if made_up else {})} for name, entry in ENTRIES.items()}, ensure_ascii=False))
    else:
        sys.exit("usage: python tests/page_saluki.py folder <work> <directory> | convert <folder> <out> [--context N] [--broken tiled] | entries <out.json>")
