# page_27b.py
# T233: Ternary Bonsai 2 27B converted as the page converts it (the conduct of a conversion answered from the folder,
# tests/conducting.py: T374.4), by the native Python of a CI runner, into a file: the ternary checkpoint (7.66 GB) that tests/page-27b.mjs runs on the
# page's forward pass. tests/page_27b.sh runs it; the development machine does not (the GGUF is 5.95 GB).
#
#   python tests/page_27b.py convert <folder> <out> [--context 4096] [--broken tiled | order]
#   python tests/page_27b.py made-up <directory>
#
# convert. <folder>: what tests/hf_fetch.py makes for the list's entry (the original's config.json and tokenizer, the
#   GGUF linked beside them). Writes <out>.bin (a memory map: the checkpoint is never whole in memory), <out>.tokenizer.bin
#   and <out>.json (the options the page would hand the engine), and says the seconds (of the whole conduct: the head,
#   the tokenizer and the template too) and the memory: the peak of the process's own memory (RssAnon, read after every
#   part of the stream fed; the checkpoint's memory map is the file system's pages).
#   --broken: a conversion wrong on purpose in a way only this model's files show, for the comparison to fail on:
#     tiled  the columns of a linear-attention layer's output matrix read in llama.cpp's order of the value heads, as
#            the other tensors of the value heads are stored (T245). A rotated GGUF holds these columns in Hugging
#            Face's order, and gguf_model() takes the mark off them (T237);
#     order  a PTQ1_0 block's values read in the order of its bytes (five digits of the first byte, then of the second),
#            where the file holds the first digit of every byte, then the second (T230's reader).
# made-up: a small model of the 27B's kind (tests/make_ternary.py's `rotated`: ternary, hybrid, three value heads to a
#   key head, a rotated basis) with references made by the engine's NumPy forward pass, in the files and layouts
#   tests/reference_27b.sh leaves, for a dry run of tests/page-27b.mjs's comparison in a minute (its numbers are not
#   held to lines: the reference here does not round the activations). And (T374.4) two folders as tests/hf_fetch.py
#   makes one for the 27B, each with a small ternary GGUF (PTQ1_0, PQ2_0): what `convert` and page-27b.mjs's convert
#   take, for a dry run of both.
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from conducting import Mapped, converted, listed  # noqa: E402
from convert import conduct as conducting_part  # noqa: E402


def own_memory():
    """The megabytes of this process that are no file's pages (RssAnon)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return 0.0


def bytes_first(raw):
    """A PTQ1_0 block read wrong on purpose: its values in the order of its bytes."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    scales = np.ascontiguousarray(blocks[:, 26:]).view(np.float16).astype(np.float32)
    digits = np.concatenate([llama2_convert.base3(blocks[:, begin:end], count).transpose(0, 2, 1).reshape(len(blocks), -1)
                             for begin, end, count in ((0, 16, 5), (16, 24, 5), (24, 26, 4))], axis=1)
    return ((digits.view(np.int8) - np.int8(1)) * scales).reshape(-1)


def tiled(header, config):
    """The mark gguf_model() takes off a rotated GGUF's output matrices, put back."""
    linear = llama2_convert.linear_layers(llama2_convert.normalize(json.loads(config)))
    after_keys, of_a_head, axis = llama2_convert.QWEN35_TILED["out_proj.weight"]
    header = json.loads(header)
    marked = [name for name in header if name.endswith("linear_attn.out_proj.weight")]
    for name in marked:
        header[name]["tiled"] = (2 * linear["key_heads"] * linear["key_dim"] if after_keys else 0, linear["key_heads"],
                                 linear["value_heads"] // linear["key_heads"], linear["value_dim"] if of_a_head else 1, axis)
    print(f"page: broken on purpose: the columns of {len(marked)} output matrices are read in llama.cpp's order of the value heads")
    return json.dumps(header)


def convert(folder, out, context, broken):
    # (the folder as tests/hf_fetch.py makes it: the first .gguf by its name, with the original's files beside it)
    peak = own_memory()

    def fed():
        nonlocal peak
        peak = max(peak, own_memory())

    hf, answerer = listed(folder, fed=fed)
    gguf = Path(folder) / hf["weights"]
    if broken == "tiled":
        # between the header the conduct makes of the GGUF's head and the conversion it makes with it: in the conduct's
        # own name for the reader (set in the part that reads the name, as a test would)
        reading = conducting_part.gguf_weights

        def marked(head, config):
            header, base = reading(head, config)
            return tiled(header, config), base

        conducting_part.gguf_weights = marked
    if broken == "order":
        assert gguf.name.endswith("PTQ1_0.gguf"), "the order of a PTQ1_0 block is broken on the PTQ1_0 file"
        llama2_convert.SOURCES["PTQ1_0"] = llama2_convert.SOURCES["PTQ1_0"]._replace(read=bytes_first)
        print("page: broken on purpose: a PTQ1_0 block's values are read in the order of its bytes")
    sink = Mapped(f"{out}.bin")
    began, size = time.perf_counter(), gguf.stat().st_size
    conversion = converted(answerer, hf, dtype="ternary", max_seq_len=context, sink=sink)
    sink.data.flush()
    seconds = time.perf_counter() - began
    Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    Path(f"{out}.json").write_text(json.dumps(options))
    shown = {**options, "rotated": options.get("rotated") and {"block": options["rotated"]["block"],
                                                               "signs": {width: f"{len(bits)} hex digits" for width, bits in options["rotated"]["signs"].items()}}}
    print(f"page: {gguf.name} ({size:,} bytes) to {out}.bin: {sink.size:,} bytes ({sink.size / 2 ** 30:.3f} GiB) of {sink.dtype}, header {sink.header}, "
          f"in {seconds:.0f} s ({size / 1e6 / seconds:.0f} MB/s of the file); the process's own memory {peak:.0f} MB at most "
          f"(with the pages of the checkpoint's file it touched {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB)")
    print(f"page: the options of the conversion: {json.dumps(shown, ensure_ascii=False)}")
    print(f"page: a template the converter read: {'yes' if 'template' in conversion.options else 'none (the list writes the format)'}; "
          f"tokenizer.bin {len(conversion.tokenizer):,} bytes")


def made_up(directory):
    """A made-up model as <directory>/page.{bin,tokenizer.bin,json} and its references in <directory>."""
    import subprocess

    from llama2_numpy import Llama
    from reference_27b import save_run
    directory = Path(directory)
    (directory / "f32").mkdir(parents=True, exist_ok=True)
    (directory / "saved").mkdir(exist_ok=True)
    subprocess.check_call([sys.executable, str(HERE / "make_ternary.py"), str(directory / "page"), "ternary", "rotated"])
    options = json.loads((directory / "page.json").read_text())
    model = Llama((directory / "page.bin").read_bytes(), (directory / "page.tokenizer.bin").read_bytes(), kernels=None, **options)
    for index, prompt in enumerate(([1, 5, 7, 9, 11], [3, 300, 2, 2, 9, 40, 41])):
        ids, rows = list(prompt), []
        for position in range(len(prompt) + 5):
            rows.append(np.array(model.forward(ids[position], position)))
            if position >= len(prompt) - 1:
                ids.append(int(rows[-1].argmax()))
        wrote = ids[len(prompt):]
        (directory / f"fork-{index}.ids").write_text(" ".join(map(str, prompt)) + "\n" + " ".join(map(str, wrote)) + "\n")
        np.asarray(rows, dtype=np.float32).tofile(directory / "f32" / f"fork-{index}.single")
        save_run(directory / "saved", index, "as 8 bits round", model, rows)
        print(f"page: made-up text {index}: {len(prompt)} tokens of a prompt, {len(wrote)} written, {len(rows)} positions")
    # a long text for page-27b.mjs's long mode (the review of T233), in the files the fork's long run leaves: the ids (the
    # prompt's, the tokens written), the positions whose rows are kept, the rows of those and then of each token written
    # but the last. The made-up model's own context is 1024 positions: 700 of them are the prompt
    long_prompt = np.random.default_rng(7).integers(1, 380, 700).tolist()
    wanted = [10, 20, 40, 100, 300, 500, 600, len(long_prompt) - 1]
    ids, kept, wrote = list(long_prompt), [], []
    for position in range(len(long_prompt) + 5):
        row = np.array(model.forward(ids[position], position))
        if position in wanted or position >= len(long_prompt):
            kept.append(row)
        if position >= len(long_prompt) - 1:
            wrote.append(int(row.argmax()))
            ids.append(wrote[-1])
    (directory / "fork-long.ids").write_text(" ".join(map(str, long_prompt)) + "\n" + " ".join(map(str, wrote)) + "\n")
    (directory / "fork-long.rows").write_text(" ".join(map(str, wanted)) + "\n")
    np.asarray(kept, dtype=np.float32).tofile(directory / "fork-long.logits")
    print(f"page: made-up long text: {len(long_prompt)} tokens, rows of {len(wanted)} positions of it and of {len(wrote) - 1} written after it")
    # T374.4: and a small ternary model as the 27B is published, for a dry run of the two conversions (`convert` here and
    # of tests/page-27b.mjs), which no other stage of the dry run reaches: a folder with the original's config.json and
    # tokenizer and a GGUF of each ternary type beside them (tests/test_ternary.py's made-up Qwen3: the same values
    # in both files, as the 27B's two files hold the same model)
    from test_gguf import unigram
    from test_ternary import bonsai
    for kind in ("PTQ1_0", "PQ2_0"):
        shape, published, file, _ = bonsai(kind)
        folder = directory / f"gguf-{kind}"
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(json.dumps(published))
        (folder / "tokenizer.json").write_bytes(unigram(shape["vocab_size"]))
        (folder / f"made-up-{kind}.gguf").write_bytes(file)
        print(f"page: made-up GGUF of {kind} blocks: {len(file):,} bytes, with the config.json and the tokenizer of its original beside it")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    named = lambda name, default: arguments[arguments.index(name) + 1] if name in arguments else default
    if arguments[:1] == ["convert"]:
        convert(arguments[1], arguments[2], int(named("--context", 4096)), named("--broken", ""))
    elif arguments[:1] == ["made-up"]:
        made_up(arguments[1])
    else:
        sys.exit("usage: python tests/page_27b.py convert <folder> <out> [--context N] [--broken tiled|order] | made-up <directory>")
