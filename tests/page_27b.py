# page_27b.py
# T233: Ternary Bonsai 2 27B converted as the page converts it (llama2_convert.Conversion, fed the file in order), by
# the native Python of a CI runner, into a file: the ternary checkpoint (7.66 GB) that tests/page-27b.mjs runs on the
# page's forward pass. tests/page_27b.sh runs it; the development machine does not (the GGUF is 5.95 GB).
#
#   python tests/page_27b.py convert <folder> <out> [--context 4096] [--broken tiled | order]
#   python tests/page_27b.py made-up <directory>
#
# convert. <folder>: what tests/hf_fetch.py makes for the list's entry (the original's config.json and tokenizer, the
#   GGUF linked beside them). Writes <out>.bin (a memory map: the checkpoint is never whole in memory), <out>.tokenizer.bin
#   and <out>.json (the options the page would hand the engine), and says the seconds and the memory: the peak of the
#   process's own memory (RssAnon, read after every piece fed; the two memory maps are the file system's pages).
#   --broken: a conversion wrong on purpose in a way only this model's files show, for the comparison to fail on:
#     tiled  the columns of a linear-attention layer's output matrix read in llama.cpp's order of the value heads, as
#            the other tensors of the value heads are stored (T245). A rotated GGUF holds these columns in Hugging
#            Face's order, and gguf_model() takes the mark off them (T237);
#     order  a PTQ1_0 block's values read in the order of its bytes (five digits of the first byte, then of the second),
#            where the file holds the first digit of every byte, then the second (T230's reader).
# made-up: a small model of the 27B's kind (tests/make_ternary.py's `rotated`: ternary, hybrid, three value heads to a
#   key head, a rotated basis) with references made by the engine's NumPy forward pass, in the files and layouts
#   tests/reference_27b.sh leaves, for a dry run of tests/page-27b.mjs's comparison in a minute (its numbers are not
#   held to lines: the reference here does not round the activations).
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion, Incomplete, gguf_weights  # noqa: E402

CHUNK = 8 << 20


def own_memory():
    """The megabytes of this process that are no file's pages (RssAnon)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return 0.0


class File:
    """The converter's sink: the checkpoint goes into <out>.bin, a memory map."""

    def __init__(self, path):
        self.path = path

    def open(self, size, header, dtype, form):
        self.size, self.header, self.dtype = size, list(header), dtype
        self.data = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


def bytes_first(raw):
    """A PTQ1_0 block read wrong on purpose: its values in the order of its bytes."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28)
    scales = np.ascontiguousarray(blocks[:, 26:]).view(np.float16).astype(np.float32)
    digits = np.concatenate([llama2_convert.base3(blocks[:, begin:end], count).transpose(0, 2, 1).reshape(len(blocks), -1)
                             for begin, end, count in ((0, 16, 5), (16, 24, 5), (24, 26, 4))], axis=1)
    return ((digits.view(np.int8) - np.int8(1)) * scales).reshape(-1)


def convert(folder, out, context, broken):
    folder = Path(folder)
    gguf = sorted(folder.glob("*.gguf"))[0]
    tokenizer = next(path for path in (folder / name for name in ("tokenizer.json", "spiece.model", "tokenizer.model")) if path.exists())
    config = (folder / "config.json").read_text()
    tokenizer_config = folder / "tokenizer_config.json"
    data = np.memmap(gguf, dtype=np.uint8, mode="r")
    size = 1 << 20
    while True:
        try:
            header, base = gguf_weights(bytes(data[:size]), config)
            break
        except Incomplete:
            size *= 2
    if broken == "tiled":
        # the mark gguf_model() takes off a rotated GGUF's output matrices, put back
        linear = llama2_convert.linear_layers(llama2_convert.normalize(json.loads(config)))
        after_keys, of_a_head, axis = llama2_convert.QWEN35_TILED["out_proj.weight"]
        header = json.loads(header)
        marked = [name for name in header if name.endswith("linear_attn.out_proj.weight")]
        for name in marked:
            header[name]["tiled"] = (2 * linear["key_heads"] * linear["key_dim"] if after_keys else 0, linear["key_heads"],
                                     linear["value_heads"] // linear["key_heads"], linear["value_dim"] if of_a_head else 1, axis)
        print(f"page: broken on purpose: the columns of {len(marked)} output matrices are read in llama.cpp's order of the value heads")
        header = json.dumps(header)
    if broken == "order":
        assert gguf.name.endswith("PTQ1_0.gguf"), "the order of a PTQ1_0 block is broken on the PTQ1_0 file"
        llama2_convert.READERS["PTQ1_0"] = (llama2_convert.READERS["PTQ1_0"][0], bytes_first)
        print("page: broken on purpose: a PTQ1_0 block's values are read in the order of its bytes")
    sink = File(f"{out}.bin")
    began, peak = time.perf_counter(), own_memory()
    conversion = Conversion(header, base, config, tokenizer.read_bytes(), tokenizer.name, dtype="ternary", max_seq_len=context,
                            start=base, tokenizer_config=tokenizer_config.read_text() if tokenizer_config.exists() else None, sink=sink)
    for start in range(base, len(data), CHUNK):
        conversion.feed(bytes(data[start:min(start + CHUNK, len(data))]))
        peak = max(peak, own_memory())
    conversion.finish()
    sink.data.flush()
    seconds = time.perf_counter() - began
    Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
    options = {key: value for key, value in conversion.options.items() if key != "template"}
    Path(f"{out}.json").write_text(json.dumps(options))
    shown = {**options, "rotated": options.get("rotated") and {"block": options["rotated"]["block"],
                                                               "signs": {width: f"{len(bits)} hex digits" for width, bits in options["rotated"]["signs"].items()}}}
    print(f"page: {gguf.name} ({len(data):,} bytes) to {out}.bin: {sink.size:,} bytes ({sink.size / 2 ** 30:.3f} GiB) of {sink.dtype}, header {sink.header}, "
          f"in {seconds:.0f} s ({len(data) / 1e6 / seconds:.0f} MB/s of the file); the process's own memory {peak:.0f} MB at most "
          f"(with the pages of the two files it touched {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB)")
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


if __name__ == "__main__":
    arguments = sys.argv[1:]
    named = lambda name, default: arguments[arguments.index(name) + 1] if name in arguments else default
    if arguments[:1] == ["convert"]:
        convert(arguments[1], arguments[2], int(named("--context", 4096)), named("--broken", ""))
    elif arguments[:1] == ["made-up"]:
        made_up(arguments[1])
    else:
        sys.exit("usage: python tests/page_27b.py convert <folder> <out> [--context N] [--broken tiled|order] | made-up <directory>")
