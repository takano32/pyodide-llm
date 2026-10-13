# reference_saluki_numpy.py (T364.1): the engine's own NumPy forward pass (Llama.forward, as it is) over Underdog Saluki
# 27B's GGUF, read by memory map the way tests/reference_27b.py reads Ternary Bonsai 2 27B's: a matrix is widened a few
# hundred megabytes at a time by the converter's readers (T362: convert/superblocks.py, held to llama.cpp's C to the
# bit), multiplied by every vector that waits for it and let go. Its machinery is reference_27b.py's (the Conductor, a
# stored matrix, the comparison); what is this file's is the model: a stock llama.cpp GGUF of a Qwen3.5 with no rotated
# basis, whose linear-attention layers' output matrices have their columns in llama.cpp's order of the value heads.
#
#   python tests/reference_saluki_numpy.py <the GGUF> <the directory with fork-<i>.ids (and what else is there)>
#   python tests/reference_saluki_numpy.py made-up <directory>
#
# What it is for. It is the narrow reference of the page's forward pass (tests/page_saluki.sh): it reads the GGUF and
# not the checkpoint the page converts it to, and it reads it the way the widening conversion does (every tensor back in
# Hugging Face's order, the output matrix's input put in the file's order of heads here, by a rule written a second
# time: grouped()), so a mistake of the conversion that keeps the blocks (rows moved as bytes, the form's "tiled") or of
# the kernels that multiply by them is not also in it. Stock llama.cpp cannot be that reference: its logits move by 0.3
# to 11 between its own two paths and between CPUs (T361).
#
# The runs, for every text (ROUNDINGS): "float32"; "as 8 bits round" (the activations of every large matrix rounded as
# the page's kernels of the blocks round them, quantize_x: a scale to 32 values, 8 bits: the run that stands for the
# page); "as llama.cpp rounds" (as stock llama.cpp's CPU rounds them, Q8_K: a scale to 256 values, and Q8_0 for the two
# small matrices of the gates: how far that alone moves the model). And on the first text the engine broken on purpose
# (BREAKS), to see what a comparison with these numbers catches.
# What it says: every run against "float32"; "float32" and "as llama.cpp rounds" against stock llama.cpp's two paths
# (fork-<i>.logits and .single, if there); and "float32" against llama.cpp with float32 activations (f32/fork-<i>.single,
# tests/reference_saluki_patch.py, if it ran): the independent check of the whole, held to TIGHT_LINES where they are
# given (the environment: "0.03,1e-4", reference_27b.py's for Bonsai; without, the numbers alone), and every broken
# run then has to be past them.
# SAVE_LOGITS=<a directory>: the logits, keys and values of "float32" and "as 8 bits round" (reference_27b.py's
# save_run()), which tests/page-27b.mjs compare reads.
#
# made-up: a small GGUF of the same kind (tests/test_kept.py's: ten types by the layer, three value heads to a key
# head, a classifier of its own) in <directory>/gguf with its original's files, ids in the driver's form, and this
# file's own pass over it saved: what the dry stage of tests/page_saluki.sh converts and compares.
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference_27b  # noqa: E402 (puts the engine's Python on the path)
import engine.model  # noqa: E402
import gguf_check  # noqa: E402
import llama2_numpy  # noqa: E402
from llama2_convert import SOURCES, gguf_tokenizer, transformed  # noqa: E402
from llama2_numpy import Llama  # noqa: E402
from reference_27b import Conductor, Distance, Matrix, Rows, Small, Stored, as_page, as_q8_0, compare, grouped, rows_of_file, save_run  # noqa: E402

# the float32 values of a matrix widened at a time (the readers of the blocks hold a few arrays of that many besides)
reference_27b.CHUNK = int(os.environ.get("CHUNK", 16 << 20))
BY_TYPE = {source.ggml: source for source in SOURCES.values()}


def as_q8_k(x):
    """x as ggml's quantize_row_q8_K_ref stores it, which every dot product with a k- or i-quantized row reads: blocks
    of 256, iscale = -127 / (the value of the largest magnitude), q = min(127, nearest(iscale * x)), d = 1 / iscale;
    the values q * d (llama.cpp at 4413a192, ggml/src/ggml-quants.c; MIT, the rule and no line)."""
    blocks = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 256)
    largest = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=1)[:, None], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = np.where(largest != 0, np.float32(-127.0) / largest, np.float32(0.0)).astype(np.float32)
        d = np.where(iscale != 0, np.float32(1.0) / iscale, np.float32(0.0)).astype(np.float32)
    q = np.minimum(np.rint(blocks * iscale), np.float32(127.0))
    return (q * d).reshape(np.shape(x)).astype(np.float32)


def nudged(seed):
    """as_page(127) of the activations moved by a millionth of their group's largest first (a random amount for each,
    of a seed): the rounding the page's kernels do, of numbers as far from the engine's as another order of float32
    sums puts them. What two such runs differ by is what forward.ts may differ by from the run that rounds alike."""
    rng, rounded = np.random.default_rng(seed), as_page(127)

    def moved(x):
        blocks = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
        noise = rng.uniform(-1e-6, 1e-6, blocks.shape).astype(np.float32) * np.abs(blocks).max(axis=1, keepdims=True)
        return rounded((blocks + noise).reshape(np.shape(x)))
    return moved


# what every large matrix reads, and what the two small ones of a linear layer's gates read (Q8_0 in the file: llama.cpp
# rounds their activation to Q8_0; the page keeps them float32)
ROUNDINGS = {"float32": (None, None), "as 8 bits round": (as_page(127), None), "as llama.cpp rounds": (as_q8_k, as_q8_0)}
# NUDGED="0,2" (the environment): for these texts two more runs, "nudged 1" and "nudged 2" (nudged())
NUDGES = {"nudged 1": 1, "nudged 2": 2}
BREAKS = {
    "untiled output": "the output matrices of the linear layers given the value heads in Hugging Face's order (their columns are in llama.cpp's)",
    "tiled": "the value heads of every tensor read in the file's order",
    "halves": "q and k read as if no part of a head were stored in halves",
    "gate": "q and its gate taken for each other",
}


def rows_of(info, data, base, first, last):
    """Rows first..last of a GGUF's tensor as float32: llama.cpp's blocks by the converter's readers."""
    source = BY_TYPE[info["type"]]
    if source.values == 1 or source.name == "Q8_0":
        return gguf_check.tensor(info, data, base, first, last)[0]
    width = info["shape"][-1]
    row = width // source.values * source.size
    start = base + info["offset"] + first * row
    return source.read(bytes(data[start:start + (last - first) * row])).reshape(last - first, width)


class Source:
    """A GGUF by memory map: its metadata, its matrices as Stored, its small tensors as arrays."""

    def __init__(self, path, positions):
        _, self.metadata, self.infos, self.data, self.base = gguf_check.read_gguf(path)
        self.positions, self.matrices = positions, {}

    def stored(self, name):
        if name not in self.matrices:
            info = self.infos[name]
            rows, width = info["shape"]
            self.matrices[name] = Stored(name, rows, width, lambda first, last: rows_of(info, self.data, self.base, first, last))
        return self.matrices[name]

    def small(self, name):
        info = self.infos[name]
        return np.array(rows_of(info, self.data, self.base, 0, info["shape"][0]), dtype=np.float32)


class Streamed(Llama):
    """The engine over a stock llama.cpp GGUF of a Qwen3.5: Llama's own constructor and forward pass, with the tensors
    taken from the GGUF where the engine takes them from its checkpoint. rounding: ROUNDINGS; broken: BREAKS."""

    def __init__(self, source, conductor, rounding="float32", broken=""):
        self.source, self.conductor, self.rounding, self.broken = source, conductor, rounding, broken
        metadata, key = source.metadata, lambda name: source.metadata[f"qwen35.{name}"]
        heads, kv_heads, head = key("attention.head_count"), key("attention.head_count_kv"), key("attention.key_length")
        values = key("ssm.time_step_rank")
        linear = {"every": key("full_attention_interval"), "key_heads": key("ssm.group_count"), "value_heads": values,
                  "key_dim": key("ssm.state_size"), "value_dim": key("ssm.inner_size") // values, "conv": key("ssm.conv_kernel")}
        vocabulary = source.infos["token_embd.weight"]["shape"][0]
        header = struct.pack("<7i", key("embedding_length"), key("feed_forward_length"), key("block_count"), heads, kv_heads,
                             -vocabulary, source.positions)
        # float16: the RoPE tables are the engine's own (rope_tables()), not a file's
        super().__init__(header, None, dtype="float16", arch="qwen35", linear=linear, head_dim=head,
                         rotary=key("rope.dimension_count"), rope_theta=float(key("rope.freq_base")),
                         rms_norm_eps=float(f"{key('attention.layer_norm_rms_epsilon'):.6g}"),
                         bos=metadata.get("tokenizer.ggml.bos_token_id", 1))

    def file_tensors(self, checkpoint, rows, dtype, shared_weights, external):
        """Llama.file_tensors() from a GGUF: the same attributes, the large matrices behind the conductor."""
        source, linear, broken = self.source, self.linear, self.broken
        heads, kv_heads, head, rot = self.n_heads, self.n_kv_heads, self.head_size, self.rotary
        key_heads, value_heads, value_dim = linear["key_heads"], linear["value_heads"], linear["value_dim"]
        keys = key_heads * linear["key_dim"]
        full = [l for l, (lines, _) in enumerate(self.slots) if not lines]
        lines = [l for l, (kind, _) in enumerate(self.slots) if kind]
        rounding, rounding_small = (nudged(NUDGES[self.rounding]), None) if self.rounding in NUDGES else ROUNDINGS[self.rounding]
        small = rounding_small or (lambda x: x)
        # the rows of the engine's matrices among the stored ones (as reference_27b.py's: the converter's "heads"
        # transform applied to the rows' numbers)
        turned = 0 if broken == "halves" else rot
        index = lambda rows, transform: transformed(np.arange(rows)[:, None], transform, head).reshape(-1)
        q_order, gate_order = index(2 * heads * head, ("heads", 2, 0, heads, turned)), index(2 * heads * head, ("heads", 2, 1, heads, 0))
        if broken == "gate":
            q_order, gate_order = gate_order, q_order
        k_order = index(kv_heads * head, ("heads", 1, 0, kv_heads, turned))
        norm_order = index(head, ("heads", 1, 0, 1, turned))
        # the value heads: Hugging Face's order from the GGUF's (llama.cpp tiles them where a key head has several)
        same = broken == "tiled"
        head_order = np.arange(value_heads) if same else grouped(key_heads, value_heads)
        v_order = np.arange(value_heads * value_dim) if same else grouped(key_heads, value_heads, value_dim)
        qkv_order = np.concatenate([np.arange(2 * keys), 2 * keys + v_order])
        # the output matrix's columns are in the file's order of the value heads (a stock GGUF: no rotated basis took
        # that mark off them), so the vector it reads goes there: place v_order[i] of it is the engine's value i
        columns = np.arange(value_heads * value_dim) if same or broken == "untiled output" else v_order

        def tiled(x):
            moved = np.empty_like(x)
            moved[columns] = x
            return moved

        def stack(name, layers, order=None, change=None):
            prepare = rounding if change is None else change if rounding is None else (lambda x: rounding(change(x)))
            return [Matrix(self.conductor, source.stored(f"blk.{l}.{name}.weight"), order, prepare) for l in layers]

        def vectors(name, layers, change=lambda v: v):
            return np.stack([change(source.small(f"blk.{l}.{name}")) for l in layers]).astype(np.float32)

        everything = range(self.n_layers)
        self.token_embedding_table = Rows(source.stored("token_embd.weight"))
        # (llama.cpp wrote the norms with the 1 the model adds to them, and ssm_a as -exp(A_log): T236)
        self.rms_att_weight = vectors("attn_norm.weight", everything)
        self.wq, self.wg = stack("attn_q", full, q_order), stack("attn_q", full, gate_order)
        self.wk, self.wv, self.wo = stack("attn_k", full, k_order), stack("attn_v", full), stack("attn_output", full)
        self.q_norm = vectors("attn_q_norm.weight", full, lambda v: v[norm_order])
        self.k_norm = vectors("attn_k_norm.weight", full, lambda v: v[norm_order])
        self.bq = self.bk = self.bv = None
        self.wqkv, self.wz = stack("attn_qkv", lines, qkv_order), stack("attn_gate", lines, v_order)
        self.wb = Small(vectors("ssm_beta.weight", lines, lambda w: w[head_order]), small)
        self.wa = Small(vectors("ssm_alpha.weight", lines, lambda w: w[head_order]), small)
        # (channels, taps) in the file; the engine's rows are the taps
        self.conv = vectors("ssm_conv1d.weight", lines, lambda w: w[qkv_order].T)
        self.dt_bias = vectors("ssm_dt.bias", lines, lambda v: v[head_order])
        self.decay = vectors("ssm_a", lines, lambda v: v[head_order])
        self.delta_norm = vectors("ssm_norm.weight", lines)
        self.wout = stack("ssm_out", lines, change=tiled)
        self.rms_ffn_weight = vectors("post_attention_norm.weight", everything)
        self.w1, self.w2, self.w3 = stack("ffn_gate", everything), stack("ffn_down", everything), stack("ffn_up", everything)
        self.rms_final_weight = source.small("output_norm.weight")
        self.wcls = Matrix(self.conductor, source.stored("output.weight"), None, rounding)


def texts_of(work):
    texts = []
    for index in range(100):
        ids = work / f"fork-{index}.ids"
        if not ids.exists():
            break
        prompt, wrote = ([int(token) for token in line.split()] for line in ids.read_text().split("\n")[:2])
        text = (work / f"prompt-{index}.txt").read_text() if (work / f"prompt-{index}.txt").exists() else None
        texts.append({"prompt": prompt, "wrote": wrote, "text": text, "ids": prompt + wrote[:-1]})
    assert texts, f"no fork-<i>.ids in {work}"
    return texts


def passed(source, texts, runs_of):
    """One pass over the file for the runs runs_of(conductor) makes ([(text, name, model)]): {(text, name): rows}."""
    conductor = Conductor()
    runs = runs_of(conductor)
    jobs = [(lambda model=model, token=token, position=position: np.array(model.forward(token, position)))
            for index, _, model in runs for position, token in enumerate(texts[index]["ids"])]
    print(f"reference: {len(jobs)} positions of {len(runs)} runs in one pass over the file", flush=True)
    began = time.perf_counter()
    results = conductor.run(jobs)
    seconds = time.perf_counter() - began
    print(f"reference: the pass took {seconds:.0f} s: {conductor.seconds['widening']:.0f} s widening "
          f"{conductor.values / 1e9:.1f} G weights ({conductor.values / max(conductor.seconds['widening'], 1e-9) / 1e6:.0f} M a second), "
          f"{conductor.seconds['multiplying']:.0f} s multiplying them by {len(jobs)} vectors", flush=True)
    logits, at = {}, 0
    for index, name, _ in runs:
        count = len(texts[index]["ids"])
        logits[(index, name)] = results[at:at + count]
        at += count
    return runs, logits


def main(gguf, work):
    texts = texts_of(work)
    source = Source(gguf, positions=max(len(text["ids"]) for text in texts) + 2)
    vocab = source.infos["token_embd.weight"]["shape"][0]
    tokenizer = None
    if any(text["text"] is not None for text in texts):
        # the engine's tokenizer, from the GGUF's vocabulary as the converter reads it, against llama.cpp's ids
        tokenizer_bin, options, _, controls, added = gguf_tokenizer(source.metadata, vocab)
        tokenizer = llama2_numpy.Tokenizer(tokenizer_bin, vocab, kind=options["tokenizer_kind"], nfc=options["nfc"],
                                           pretokenizer=options["pretokenizer"], ignore_merges=options["ignore_merges"])
        specials = sorted({*controls, *added} - {""}, key=lambda token: (-len(token), token))
        for index, text in enumerate(texts):
            if text["text"] is not None:
                ids = tokenizer.encode(text["text"], specials=specials)
                text["tokens"] = ids == text["prompt"]
                said = "the same ids as llama.cpp" if text["tokens"] else f"{ids}, llama.cpp {text['prompt']}"
                print(f"reference: text {index}: the engine's tokenizer gives {said}")
    engine.model.Tokenizer = lambda *arguments, **named: tokenizer  # (every run's Llama takes this one)
    for index, text in enumerate(texts):
        text["batch"] = rows_of_file(work / f"fork-{index}.logits", vocab)
        text["single"] = rows_of_file(work / f"fork-{index}.single", vocab)
        text["f32"] = rows_of_file(work / "f32" / f"fork-{index}.single", vocab)
    print(f"reference: llama.cpp with float32 activations wrote {[None if text['f32'] is None else len(text['f32']) for text in texts]} "
          f"positions of {[len(text['ids']) for text in texts]}", flush=True)

    again = [int(index) for index in os.environ.get("NUDGED", "").split(",") if index]
    runs, logits = passed(source, texts, lambda conductor: (
        [(index, name, Streamed(source, conductor, rounding=name)) for index in range(len(texts)) for name in ROUNDINGS]
        + [(index, name, Streamed(source, conductor, rounding=name)) for index in again for name in NUDGES]
        + [(0, name, Streamed(source, conductor, broken=name)) for name in BREAKS]))
    if os.environ.get("SAVE_LOGITS"):
        saved = Path(os.environ["SAVE_LOGITS"])
        saved.mkdir(parents=True, exist_ok=True)
        for index, name, model in runs:
            if name in ("float32", "as 8 bits round", *NUDGES):
                save_run(saved, index, name, model, logits[(index, name)])
        files = sorted(saved.glob("engine-*"))
        print(f"reference: the logits, keys and values of the runs for the page are in {saved}: {len(files)} files, "
              f"{sum(file.stat().st_size for file in files) / 1e6:.0f} MB", flush=True)

    lines = [float(value) for value in os.environ["TIGHT_LINES"].split(",")] if os.environ.get("TIGHT_LINES") else None
    failed = []
    for index, text in enumerate(texts):
        prompt, ours = len(text["prompt"]), logits[(index, "float32")]
        greedy = [int(np.argmax(row)) for row in ours[prompt - 1:]]
        print(f"reference: text {index}: {prompt} tokens of a prompt, then {text['wrote']}; float32's greedy tokens after the same ones: {greedy} "
              f"({sum(a == b for a, b in zip(greedy, text['wrote']))} of {len(text['wrote'])} the same)")
        if text.get("tokens") is False:
            failed.append(f"text {index}: the tokenizer")
        compare(f"text {index}, as 8 bits round against float32 (the page's rounding of the activations)", logits[(index, "as 8 bits round")], ours)
        compare(f"text {index}, as llama.cpp rounds against float32 (llama.cpp's rounding of the activations)", logits[(index, "as llama.cpp rounds")], ours)
        if index in again:
            # (how far the same rounding is from itself where the numbers before it differ in their last bits)
            for name in NUDGES:
                compare(f"text {index}, {name} against as 8 bits round (the same rounding of activations a millionth off)", logits[(index, name)], logits[(index, "as 8 bits round")])
            compare(f"text {index}, nudged 1 against nudged 2", logits[(index, "nudged 1")], logits[(index, "nudged 2")])
            by = [float(np.max(np.abs(a - b))) for a, b in zip(logits[(index, "nudged 1")], logits[(index, "as 8 bits round")])]
            print(f"reference: text {index}, nudged 1 against as 8 bits round, by position: {' '.join(f'{value:.2f}' for value in by)}")
        for name, theirs in (("llama.cpp a token at a time", text["single"]), ("llama.cpp's batch", text["batch"])):
            if theirs is not None:
                compare(f"text {index}, float32 against {name}", ours, theirs)
                compare(f"text {index}, as llama.cpp rounds against {name}", logits[(index, "as llama.cpp rounds")], theirs)
        if text["single"] is not None and text["batch"] is not None:
            compare(f"text {index}, llama.cpp's batch against llama.cpp a token at a time", text["batch"][:prompt], text["single"])
        if text["f32"] is not None and len(text["f32"]):
            near = compare(f"text {index}, float32 against llama.cpp with float32 activations", ours, text["f32"])
            if text["single"] is not None:
                compare(f"text {index}, llama.cpp with float32 activations against llama.cpp as it is (its rounding)", text["f32"], text["single"])
            if lines:
                ok = near.worst <= lines[0] and near.kl_worst <= lines[1] and all(gap <= 2 * near.worst for gap in near.gaps)
                print(f"reference: text {index}: {'ok' if ok else 'FAILED'} against llama.cpp with float32 activations: {near.worst:.4f} against "
                      f"the line {lines[0]}, the KL at most {near.kl_worst:.1e} against {lines[1]:.0e}, {near.count} positions", flush=True)
                if not ok:
                    failed.append(f"text {index} against llama.cpp with float32 activations")
    # the broken runs: how far each is from the right one, beside what the page's rounding of the activations moves it
    # by; held to the lines (each must be past them) where there are lines
    unbroken, single = logits[(0, "float32")], texts[0]["f32"]
    floor = Distance(logits[(0, "as 8 bits round")], unbroken)
    for name, what in BREAKS.items():
        far = compare(f"broken, {what}, against float32", logits[(0, name)], unbroken)
        print(f"reference: broken, {what}: {far.worst / max(floor.worst, 1e-9):.1f} times what the page's rounding moves the logits by "
              f"({floor.worst:.4f}), {far.kl_worst / max(floor.kl_worst, 1e-12):.1f} times its largest KL ({floor.kl_worst:.1e})", flush=True)
        if single is not None and len(single):
            near = compare(f"broken, {what}, against llama.cpp with float32 activations", logits[(0, name)], single)
            if lines and not (near.worst > lines[0] or near.kl_worst > lines[1]):
                failed.append(f"broken, {what}: within the lines of llama.cpp with float32 activations")
    if failed:
        print(f"reference: FAILED: {', '.join(failed)}")
        return 1
    print("reference: ok: the engine's NumPy forward pass over the GGUF, its runs and its broken ones are written")
    return 0


def made_up(directory):
    """<directory>/gguf: a small GGUF of the kind with its original's config.json and tokenizer (what tests/page_saluki.py
    convert and tests/page-27b.mjs convert take); <directory>: ids of two texts in the driver's form and this file's own
    pass over the GGUF, as SAVE_LOGITS leaves it, with f32/fork-<i>.single the float32 run's logits (the place of
    llama.cpp with float32 activations: the comparison's other reference, here the same engine)."""
    from test_kept import Model
    made = Model(seq_len=128)  # (a context the format of a question fits)
    folder = directory / "gguf"
    (directory / "f32").mkdir(parents=True, exist_ok=True)
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text(json.dumps(made.config))
    (folder / "tokenizer.json").write_bytes(made.vocabulary)
    (folder / "made-up-saluki.gguf").write_bytes(made.file)
    prompts = ([1, 5, 7, 9, 11], [3, 30, 2, 2, 9, 20, 21])
    source = Source(folder / "made-up-saluki.gguf", positions=16)
    engine.model.Tokenizer = lambda *arguments, **named: None
    # the tokens written: greedy by the float32 run, a position at a time (the Conductor serves one job as well)
    for index, prompt in enumerate(prompts):
        conductor, ids = Conductor(), list(prompt)
        writer = Streamed(source, conductor)
        for position in range(len(prompt) + 4):
            row = conductor.run([lambda token=ids[position], position=position: np.array(writer.forward(token, position))])[0]
            if position >= len(prompt) - 1:
                ids.append(int(row.argmax()))
        (directory / f"fork-{index}.ids").write_text(" ".join(map(str, prompt)) + "\n" + " ".join(map(str, ids[len(prompt):])) + "\n")
    texts = texts_of(directory)
    runs, logits = passed(source, texts, lambda conductor: [(index, name, Streamed(source, conductor, rounding=name))
                                                           for index in range(len(texts)) for name in ("float32", "as 8 bits round")])
    (directory / "saved").mkdir(exist_ok=True)
    for index, name, model in runs:
        save_run(directory / "saved", index, name, model, logits[(index, name)])
        if name == "float32":
            np.asarray(logits[(index, name)], dtype=np.float32).tofile(directory / "f32" / f"fork-{index}.single")
    # a broken run changes the logits, and the right one is the converter's reading of the same file
    broken = passed(source, texts[:1], lambda conductor: [(0, name, Streamed(source, conductor, broken=name)) for name in BREAKS])[1]
    for name in BREAKS:
        far = Distance(broken[(0, name)], logits[(0, "float32")])
        print(f"reference: made-up, broken ({name}): {far.worst:.3e} from float32")
        assert far.worst > 1e-3, name
    widened = made.converted(blocks=False, dtype="float32")
    wide = Llama(bytes(widened.checkpoint), bytes(widened.tokenizer), **widened.options)
    rows = np.stack([wide.forward(token, position) for position, token in enumerate(texts[0]["ids"])])
    near = Distance(logits[(0, "float32")], rows)
    largest = float(np.abs(rows).max())
    print(f"reference: made-up: this file's float32 pass over the GGUF is {near.worst:.3e} from the engine on the widened conversion "
          f"(the largest logit is {largest:.2f}), the same largest logit at {near.same} of {near.count} positions")
    assert near.worst <= 2e-4 * max(largest, 1.0), near.worst
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "made-up":
        sys.exit(made_up(Path(sys.argv[2])))
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
