# reference_27b.py
# T238: the engine's own NumPy forward pass (public/llama2_numpy.py's Llama.forward, as it is) over Ternary Bonsai 2
# 27B, whose matrices widened to float32 are 107 GB: the GGUF is read by memory map and a matrix is widened a few
# hundred megabytes at a time, multiplied and let go. What is checked is the engine's computation on the real model:
# the hybrid attention, the rotated basis (T237), the order of the value heads in a GGUF (llama.cpp stores them
# another way where they outnumber the key heads) and how q, its gate and k are stored, against what Prism ML's fork
# of llama.cpp computes (tests/reference_27b_fork.cpp wrote its ids and logits; tests/reference_27b.sh runs both).
#
#   python tests/reference_27b.py <the GGUF> <the directory with fork-<i>.ids, fork-<i>.logits, prompt-<i>.txt>
#
# One pass over the file serves every position of every text: a matrix is widened once and multiplied by all the
# vectors that wait for it (Conductor). The forward pass is the engine's, a position at a time, each in a thread of
# its own that stops where it multiplies by a matrix of the file; the threads run one after the other in the order of
# the positions, so a position finds the state and the keys the one before it left.
#
# The time (the estimate, before any run): one pass widens the 25.6 G weights a token multiplies by (all but the
# embedding) and multiplies them by C vectors, 25.6 G x C multiply-adds: about 25.6e9 / (1e8 values a second of
# NumPy's table look-ups) = 4 to 5 minutes of widening, and with C = 240 vectors 6.1e12 multiply-adds, 2 to 4
# minutes of BLAS on 4 cores. A token by itself would cost the same widening: 5 minutes a token.
# Measured (run 36913721685, Xeon 6973P-C, 4 logical cores): the pass over 256 positions of 11 runs took 257 s, 157 s
# of it widening 26.6 G weights (170 M a second) and 41 s multiplying (165 G multiply-adds a second).
#
# The fork multiplies a ternary matrix by an activation it first rounds to Q8_0 (blocks of 32, a float16 scale,
# int8 values: ggml's vec_dot type of PTQ1_0), and the two small BF16 matrices of a linear-attention layer by one
# rounded to bfloat16. So it is not a float32 reference, and float32 rounding is not the size of the difference: the
# rounding of the activations is. Two runs here measure it: "float32" (the engine as the page will run it without
# its own rounding of the activations) and "as the fork rounds" (the same activations rounded the fork's way before
# every matrix). The second is what the fork computes up to the order of its sums, and every rounding that falls the
# other way after that; the first differs from it by the rounding itself.
#
# Runs with the engine broken on purpose (the first text only) show what the comparison catches: the value heads
# read in the GGUF's order, no signs, no rotation, the embedding's rows not turned back, q and k read as if no part
# of a head were stored in halves.
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import gguf_check  # noqa: E402
import llama2_convert  # noqa: E402
import llama2_numpy  # noqa: E402
from llama2_convert import gguf_rotated, gguf_tokenizer, transformed  # noqa: E402
from llama2_numpy import Llama, rotate  # noqa: E402

PQ2_0, PTQ1_0 = 142, 143
CHUNK = 64 << 20  # the float32 values of a matrix widened at a time (256 MB)


# ------------------------------------------------------------------------------------------- PTQ1_0
# A block of 128 weights in 28 bytes (the fork's ggml/src/ggml-common.h, block_ptq1_0, and dequantize_row_ptq1_0 of
# ggml/src/ggml-quants.c at 88c4bc60; MIT, the rule and no line): qs[24], five ternary digits a byte, then qh[2],
# four a byte, then a float16 scale d. Digit n of a byte b is ((b * 3 ** n mod 256) * 3) >> 8 (0, 1, 2: -d, 0, +d).
# The weights are not in the order of the bytes: of qs's first 16 bytes the digits 0 are weights 0 to 15, the digits
# 1 weights 16 to 31, ... (80 weights), of its last 8 bytes the same (40 weights), of qh's two bytes the same (8).
def digits(count):
    byte = np.arange(256, dtype=np.uint32)[:, None]
    return ((((byte * 3 ** np.arange(count, dtype=np.uint32)) & 0xFF) * 3) >> 8).astype(np.float32) - 1


FIVE, FOUR = digits(5), digits(4)


def widen_ptq1_0(raw):
    blocks = np.asarray(raw).reshape(-1, 28)
    out = np.empty((blocks.shape[0], 128), dtype=np.float32)
    out[:, :80] = FIVE[blocks[:, :16]].transpose(0, 2, 1).reshape(-1, 80)
    out[:, 80:120] = FIVE[blocks[:, 16:24]].transpose(0, 2, 1).reshape(-1, 40)
    out[:, 120:] = FOUR[blocks[:, 24:26]].transpose(0, 2, 1).reshape(-1, 8)
    out *= gguf_check.half(np.ascontiguousarray(blocks[:, 26:28]))
    return out


def pack_ptq1_0(digits_of_weights, scales):
    """The blocks of weights given as digits 0, 1, 2 (blocks, 128) with their float16 scales: what the fork's
    quantize_row_ptq1_0 writes (a byte is ceil(its digits as a base-3 number * 256 / 243)), for the tests."""
    d = np.asarray(digits_of_weights, dtype=np.int64).reshape(-1, 128)
    byte = lambda rows, count: -(-(rows * 3 ** np.arange(4, 4 - count, -1)).sum(axis=-1) * 256 // 243)
    parts = [byte(d[:, :80].reshape(-1, 5, 16).transpose(0, 2, 1), 5),
             byte(d[:, 80:120].reshape(-1, 5, 8).transpose(0, 2, 1), 5),
             byte(d[:, 120:].reshape(-1, 4, 2).transpose(0, 2, 1), 4)]
    packed = np.concatenate(parts, axis=1).astype(np.uint8)
    return np.concatenate([packed, np.asarray(scales, dtype=np.float16).reshape(-1, 1).view(np.uint8)], axis=1).tobytes()


BLOCKS = {PTQ1_0: (128, 28, widen_ptq1_0), PQ2_0: (128, 34, gguf_check.widen_pq2_0)}


def rows_of(info, data, base, first, last):
    """Rows first..last of a GGUF's tensor as float32 (a vector: all of it)."""
    if info["type"] not in BLOCKS:
        return gguf_check.tensor(info, data, base, first, last)[0]
    values, size, widen = BLOCKS[info["type"]]
    width = info["shape"][-1]
    row = width // values * size
    start = base + info["offset"] + first * row
    return widen(data[start:start + (last - first) * row]).reshape(last - first, width)


# ------------------------------------------------------------------------------------------- one pass for all
class Conductor:
    """Runs jobs (calls of the engine's forward pass) as threads, one at a time and in their order, each as far as its
    next product with a stored matrix; then computes that product for all of them at once. Every job has to multiply
    by the same matrices in the same order (the same model), and a job may read what the jobs before it left when
    they passed the same place (the state and the keys of the position before)."""

    def __init__(self):
        self.back = threading.Semaphore(0)
        self.seconds = {"widening": 0.0, "multiplying": 0.0}
        self.values = 0

    def ask(self, stored, x):
        worker = threading.current_thread()
        worker.request = (stored, x)
        self.back.release()
        worker.go.acquire()
        return worker.answer

    def run(self, jobs):
        def body(job):
            worker = threading.current_thread()
            worker.go.acquire()
            try:
                worker.result = job()
            except BaseException as error:  # noqa: BLE001 (handed to the caller below)
                worker.error = error
            worker.request = None
            self.back.release()

        workers = []
        for job in jobs:
            worker = threading.Thread(target=body, args=(job,), daemon=True)
            worker.go, worker.request, worker.result, worker.error = threading.Semaphore(0), None, None, None
            worker.start()
            workers.append(worker)
        waiting = workers
        while waiting:
            for worker in waiting:  # in the order of the jobs, one at a time
                worker.go.release()
                self.back.acquire()
                if worker.error is not None:
                    raise worker.error
            waiting = [worker for worker in waiting if worker.request is not None]
            if waiting:
                stored = waiting[0].request[0]
                if any(worker.request[0] is not stored for worker in waiting):
                    raise RuntimeError("the jobs do not multiply by the same matrices in the same order")
                products = stored.times(np.stack([worker.request[1] for worker in waiting], axis=1), self)
                for column, worker in enumerate(waiting):
                    worker.answer = products[:, column].copy()
        return [worker.result for worker in workers]


class Stored:
    """A matrix as it is stored: read(first, last) gives rows of it in float32."""

    def __init__(self, name, rows, width, read):
        self.name, self.rows, self.width, self.read = name, rows, width, read

    def times(self, vectors, conductor):
        out = np.empty((self.rows, vectors.shape[1]), dtype=np.float32)
        step = max(1, CHUNK // self.width)
        for first in range(0, self.rows, step):
            last = min(first + step, self.rows)
            began = time.perf_counter()
            weights = self.read(first, last)
            middle = time.perf_counter()
            out[first:last] = weights @ vectors
            conductor.seconds["widening"] += middle - began
            conductor.seconds["multiplying"] += time.perf_counter() - middle
            conductor.values += weights.size
        return out


class Matrix:
    """A stored matrix as one run of the engine multiplies by it: order[i] is the stored row that is the engine's row
    i (None: the same rows), prepare is what the run does to a vector before the product."""

    def __init__(self, conductor, stored, order=None, prepare=None):
        self.conductor, self.stored, self.order, self.prepare = conductor, stored, order, prepare

    def __matmul__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (self.stored.width,):
            raise ValueError(f"{self.stored.name} takes {self.stored.width} values, not {x.shape}")
        product = self.conductor.ask(self.stored, x if self.prepare is None else self.prepare(x))
        return product if self.order is None else product[self.order]


class Small:
    """A small float32 matrix whose vector a run prepares too (a linear-attention layer's gates)."""

    def __init__(self, values, prepare):
        self.values, self.prepare = values, prepare

    def __getitem__(self, a):
        return Small(self.values[a], self.prepare)

    def __matmul__(self, x):
        return self.values @ self.prepare(np.asarray(x, dtype=np.float32))


class Rows:
    """The embedding: one stored row at a time."""

    def __init__(self, stored, change=None):
        self.stored, self.change = stored, change

    def __getitem__(self, token):
        row = self.stored.read(int(token), int(token) + 1)[0]
        return row if self.change is None else self.change(row)


# ------------------------------------------------------------------------------------------- as the fork rounds
def as_q8_0(x):
    """x as ggml's quantize_row_q8_0 stores it and its dot products read it: blocks of 32, d = max |x| / 127 kept as
    a float16, q = round(x / d) half away from zero; the values q * d."""
    blocks = x.reshape(-1, 32)
    d = np.abs(blocks).max(axis=1, keepdims=True) / np.float32(127.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(d > 0, blocks / d, np.float32(0.0))
    q = np.trunc(scaled + np.copysign(np.float32(0.5), scaled))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape).astype(np.float32)


def as_bf16(x):
    """x rounded to bfloat16 (to the nearest, ties to even), as ggml converts the activation a BF16 matrix reads."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + (np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


# ------------------------------------------------------------------------------------------- the model
def grouped(key_heads, value_heads, size=1):
    """For every row of a tensor over the value heads in Hugging Face's order (the heads of a key head together: the
    engine's), the row of llama.cpp's GGUF that holds it. llama.cpp writes every key head's first value head, then
    every key head's second, ... (conversion/qwen.py's _reorder_v_heads at 88c4bc60: grouped to tiled), so head
    g = key * (V / K) + j of Hugging Face is head j * K + key of the file. size: the rows of a head."""
    per = value_heads // key_heads
    heads = np.array([(g % per) * key_heads + g // per for g in range(value_heads)])
    return (heads[:, None] * size + np.arange(size)).reshape(-1)


class Streamed(Llama):
    """The engine over a GGUF of a Qwen3.5 (Bonsai 2's Qwen3.8): Llama's own constructor and forward pass, with the
    tensors taken from the GGUF where the engine takes them from its checkpoint. broken: what to compute wrong."""

    def __init__(self, source, conductor, rounding=False, broken=""):
        self.source, self.conductor, self.rounding, self.broken = source, conductor, rounding, broken
        metadata, key = source.metadata, lambda name: source.metadata[f"qwen35.{name}"]
        heads, kv_heads, head = key("attention.head_count"), key("attention.head_count_kv"), key("attention.key_length")
        values = key("ssm.time_step_rank")
        linear = {"every": key("full_attention_interval"), "key_heads": key("ssm.group_count"), "value_heads": values,
                  "key_dim": key("ssm.state_size"), "value_dim": key("ssm.inner_size") // values, "conv": key("ssm.conv_kernel")}
        rotated = gguf_rotated(metadata, source.infos, linear["value_heads"] != linear["key_heads"])
        if broken == "no signs":
            rotated = {**rotated, "signs": {width: "00" * (len(bits) // 2) for width, bits in rotated["signs"].items()}}
        if broken == "no rotation":
            rotated = None
        header = struct.pack("<7i", key("embedding_length"), key("feed_forward_length"), key("block_count"), heads, kv_heads,
                             -len(metadata["tokenizer.ggml.tokens"]), source.positions)
        # float16: the RoPE tables are the engine's own (partial_tables), not a file's
        super().__init__(header, None, dtype="float16", arch="qwen35", linear=linear, head_dim=head,
                         rotary=key("rope.dimension_count"), rope_theta=float(key("rope.freq_base")),
                         rms_norm_eps=float(f"{key('attention.layer_norm_rms_epsilon'):.6g}"), rotated=rotated,
                         bos=metadata["tokenizer.ggml.bos_token_id"])

    def qwen35_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, frequencies):
        """Llama.qwen35_tensors() from a GGUF: the same attributes, the large matrices behind the conductor."""
        source, linear, broken = self.source, self.linear, self.broken
        heads, kv_heads, head, rot = self.n_heads, self.n_kv_heads, self.head_size, self.rotary
        key_heads, value_heads = linear["key_heads"], linear["value_heads"]
        keys = key_heads * linear["key_dim"]
        full = [l for l, (lines, _) in enumerate(self.slots) if not lines]
        lines = [l for l, (kind, _) in enumerate(self.slots) if kind]
        prepare = as_q8_0 if self.rounding else None
        small = as_bf16 if self.rounding else (lambda x: x)
        # the rows of the engine's matrices among the stored ones: q and its gate are one matrix (a head's q, then its
        # gate), and of the part of a head that RoPE turns the halves are stored one after the other (the converter's
        # "heads" transform says which stored row becomes which, here applied to the rows' numbers)
        turned = 0 if broken == "halves" else rot
        index = lambda rows, transform: transformed(np.arange(rows)[:, None], transform, head).reshape(-1)
        q_order, gate_order = index(2 * heads * head, ("heads", 2, 0, heads, turned)), index(2 * heads * head, ("heads", 2, 1, heads, 0))
        k_order = index(kv_heads * head, ("heads", 1, 0, kv_heads, turned))
        norm_order = index(head, ("heads", 1, 0, 1, turned))
        # the value heads: Hugging Face's order from the GGUF's
        same = broken == "tiled"
        head_order = np.arange(value_heads) if same else grouped(key_heads, value_heads)
        v_order = np.arange(value_heads * linear["value_dim"]) if same else grouped(key_heads, value_heads, linear["value_dim"])
        qkv_order = np.concatenate([np.arange(2 * keys), 2 * keys + v_order])

        def stack(name, layers, order=None):
            return [Matrix(self.conductor, source.stored(f"blk.{l}.{name}.weight"), order, prepare) for l in layers]

        def vectors(name, layers, change=lambda v: v):
            return np.stack([change(source.small(f"blk.{l}.{name}")) for l in layers]).astype(np.float32)

        everything = range(self.n_layers)
        table = source.stored("token_embd.weight")
        self.token_embedding_table = Rows(table, None if broken != "embedding" else (
            lambda row: rotate(row, self.rotated["signs"][self.dim], self.rotated["block"])))
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
        # the output matrix reads the heads in Hugging Face's order, as stored (prism.hadamard.gdn_v_grouped)
        self.wout = stack("ssm_out", lines)
        self.rms_ffn_weight = vectors("post_attention_norm.weight", everything)
        self.w1, self.w2, self.w3 = stack("ffn_gate", everything), stack("ffn_down", everything), stack("ffn_up", everything)
        self.rms_final_weight = source.small("output_norm.weight")
        self.wcls = Matrix(self.conductor, source.stored("output.weight"), None, prepare)
        self.freq_cis_real, self.freq_cis_imag = self.partial_tables(frequencies)


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


# ------------------------------------------------------------------------------------------- the comparison
def compare(name, ours, theirs, wrote=None):
    """One line: the largest difference of the logits over the positions, the positions with the same largest logit,
    and, where that differs, by how much the other's own first two differ there."""
    worst, same, close = 0.0, 0, []
    for position, (a, b) in enumerate(zip(ours, theirs)):
        worst = max(worst, float(np.max(np.abs(a - b))))
        if int(np.argmax(a)) == int(np.argmax(b)):
            same += 1
        else:
            first, second = np.sort(b)[-1], np.sort(b)[-2]
            close.append(f"position {position}: its first two are {first - second:.4f} apart")
    line = f"reference: {name}: largest difference {worst:.4f}, the same largest logit at {same} of {len(theirs)} positions"
    if close:
        line += " (" + "; ".join(close[:4]) + (" ..." if len(close) > 4 else "") + ")"
    if wrote is not None:
        line += f", {wrote}"
    print(line, flush=True)
    return worst, same


def main():
    gguf, work = Path(sys.argv[1]), Path(sys.argv[2])
    texts = []
    for index in range(100):
        ids = work / f"fork-{index}.ids"
        if not ids.exists():
            break
        prompt, wrote = ([int(token) for token in line.split()] for line in ids.read_text().split("\n")[:2])
        text = (work / f"prompt-{index}.txt").read_text() if (work / f"prompt-{index}.txt").exists() else None
        texts.append({"prompt": prompt, "wrote": wrote, "text": text})
    assert texts, f"no fork-<i>.ids in {work}"
    source = Source(gguf, positions=max(len(t["prompt"]) + len(t["wrote"]) for t in texts) + 1)
    vocab = len(source.metadata["tokenizer.ggml.tokens"])
    for index, text in enumerate(texts):
        text["ids"] = text["prompt"] + text["wrote"][:-1]
        text["fork"] = np.fromfile(work / f"fork-{index}.logits", dtype=np.float32).reshape(-1, vocab)
        text["single"] = np.fromfile(work / f"fork-{index}.single", dtype=np.float32).reshape(-1, vocab)
        assert len(text["fork"]) == len(text["ids"]), (len(text["fork"]), len(text["ids"]))

    # the tokenizer: the engine's, from the GGUF's vocabulary as the converter reads it, against the fork's ids
    began = time.perf_counter()
    tokenizer_bin, options, _, controls, added = gguf_tokenizer(source.metadata, vocab)
    tokenizer = llama2_numpy.Tokenizer(tokenizer_bin, vocab, kind=options["tokenizer_kind"], nfc=options["nfc"],
                                       pretokenizer=options["pretokenizer"], ignore_merges=options["ignore_merges"])
    specials = sorted({*controls, *added} - {""}, key=lambda token: (-len(token), token))
    for index, text in enumerate(texts):
        if text["text"] is not None:
            ids = tokenizer.encode(text["text"], specials=specials)
            said = "the same ids as the fork" if ids == text["prompt"] else f"{ids}, the fork {text['prompt']}"
            print(f"reference: text {index}: the engine's tokenizer gives {said}")
            text["tokens"] = ids == text["prompt"]
    print(f"reference: the tokenizer in {time.perf_counter() - began:.1f} s", flush=True)
    # every run's Llama takes this tokenizer (building 248320 pieces once is enough)
    llama2_numpy.Tokenizer = lambda *arguments, **named: tokenizer

    conductor = Conductor()
    runs = [(index, name, Streamed(source, conductor, rounding=rounding))
            for index in range(len(texts)) for name, rounding in (("float32", False), ("as the fork rounds", True))]
    breaks = {"tiled": "the value heads in the GGUF's order", "no signs": "no signs", "no rotation": "no rotation",
              "embedding": "the embedding's rows not turned back", "halves": "q and k as if no part were stored in halves"}
    runs += [(0, name, Streamed(source, conductor, broken=broken)) for broken, name in breaks.items()]
    jobs = [(lambda model=model, token=token, position=position: np.array(model.forward(token, position)))
            for index, _, model in runs for position, token in enumerate(texts[index]["ids"])]
    print(f"reference: {len(jobs)} positions of {len(runs)} runs in one pass over {gguf.name}", flush=True)
    began = time.perf_counter()
    results = conductor.run(jobs)
    seconds = time.perf_counter() - began
    print(f"reference: the pass took {seconds:.0f} s: {conductor.seconds['widening']:.0f} s widening "
          f"{conductor.values / 1e9:.1f} G weights ({conductor.values / conductor.seconds['widening'] / 1e6:.0f} M a second), "
          f"{conductor.seconds['multiplying']:.0f} s multiplying them by {len(jobs)} vectors "
          f"({conductor.values * len(jobs) / conductor.seconds['multiplying'] / 1e9:.0f} G multiply-adds a second)", flush=True)

    logits, at = {}, 0
    for index, name, _ in runs:
        count = len(texts[index]["ids"])
        logits[(index, name)] = results[at:at + count]
        at += count

    failed = []
    for index, text in enumerate(texts):
        prompt = len(text["prompt"])
        ours, rounded, fork = logits[(index, "float32")], logits[(index, "as the fork rounds")], text["fork"]
        greedy = [int(np.argmax(row)) for row in ours[prompt - 1:]]
        wrote = f"greedy: {sum(a == b for a, b in zip(greedy, text['wrote']))} of {len(text['wrote'])} tokens the fork's"
        print(f"reference: text {index}: {prompt} tokens of a prompt, {len(text['wrote'])} written by the fork: {text['wrote']}")
        # the fork's own two paths: the prompt as one batch and a token at a time
        own, _ = compare(f"text {index}, the fork's batch against the fork a token at a time", fork[:prompt], text["single"])
        rounding, _ = compare(f"text {index}, float32 against as the fork rounds (the rounding of the activations)", ours, rounded)
        near, near_same = compare(f"text {index}, as the fork rounds against the fork", rounded, fork)
        far, same = compare(f"text {index}, float32 against the fork", ours, fork, wrote)
        # the line: the float32 engine is no farther from the fork than three times what the rounding of the
        # activations moves the engine itself (and the fork's two paths), and where its largest logit is another, the
        # fork's own first two are closer than twice that difference. Not float32 rounding: the fork is not one set of
        # numbers to that precision (run 36913721685: its batch and its token at a time differ by 0.097 to 0.134 on
        # the three texts, the engine from the fork by 0.095 to 0.149, the rounding moves the engine by 0.096 to 0.284;
        # a wrong order of heads, no signs, no rotation or an embedding not turned back by 18 to 20, q and k read
        # without their halves by 3.3)
        line = 3 * max(rounding, own)
        margins = [float(np.sort(b)[-1] - np.sort(b)[-2]) for a, b in zip(ours, fork) if int(np.argmax(a)) != int(np.argmax(b))]
        ok = far <= line and all(margin <= 2 * far for margin in margins) and text.get("tokens", True)
        print(f"reference: text {index}: {'ok' if ok else 'FAILED'}: {far:.4f} against the line {line:.4f} "
              f"(3 x the larger of the rounding {rounding:.4f} and the fork's two paths {own:.4f})", flush=True)
        if not ok:
            failed.append(f"text {index}")
    fork = texts[0]["fork"]
    reference, _ = compare("text 0, float32 against the fork again", logits[(0, "float32")], fork)
    for name in breaks.values():
        far, same = compare(f"broken, {name}, against the fork", logits[(0, name)], fork)
        caught = far > 10 * reference
        print(f"reference: broken, {name}: {'caught' if caught else 'NOT CAUGHT'} ({far / reference:.0f} times the "
              f"unbroken engine's difference)", flush=True)
        if not caught:
            failed.append(f"broken, {name}")
    if failed:
        print(f"reference: FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("reference: ok: the engine's NumPy forward pass computes the 27B as the fork does")


if __name__ == "__main__":
    main()
