# The conversion itself: the file fed in its own order, each tensor converted as it arrives and written to its place.
import math

import numpy as np

from convert.checkpoint import EITHER, Writer, check_dtype, checkpoint_size, dtype_name
from convert.readers import READERS, bfloat16, q8_0
from convert.sources import header_rotated
from convert.config import (PARTLY_TURNED, check_config, checkpoint_header, head_size, normalize, query_scale,
                            rotary_dim)
from convert.plan import checkpoint_form, conversion_plan, name_prefix, rope_table, source_shape, transformed
from convert.gguf import rope_freqs_agree

# Pieces of at most this many values are converted at a time: 4 MB as float32. Measured on llm-jp-3-150m, the
# peak is the output plus 14 MB with this, plus 52 MB with pieces four times as large, at the same speed.
PIECE = 1024 * 1024


def convert_weights(source, config, dtype, max_seq_len, out, progress=None, quantize_rows=None):
    """Fill out, a writable buffer of checkpoint_size() bytes, from source (Safetensors or Arrays).

    progress(values done, values in all) is called after every piece. quantize_rows: see Writer.
    """
    for done, total in convert_pieces(source, config, dtype, max_seq_len, out, quantize_rows):
        if progress:
            progress(done, total)
    return checkpoint_header(config, source, max_seq_len)


def convert_pieces(source, config, dtype, max_seq_len, out, quantize_rows=None):
    """convert_weights() as a generator that yields (values done, values in all) after every piece: whoever drives
    it can show the progress, let other work in between, and stop half way (the worker of the page does all three)."""
    config = normalize(config)
    check_config(config)
    header = checkpoint_header(config, source, max_seq_len)
    # the size the heads of wq and wk are interleaved by: another size turns rows across heads without a word
    size = head_size(config)
    form = checkpoint_form(config, source)
    writer = Writer(out, header, dtype, form, quantize_rows=quantize_rows)

    plan, shapes = conversion_plan(header, form, name_prefix(source, form["arch"]),
                                   rotary_dim(config) if form["arch"] in PARTLY_TURNED else 0, query_scale(config))
    total, done = sum(math.prod(shape) for shape in shapes), 0

    for index, (parts, shape) in enumerate(zip(plan, shapes)):
        if parts is None:
            # the RoPE tables (left out of an int8 checkpoint): cos, then sin
            writer.write(index, 0, rope_table(config, header, plan[:index].count(None)))
            done += math.prod(shape)
            yield done, total
            continue
        first = 0
        for name, transform in parts:
            found = source.shape(name) if name in source else None
            expected = source_shape(shape[1:] if len(parts) > 1 else shape, transform)
            if found != expected:
                raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {expected}.")
            rows = found[0] if len(found) > 1 else 1
            row = math.prod(found) // rows
            # a transform needs the whole tensor (a small one); everything else goes piece by piece
            step = rows if transform or len(found) == 1 else max(1, PIECE // row)
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                values = source.rows(name, 0, found[0]) if len(found) == 1 else source.rows(name, start, stop)
                values = transformed(values, transform, size)
                writer.write(index, first, values)
                first += values.size
                done += values.size
                yield done, total
        assert first == math.prod(shape), name


class Stream:
    """The same conversion in the order of the file: feed() takes the bytes of a .safetensors file from its beginning
    to its end, in chunks of any size, and every tensor goes to its place in the checkpoint as soon as its rows are
    there. For a download: reading in the order of the output would mean hundreds of range requests, a second each.

    header: the JSON of the file (its first 8 bytes say how long it is), base: where the tensors begin, start: the
    position in the file of the first byte that feed() will get. out: a buffer of checkpoint_size() bytes, or None
    to have one made (self.out). sink and quantize_rows: see Writer. bfloat16: the widening of bfloat16 on the SIMD
    kernels (llama2_numpy.kernel_widener, T123), the same float32 as this file's bfloat16(). q8_0: the widening of
    GGUF's Q8_0 on the kernels (llama2_numpy.kernel_q8_0, T136), the same float32 as this file's q8_0(). readers:
    more of READERS on the kernels, by the GGUF's type (llama2_numpy.kernel_ternary_readers, T273: PQ2_0 and PTQ1_0),
    the same float32 as this file's.
    """

    def __init__(self, header, base, config, dtype, max_seq_len, out=None, start=0, sink=None, quantize_rows=None,
                 bfloat16=None, q8_0=None, readers=None):
        config = normalize(config)
        self.bfloat16, self.q8_0, self.readers = bfloat16, q8_0, readers or {}
        check_config(config)
        self.tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        self.rotated = header_rotated(header)  # T237: what checkpoint_form() asks
        self.header = checkpoint_header(config, self, max_seq_len)
        self.head_size = head_size(config)
        # what lays out the checkpoint and sizes the forward pass that the header does not say (T115, T124, T144):
        # the options say it, and the worker's footprint() reads it
        self.form = checkpoint_form(config, self)
        if callable(dtype):
            # T115: chosen once the header is known, from the size each quantized dtype would take (the worker's
            # automatic choice: int8 where the forward pass fits a 32-bit memory, six bits where it does not)
            sizes = {name: self.size(name) for name in EITHER}
            dtype = str(dtype(list(self.header), self.form, sizes))
        check_dtype(dtype)
        self.dtype = dtype_name(dtype)
        if out is None and sink is None:
            out = bytearray(self.size(dtype))
        self.out = out  # None when the checkpoint goes to sink
        self.writer = Writer(self.out, self.header, dtype, self.form, sink=sink, quantize_rows=quantize_rows)
        plan, shapes = conversion_plan(self.header, self.form, name_prefix(self, self.form["arch"]),
                                       rotary_dim(config) if self.form["arch"] in PARTLY_TURNED else 0, query_scale(config))
        self.total, self.done = sum(math.prod(shape) for shape in shapes), 0
        wanted = {}
        for index, (parts, shape) in enumerate(zip(plan, shapes)):
            if parts is None:
                self.writer.write(index, 0, rope_table(config, self.header, plan[:index].count(None)))
                self.done += math.prod(shape)
                continue
            first = 0
            for name, transform in parts:
                found = tuple(self.tensors[name]["shape"]) if name in self.tensors else None
                expected = source_shape(shape[1:] if len(parts) > 1 else shape, transform)
                if found != expected:
                    raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {expected}.")
                if self.tensors[name]["dtype"] not in READERS:
                    raise ValueError(f"{name} is stored as {self.tensors[name]['dtype']}: only float32, float16 and bfloat16 are supported.")
                # GPT-2's c_attn holds q, k and v in one matrix, so one tensor of the file can feed several
                wanted.setdefault(name, []).append((index, first, transform))
                first += math.prod(shape[1:] if len(parts) > 1 else shape)
        # what to do with each stretch of the file, in the order of the file
        self.steps = []
        for name, info in sorted(self.tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
            begin, end = info["data_offsets"]
            # T136: a table to check as it passes, not to convert (gguf_weights)
            target = wanted.get(name) or ("check" if info.get("rope_freqs") else None)
            self.steps.append((base + begin, base + end, name, target))
        self.position, self.step, self.pending, self.first = start, 0, bytearray(), 0
        self.config = config

    def __contains__(self, name):  # what checkpoint_header() asks
        return name in self.tensors

    def size(self, dtype):
        """The bytes of the checkpoint in that dtype."""
        return checkpoint_size(self.header, dtype, self.form)

    def feed(self, data):
        """data: the next bytes of the file. Returns (values done, values in all)."""
        data = memoryview(data.to_py() if hasattr(data, "to_py") else data).cast("B")
        offset = 0
        while offset < len(data) and self.step < len(self.steps):
            begin, end, name, target = self.steps[self.step]
            here = self.position + offset
            if here < begin:  # the JSON header, padding, or a tensor nobody needs
                offset += min(begin - here, len(data) - offset)
                continue
            take = min(end - here, len(data) - offset)
            if target == "check":
                self.pending += data[offset:offset + take]
                if here + take == end:
                    rope_freqs_agree(np.frombuffer(bytes(self.pending), dtype=np.float32), self.config)
            elif target is not None:
                self.pending += data[offset:offset + take]
                self.convert(name, target, last=here + take == end)
            offset += take
            if here + take == end:
                self.step, self.pending, self.first = self.step + 1, bytearray(), 0
        self.position += len(data)
        return self.done, self.total

    def convert(self, name, targets, last):
        info = self.tensors[name]
        itemsize, reader = READERS[info["dtype"]]
        if info["dtype"] == "BF16" and self.bfloat16 is not None:
            reader = self.bfloat16
        if info["dtype"] == "Q8_0" and self.q8_0 is not None:
            reader = self.q8_0
        reader = self.readers.get(info["dtype"]) or reader
        shape = tuple(info["shape"])
        # T136's third stage: a GPT-2's Conv1D matrix, which the GGUF holds as (out, in), is read in that shape
        stored = tuple(reversed(shape)) if info.get("transposed") else shape
        row = int((math.prod(stored[1:]) if len(stored) > 1 else int(stored[0])) * itemsize)
        # the head permutation needs its whole matrix (a small one); everything else goes row by row, as it comes.
        # A GGUF's tensor held in another order than Hugging Face's is put back whole too
        again = info.get("turned") or info.get("split") or info.get("transposed") or info.get("tiled")
        whole = any(transform for _, _, transform in targets) or len(targets) > 1 or bool(again)
        rows = len(self.pending) // row if not whole or last else 0
        if whole and last:
            rows = stored[0] if len(stored) > 1 else 1  # a vector is one row of its own length
        if rows == 0 or (len(self.pending) < PIECE and not last):
            return
        values = reader(bytes(self.pending[:rows * row]))
        del self.pending[:rows * row]
        if len(stored) > 1:
            values = values.reshape(rows, *stored[1:])
        if info.get("transposed"):
            values = values.T  # back to (in, out), which the plan transposes as it does a safetensors' own
        if info.get("turned"):
            # a GGUF of a Llama holds q and k turned already (llama.cpp's convert does what permute_heads does):
            # back to Hugging Face's order, so that the plan below turns them once, like everything else
            values = unturned(values, info["turned"])
        if info.get("split"):
            values = unsplit(values, info["split"])
        if info.get("tiled"):
            values = untiled(values, *info["tiled"])
        for index, first, transform in targets:
            out = transformed(values, left_to_do(transform, info.get("done")), self.head_size)
            self.writer.write(index, first + self.first, out)
            self.done += out.size
        self.first += 0 if whole else values.size

    def finish(self):
        if self.step < len(self.steps) or self.done != self.total:
            raise ValueError("The file ended before all of its tensors were read.")
        return self.header


def left_to_do(transform, done):
    """transform without the step a GGUF's tensor comes with (gguf_model()'s "done", T236: the 1 llama.cpp adds to a
    Qwen3.5's norms, its -exp(A_log)). A tensor said to come with a step the plan does not have for it is refused: the
    table of names and the plan would have drifted apart, and the values would go through changed once too often."""
    if not done:
        return transform
    steps = () if transform is None else transform if isinstance(transform[0], tuple) else (transform,)
    left = tuple(step for step in steps if step[0] != done)
    if len(left) != len(steps) - 1:
        raise ValueError(f"A tensor of this GGUF comes with the step {done!r} done, which the conversion has not for it.")
    return left or None


def unturned(w, heads):
    """The inverse of permute_heads: adjacent pairs of each head back to [first halves, second halves]."""
    rows = w.shape[0] // heads
    return w.reshape(heads, rows // 2, 2, -1).transpose(0, 2, 1, 3).reshape(w.shape)


def unsplit(w, heads):
    """GPT-NeoX's query_key_value (or its bias) as llama.cpp stores it, [all of q; all of k; all of v], back to
    Hugging Face's order, q, k and v of the first head, then of the second, ... (T136's third stage)."""
    return w.reshape(3, heads, w.shape[0] // 3 // heads, -1).swapaxes(0, 1).reshape(w.shape)


def untiled(w, first, key_heads, per, size, axis=0):
    """The value heads of a Qwen3.5's linear-attention layer as llama.cpp stores them where a key head has per (more
    than one) of them, back to Hugging Face's order (T245). Hugging Face holds them grouped by their key head: the per
    value heads of key head 0, then those of key head 1, ... llama.cpp writes them tiled, every key head's first value
    head, then every key head's second, ..., so that its broadcast of the key heads over the value heads is a plain
    repeat (conversion/qwen.py's _reorder_v_heads at dcd387a4): the value head at place j * key_heads + h of the GGUF
    is Hugging Face's value head h * per + j. w: the whole tensor. The heads are what it has from first on along axis,
    size entries to a head; what stands before first (q and k in in_proj_qkv and in the convolution's channels) stays."""
    w = np.moveaxis(np.asarray(w), axis, 0)
    if w.shape[0] - first != key_heads * per * size:
        raise ValueError(f"{w.shape[0] - first} are not {key_heads * per} value heads of {size}.")
    heads = w[first:].reshape(per, key_heads, size, *w.shape[1:]).swapaxes(0, 1).reshape(-1, *w.shape[1:])
    return np.moveaxis(np.concatenate([w[:first], heads]), 0, axis)
