# llama2_convert.py
# Hugging Face Llama checkpoint -> what llama2_numpy.py loads, with nothing but NumPy, one piece of a tensor at a
# time: the weights are read through read(offset, length) and written into a buffer that has its final size from
# the start, float32, float16 or int8. So it never holds more than the output and a few megabytes, which is what
# lets the same code run when the site is built (convert_hf.py, quantize.py) and inside the browser, where the
# WebAssembly memory has 32 bits and never shrinks.
import json
import struct

import numpy as np

# Pieces of at most this many values are converted at a time: 4 MB as float32. Measured on llm-jp-3-150m, the
# peak is the output plus 14 MB with this, plus 52 MB with pieces four times as large, at the same speed.
PIECE = 1024 * 1024


# ------------------------------------------------------------------------------------ the checkpoint format
def group_size(row_length):
    size = 32
    while row_length % size:
        size //= 2
    return size


def layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
    """(shape, is a matrix) of every tensor, in file order. llama2_numpy.py reads the same order.

    is a matrix: True for what int8 quantizes, False for the norm weights, None for the RoPE tables.
    """
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size
    tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
               ((n_layers, dim, dim), True), ((n_layers, kv_dim, dim), True), ((n_layers, kv_dim, dim), True),
               ((n_layers, dim, dim), True), ((n_layers, dim), False),
               ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
               ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
    if vocab_size < 0:
        tensors.append(((abs(vocab_size), dim), True))
    return tensors


def tensor_bytes(shape, is_matrix, dtype):
    """How many bytes a tensor of layout() takes in a checkpoint of that dtype."""
    count = int(np.prod(shape))
    if np.dtype(dtype) != np.int8:
        return count * np.dtype(dtype).itemsize
    if is_matrix is None:
        return 0  # int8 checkpoints leave the RoPE tables out
    # int8 values and one float32 scale per group; the norm weights stay float32
    return count + 4 * (count // group_size(shape[-1])) if is_matrix else 4 * count


def checkpoint_size(header, dtype):
    return 28 + sum(tensor_bytes(shape, is_matrix, dtype) for shape, is_matrix in layout(*header))


def quantize(values):
    """float32 values, whole rows -> (int8 values, float32 scales), one scale per group of the row."""
    groups = values.reshape(-1, group_size(values.shape[-1]))
    scales = (np.abs(groups).max(axis=1) / 127.0).astype(np.float32)
    inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
    return np.rint(groups * inverse[:, None]).astype(np.int8), scales


class Writer:
    """Puts pieces of the tensors of layout(), in any order, where they belong in the checkpoint buffer."""

    def __init__(self, out, header, dtype):
        self.out, self.dtype = np.frombuffer(out, dtype=np.uint8), np.dtype(dtype)
        assert self.out.size == checkpoint_size(header, dtype), "the buffer has not the size of the checkpoint"
        self.out[:28] = np.frombuffer(struct.pack("<7i", *header), dtype=np.uint8)
        self.tensors, offset = [], 28
        for shape, is_matrix in layout(*header):
            self.tensors.append((offset, shape, is_matrix))
            offset += tensor_bytes(shape, is_matrix, dtype)

    def put(self, offset, array):
        raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        self.out[offset:offset + raw.size] = raw

    def write(self, index, first, values):
        """values: whole rows of tensor number index, beginning at its element number first."""
        offset, shape, is_matrix = self.tensors[index]
        if self.dtype != np.int8:
            self.put(offset + first * self.dtype.itemsize, np.asarray(values).astype(self.dtype, copy=False))
        elif is_matrix:
            quantized, scales = quantize(np.asarray(values, dtype=np.float32).reshape(-1, shape[-1]))
            self.put(offset + first, quantized)
            self.put(offset + int(np.prod(shape)) + 4 * (first // group_size(shape[-1])), scales)
        elif is_matrix is False:
            self.put(offset + 4 * first, np.asarray(values, dtype=np.float32))


# ------------------------------------------------------------------------------------------ the weights
def bfloat16(raw):
    # NumPy has no bfloat16, but a bfloat16 is exactly the upper half of a float32: widening is a shift
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    wide <<= 16
    return wide.view(np.float32)


READERS = {"F32": (4, lambda raw: np.frombuffer(raw, dtype=np.float32)),
           "F16": (2, lambda raw: np.frombuffer(raw, dtype=np.float16)), "BF16": (2, bfloat16)}


class Safetensors:
    """The tensors of a .safetensors file behind read(offset, length): a local file, a File of the browser, a URL."""

    def __init__(self, read):
        self.read = read
        (header_size,) = struct.unpack("<Q", bytes(read(0, 8)))
        if not 2 <= header_size <= 100_000_000:
            raise ValueError("This is not a safetensors file.")
        try:
            self.tensors = {name: info for name, info in json.loads(bytes(read(8, header_size))).items() if name != "__metadata__"}
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        self.base = 8 + header_size

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def rows(self, name, start, stop):
        """Rows start..stop of a tensor (all of a vector), as float32 or float16."""
        info = self.tensors[name]
        if info["dtype"] not in READERS:
            raise ValueError(f"{name} is stored as {info['dtype']}: only float32, float16 and bfloat16 are supported.")
        itemsize, reader = READERS[info["dtype"]]
        shape = self.shape(name)
        row = int(np.prod(shape[1:]))
        begin = self.base + info["data_offsets"][0] + start * row * itemsize
        return reader(self.read(begin, (stop - start) * row * itemsize)).reshape(stop - start, *shape[1:])


class Arrays:
    """The same interface for tensors that are in memory already (a PyTorch checkpoint, a test)."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name].shape)

    def rows(self, name, start, stop):
        return self.tensors[name][start:stop]


def check_config(config):
    """ValueError, in words for the visitor, unless this config.json describes a model the engine can run."""
    def refuse(reason):
        raise ValueError(f"This model cannot be converted: {reason}.")

    if config.get("model_type") != "llama":
        refuse(f"it is a {config.get('model_type', 'model of unknown type')}, and only Llama models are supported")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
                "max_position_embeddings"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            refuse(f"its config.json has no usable {key}")
    dim, n_heads = config["hidden_size"], config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    if dim % n_heads or n_heads % n_kv_heads or config.get("head_dim", dim // n_heads) != dim // n_heads or dim // n_heads % 2:
        refuse("its attention heads do not divide the hidden size the way llama2.c expects")
    if config.get("rope_scaling"):
        refuse("it uses RoPE scaling")
    if config.get("hidden_act", "silu") != "silu":
        refuse(f"its activation is {config['hidden_act']}, not silu")
    if config.get("attention_bias") or config.get("mlp_bias"):
        refuse("its layers have biases")


def checkpoint_header(config, source, max_seq_len):
    """The 7 ints of the legacy header. A negative vocabulary size signals a classifier of its own (llama2.c)."""
    shared_classifier = config.get("tie_word_embeddings", False) or "lm_head.weight" not in source
    vocab_size = config["vocab_size"]
    # the KV cache grows with seq_len, so a long context can be cut down for the browser
    return (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
            config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size if shared_classifier else -vocab_size, min(config["max_position_embeddings"], max_seq_len))


def permute_heads(w, heads, head_size):
    # Hugging Face stores each head of wq/wk as [first halves, second halves] (rotate_half);
    # llama2.c rotates adjacent pairs, so interleave the two halves again
    return w.reshape(heads, 2, head_size // 2, w.shape[1]).transpose(0, 2, 1, 3).reshape(w.shape)


def conversion_plan(header):
    """For every tensor of layout(): the tensors of the Hugging Face checkpoint it is made of, in order, as
    (name, heads to permute or None); None stands for a RoPE table. And the shapes of layout()."""
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header

    def layers(name, heads=None):
        return [(f"model.layers.{layer}.{name}.weight", heads) for layer in range(n_layers)]

    plan = [[("model.embed_tokens.weight", None)], layers("input_layernorm"),
            layers("self_attn.q_proj", n_heads), layers("self_attn.k_proj", n_kv_heads), layers("self_attn.v_proj"),
            layers("self_attn.o_proj"), layers("post_attention_layernorm"),
            layers("mlp.gate_proj"), layers("mlp.down_proj"), layers("mlp.up_proj"), [("model.norm.weight", None)],
            None, None]
    if vocab_size < 0:
        plan.append([("lm_head.weight", None)])
    return plan, [shape for shape, _ in layout(*header)]


def rope_table(config, header, which):
    """The cos (which = 0) or sin (1) table of the legacy format, for float32 and float16 checkpoints."""
    head_size, seq_len = header[0] // header[3], header[6]
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    frequencies = 1.0 / config.get("rope_theta", 10000.0) ** (np.arange(0, head_size, 2, dtype=np.float64) / head_size)
    return (np.cos if which == 0 else np.sin)(positions * frequencies)


def convert_weights(source, config, dtype, max_seq_len, out, progress=None):
    """Fill out, a writable buffer of checkpoint_size() bytes, from source (Safetensors or Arrays).

    progress(values done, values in all) is called after every piece.
    """
    for done, total in convert_pieces(source, config, dtype, max_seq_len, out):
        if progress:
            progress(done, total)
    return checkpoint_header(config, source, max_seq_len)


def convert_pieces(source, config, dtype, max_seq_len, out):
    """convert_weights() as a generator that yields (values done, values in all) after every piece: whoever drives
    it can show the progress, let other work in between, and stop half way (the worker of the page does all three)."""
    check_config(config)
    header = checkpoint_header(config, source, max_seq_len)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    head_size = dim // n_heads
    writer = Writer(out, header, dtype)

    plan, shapes = conversion_plan(header)
    total, done = sum(int(np.prod(shape)) for shape in shapes), 0
    permute_reverse = lambda w, heads: permute_heads(w, heads, head_size)

    for index, (parts, shape) in enumerate(zip(plan, shapes)):
        if parts is None:
            # the RoPE tables (left out of an int8 checkpoint): cos, then sin
            writer.write(index, 0, rope_table(config, header, plan[:index].count(None)))
            done += int(np.prod(shape))
            yield done, total
            continue
        first = 0
        for name, heads in parts:
            found = source.shape(name) if name in source else None
            expected = shape[1:] if len(parts) > 1 else shape
            if found != tuple(expected):
                raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {tuple(expected)}.")
            rows = found[0] if len(found) > 1 else 1
            row = int(np.prod(found)) // rows
            # the head permutation needs its whole matrix (a small one); everything else goes piece by piece
            step = rows if heads or len(found) == 1 else max(1, PIECE // row)
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                values = source.rows(name, 0, found[0]) if len(found) == 1 else source.rows(name, start, stop)
                if heads:
                    values = permute_reverse(values, heads)
                writer.write(index, first, values)
                first += values.size
                done += values.size
                yield done, total
        assert first == int(np.prod(shape)), name


class Stream:
    """The same conversion in the order of the file: feed() takes the bytes of a .safetensors file from its beginning
    to its end, in chunks of any size, and every tensor goes to its place in the checkpoint as soon as its rows are
    there. For a download: reading in the order of the output would mean hundreds of range requests, a second each.

    header: the JSON of the file (its first 8 bytes say how long it is), base: where the tensors begin, start: the
    position in the file of the first byte that feed() will get. out: a buffer of checkpoint_size() bytes, or None
    to have one made (self.out).
    """

    def __init__(self, header, base, config, dtype, max_seq_len, out=None, start=0):
        check_config(config)
        self.tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        self.header = checkpoint_header(config, self, max_seq_len)
        self.head_size = self.header[0] // self.header[3]
        self.out = bytearray(checkpoint_size(self.header, dtype)) if out is None else out
        self.writer = Writer(self.out, self.header, dtype)
        plan, shapes = conversion_plan(self.header)
        self.total, self.done = sum(int(np.prod(shape)) for shape in shapes), 0
        wanted = {}
        for index, (parts, shape) in enumerate(zip(plan, shapes)):
            if parts is None:
                self.writer.write(index, 0, rope_table(config, self.header, plan[:index].count(None)))
                self.done += int(np.prod(shape))
                continue
            first = 0
            for name, heads in parts:
                found = tuple(self.tensors[name]["shape"]) if name in self.tensors else None
                expected = tuple(shape[1:] if len(parts) > 1 else shape)
                if found != expected:
                    raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {expected}.")
                if self.tensors[name]["dtype"] not in READERS:
                    raise ValueError(f"{name} is stored as {self.tensors[name]['dtype']}: only float32, float16 and bfloat16 are supported.")
                wanted[name] = (index, first, heads)
                first += int(np.prod(found))
        # what to do with each stretch of the file, in the order of the file
        self.steps = []
        for name, info in sorted(self.tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
            begin, end = info["data_offsets"]
            self.steps.append((base + begin, base + end, name, wanted.get(name)))
        self.position, self.step, self.pending, self.first = start, 0, bytearray(), 0
        self.size = max((end for _, end, _, _ in self.steps), default=base)

    def __contains__(self, name):  # what checkpoint_header() asks
        return name in self.tensors

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
            if target is not None:
                self.pending += data[offset:offset + take]
                self.convert(name, target, last=here + take == end)
            offset += take
            if here + take == end:
                self.step, self.pending, self.first = self.step + 1, bytearray(), 0
        self.position += len(data)
        return self.done, self.total

    def convert(self, name, target, last):
        index, first, heads = target
        info = self.tensors[name]
        itemsize, reader = READERS[info["dtype"]]
        shape = tuple(info["shape"])
        row = (int(np.prod(shape[1:])) if len(shape) > 1 else int(shape[0])) * itemsize
        # the head permutation needs its whole matrix (a small one); everything else goes row by row, as it comes
        rows = len(self.pending) // row if not heads or last else 0
        if heads and last:
            rows = shape[0]
        if rows == 0 or (len(self.pending) < PIECE and not last):
            return
        values = reader(bytes(self.pending[:rows * row])).reshape(rows, *shape[1:]) if len(shape) > 1 else reader(bytes(self.pending[:rows * row]))
        del self.pending[:rows * row]
        if heads:
            values = permute_heads(values, heads, self.head_size)
        self.writer.write(index, first + self.first, values)
        self.first += values.size
        self.done += values.size

    def finish(self):
        if self.step < len(self.steps) or self.done != self.total:
            raise ValueError("The file ended before all of its tensors were read.")
        return self.header


# ---------------------------------------------------------------------------------------- the tokenizer
UNMATCHABLE = -1e9  # control, unknown and byte pieces must never match user text: llama2_numpy.py skips such scores


def tokenizer_bin(pieces, vocab_size):
    """llama2.c's tokenizer.bin from (text, score, matchable) pieces."""
    rows = [(score if matchable else UNMATCHABLE, text.replace("▁", " ").encode("utf-8")) for text, score, matchable in pieces]
    if len(rows) > vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}.")
    # A model can have a few more embedding rows than the tokenizer has pieces (padding to a round number). Many
    # more means the tokenizer of another model, which would convert fine and then write nonsense.
    if len(rows) < 0.9 * vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}: "
                         f"they do not belong together.")
    rows += [(UNMATCHABLE, b"")] * (vocab_size - len(rows))
    out = [struct.pack("<i", max(len(text) for _, text in rows))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in rows]
    return b"".join(out)


def tokenizer_json_pieces(tokenizer):
    if tokenizer["model"]["type"] != "Unigram":
        raise ValueError(f"This tokenizer.json is a {tokenizer['model']['type']} model: only Unigram ones are supported "
                         f"(or a sentencepiece tokenizer.model).")
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id, (text, score) in enumerate(tokenizer["model"]["vocab"]):
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        yield text, score, not (is_byte or text in special or id == tokenizer["model"].get("unk_id"))


def tokenizer_json_options(tokenizer):
    """What the engine has to know about this tokenizer: Llama(tokenizer_kind=, nfkc=)."""
    normalizers = json.dumps(tokenizer.get("normalizer") or {})
    # "Precompiled" is sentencepiece's character map, nmt_nfkc in practice
    return {"tokenizer_kind": "unigram", "nfkc": '"NFKC"' in normalizers or '"Precompiled"' in normalizers}


def protobuf_fields(data):
    """Yield (field number, value) of one protobuf message; nested messages come back as bytes."""
    i = 0

    def varint():
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < len(data):
        key = varint()
        field, wire_type = key >> 3, key & 7
        if wire_type == 0:
            yield field, varint()
        elif wire_type == 1:
            yield field, data[i:i + 8]
            i += 8
        elif wire_type == 2:
            size = varint()
            yield field, data[i:i + size]
            i += size
        elif wire_type == 5:
            yield field, data[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def sentencepiece_pieces(model):
    """(text, score, matchable) of a sentencepiece model (spiece.model, tokenizer.model), given as bytes."""
    NORMAL, USER_DEFINED = 1, 4
    for field, value in protobuf_fields(model):
        if field == 1:  # ModelProto.pieces
            piece = dict(protobuf_fields(value))
            score = struct.unpack("<f", piece[2])[0] if 2 in piece else 0.0
            yield piece.get(1, b"").decode("utf-8"), score, piece.get(3, NORMAL) in (NORMAL, USER_DEFINED)


def sentencepiece_options(model):
    """Llama(tokenizer_kind=, nfkc=) from the trainer and normalizer specs of a sentencepiece model."""
    UNIGRAM, BPE = 1, 2
    kind, normalizer = UNIGRAM, "nmt_nfkc"  # sentencepiece's own defaults
    for field, value in protobuf_fields(model):
        if field == 2:  # trainer_spec.model_type
            kind = dict(protobuf_fields(value)).get(3, UNIGRAM)
        elif field == 3:  # normalizer_spec.name
            normalizer = dict(protobuf_fields(value)).get(1, b"nmt_nfkc").decode("utf-8")
    if kind not in (UNIGRAM, BPE):
        raise ValueError("This sentencepiece model is neither unigram nor BPE.")
    return {"tokenizer_kind": "unigram" if kind == UNIGRAM else "bpe", "nfkc": "nfkc" in normalizer}


# ------------------------------------------------------------------------------------------ in the browser
class Conversion:
    """A Hugging Face model converted inside the page, from the visitor's disk or from huggingface.co.

    header: the JSON at the beginning of model.safetensors (text), base: where its tensors begin, config: the text of
    config.json, tokenizer: the bytes of tokenizer.json or of a sentencepiece model. Then feed() the bytes of the
    file in order, beginning at start, and finish(). checkpoint, tokenizer and options are what Llama() takes.
    """

    def __init__(self, header, base, config, tokenizer, tokenizer_name, dtype="int8", max_seq_len=4096, start=0):
        try:
            self.config = json.loads(config)
        except ValueError:
            raise ValueError("config.json is not JSON.") from None
        if not isinstance(self.config, dict):
            raise ValueError("config.json is not the configuration of a model.")
        check_config(self.config)
        if np.dtype(dtype) not in (np.float32, np.float16, np.int8):
            raise ValueError(f"dtype must be float32, float16 or int8, not {dtype}.")
        try:
            header = json.loads(header)
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        vocab_size = self.config["vocab_size"]
        tokenizer = bytes(tokenizer.to_py() if hasattr(tokenizer, "to_py") else tokenizer)
        if tokenizer_name.lower().endswith(".json"):
            try:
                parsed = json.loads(tokenizer)
            except ValueError:
                raise ValueError("tokenizer.json is not JSON.") from None
            self.tokenizer, options = tokenizer_bin(tokenizer_json_pieces(parsed), vocab_size), tokenizer_json_options(parsed)
        else:
            self.tokenizer, options = tokenizer_bin(sentencepiece_pieces(tokenizer), vocab_size), sentencepiece_options(tokenizer)
        bos = self.config.get("bos_token_id", 1)
        eos = self.config.get("eos_token_id", 2)
        stop = [token for token in [bos, *(eos if isinstance(eos, list) else [eos])] if isinstance(token, int)]
        self.options = {**options, "dtype": np.dtype(dtype).name, "rope_theta": float(self.config.get("rope_theta", 10000.0)),
                        "bos": bos if isinstance(bos, int) else 1, "stop_tokens": stop}
        # a context longer than max_seq_len is cut: the RoPE tables and the scratch of the attention grow with it
        self.stream = Stream(header, int(base), self.config, dtype, int(max_seq_len), start=int(start))
        self.checkpoint = self.stream.out

    def feed(self, data):
        done, total = self.stream.feed(data)
        return done / total

    def finish(self):
        self.stream.finish()
