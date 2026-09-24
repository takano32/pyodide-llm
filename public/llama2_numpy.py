# llama2_numpy.py
# Llama 2 inference with NumPy. A port of tairov/llama2.py (itself a port of karpathy/llama2.c) in which
# every loop over vector elements became a NumPy call, so the interpreter only sequences the layers.
import codecs
import math
import re
import struct
import time
import unicodedata

import numpy as np

BOS = 1  # beginning-of-sequence token, also what the model emits when a story is over


def byte_chars():
    """GPT-2's byte <-> character table: every byte becomes one printable character, so that a byte-level BPE
    vocabulary is plain text. 0x20 is "\u0120", 0x0A is "\u010a"."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, chars)}


BYTE_CHARS = byte_chars()
CHAR_BYTES = {char: byte for byte, char in BYTE_CHARS.items()}

CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def letter(char):
    return unicodedata.category(char)[0] == "L"


def number(char):
    return unicodedata.category(char)[0] == "N"


def pretokenize(text, pattern):
    r"""Split text the way the tokenizer.json's pre_tokenizer does, without a regex engine: those patterns need
    \p{L} and \p{N}, which the standard re module has not.

    "gpt2" is what ByteLevel(use_regex) applies:
        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    "gpt2-digits" is the same after Digits(individual_digits), which SmolLM2 puts in front of it.
    "qwen" is the pattern Qwen2 spells out (the contractions match whatever the case, digits come one by one,
    and a piece of anything-but-a-line-break may lead a word):
        (?i:'s|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    All three are checked against the real patterns in tests/test_tokenizer.py.
    """
    if pattern == "gpt2-digits":
        parts = []
        for chunk in re.findall(r"\d|\D+", text):
            parts += pretokenize(chunk, "gpt2")
        return parts
    qwen = pattern == "qwen"
    parts, i, n = [], 0, len(text)
    while i < n:
        char = text[i]
        # 's, 't, ... (Qwen matches them whatever the case)
        if char == "'":
            rest = text[i:i + 3].lower() if qwen else text[i:i + 3]
            found = next((c for c in CONTRACTIONS if rest.startswith(c)), None)
            if found:
                parts.append(text[i:i + len(found)])
                i += len(found)
                continue
        # letters, with one character in front of them: a space, or (Qwen) anything but a line break
        start = i
        lead = i + 1 if i + 1 < n and (char == " " or (qwen and not letter(char) and not number(char)
                                                      and char not in "\r\n")) else i
        if lead < n and letter(text[lead]):
            i = lead
            while i < n and letter(text[i]):
                i += 1
            parts.append(text[start:i])
            continue
        # digits: Qwen takes them one by one, GPT-2 takes a run and may put a space in front
        if qwen and number(char):
            parts.append(char)
            i += 1
            continue
        if not qwen:
            lead = i + 1 if char == " " and i + 1 < n and number(text[i + 1]) else i
            if lead < n and number(text[lead]):
                i = lead
                while i < n and number(text[i]):
                    i += 1
                parts.append(text[start:i])
                continue
        # everything else that is not a space, with a space allowed in front of it
        lead = i + 1 if char == " " and i + 1 < n and not text[i + 1].isspace() else i
        if lead < n and not text[lead].isspace() and not letter(text[lead]) and not number(text[lead]):
            i = lead
            while i < n and not text[i].isspace() and not letter(text[i]) and not number(text[i]):
                i += 1
            if qwen:  # the line breaks that follow belong to the same piece
                while i < n and text[i] in "\r\n":
                    i += 1
            parts.append(text[start:i])
            continue
        # whitespace. Qwen keeps a run that ends in line breaks whole; otherwise a run that is followed by a
        # word leaves its last character to that word, and a run at the end of the text stays whole.
        i = start
        if qwen:
            j = i
            while j < n and text[j].isspace() and text[j] not in "\r\n":
                j += 1
            if j < n and text[j] in "\r\n":
                while j < n and text[j] in "\r\n":
                    j += 1
                parts.append(text[i:j])
                i = j
                continue
        j = i
        while j < n and text[j].isspace():
            j += 1
        end = j if j == n or j - 1 == i else j - 1
        parts.append(text[i:end])
        i = end
    return parts


class Tokenizer:
    """llama2.c's tokenizer.bin: sentencepiece pieces with their scores.

    kind="bpe" merges the best-scoring adjacent pair, like llama2.c (Llama 2 vocabulary);
    kind="unigram" picks the segmentation with the best total score (sentencepiece unigram models);
    kind="bytebpe" is Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). It merges by the rank of the merge
    that makes the piece, which is the same loop as "bpe" because the converter writes minus the rank as the
    score. Its vocabulary is written in the byte <-> character table above, so the pieces are plain text.
    """

    UNMATCHABLE = -1e8  # convert_hf.py gives control and byte pieces a score below this

    def __init__(self, data, vocab_size, kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2"):
        self.kind, self.nfkc, self.nfc, self.pretokenizer = kind, nfkc, nfc, pretokenizer
        self.vocab, self.scores = [], []
        offset = 4  # skip max_token_length
        for _ in range(vocab_size):
            score, length = struct.unpack_from("<fi", data, offset)
            offset += 8
            self.vocab.append(bytes(data[offset:offset + length]))
            self.scores.append(score)
            offset += length
        # first occurrence wins, like list.index() in llama2.py
        self.index = {}
        for i, piece in enumerate(self.vocab):
            self.index.setdefault(piece, i)
        # raw byte tokens look like b"<0x0A>"; they spell out whatever the vocabulary lacks
        self.byte_tokens = [self.index.get(b"<0x%02X>" % byte, byte + 3) for byte in range(256)]
        self.max_piece_chars = max(len(piece.decode("utf-8", "ignore")) for piece in self.vocab)
        # byte-level pieces are text, and the merging works on that text rather than on bytes
        self.text_index = {}
        if self.kind == "bytebpe":
            for i, piece in enumerate(self.vocab):
                self.text_index.setdefault(piece.decode("utf-8", "replace"), i)
        self.unknown_score = min(score for score in self.scores if score > self.UNMATCHABLE) - 10.0

    def encode(self, text, specials=()):
        """specials: pieces such as "</s>" that stand for their token wherever they are written (a chat template
        puts them between the turns). As plain text they would be spelled out letter by letter."""
        tokens, first = [], True
        for part in re.split("(" + "|".join(re.escape(special) for special in specials) + ")", text) if specials else [text]:
            if part in specials:
                tokens.append(self.index[part.encode("utf-8")])
            elif part:
                if self.nfkc:
                    part = unicodedata.normalize("NFKC", part)
                if self.nfc:
                    part = unicodedata.normalize("NFC", part)
                if self.kind == "bytebpe":
                    # no dummy prefix: a byte-level vocabulary spells the space out as a character of its own
                    tokens += self.encode_bytebpe(part)
                else:
                    # sentencepiece's dummy prefix: the model saw every text start with a space (but not the
                    # text after a special token)
                    part = " " + part if first else part
                    tokens += self.encode_unigram(part) if self.kind == "unigram" else self.encode_bpe(part)
            first = False
        return tokens

    def encode_bpe(self, text):
        # First encode every individual character; a character the vocabulary lacks becomes its UTF-8 bytes
        tokens = []
        for char in text:
            piece = char.encode("utf-8")
            if piece in self.index:
                tokens.append(self.index[piece])
            else:
                tokens.extend(self.byte_tokens[byte] for byte in piece)

        # Merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while True:
            best_score, best_id, best_idx = -1e10, -1, -1
            for i in range(len(tokens) - 1):
                id = self.index.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if id is not None and self.scores[id] > best_score:
                    best_score, best_id, best_idx = self.scores[id], id, i
            if best_idx == -1:
                return tokens
            tokens[best_idx:best_idx + 2] = [best_id]

    def encode_bytebpe(self, text):
        # The pre-tokenizer keeps merges inside a word: the pieces never cross from a word into the next.
        tokens = []
        for part in pretokenize(text, self.pretokenizer):
            symbols = [BYTE_CHARS[byte] for byte in part.encode("utf-8")]
            while len(symbols) > 1:
                best_score, best_id, best_idx = self.UNMATCHABLE, -1, -1
                for i in range(len(symbols) - 1):
                    id = self.text_index.get(symbols[i] + symbols[i + 1])
                    if id is not None and self.scores[id] > best_score:
                        best_score, best_id, best_idx = self.scores[id], id, i
                if best_idx == -1:
                    break
                symbols[best_idx:best_idx + 2] = [self.vocab[best_id].decode("utf-8", "replace")]
            tokens += [self.text_index[symbol] for symbol in symbols]
        return tokens

    def encode_unigram(self, text):
        # Viterbi: best[j] is the best total score of any segmentation of text[:j]
        best = [0.0] + [-math.inf] * len(text)
        back = [None] * (len(text) + 1)
        for i in range(len(text)):
            if best[i] == -math.inf:
                continue
            for j in range(i + 1, min(len(text), i + self.max_piece_chars) + 1):
                id = self.index.get(text[i:j].encode("utf-8"))
                if id is not None and self.scores[id] > self.UNMATCHABLE and best[i] + self.scores[id] > best[j]:
                    best[j], back[j] = best[i] + self.scores[id], (i, [id])
            # a character the vocabulary lacks becomes its UTF-8 bytes, at a penalty
            if back[i + 1] is None or back[i + 1][0] != i:
                score = best[i] + self.unknown_score
                if score > best[i + 1]:
                    best[i + 1], back[i + 1] = score, (i, [self.byte_tokens[byte] for byte in text[i].encode("utf-8")])
        tokens, j = [], len(text)
        while j > 0:
            i, ids = back[j]
            tokens[:0] = ids
            j = i
        return tokens

    def decode(self, prev_token, token, bos=1):
        piece = self.vocab[token]
        if self.kind == "bytebpe":
            # back through the byte <-> character table; a piece that is not written in it (an added token such
            # as <|im_end|>) is its own text
            text = piece.decode("utf-8", "replace")
            return bytes(CHAR_BYTES[char] for char in text) if all(char in CHAR_BYTES for char in text) else piece
        # Following the first token, sentencepiece decoder strips the leading whitespace (the dummy prefix)
        if prev_token == bos and piece.startswith(b" "):
            piece = piece[1:]
        # Some tokens designate raw bytes, and look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece


def partial_rope(heads, cos, sin, rotary):
    """GPT-NeoX rotates the first rotary values of every head and leaves the rest alone."""
    turned = rope(heads[:, :rotary], cos[:rotary // 2], sin[:rotary // 2])
    return np.concatenate([turned, heads[:, rotary:]], axis=1) if rotary < heads.shape[1] else turned


def rmsnorm(x, weight):
    return weight * (x / np.sqrt(x.dot(x) / x.size + 1e-5))


def layernorm(x, weight, bias):
    """GPT-2 normalizes by the mean and the variance, and adds a bias."""
    centred = x - x.mean()
    return weight * (centred / np.sqrt(centred.dot(centred) / x.size + 1e-5)) + bias


def gelu(x):
    """GPT-2's gelu_new: the tanh approximation, written with exp so that the kernel can do the same."""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * (1.0 / (1.0 + np.exp(-2.0 * inner)))


def rope(x, cos, sin):
    # Rotate each pair (x[2i], x[2i+1]) of every head by the angle for this position
    pairs = x.reshape(-1, cos.size, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = x0 * cos - x1 * sin
    out[..., 1] = x0 * sin + x1 * cos
    return out.reshape(-1, 2 * cos.size)


REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens
# The KV cache starts with room for this many positions and doubles when a run gets there: a context of 4096
# tokens is 200 MB of cache for llm-jp-3-150m, which a short text should not have to pay for (and WebAssembly
# never gives memory back)
KV_START = 256
# The classifier's input has a few channels that the final norm's weight blows up (openai-community/gpt2: 12 to 17
# times, 316 against a median of 0.3). With the int8 kernels the activations are quantized in groups of 32, so one
# such channel sets the scale of its group and the other 31 round to nothing: GPT-2 lost 17% of perplexity to
# that (T92). These many channels, the largest of the norm's weight, are taken out of the vector before it is
# quantized and multiplied in float32 by their own columns of the classifier (vocab_size * 8 multiply-adds next to
# vocab_size * dim). With them GPT-2 is back to +0.35%. Only a norm whose largest weight is OUTLIER_RATIO times
# its median gets this (GPT-2: 13.9; tiny-lm 1.1, llm-jp-3 150M 1.3, Pythia 160M 1.3, rinna GPT-2 1.4, SmolLM2 135M
# 1.9): the others gain nothing from it (tiny-lm stays at +0.31%) and would pay about 3% of speed.
OUTLIER_CHANNELS = 8
OUTLIER_RATIO = 4.0


# what T52 can leave out, each of them something that already has a fallback
SWITCHES = ("kernels", "int8", "relaxed", "sampler")


def load_kernels(path, without_relaxed=False):
    """The WASM SIMD kernels of kernels/*.ts, as ctypes functions, or None when they cannot be used.

    They are Emscripten side modules: ctypes.CDLL links them into Pyodide's own memory, so they work in place
    on NumPy arrays. Anything may go wrong here (no file, not Pyodide, a future Emscripten that loads side
    modules differently), and then NumPy does the work as before.
    """
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        i32, p = ctypes.c_int32, ctypes.c_void_p
        signatures = dict(matmul_f32=[p, p, p, i32, i32, i32], quantize_x=[p, p, p, i32, i32],
                          matmul_q8=[p, p, p, p, p, i32, i32, i32], rmsnorm=[p, p, p, i32], rope=[p, p, p, i32, i32, i32],
                          attention=[p, p, p, p, p, i32, i32, i32, i32], swiglu=[p, p, p, i32], add_inplace=[p, p, i32],
                          add_columns=[p, p, p, i32, i32],
                          layernorm=[p, p, p, p, i32], gelu=[p, p, p, i32],
                          penalize=[p, p, i32, ctypes.c_float],
                          sample=[p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_double, p, p])
        kernels = {}
        for name, argtypes in signatures.items():
            kernels[name] = getattr(lib, name)
            kernels[name].argtypes, kernels[name].restype = argtypes, i32 if name == "sample" else None
    except Exception:
        return None
    if without_relaxed:
        return kernels
    try:
        # a browser without relaxed SIMD (shipping Safari) refuses to compile this one: then int8 uses matmul_q8
        relaxed = ctypes.CDLL(path.replace(".so", "_relaxed.wasmlib")).matmul_q8r
        relaxed.argtypes, relaxed.restype = [p, p, p, p, p, p, i32, i32, i32], None
        kernels["matmul_q8r"] = relaxed
    except Exception:
        pass
    return kernels


def kernel_quantizer(path):
    """llama2_convert.quantize() on the SIMD kernels (T89): int8 values in groups of 32 and one float32 scale per
    group, the same bytes as NumPy's, six times faster (quantize_x with no bias: the activations' quantizer is the
    same computation). For the converter's quantize_rows; None where the kernels cannot be loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    quantize_x = kernels["quantize_x"]

    def quantize_rows(values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        quantized = np.empty(values.size, dtype=np.int8)
        scales = np.empty(values.size // 32, dtype=np.float32)
        quantize_x(quantized.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size, 0)
        return quantized.reshape(-1, 32), scales

    return quantize_rows


class Tensor:
    """Where a tensor of the checkpoint is, when the weights live outside Python (T93: the forward pass runs in
    public/forward.js on its own WebAssembly memory). kind: "int8" (values, then one float32 scale per group of
    the last dimension at scales), "f32" or "f16". Offsets count from the start of the checkpoint file."""

    __slots__ = ("kind", "offset", "shape", "group", "scales")

    def __init__(self, kind, offset, shape, group=0, scales=0):
        self.kind, self.offset, self.shape, self.group, self.scales = kind, offset, tuple(shape), group, scales

    def plan(self):
        return {"kind": self.kind, "offset": self.offset, "shape": list(self.shape), "group": self.group,
                "scales": self.scales}


# the attributes of Llama that are tensors of the file, in no particular order
TENSOR_NAMES = ("token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo", "rms_ffn_weight", "w1", "w2", "w3",
                "rms_final_weight", "freq_cis_real", "freq_cis_imag", "wcls", "bq", "bk", "bv", "positions",
                "ln_att_bias", "ln_ffn_bias", "ln_final_bias", "bo", "b1", "b2")


def outlier_channels(weight, count=OUTLIER_CHANNELS, ratio=OUTLIER_RATIO):
    """The count channels with the largest final-norm weight, in order, or none when the weight has no outliers
    (its largest is less than ratio times its median)."""
    size = np.abs(np.asarray(weight, dtype=np.float32))
    if size.max() < ratio * np.median(size):
        return np.zeros(0, dtype=np.intp)
    return np.sort(np.argsort(-size)[:count])


def outlier_columns(classifier, channels):
    """The columns of the int8 classifier for these channels, as float32, one column per row (len(channels),
    vocab_size): what the add_columns kernel multiplies (see OUTLIER_CHANNELS)."""
    values, scales = classifier  # (vocab_size, dim / group, group) int8 and (vocab_size, dim / group, 1) float32
    group = values.shape[-1]
    columns = np.stack([values[:, c // group, c % group].astype(np.float32) * scales[:, c // group, 0] for c in channels])
    return np.ascontiguousarray(columns)


def checkpoint_dtype(header, size, bias=False, arch="llama"):
    """"float32", "float16" or "int8": what a checkpoint file of size bytes with this header (7 ints) holds.

    The legacy format does not say, but the header fixes the size of each variant. Anything else is no checkpoint
    this engine can read, and the ValueError says so before hundreds of megabytes are read for nothing.
    bias and arch are what the file cannot say either (see Llama.__init__): the tensors differ with them.
    """
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (int(value) for value in header)
    limit = 1 << 24
    if not (0 < dim < limit and 0 < hidden_dim < limit and 0 < n_layers < 4096 and 0 < n_kv_heads <= n_heads <= dim
            and 0 < abs(vocab_size) < limit and 0 < seq_len < limit and dim % n_heads == 0 and n_heads % n_kv_heads == 0):
        raise ValueError("This is not a llama2.c checkpoint: the header makes no sense.")
    kv_dim = n_kv_heads * (dim // n_heads)
    rope = 2 * seq_len * (dim // n_heads // 2)
    if arch in ("gpt2", "neox"):
        # the same tensors in the same order as gpt2_tensors() and llama2_convert.layout(arch=): q, k, v, o, the two
        # FFN matrices (no gate), and for GPT-2 the table of positions in place of the RoPE tables
        matrices = [(abs(vocab_size), dim)] + [(n_layers * dim, dim)] * 4 + [(n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim)]
        if arch == "gpt2":
            matrices.append((seq_len, dim))
            rope = 0
        # LayerNorm weights and biases (two per layer, one at the end), the biases of q, k, v, o and the FFN
        vectors = n_layers * (4 * dim + 3 * dim + dim + hidden_dim + dim) + 2 * dim
    else:
        # the same tensors in the same order as llama_tensors() and quantize.py: (rows, row length) of the matrices
        matrices = [(abs(vocab_size), dim), (n_layers * dim, dim), (n_layers * kv_dim, dim), (n_layers * kv_dim, dim),
                    (n_layers * dim, dim), (n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim),
                    (n_layers * hidden_dim, dim)]
        vectors = 2 * n_layers * dim + dim + (n_layers * (dim + 2 * kv_dim) if bias else 0)
    if vocab_size < 0:
        matrices.append((abs(vocab_size), dim))
    floats = sum(rows * length for rows, length in matrices) + vectors + rope

    def group(length):
        size = 32
        while length % size:
            size //= 2
        return size

    # quantize.py: int8 values and a float32 scale per group; the vectors stay float32, the RoPE tables are left out
    int8 = sum(rows * length + 4 * (rows * length // group(length)) for rows, length in matrices) + 4 * vectors
    sizes = {28 + 4 * floats: "float32", 28 + 2 * floats: "float16", 28 + int8: "int8"}
    if size not in sizes:
        raise ValueError(f"This is not a llama2.c checkpoint: its header asks for {28 + 4 * floats} bytes as float32, "
                         f"{28 + 2 * floats} as float16 or {28 + int8} as int8, and the file has {size}.")
    return sizes[size]


def check_tokenizer(tokenizer, header):
    """ValueError unless tokenizer (a tokenizer.bin) holds exactly the vocabulary of the checkpoint with this header.

    The engine would read the first pieces of a larger vocabulary without complaint, and write nonsense.
    """
    vocab_size = abs(int(list(header)[5]))
    offset, pieces = 4, 0
    while offset + 8 <= len(tokenizer):
        _, length = struct.unpack_from("<fi", tokenizer, offset)
        if length < 0 or offset + 8 + length > len(tokenizer):
            raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
        offset += 8 + length
        pieces += 1
    if offset != len(tokenizer) or pieces == 0:
        raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
    if pieces != vocab_size:
        raise ValueError(f"This tokenizer.bin holds {pieces} pieces, but the checkpoint has a vocabulary of "
                         f"{vocab_size}: they do not belong together.")


class Llama:
    def __init__(self, checkpoint, tokenizer, dtype="float32", rope_theta=10000.0,
                 tokenizer_kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", bias=False, arch="llama",
                 rotary=0, parallel_residual=False, bos=BOS, stop_tokens=(BOS,), kernels=None, specials=(),
                 disable=(), external=None):
        """checkpoint: llama2.c "legacy" format, a 7 int header then the weights.

        dtype="float16" and dtype="int8" are this project's smaller variants (convert_hf.py, quantize.py).
        arch="neox": GPT-NeoX, which is arch="gpt2" with RoPE over the first rotary values of every head
        (rotary=0 means all of them) and, when parallel_residual is on, the attention and the FFN both reading
        the same x instead of one after the other.
        arch="gpt2": LayerNorm instead of RMSNorm, GELU instead of SwiGLU (and no gate matrix), a learned table
        of positions instead of RoPE, and a bias after every projection. The tensors of the file differ with it,
        so it is llama2_convert.layout(arch=) that says what is there.
        bias=True: the checkpoint ends with a bias for q, k and v of every layer, which is added after those
        projections (Qwen2). The legacy header cannot say so, so the caller does, like the tokenizer settings.
        bos starts every sequence; generation ends when the model emits one of stop_tokens.
        kernels is the path of simdkernel.so; without it, or when it cannot be loaded, NumPy does the math.
        disable: the optimizations to leave out, to measure what each one is worth (T52). Only what already has
        a fallback: "kernels" (NumPy does everything), "int8" (the weights are widened to float32 and the
        float32 kernel multiplies them), "relaxed" (matmul_q8 instead of matmul_q8r) and "sampler" (NumPy
        samples). Anything else is refused, so that a typo never quietly measures the wrong thing.
        external (T93): the weights are not in Python but in a WebAssembly memory of public/forward.js, which also
        runs the forward pass. checkpoint is then None, and external has size (of the file), read(offset, length)
        (a few bytes: the header, the final norm) and start(plan), which gets where every tensor is and returns an
        object with forward(token, pos, need_logits, logits) and backend. Python keeps the tokenizer, generate()
        and the sampling. The NumPy forward cannot run on weights it does not have: disable "kernels" without it.
        """
        if external is not None:
            head = external.read(0, 28)
            checkpoint = bytes(head.to_py() if hasattr(head, "to_py") else head)
            if "kernels" in tuple(str(name) for name in disable):
                raise ValueError("Without the kernels the weights have to be in Python: load them there instead.")
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads,
         self.n_kv_heads, vocab_size, self.seq_len) = struct.unpack_from("<7i", checkpoint, 0)
        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = vocab_size > 0
        self.vocab_size = abs(vocab_size)
        self.head_size = self.dim // self.n_heads
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        kv_dim = self.n_kv_heads * self.head_size

        disable = tuple(str(name) for name in disable)
        unknown = [name for name in disable if name not in SWITCHES]
        if unknown:
            raise ValueError(f"There is no optimization called {unknown[0]!r}: {', '.join(SWITCHES)}.")
        self.disabled = disable
        dtype = np.dtype(dtype)
        offset = 28
        # The int8 kernels work on groups of 32 only
        suitable = dtype != np.int8 or (dim % 32 == 0 and kv_dim % 32 == 0 and hidden_dim % 32 == 0)
        kernels = load_kernels(kernels, "relaxed" in disable) if kernels and "kernels" not in disable and \
            (suitable or external is not None) else None
        # int8 kernels compute on the int8 weights directly: they are never widened, a quarter of the memory
        keep_int8 = (kernels is not None or external is not None) and suitable and dtype == np.int8 and "int8" not in disable

        def take(*shape, matrix=True, widen=True):
            nonlocal offset
            count = math.prod(shape)
            if external is not None:
                # only where it is: public/forward.js reads it (and widens what has to be widened) itself
                if dtype == np.int8 and matrix:
                    group = 32
                    while shape[-1] % group:
                        group //= 2
                    tensor = Tensor("int8", offset, shape, group, offset + count)
                    offset += count + 4 * (count // group)
                    return tensor
                tensor = Tensor("f16" if dtype == np.float16 else "f32", offset, shape)
                offset += count * (2 if dtype == np.float16 else 4)
                return tensor
            if dtype == np.int8 and matrix:
                # quantize.py: int8 values, then one float32 scale per group
                group = 32
                while shape[-1] % group:
                    group //= 2
                values = np.frombuffer(checkpoint, dtype=np.int8, count=count, offset=offset)
                scales = np.frombuffer(checkpoint, dtype=np.float32, count=count // group, offset=offset + count)
                offset += count + scales.nbytes
                if not widen:
                    values, scales = values.reshape(*shape[:-1], -1, group), scales.reshape(*shape[:-1], -1, 1)
                    # With the kernels every tensor stays a view into the checkpoint buffer. Otherwise this is the
                    # embedding table next to widened copies: copy it, so that the buffer can be freed.
                    return (values, scales) if keep_int8 else (values.copy(), scales.copy())
                return (values.reshape(-1, group).astype(np.float32) * scales[:, None]).reshape(shape)
            # float32 weights are views into the checkpoint buffer: nothing is copied. float16 is widened.
            array = np.frombuffer(checkpoint, dtype=np.float32 if dtype == np.int8 else dtype, count=count, offset=offset)
            offset += array.nbytes
            if dtype == np.float16 and not widen:
                return array.reshape(shape).copy()
            return array.astype(np.float32, copy=dtype == np.int8 and not keep_int8).reshape(shape)

        self.arch, self.parallel_residual = arch, parallel_residual
        # how many values of each head RoPE turns: all of them unless the model says otherwise
        self.rotary = int(rotary) if rotary else self.head_size
        self.positions = None
        self.ln_att_bias = self.ln_ffn_bias = self.ln_final_bias = None
        self.bo = self.b1 = self.b2 = None
        if arch in ("gpt2", "neox"):
            self.gpt2_tensors(take, shared_weights, keep_int8, kv_dim, dtype, rope_theta)
        else:
            self.llama_tensors(take, shared_weights, keep_int8, kv_dim, bias, dtype, rope_theta)
        self.backend = "NumPy"
        if external is not None:
            if offset != int(external.size):
                raise ValueError(f"The checkpoint has {int(external.size)} bytes, and its header asks for {offset} "
                                 f"as {dtype.name}.")
            self.forward = self.external_forward(external, keep_int8, disable)
            if kernels and "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        elif kernels:
            self.forward = self.kernel_forward(kernels, keep_int8)
            if "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        else:
            self.key_cache = np.zeros((n_layers, self.n_kv_heads, min(KV_START, self.seq_len), self.head_size), dtype=np.float32)
            self.value_cache = np.zeros_like(self.key_cache)
        if (kernels or external is not None) and "sampler" in disable:
            self.backend += ", NumPy sampling"
        if disable:
            # the line has to say what the numbers are the numbers of
            self.backend += " (without " + ", ".join(name for name in SWITCHES if name in disable) + ")"
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, kind=tokenizer_kind, nfkc=nfkc, nfc=nfc,
                                   pretokenizer=pretokenizer)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.specials = tuple(str(special) for special in specials)  # see Tokenizer.encode()
        self.stats = {}
        self._run = 0

    def llama_tensors(self, take, shared_weights, keep_int8, kv_dim, bias, dtype, rope_theta):
        """The tensors of a Llama (and of a Qwen2, which adds the q, k and v biases at the end), in file order."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        # With a separate classifier the embedding table is only ever read one row at a time, so an int8 or
        # float16 table stays as it is (a quarter or half of the memory) and forward() widens the row it needs.
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        self.rms_att_weight = take(n_layers, dim, matrix=False)
        self.wq = take(n_layers, dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wo = take(n_layers, dim, dim, widen=not keep_int8)
        self.rms_ffn_weight = take(n_layers, dim, matrix=False)
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.w3 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.rms_final_weight = take(dim, matrix=False)
        if dtype != np.int8:
            self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
            self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        # the q, k and v biases go last, so that a checkpoint without them is the file it always was
        self.bq = take(n_layers, dim, matrix=False) if bias else None
        self.bk = take(n_layers, kv_dim, matrix=False) if bias else None
        self.bv = take(n_layers, kv_dim, matrix=False) if bias else None
        if dtype != np.float32:
            # half precision is too coarse for the rotation angles, and int8 files leave the RoPE tables out
            angles = np.arange(self.seq_len)[:, None] / rope_theta ** (np.arange(0, self.head_size, 2) / self.head_size)
            self.freq_cis_real, self.freq_cis_imag = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def gpt2_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, rope_theta):
        """The tensors of a GPT-2 or a GPT-NeoX, in the order llama2_convert.layout() writes them. The two
        differ in one place: GPT-2 has a learned table of positions, GPT-NeoX the RoPE tables (left out of an
        int8 checkpoint, as everywhere)."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        vector = lambda n=dim: take(n_layers, n, matrix=False)
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        if self.arch == "neox":
            if dtype != np.int8:
                self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
                self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        else:
            self.positions = take(self.seq_len, dim, widen=True)
        self.rms_att_weight, self.ln_att_bias = vector(), vector()
        self.wq = take(n_layers, dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.bq, self.bk, self.bv = vector(), vector(kv_dim), vector(kv_dim)
        self.wo = take(n_layers, dim, dim, widen=not keep_int8)
        self.bo = vector()
        self.rms_ffn_weight, self.ln_ffn_bias = vector(), vector()
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.b1 = vector(hidden_dim)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.b2 = vector()
        self.rms_final_weight = take(dim, matrix=False)
        self.ln_final_bias = take(dim, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        self.w3 = None
        if self.arch == "gpt2":
            # no rotation: the position is a row of a learned table, added to the embedding
            self.freq_cis_real = self.freq_cis_imag = np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32)
        elif dtype != np.float32:
            # the angles of the rotated part only, in a table of the same shape (the rest is never read)
            angles = np.arange(self.seq_len)[:, None] / rope_theta ** (np.arange(0, self.rotary, 2) / self.rotary)
            tables = [np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32) for _ in range(2)]
            for table, values in zip(tables, (np.cos(angles), np.sin(angles))):
                table[:, :self.rotary // 2] = values
            self.freq_cis_real, self.freq_cis_imag = tables

    def embedding(self, token):
        if isinstance(self.token_embedding_table, tuple):
            values, scales = self.token_embedding_table
            return (values[token] * scales[token]).reshape(self.dim)
        return self.token_embedding_table[token].astype(np.float32)

    def external_forward(self, external, int8, disable):
        """forward() in public/forward.js (T93): Python hands over where every tensor is, and the few small
        arrays it computes itself (the RoPE tables of a checkpoint that leaves them out, the outlier channels of
        T92), and gets the logits back into one array of its own, which the sampling kernels then read."""
        tensors = {name: getattr(self, name).plan() for name in TENSOR_NAMES if isinstance(getattr(self, name, None), Tensor)}
        derived = {name: np.ascontiguousarray(getattr(self, name), dtype=np.float32).tobytes()
                   for name in ("freq_cis_real", "freq_cis_imag") if isinstance(getattr(self, name), np.ndarray)}
        channels = []
        if int8:
            final = self.rms_final_weight
            raw = external.read(final.offset, self.dim * 4)
            weight = np.frombuffer(bytes(raw.to_py() if hasattr(raw, "to_py") else raw), dtype=np.float32)
            channels = [int(c) for c in outlier_channels(weight, min(OUTLIER_CHANNELS, self.dim))]
        plan = {"arch": self.arch, "dim": self.dim, "hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
                "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads, "head_size": self.head_size,
                "vocab_size": self.vocab_size, "seq_len": self.seq_len, "rotary": self.rotary,
                "parallel_residual": bool(self.parallel_residual), "kv_start": KV_START,
                "shared_classifier": self.wcls is self.token_embedding_table, "int8": bool(int8),
                "relaxed": "relaxed" not in disable, "tensors": tensors, "derived": derived, "outliers": channels}
        engine = external.start(plan)
        self.backend = str(engine.backend)
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        engine.bind(logits)
        self._external = (engine, logits)  # keep both alive: JS writes into the array
        run = engine.forward

        def forward(token, pos, need_logits=True):
            run(token, pos, need_logits)
            return logits if need_logits else None

        return forward

    def release(self):
        """Let go of what JavaScript holds for this engine (the forward pass of forward.js and the array it fills).
        The worker calls it before it drops a model (T93)."""
        external = getattr(self, "_external", None)
        if external is not None:
            external[0].release()
            self._external = None

    def kernel_forward(self, kernels, int8):
        """forward() on the SIMD kernels: Python still sequences the layers, every operation is one kernel call.

        NumPy owns all the memory. The kernels get addresses, taken once here because array.ctypes.data costs
        microseconds, and work in place: nothing is copied. About 100 calls per token, 3 to 12 us each.
        """
        dim, hidden_dim, n_layers, n_heads, head_size = self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.head_size
        n_kv_heads, kv_dim = self.n_kv_heads, self.n_kv_heads * self.head_size
        x, xb, xb2, q, before = (np.zeros(dim, dtype=np.float32) for _ in range(5))
        hb, hb2 = np.zeros(hidden_dim, dtype=np.float32), np.zeros(hidden_dim, dtype=np.float32)
        # the attention kernel keeps the scores of all heads: it walks the cache once, not once per head
        att, logits = np.zeros(self.seq_len * n_heads, dtype=np.float32), np.zeros(self.vocab_size, dtype=np.float32)
        # per layer [positions][kv_dim], unlike the NumPy forward: k and v of a position are written straight into
        # their rows. One array per layer, so that growing (see KV_START) never needs a second copy of all of it.
        capacity = min(KV_START, self.seq_len)
        key_cache = [np.zeros((capacity, kv_dim), dtype=np.float32) for _ in range(n_layers)]
        value_cache = [np.zeros((capacity, kv_dim), dtype=np.float32) for _ in range(n_layers)]
        xq, xs = np.zeros(max(dim, hidden_dim), dtype=np.int8), np.zeros(max(dim, hidden_dim) // 32, dtype=np.float32)
        address = lambda array: array.ctypes.data
        x_p, xb_p, xb2_p, q_p, hb_p, hb2_p, att_p, logits_p, xq_p, xs_p = map(address, (x, xb, xb2, q, hb, hb2, att, logits, xq, xs))
        before_p, parallel = address(before), self.parallel_residual
        key_p, value_p = [address(layer) for layer in key_cache], [address(layer) for layer in value_cache]
        cos_p, sin_p = address(self.freq_cis_real), address(self.freq_cis_imag)
        att_w, ffn_w, final_w = map(address, (self.rms_att_weight, self.rms_ffn_weight, self.rms_final_weight))
        rmsnorm, rope, attention = kernels["rmsnorm"], kernels["rope"], kernels["attention"]
        swiglu, add_inplace, quantize_x = kernels["swiglu"], kernels["add_inplace"], kernels["quantize_x"]
        neox, gpt2 = self.arch == "neox", self.arch == "gpt2"
        layer_norm, rotary = neox or gpt2, self.rotary
        # only a GPT-2 or a GPT-NeoX has these; a Llama never reads them
        positions_p = att_b = ffn_b = final_b = bo_p = b1_p = b2_p = 0
        # GPT-2 and GPT-NeoX: LayerNorm instead of RMSNorm (a bias comes with each), GELU instead of the gated
        # SwiGLU, and either a learned table of positions (GPT-2) or a rotation of part of each head (NeoX)
        if layer_norm:
            layer_norm_kernel, gelu_kernel = kernels["layernorm"], kernels["gelu"]
            att_b, ffn_b, final_b = map(address, (self.ln_att_bias, self.ln_ffn_bias, self.ln_final_bias))
            bo_p, b1_p, b2_p = map(address, (self.bo, self.b1, self.b2))
            positions_p = address(self.positions) if self.positions is not None else 0  # an attribute: it stays alive
        self._kernel_buffers = (x, xb, xb2, q, hb, hb2, att, logits, key_cache, value_cache, xq, xs, before)  # keep them alive

        if int8:
            relaxed = kernels.get("matmul_q8r")
            self.backend = "SIMD kernels, int8" + (", relaxed SIMD" if relaxed else "")
            # relaxed SIMD multiplies int8 by 7-bit unsigned: activations get a bias of 64, which
            # corrections = scale * sum(group) takes out again, as dot(w, q - 64) = dot(w, q) - 64 * sum(w)
            bias = 64 if relaxed else 0
            self._corrections = []

            def pointers(tensor):
                values, scales = tensor
                corrections = (scales[..., 0] * values.sum(axis=-1, dtype=np.int32)).astype(np.float32) if relaxed else scales
                self._corrections.append(corrections)
                return values, scales, corrections

            def matmul_for(tensor, n, d):
                values, scales, corrections = pointers(tensor)
                layers = [(address(values[l]), address(scales[l]), address(corrections[l])) for l in range(len(values))] \
                    if values.ndim == 4 else [(address(values), address(scales), address(corrections))]

                def matmul(out_p, in_p, l, same_input=False):
                    values_p, scales_p, corrections_p = layers[l]
                    if not same_input:  # q, k, v (and w1, w3) share their input: quantize it once
                        quantize_x(xq_p, xs_p, in_p, n, bias)
                    if relaxed:
                        relaxed(out_p, xq_p, xs_p, values_p, scales_p, corrections_p, n, 0, d)
                    else:
                        kernels["matmul_q8"](out_p, xq_p, xs_p, values_p, scales_p, n, 0, d)
                return matmul
        else:
            self.backend = "SIMD kernels, float32"

            def matmul_for(tensor, n, d):
                layers = [address(tensor[l]) for l in range(len(tensor))] if tensor.ndim == 3 else [address(tensor)]

                def matmul(out_p, in_p, l, same_input=False):
                    kernels["matmul_f32"](out_p, in_p, layers[l], n, 0, d)
                return matmul

        wq, wo = matmul_for(self.wq, dim, dim), matmul_for(self.wo, dim, dim)
        wk, wv = matmul_for(self.wk, dim, kv_dim), matmul_for(self.wv, dim, kv_dim)
        w1 = matmul_for(self.w1, dim, hidden_dim)
        w3 = matmul_for(self.w3, dim, hidden_dim) if self.w3 is not None else None
        w2, wcls = matmul_for(self.w2, hidden_dim, dim), matmul_for(self.wcls, dim, self.vocab_size)
        channels = outlier_channels(self.rms_final_weight, min(OUTLIER_CHANNELS, dim)) if int8 else ()
        if len(channels):
            # the classifier reads xb: its outlier channels go through add_columns instead (see OUTLIER_CHANNELS)
            columns = outlier_columns(self.wcls, channels)
            picked = np.zeros(len(channels), dtype=np.float32)
            self._outlier_buffers = (columns, picked)  # keep them alive
            columns_p, picked_p, count = address(columns), address(picked), len(channels)
            add_columns, classify, vocab_size = kernels["add_columns"], wcls, self.vocab_size

            def wcls(out_p, in_p, l):
                picked[:] = xb[channels]
                xb[channels] = 0.0
                classify(out_p, in_p, l)
                add_columns(out_p, columns_p, picked_p, count, vocab_size)
        row_bytes, kv_row_bytes, half_bytes = dim * 4, kv_dim * 4, head_size // 2 * 4
        hidden_row_bytes = hidden_dim * 4
        # Qwen2's bias on q, k and v: one add_inplace each, on the float32 vectors the matmul just wrote
        bq_p, bk_p, bv_p = (address(tensor) if tensor is not None else 0 for tensor in (self.bq, self.bk, self.bv))

        def grow(pos):
            nonlocal capacity
            capacity = min(max(2 * capacity, pos + 1), self.seq_len)
            for cache, addresses in ((key_cache, key_p), (value_cache, value_p)):
                for l in range(n_layers):
                    larger = np.zeros((capacity, kv_dim), dtype=np.float32)
                    larger[:len(cache[l])] = cache[l]
                    cache[l], addresses[l] = larger, address(larger)  # the smaller one is freed here, layer by layer

        def forward(token, pos, need_logits=True):
            if pos >= capacity:
                grow(pos)
            x[:] = self.embedding(token)
            if positions_p:
                add_inplace(x_p, positions_p + pos * row_bytes, dim)
            cos, sin = cos_p + pos * half_bytes, sin_p + pos * half_bytes
            for l in range(n_layers):
                keys, values = key_p[l], value_p[l]
                k_p, v_p = keys + pos * kv_row_bytes, values + pos * kv_row_bytes
                if layer_norm:
                    layer_norm_kernel(xb_p, x_p, att_w + l * row_bytes, att_b + l * row_bytes, dim)
                else:
                    rmsnorm(xb_p, x_p, att_w + l * row_bytes, dim)
                if parallel:
                    before[:] = x  # GPT-NeoX reads this layer's input in both branches
                wq(q_p, xb_p, l)
                wk(k_p, xb_p, l, True)
                wv(v_p, xb_p, l, True)
                if bq_p:
                    add_inplace(q_p, bq_p + l * row_bytes, dim)
                    add_inplace(k_p, bk_p + l * kv_row_bytes, kv_dim)
                    add_inplace(v_p, bv_p + l * kv_row_bytes, kv_dim)
                if not gpt2:
                    rope(q_p, cos, sin, n_heads, head_size, rotary)
                    rope(k_p, cos, sin, n_kv_heads, head_size, rotary)
                attention(xb_p, q_p, keys, values, att_p, pos, n_heads, n_kv_heads, head_size)
                wo(xb2_p, xb_p, l)
                add_inplace(x_p, xb2_p, dim)
                if layer_norm:
                    add_inplace(x_p, bo_p + l * row_bytes, dim)
                    layer_norm_kernel(xb_p, before_p if parallel else x_p, ffn_w + l * row_bytes,
                                      ffn_b + l * row_bytes, dim)
                    w1(hb_p, xb_p, l)
                    gelu_kernel(hb_p, hb_p, b1_p + l * hidden_row_bytes, hidden_dim)
                    w2(xb2_p, hb_p, l)
                    add_inplace(x_p, xb2_p, dim)
                    add_inplace(x_p, b2_p + l * row_bytes, dim)
                    continue
                rmsnorm(xb_p, x_p, ffn_w + l * row_bytes, dim)
                w1(hb_p, xb_p, l)
                w3(hb2_p, xb_p, l, True)
                swiglu(hb_p, hb_p, hb2_p, hidden_dim)
                w2(xb2_p, hb_p, l)
                add_inplace(x_p, xb2_p, dim)
            if not need_logits:
                return None
            if layer_norm:
                layer_norm_kernel(xb_p, x_p, final_w, final_b, dim)
            else:
                rmsnorm(xb_p, x_p, final_w, dim)
            wcls(logits_p, xb_p, 0)
            return logits

        return forward

    def forward(self, token, pos, need_logits=True):
        n_kv_heads, head_size = self.n_kv_heads, self.head_size
        kv_mul = self.n_heads // n_kv_heads  # >1 with grouped-query attention
        if pos >= self.key_cache.shape[2]:
            # room for twice as many positions (see KV_START)
            positions = min(max(2 * self.key_cache.shape[2], pos + 1), self.seq_len)
            for name in ("key_cache", "value_cache"):
                cache = getattr(self, name)
                larger = np.zeros((*cache.shape[:2], positions, head_size), dtype=np.float32)
                larger[:, :, :cache.shape[2]] = cache
                setattr(self, name, larger)
        scale = np.float32(1.0 / math.sqrt(head_size))
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]
        neox, gpt2 = self.arch == "neox", self.arch == "gpt2"
        layer_norm = neox or gpt2
        # GPT-2 and GPT-NeoX normalize by the mean as well, and have a bias on every projection
        norm = (lambda v, w, b: layernorm(v, w, b)) if layer_norm else (lambda v, w, b: rmsnorm(v, w))
        if gpt2:
            turn = lambda v, c, s: v.reshape(-1, head_size)
        elif neox:
            # only the first self.rotary of every head are rotated, the rest go through untouched
            turn = lambda v, c, s: partial_rope(v.reshape(-1, head_size), c, s, self.rotary)
        else:
            turn = rope

        # Copy the token embedding into x, and (GPT-2) the row of this position
        x = self.embedding(token)
        if gpt2:
            x = x + self.positions[pos]

        # Forward all the layers
        for l in range(self.n_layers):
            # QKV matmuls for this position, RoPE on q and k, k and v go to the kv cache
            xb = norm(x, self.rms_att_weight[l], self.ln_att_bias[l] if layer_norm else None)
            qv, kv, vv = self.wq[l] @ xb, self.wk[l] @ xb, self.wv[l] @ xb
            if self.bq is not None:  # Qwen2 and GPT-2 add a bias to q, k and v
                qv, kv, vv = qv + self.bq[l], kv + self.bk[l], vv + self.bv[l]
            q = turn(qv, cos, sin).reshape(n_kv_heads, kv_mul, head_size)
            self.key_cache[l, :, pos] = turn(kv, cos, sin)
            self.value_cache[l, :, pos] = vv.reshape(n_kv_heads, head_size)

            # Multihead attention over all timesteps so far, all heads at once
            keys = self.key_cache[l, :, :pos + 1]      # (n_kv_heads, pos + 1, head_size)
            values = self.value_cache[l, :, :pos + 1]
            att = (q @ keys.transpose(0, 2, 1)) * scale  # (n_kv_heads, kv_mul, pos + 1)
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            # Output projection and residual connection
            attended = self.wo[l] @ (att @ values).reshape(self.dim)
            if layer_norm:
                attended = attended + self.bo[l]
            # GPT-NeoX with use_parallel_residual: both branches read the x this layer began with
            before = x
            x = x + attended

            # FFN: w2(silu(w1(x)) * w3(x)), or GPT-2's w2(gelu(w1(x))), and residual connection
            xb = norm(before if self.parallel_residual else x, self.rms_ffn_weight[l],
                      self.ln_ffn_bias[l] if layer_norm else None)
            hb = self.w1[l] @ xb
            if layer_norm:
                x = x + self.w2[l] @ gelu(hb + self.b1[l]) + self.b2[l]
            else:
                hb = hb / (1.0 + np.exp(-hb)) * (self.w3[l] @ xb)
                x = x + self.w2[l] @ hb

        if not need_logits:
            return None
        # Final norm, then the classifier into logits (60% of all the multiply-adds of stories15M)
        return self.wcls @ norm(x, self.rms_final_weight, self.ln_final_bias)


    def kernel_sampler(self, kernels):
        """penalize() and sample() on the kernels: the same as the methods below, which stay for NumPy alone.

        Sorting the candidates of the nucleus was most of the time of a step for a small model with a large
        vocabulary, and a repetition penalty, which flattens the distribution, made it worse.
        """
        probabilities, order = np.empty(self.vocab_size, dtype=np.float32), np.empty(self.vocab_size, dtype=np.int32)
        recent = np.empty(REPETITION_WINDOW, dtype=np.int32)
        probabilities_p, order_p, recent_p = probabilities.ctypes.data, order.ctypes.data, recent.ctypes.data
        self._sampler_buffers = (probabilities, order, recent)  # keep them alive: the kernels only know addresses
        kernel_penalize, kernel_sample = kernels["penalize"], kernels["sample"]
        known = {}  # id(logits) -> (logits, address): forward() returns the same array every time

        def address(logits):
            entry = known.get(id(logits))
            if entry is None or entry[0] is not logits:
                if logits.dtype != np.float32 or not logits.flags.c_contiguous:
                    raise TypeError("the kernels need contiguous float32 logits")
                known.clear()
                entry = known[id(logits)] = (logits, logits.ctypes.data)
            return entry[1]

        seen = [None, 0]  # the list that recent[] mirrors, and its length then

        def penalize(logits, history, penalty):
            # recent[] is a ring of the latest tokens. generate() appends one token per step, and then one number
            # is written here: copying 64 of them from a list costs more than the kernel takes
            size = len(history)
            if seen[0] is history and size == seen[1] + 1:
                recent[(size - 1) % REPETITION_WINDOW] = history[-1]
            else:
                for position in range(max(size - REPETITION_WINDOW, 0), size):
                    recent[position % REPETITION_WINDOW] = history[position]
            seen[0], seen[1] = history, size
            kernel_penalize(address(logits), recent_p, min(size, REPETITION_WINDOW), penalty)

        def sample(logits, temperature, topp, rng):
            if temperature == 0.0:
                return int(np.argmax(logits))
            # the random number is drawn here, so that a seed gives the same text again
            return kernel_sample(address(logits), logits.size, temperature, topp, rng.random(), probabilities_p, order_p)

        return penalize, sample

    def penalize(self, logits, history, penalty):
        """Make the tokens of the last steps less likely: tiny models love to loop."""
        recent = np.unique(history[-REPETITION_WINDOW:])
        logits[recent] = np.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)

    def sample(self, logits, temperature, topp, rng):
        if temperature == 0.0:
            # Greedy argmax sampling: take the token with the highest probability
            return int(np.argmax(logits))
        nucleus = 0.0 < topp < 1.0
        if nucleus:
            # exp() over a vocabulary of 50000 or 100000 tokens costs as much as half a forward pass, and nearly all
            # of it goes to tokens that cannot be drawn: leave out, while still cheap, whatever is less than a ten
            # millionth as probable as the best token (together far below one percent of the probability mass)
            best = logits.max()
            candidates = np.flatnonzero(logits >= best + temperature * math.log(1e-7))
            probabilities = np.exp((logits[candidates] - best) / temperature).astype(np.float64)
        else:
            candidates = np.arange(logits.size)
            probabilities = np.exp((logits - logits.max()) / temperature).astype(np.float64)
        probabilities /= probabilities.sum()
        if nucleus:
            # Top-p (nucleus) sampling: only the most probable tokens whose probabilities add up to topp.
            # Tokens below (1 - topp) / (n - 1) cannot be part of that set (llama2.c), so they need not be sorted.
            likely = np.flatnonzero(probabilities >= (1.0 - topp) / max(probabilities.size - 1, 1))
            likely = likely[np.argsort(-probabilities[likely])]
            candidates, probabilities = candidates[likely], probabilities[likely]
            cumulative = np.cumsum(probabilities)
            cumulative = cumulative[:np.searchsorted(cumulative, topp) + 1]
        else:
            cumulative = np.cumsum(probabilities)
        # one random number on the cumulative distribution; Generator.choice() would cost a third of a millisecond
        chosen = np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")
        return int(candidates[min(chosen, cumulative.size - 1)])

    def generate(self, prompt="", steps=256, temperature=0.0, topp=0.9, repetition_penalty=1.0, seed=None, echo=True):
        """Yield the text piece by piece, as it is generated. echo=False leaves the prompt out of it (an instruction
        wrapped in a template, which nobody wants to read back)."""
        prompt_tokens = self.tokenizer.encode(prompt, self.specials) if prompt else []
        # Right now we cannot run for more than seq_len steps
        if steps <= 0 or steps > self.seq_len:
            steps = self.seq_len
        if len(prompt_tokens) >= steps:
            raise ValueError(f"The prompt is {len(prompt_tokens)} tokens long, but only {steps - 1} fit.")
        rng = np.random.default_rng(seed)
        # a character can be split over several tokens
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

        self._run += 1
        run = self._run
        token, count, sampled, forced = self.bos, 0, 0, 0
        history = [self.bos]
        start = sampling_start = time.perf_counter()
        first_token = None
        try:
            for pos in range(steps):
                if pos < len(prompt_tokens):
                    # Still processing the prompt: force the next token, and the logits are not needed
                    self.forward(token, pos, need_logits=False)
                    next_token = prompt_tokens[pos]
                    forced += 1
                    sampling_start = time.perf_counter()
                else:
                    logits = self.forward(token, pos)
                    if repetition_penalty != 1.0:
                        self.penalize(logits, history, repetition_penalty)
                    next_token = self.sample(logits, temperature, topp, rng)
                    sampled += 1
                    if first_token is None:
                        first_token = time.perf_counter()
                    # The BOS token delimits sequences: the story is over
                    if next_token in self.stop_tokens:
                        break
                text = utf8.decode(self.tokenizer.decode(token, next_token, self.bos))
                token = next_token
                history.append(token)
                count += 1
                if text and (echo or pos >= len(prompt_tokens)):
                    yield text
            text = utf8.decode(b"", final=True)
            if text:
                yield text
        finally:
            # an abandoned generator that is finalized late must not overwrite the stats of a newer run
            if run == self._run:
                now = time.perf_counter()
                self.stats = {
                    "tokens": count,
                    "seconds": now - start,
                    # the prompt is not counted: its tokens skip the classifier, so they are much cheaper
                    "tokens_per_second": sampled / (now - sampling_start) if now > sampling_start else 0.0,
                    # The prompt costs a forward pass per token too, and it is what the reader waits for first.
                    # sampled counts the sampling steps, one more than "tokens" shows when a stop token ended it.
                    "sampled": sampled,
                    "prompt_tokens": forced,
                    "prompt_seconds": sampling_start - start,
                    "prompt_tokens_per_second": forced / (sampling_start - start) if sampling_start > start else 0.0,
                    "first_token_seconds": first_token - start if first_token is not None else 0.0,
                }
