# llama2_numpy.py
# Llama 2 inference with NumPy. A port of tairov/llama2.py (itself a port of karpathy/llama2.c) in which
# every loop over vector elements became a NumPy call, so the interpreter only sequences the layers.
import codecs
import math
import struct
import time
import unicodedata

import numpy as np

BOS = 1  # beginning-of-sequence token, also what the model emits when a story is over


class Tokenizer:
    """llama2.c's tokenizer.bin: sentencepiece pieces with their scores.

    kind="bpe" merges the best-scoring adjacent pair, like llama2.c (Llama 2 vocabulary);
    kind="unigram" picks the segmentation with the best total score (sentencepiece unigram models).
    """

    UNMATCHABLE = -1e8  # convert_hf.py gives control and byte pieces a score below this

    def __init__(self, data, vocab_size, kind="bpe", nfkc=False):
        self.kind, self.nfkc = kind, nfkc
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
        self.unknown_score = min(score for score in self.scores if score > self.UNMATCHABLE) - 10.0

    def encode(self, text):
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        # sentencepiece's dummy prefix: the model saw every text start with a space
        text = " " + text
        return self.encode_unigram(text) if self.kind == "unigram" else self.encode_bpe(text)

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
        # Following the first token, sentencepiece decoder strips the leading whitespace (the dummy prefix)
        if prev_token == bos and piece.startswith(b" "):
            piece = piece[1:]
        # Some tokens designate raw bytes, and look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece


def rmsnorm(x, weight):
    return weight * (x / np.sqrt(x.dot(x) / x.size + 1e-5))


def rope(x, cos, sin):
    # Rotate each pair (x[2i], x[2i+1]) of every head by the angle for this position
    pairs = x.reshape(-1, cos.size, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = x0 * cos - x1 * sin
    out[..., 1] = x0 * sin + x1 * cos
    return out.reshape(-1, 2 * cos.size)


REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens


def load_kernels(path):
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
                          matmul_q8=[p, p, p, p, p, i32, i32, i32], rmsnorm=[p, p, p, i32], rope=[p, p, p, i32, i32],
                          attention=[p, p, p, p, p, i32, i32, i32, i32], swiglu=[p, p, p, i32], add_inplace=[p, p, i32],
                          penalize=[p, p, i32, ctypes.c_float],
                          sample=[p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_double, p, p])
        kernels = {}
        for name, argtypes in signatures.items():
            kernels[name] = getattr(lib, name)
            kernels[name].argtypes, kernels[name].restype = argtypes, i32 if name == "sample" else None
    except Exception:
        return None
    try:
        # a browser without relaxed SIMD (shipping Safari) refuses to compile this one: then int8 uses matmul_q8
        relaxed = ctypes.CDLL(path.replace(".so", "_relaxed.wasmlib")).matmul_q8r
        relaxed.argtypes, relaxed.restype = [p, p, p, p, p, p, i32, i32, i32], None
        kernels["matmul_q8r"] = relaxed
    except Exception:
        pass
    return kernels


def checkpoint_dtype(header, size):
    """"float32", "float16" or "int8": what a checkpoint file of size bytes with this header (7 ints) holds.

    The legacy format does not say, but the header fixes the size of each variant. Anything else is no checkpoint
    this engine can read, and the ValueError says so before hundreds of megabytes are read for nothing.
    """
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (int(value) for value in header)
    limit = 1 << 24
    if not (0 < dim < limit and 0 < hidden_dim < limit and 0 < n_layers < 4096 and 0 < n_kv_heads <= n_heads <= dim
            and 0 < abs(vocab_size) < limit and 0 < seq_len < limit and dim % n_heads == 0 and n_heads % n_kv_heads == 0):
        raise ValueError("This is not a llama2.c checkpoint: the header makes no sense.")
    kv_dim = n_kv_heads * (dim // n_heads)
    # the same tensors in the same order as Llama.__init__ and quantize.py: (rows, row length) of the matrices
    matrices = [(abs(vocab_size), dim), (n_layers * dim, dim), (n_layers * kv_dim, dim), (n_layers * kv_dim, dim),
                (n_layers * dim, dim), (n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim),
                (n_layers * hidden_dim, dim)]
    if vocab_size < 0:
        matrices.append((abs(vocab_size), dim))
    vectors = 2 * n_layers * dim + dim
    rope = 2 * seq_len * (dim // n_heads // 2)
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
                 tokenizer_kind="bpe", nfkc=False, bos=BOS, stop_tokens=(BOS,), kernels=None):
        """checkpoint: llama2.c "legacy" format, a 7 int header then the weights.

        dtype="float16" and dtype="int8" are this project's smaller variants (convert_hf.py, quantize.py).
        bos starts every sequence; generation ends when the model emits one of stop_tokens.
        kernels is the path of simdkernel.so; without it, or when it cannot be loaded, NumPy does the math.
        """
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads,
         self.n_kv_heads, vocab_size, self.seq_len) = struct.unpack_from("<7i", checkpoint, 0)
        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = vocab_size > 0
        self.vocab_size = abs(vocab_size)
        self.head_size = self.dim // self.n_heads
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        kv_dim = self.n_kv_heads * self.head_size

        dtype = np.dtype(dtype)
        offset = 28
        # The int8 kernels work on groups of 32 only
        suitable = dtype != np.int8 or (dim % 32 == 0 and kv_dim % 32 == 0 and hidden_dim % 32 == 0)
        kernels = load_kernels(kernels) if kernels and suitable else None
        # int8 kernels compute on the int8 weights directly: they are never widened, a quarter of the memory
        keep_int8 = kernels is not None and dtype == np.int8

        def take(*shape, matrix=True, widen=True):
            nonlocal offset
            count = math.prod(shape)
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
        if dtype != np.float32:
            # half precision is too coarse for the rotation angles, and int8 files leave the RoPE tables out
            angles = np.arange(self.seq_len)[:, None] / rope_theta ** (np.arange(0, self.head_size, 2) / self.head_size)
            self.freq_cis_real, self.freq_cis_imag = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

        self.backend = "NumPy"
        if kernels:
            self.forward = self.kernel_forward(kernels, keep_int8)
            self.penalize, self.sample = self.kernel_sampler(kernels)
        else:
            self.key_cache = np.zeros((n_layers, self.n_kv_heads, self.seq_len, self.head_size), dtype=np.float32)
            self.value_cache = np.zeros_like(self.key_cache)
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, kind=tokenizer_kind, nfkc=nfkc)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.stats = {}
        self._run = 0

    def embedding(self, token):
        if isinstance(self.token_embedding_table, tuple):
            values, scales = self.token_embedding_table
            return (values[token] * scales[token]).reshape(self.dim)
        return self.token_embedding_table[token].astype(np.float32)

    def kernel_forward(self, kernels, int8):
        """forward() on the SIMD kernels: Python still sequences the layers, every operation is one kernel call.

        NumPy owns all the memory. The kernels get addresses, taken once here because array.ctypes.data costs
        microseconds, and work in place: nothing is copied. About 100 calls per token, 3 to 12 us each.
        """
        dim, hidden_dim, n_layers, n_heads, head_size = self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.head_size
        n_kv_heads, kv_dim = self.n_kv_heads, self.n_kv_heads * self.head_size
        x, xb, xb2, q = (np.zeros(dim, dtype=np.float32) for _ in range(4))
        hb, hb2 = np.zeros(hidden_dim, dtype=np.float32), np.zeros(hidden_dim, dtype=np.float32)
        att, logits = np.zeros(self.seq_len, dtype=np.float32), np.zeros(self.vocab_size, dtype=np.float32)
        # [layers][seq][kv_dim], unlike the NumPy forward: k and v of a position are written straight into their rows
        key_cache = np.zeros((n_layers, self.seq_len, kv_dim), dtype=np.float32)
        value_cache = np.zeros_like(key_cache)
        xq, xs = np.zeros(max(dim, hidden_dim), dtype=np.int8), np.zeros(max(dim, hidden_dim) // 32, dtype=np.float32)
        address = lambda array: array.ctypes.data
        x_p, xb_p, xb2_p, q_p, hb_p, hb2_p, att_p, logits_p, xq_p, xs_p = map(address, (x, xb, xb2, q, hb, hb2, att, logits, xq, xs))
        key_p, value_p, cos_p, sin_p = map(address, (key_cache, value_cache, self.freq_cis_real, self.freq_cis_imag))
        att_w, ffn_w, final_w = map(address, (self.rms_att_weight, self.rms_ffn_weight, self.rms_final_weight))
        rmsnorm, rope, attention = kernels["rmsnorm"], kernels["rope"], kernels["attention"]
        swiglu, add_inplace, quantize_x = kernels["swiglu"], kernels["add_inplace"], kernels["quantize_x"]
        self._kernel_buffers = (x, xb, xb2, q, hb, hb2, att, logits, key_cache, value_cache, xq, xs)  # keep them alive

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
        w1, w3 = matmul_for(self.w1, dim, hidden_dim), matmul_for(self.w3, dim, hidden_dim)
        w2, wcls = matmul_for(self.w2, hidden_dim, dim), matmul_for(self.wcls, dim, self.vocab_size)
        layer_bytes, row_bytes, kv_row_bytes, half_bytes = self.seq_len * kv_dim * 4, dim * 4, kv_dim * 4, head_size // 2 * 4

        def forward(token, pos, need_logits=True):
            x[:] = self.embedding(token)
            cos, sin = cos_p + pos * half_bytes, sin_p + pos * half_bytes
            for l in range(n_layers):
                keys, values = key_p + l * layer_bytes, value_p + l * layer_bytes
                k_p, v_p = keys + pos * kv_row_bytes, values + pos * kv_row_bytes
                rmsnorm(xb_p, x_p, att_w + l * row_bytes, dim)
                wq(q_p, xb_p, l)
                wk(k_p, xb_p, l, True)
                wv(v_p, xb_p, l, True)
                rope(q_p, cos, sin, n_heads, head_size)
                rope(k_p, cos, sin, n_kv_heads, head_size)
                attention(xb_p, q_p, keys, values, att_p, pos, n_heads, n_kv_heads, head_size)
                wo(xb2_p, xb_p, l)
                add_inplace(x_p, xb2_p, dim)
                rmsnorm(xb_p, x_p, ffn_w + l * row_bytes, dim)
                w1(hb_p, xb_p, l)
                w3(hb2_p, xb_p, l, True)
                swiglu(hb_p, hb_p, hb2_p, hidden_dim)
                w2(xb2_p, hb_p, l)
                add_inplace(x_p, xb2_p, dim)
            if not need_logits:
                return None
            rmsnorm(xb_p, x_p, final_w, dim)
            wcls(logits_p, xb_p, 0)
            return logits

        return forward

    def forward(self, token, pos, need_logits=True):
        n_kv_heads, head_size = self.n_kv_heads, self.head_size
        kv_mul = self.n_heads // n_kv_heads  # >1 with grouped-query attention
        scale = np.float32(1.0 / math.sqrt(head_size))
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]

        # Copy the token embedding into x
        x = self.embedding(token)

        # Forward all the layers
        for l in range(self.n_layers):
            # QKV matmuls for this position, RoPE on q and k, k and v go to the kv cache
            xb = rmsnorm(x, self.rms_att_weight[l])
            q = rope(self.wq[l] @ xb, cos, sin).reshape(n_kv_heads, kv_mul, head_size)
            self.key_cache[l, :, pos] = rope(self.wk[l] @ xb, cos, sin)
            self.value_cache[l, :, pos] = (self.wv[l] @ xb).reshape(n_kv_heads, head_size)

            # Multihead attention over all timesteps so far, all heads at once
            keys = self.key_cache[l, :, :pos + 1]      # (n_kv_heads, pos + 1, head_size)
            values = self.value_cache[l, :, :pos + 1]
            att = (q @ keys.transpose(0, 2, 1)) * scale  # (n_kv_heads, kv_mul, pos + 1)
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            # Output projection and residual connection
            x += self.wo[l] @ (att @ values).reshape(self.dim)

            # FFN: w2(silu(w1(x)) * w3(x)), and residual connection
            xb = rmsnorm(x, self.rms_ffn_weight[l])
            hb = self.w1[l] @ xb
            hb = hb / (1.0 + np.exp(-hb)) * (self.w3[l] @ xb)
            x += self.w2[l] @ hb

        if not need_logits:
            return None
        # Final rmsnorm, then the classifier into logits (60% of all the multiply-adds of stories15M)
        return self.wcls @ rmsnorm(x, self.rms_final_weight)

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

    def generate(self, prompt="", steps=256, temperature=0.0, topp=0.9, repetition_penalty=1.0, seed=None):
        """Yield the text piece by piece, as it is generated."""
        prompt_tokens = self.tokenizer.encode(prompt) if prompt else []
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
                if text:
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
