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


class Llama:
    def __init__(self, checkpoint, tokenizer, dtype="float32", rope_theta=10000.0,
                 tokenizer_kind="bpe", nfkc=False, bos=BOS, stop_tokens=(BOS,)):
        """checkpoint: llama2.c "legacy" format, a 7 int header then the weights.

        dtype="float16" is this project's half-size variant of that format (convert_hf.py writes it).
        bos starts every sequence; generation ends when the model emits one of stop_tokens.
        """
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads,
         self.n_kv_heads, vocab_size, self.seq_len) = struct.unpack_from("<7i", checkpoint, 0)
        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = vocab_size > 0
        self.vocab_size = abs(vocab_size)
        self.head_size = self.dim // self.n_heads
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        kv_dim = self.n_kv_heads * self.head_size

        # float32 weights are views into the checkpoint buffer: nothing is copied. float16 is widened once.
        weights = np.frombuffer(checkpoint, dtype=dtype, offset=28).astype(np.float32, copy=False)
        offset = 0

        def take(*shape):
            nonlocal offset
            count = math.prod(shape)
            array = weights[offset:offset + count].reshape(shape)
            offset += count
            return array

        self.token_embedding_table = take(self.vocab_size, dim)
        self.rms_att_weight = take(n_layers, dim)
        self.wq = take(n_layers, dim, dim)
        self.wk = take(n_layers, kv_dim, dim)
        self.wv = take(n_layers, kv_dim, dim)
        self.wo = take(n_layers, dim, dim)
        self.rms_ffn_weight = take(n_layers, dim)
        self.w1 = take(n_layers, hidden_dim, dim)
        self.w2 = take(n_layers, dim, hidden_dim)
        self.w3 = take(n_layers, hidden_dim, dim)
        self.rms_final_weight = take(dim)
        self.freq_cis_real = take(self.seq_len, self.head_size // 2)
        self.freq_cis_imag = take(self.seq_len, self.head_size // 2)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim)
        if np.dtype(dtype) != np.float32:
            # half precision is too coarse for the rotation angles: compute the RoPE tables again
            angles = np.arange(self.seq_len)[:, None] / rope_theta ** (np.arange(0, self.head_size, 2) / self.head_size)
            self.freq_cis_real, self.freq_cis_imag = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

        self.key_cache = np.zeros((n_layers, self.n_kv_heads, self.seq_len, self.head_size), dtype=np.float32)
        self.value_cache = np.zeros_like(self.key_cache)
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, kind=tokenizer_kind, nfkc=nfkc)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.stats = {}
        self._run = 0

    def forward(self, token, pos, need_logits=True):
        n_kv_heads, head_size = self.n_kv_heads, self.head_size
        kv_mul = self.n_heads // n_kv_heads  # >1 with grouped-query attention
        scale = np.float32(1.0 / math.sqrt(head_size))
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]

        # Copy the token embedding into x
        x = self.token_embedding_table[token].copy()

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

    def sample(self, logits, temperature, topp, rng):
        if temperature == 0.0:
            # Greedy argmax sampling: take the token with the highest probability
            return int(np.argmax(logits))
        probabilities = np.exp((logits - logits.max()) / temperature)
        probabilities /= probabilities.sum()
        candidates = np.arange(probabilities.size)
        if 0.0 < topp < 1.0:
            # Top-p (nucleus) sampling: only the most probable tokens whose probabilities add up to topp.
            # Tokens below (1 - topp) / (n - 1) cannot be part of that set, so they need not be sorted.
            candidates = np.flatnonzero(probabilities >= (1.0 - topp) / (probabilities.size - 1))
            candidates = candidates[np.argsort(-probabilities[candidates])]
            candidates = candidates[:np.searchsorted(np.cumsum(probabilities[candidates]), topp) + 1]
        chosen = probabilities[candidates].astype(np.float64)
        return int(rng.choice(candidates, p=chosen / chosen.sum()))

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
        token, count, sampled = self.bos, 0, 0
        history = [self.bos]
        start = sampling_start = time.perf_counter()
        try:
            for pos in range(steps):
                if pos < len(prompt_tokens):
                    # Still processing the prompt: force the next token, and the logits are not needed
                    self.forward(token, pos, need_logits=False)
                    next_token = prompt_tokens[pos]
                    sampling_start = time.perf_counter()
                else:
                    logits = self.forward(token, pos)
                    if repetition_penalty != 1.0:
                        # make the tokens of the last 64 steps less likely: tiny models love to loop
                        recent = np.unique(history[-64:])
                        logits[recent] = np.where(logits[recent] > 0, logits[recent] / repetition_penalty,
                                                  logits[recent] * repetition_penalty)
                    next_token = self.sample(logits, temperature, topp, rng)
                    sampled += 1
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
                }


async def fetch(url, progress=None, size=0):
    """Download into one preallocated buffer, reporting progress(received, total) along the way.

    size is the expected number of bytes; it is needed when the server compresses the response, because
    Content-Length then counts the compressed bytes. A wrong size only costs a reallocation.
    """
    from pyodide.http import pyfetch

    response = await pyfetch(url)
    response.raise_for_status()
    headers = response.js_response.headers
    total = size if size or headers.get("content-encoding") else int(headers.get("content-length") or 0)
    if progress is None or not total:
        return await response.bytes()
    # chunks go straight into the buffer, so the body never exists twice in memory
    data = bytearray(total)
    received = 0
    reader = response.js_response.body.getReader()
    while True:
        result = await reader.read()
        if result.done:
            break
        chunk = result.value
        if received + chunk.length > len(data):
            data.extend(bytes(received + chunk.length - len(data)))
        chunk.assign_to(memoryview(data)[received:received + chunk.length])
        received += chunk.length
        progress(received, max(total, received))
    del data[received:]
    return data


async def load(checkpoint_url, tokenizer_url, progress=None, size=0, **options):
    """Fetch the model straight into memory (Pyodide only). options are passed on to Llama()."""
    return Llama(await fetch(checkpoint_url, progress, size), await fetch(tokenizer_url), **options)
