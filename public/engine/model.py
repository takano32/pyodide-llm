# Llama: a model loaded from a checkpoint, and its forward pass in NumPy (the reference; the page's is forward.js).
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
import struct

import numpy as np

from engine.tokenizer import BOS, Tokenizer
from engine.layout import Dims, convolution_form, file_size, linear_form, linear_widths
from engine.layers import (RMS_EPS, delta_rule, gelu, head_norm, l2_heads, layernorm, partial_rope, rmsnorm, rope,
                           rope_frequencies, rope_magnitude, rotate, rotated_form, rotated_widths, silu, softplus,
                           unrotate)
from engine.kernels import load_kernels
from engine.packing import PACKED
from engine.checkpoint import OUTLIER_CHANNELS, Tensor, outlier_channels
from engine.tensors import TensorOrder
from engine.sampling import REPETITION_WINDOW, Sampling
from engine.generation import Generation

# The KV cache starts with room for this many positions and doubles when a run gets there: a context of 4096
# tokens is 200 MB of cache for llm-jp-3-150m, which a short text should not have to pay for (and WebAssembly
# never gives memory back)
KV_START = 256


# what T52 can leave out, each of them something that already has a fallback
SWITCHES = ("kernels", "int8", "relaxed", "sampler", "kv16")


# T108: how many tokens of a prompt forward_many() takes at once: forward.js's BATCH. The worker cannot answer a
# message (stop, a new model) while one call runs, and a block of 16 keeps that under a second on a 1.5B model.
# T147: forward.js says how many it takes (promptBlock: more where the GPU takes the prompt)
PROMPT_BLOCK = 16


class Llama(TensorOrder, Sampling, Generation):
    forward_many = None  # T108: forward.js's forwardMany(tokens, pos) for a prompt, where there is one
    prompt_block = staticmethod(lambda: PROMPT_BLOCK)  # T147: how many tokens forward_many() takes at once, now
    # T152: forward.js's generateMany(token, pos, history, count, temperature, topp, penalty, randoms, stops), which
    # runs count steps of generate() on the GPU (the forward pass and the sampling, with the random numbers drawn here)
    # and returns the tokens it sampled (a stop token last), or None where it did not (the CPU then takes the step);
    # token_block(): how many steps it takes at once now, 0 where the CPU is faster (or there is no GPU)
    generate_many = None
    token_block = staticmethod(lambda: 0)

    def __init__(self, checkpoint, tokenizer, dtype="float32", rope_theta=10000.0,
                 tokenizer_kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", bias=False, arch="llama",
                 rotary=0, parallel_residual=False, bos=BOS, stop_tokens=(BOS,), kernels=None, specials=(),
                 disable=(), external=None, rope_scaling=None, ignore_merges=False, collapse=False,
                 unknown=None, qk_norm=False, head_dim=0, rms_norm_eps=RMS_EPS, linear=None, rotated=None,
                 convolution=None, unturned=()):
        """checkpoint: llama2.c "legacy" format, a 7 int header then the weights.

        dtype="float16" and dtype="int8" are this project's smaller variants (convert_hf.py, quantize.py),
        dtype="int6" (T98) is int8 with six bits a value (pack6), and dtype="ternary" (T230) two bits a value with a
        scale per group of 128 (pack_ternary): the weights of a ternary model as they are.
        arch="neox": GPT-NeoX, which is arch="gpt2" with RoPE over the first rotary values of every head
        (rotary=0 means all of them) and, when parallel_residual is on, the attention and the FFN both reading
        the same x instead of one after the other.
        arch="qwen35" (T229): Qwen3.5's hybrid attention, see the comment on it in engine/layers.py: a Qwen3 of whose layers
        all but every linear["every"]-th are Gated DeltaNet layers with a state in place of keys and values, and whose
        full-attention layers gate their output. linear: the numbers of those layers (FORM's, the file cannot say
        them); rotary and head_dim as below. The state follows the positions: a run begins at position 0, which
        clears it, and goes on one position after the other (forward() refuses any other).
        arch="lfm2" (T260): Liquid AI's LFM2, see the comment on it in engine/layers.py: a Qwen3 some of whose layers
        are convolution layers, which keep the last taps - 1 tokens' values in place of keys and values.
        convolution: which layers those are and their taps (FORM's, the file cannot say them). The state follows the
        positions as a Qwen3.5's does.
        arch="gpt2": LayerNorm instead of RMSNorm, GELU instead of SwiGLU (and no gate matrix), a learned table
        of positions instead of RoPE, and a bias after every projection. The tensors of the file differ with it,
        so it is engine/layout.py that says what is there.
        bias=True: the checkpoint ends with a bias for q, k and v of every layer, which is added after those
        projections (Qwen2). The legacy header cannot say so, so the caller does, like the tokenizer settings.
        qk_norm=True (T124): after them come the RMSNorm weights of q and k (one head's size each, per layer), and
        every head of q and k is normalized with them before RoPE (Qwen3). The caller says so, like bias.
        head_dim: the size of a head where it is not dim / n_heads (T124: Qwen3 0.6B has 16 heads of 128 in a dim of
        1024): q and the attention's output are then n_heads * head_dim wide. rms_norm_eps: config.json's, the epsilon
        of every RMSNorm (the kernels take it too).
        rotated (T237): the matrices are in a rotated basis (the comment above hadamard()): its block and the signs
        of every width (FORM's, the file cannot say them). The same tensors in the same places: only what they are
        multiplied by changes, and the row of the embedding is turned back.
        unturned (T255): the layers whose q and k RoPE does not turn (SmolLM3: every fourth), where there are any.
        rope_scaling: config.json's, for the RoPE tables that are not in the file (int8, float16); see
        rope_frequencies(). ignore_merges: a byte-level BPE takes a pre-tokenized piece that is in the vocabulary
        whole (Llama 3).
        bos starts every sequence; generation ends when the model emits one of stop_tokens.
        kernels is the path of simdkernel.so: the sampling runs on it (penalize, sample). The forward pass on the
        kernels is public/forward.js's (external, below); without external, NumPy computes the forward pass.
        disable: the optimizations to leave out, to measure what each one is worth (T52). Only what already has
        a fallback: "kernels" (NumPy does everything), "int8" (the weights are widened to float32 and the
        float32 kernel multiplies them), "relaxed" (matmul_q8 instead of matmul_q8r), "sampler" (NumPy
        samples) and "kv16" (an int8 model keeps its keys and values in float32 on a shared memory too, T110). Anything else is refused, so that a typo never quietly measures the wrong thing.
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
        self.head_size = int(head_dim) or self.dim // self.n_heads
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        self.q_dim, kv_dim = self.n_heads * self.head_size, self.n_kv_heads * self.head_size

        disable = tuple(str(name) for name in disable)
        unnamed = [name for name in disable if name not in SWITCHES]
        if unnamed:
            raise ValueError(f"There is no optimization called {unnamed[0]!r}: {', '.join(SWITCHES)}.")
        self.disabled = disable
        # int6 (T98) and ternary (T230) are int8 with the values packed: from here on it is int8, except where the
        # bytes are read
        packing = str(dtype) if str(dtype) in PACKED else None
        dtype = np.dtype(np.int8 if packing else dtype)
        # The int8 kernels work on groups of 32 only
        self.linear = linear_form(linear)
        # (T229: and a linear-attention layer's output matrix, whose rows are as long as its value heads together)
        suitable = dtype != np.int8 or (dim % 32 == 0 and self.q_dim % 32 == 0 and kv_dim % 32 == 0 and hidden_dim % 32 == 0
                                        and (self.linear is None or linear_widths(self.linear)[2] % 32 == 0))
        kernels = load_kernels(kernels, "relaxed" in disable) if kernels and "kernels" not in disable and \
            (suitable or external is not None) else None
        # int8 kernels compute on the int8 weights directly: they are never widened, a quarter of the memory
        # (only forward.js computes on them: the NumPy forward widens every matrix)
        keep_int8 = external is not None and suitable and dtype == np.int8 and "int8" not in disable

        self.arch, self.parallel_residual = arch, parallel_residual
        # T237: a rotated basis turns what every matrix reads (turned), and the embedding's row back
        self.rotated = rotated_form(rotated, rotated_widths(dim, self.q_dim, hidden_dim, self.linear))
        if self.rotated is not None and arch in ("gpt2", "neox", "lfm2"):
            raise ValueError("A rotated basis is a Llama's, a Qwen's or a Qwen3.5's: no GPT-2, GPT-NeoX or LFM2 has one.")
        block = self.rotated and self.rotated["block"]
        self.turned = (lambda v: v) if self.rotated is None else (lambda v: rotate(v, self.rotated["signs"][v.size], block))
        self.rms_norm_eps = float(rms_norm_eps)
        # how many values of each head RoPE turns: all of them unless the model says otherwise
        self.rotary = int(rotary) if rotary else self.head_size
        # T255: the layers whose q and k go into the scores as their matrices (and norms) leave them
        unturned = unturned.to_py() if hasattr(unturned, "to_py") else unturned
        self.unturned = tuple(sorted({int(layer) for layer in unturned or ()}))
        if self.unturned and (arch != "llama" or not 0 <= self.unturned[0] <= self.unturned[-1] < self.n_layers):
            raise ValueError("The layers RoPE leaves alone are layers of a Llama, and none of another architecture.")
        self.positions = None
        self.q_norm = self.k_norm = self.wg = None
        if (arch == "qwen35") != (self.linear is not None) or (self.linear and n_layers < self.linear["every"]):
            raise ValueError("A hybrid model (qwen35) and the numbers of its linear layers go together.")
        # T260: an LFM2's convolution layers
        self.convolution = convolution_form(convolution, n_layers)
        if (arch == "lfm2") != (self.convolution is not None):
            raise ValueError("An LFM2 (lfm2) and its convolution layers go together.")
        # what the file holds (engine/layout.py): its rows, and for every layer (does it keep a state: a linear-attention
        # layer or a convolution one, its place in the stacks of its kind's tensors)
        dims = Dims((dim, hidden_dim, n_layers, self.n_heads, self.n_kv_heads, vocab_size, self.seq_len),
                    {"arch": arch, "bias": bias, "qk_norm": qk_norm, "head_dim": head_dim, "linear": self.linear,
                     "convolution": self.convolution})
        self.slots, self.rows = dims.slots, dims.rows()
        self.ln_att_bias = self.ln_ffn_bias = self.ln_final_bias = None
        self.bq = self.bk = self.bv = self.bo = self.b1 = self.b2 = None
        self.w3 = None
        # a dict from Python, or a JavaScript object from the worker
        rope_scaling = rope_scaling.to_py() if hasattr(rope_scaling, "to_py") else rope_scaling
        frequencies = lambda width: rope_frequencies(width, rope_theta, rope_scaling)
        self.rope_magnitude = rope_magnitude(rope_scaling)
        # each row as an attribute of its name: where it is, with the weights outside Python (public/forward.js reads
        # them, and widens what has to be widened, itself), or an array
        stored = packing or dtype.name
        self.file_tensors(checkpoint, self.rows, stored, shared_weights, external is not None)
        self.rope_tables(stored, frequencies)
        self.backend = "NumPy"
        if external is not None:
            if file_size(self.rows, stored) != int(external.size):
                raise ValueError(f"The checkpoint has {int(external.size)} bytes, and its header asks for "
                                 f"{file_size(self.rows, stored)} as {dtype.name}.")
            self.forward = self.external_forward(external, keep_int8, disable)
            if kernels and "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        else:
            # NumPy's forward pass; the forward pass on the kernels is forward.js's (external), since T93
            # (the linear-attention layers and the convolution layers have no keys and values)
            attending = sum(not lines for lines, _ in self.slots)
            self.key_cache = np.zeros((attending, self.n_kv_heads, min(KV_START, self.seq_len), self.head_size), dtype=np.float32)
            self.value_cache = np.zeros_like(self.key_cache)
            if self.linear is not None:
                # their state, of a size the context does not change: a matrix for every value head, and the last
                # conv - 1 tokens' q, k and v before the convolution (the oldest first). state_at: the next position
                lines, mixed = n_layers - attending, linear_widths(self.linear)[0]
                self.delta_state = np.zeros((lines, self.linear["value_heads"], self.linear["key_dim"],
                                             self.linear["value_dim"]), dtype=np.float32)
                self.conv_state = np.zeros((lines, self.linear["conv"] - 1, mixed), dtype=np.float32)
                self.state_at = 0
            if self.convolution is not None:
                # T260: theirs is the last taps - 1 tokens' values before the convolution alone (the oldest first)
                self.conv_state = np.zeros((n_layers - attending, self.convolution["taps"] - 1, dim), dtype=np.float32)
                self.state_at = 0
            if kernels and "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        if (kernels or external is not None) and "sampler" in disable:
            self.backend += ", NumPy sampling"
        if disable:
            # the line has to say what the numbers are the numbers of
            self.backend += " (without " + ", ".join(name for name in SWITCHES if name in disable) + ")"
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, kind=tokenizer_kind, nfkc=nfkc, nfc=nfc,
                                   pretokenizer=pretokenizer, ignore_merges=ignore_merges, collapse=collapse,
                                   unknown=unknown)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.specials = tuple(str(special) for special in specials)  # see Tokenizer.encode()
        self.stats = {}
        self._run = 0

    def embedding(self, token):
        if isinstance(self.token_embedding_table, tuple):
            values, scales = self.token_embedding_table
            return (values[token] * scales[token]).reshape(self.dim)
        return self.token_embedding_table[token].astype(np.float32)

    def external_forward(self, external, int8, disable):
        """forward() in public/forward.js (T93): Python hands over where every tensor is, and the few small
        arrays it computes itself (the RoPE tables of a checkpoint that leaves them out, the outlier channels of
        T92), and gets the logits back into one array of its own, which the sampling kernels then read."""
        # (the classifier of a model that has no other is its embedding, under both names)
        held = {name: getattr(self, name) for name in (*(row.name for row in self.rows), "wcls")}
        tensors = {name: tensor.plan() for name, tensor in held.items() if isinstance(tensor, Tensor)}
        derived = {name: np.ascontiguousarray(getattr(self, name), dtype=np.float32).tobytes()
                   for name in ("freq_cis_real", "freq_cis_imag") if isinstance(getattr(self, name), np.ndarray)}
        if self.rotated is not None:
            # T237: the signs of every width, with the transform's 1 / sqrt(block) in them (what the kernel multiplies by)
            scale = np.float32(1.0 / math.sqrt(self.rotated["block"]))
            derived.update({f"signs.{width}": (signs * scale).tobytes() for width, signs in self.rotated["signs"].items()})
        channels = []
        # (T237: not in a rotated basis, where the classifier reads R of its input: the rotation spreads a channel
        # over its block, and a column of the stored matrix is no channel's)
        if int8 and self.rotated is None:
            final = self.rms_final_weight
            raw = external.read(final.offset, self.dim * 4)
            weight = np.frombuffer(bytes(raw.to_py() if hasattr(raw, "to_py") else raw), dtype=np.float32)
            channels = [int(c) for c in outlier_channels(weight, min(OUTLIER_CHANNELS, self.dim))]
        plan = {"arch": self.arch, "dim": self.dim, "hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
                "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads, "head_size": self.head_size,
                "vocab_size": self.vocab_size, "seq_len": self.seq_len, "rotary": self.rotary,
                "parallel_residual": bool(self.parallel_residual), "kv_start": KV_START, "rms_norm_eps": self.rms_norm_eps,
                "shared_classifier": self.wcls is self.token_embedding_table, "int8": bool(int8),
                "relaxed": "relaxed" not in disable, "tensors": tensors, "derived": derived, "outliers": channels,
                # T229: a Qwen3.5's linear-attention layers (None: none)
                "linear": self.linear,
                # T255: the layers RoPE leaves alone
                "unturned": list(self.unturned),
                # T260: an LFM2's convolution layers (None: none)
                "convolution": self.convolution,
                # T237: the block of a rotated basis (0: the model's own basis); its signs are in derived
                "rotated": self.rotated["block"] if self.rotated else 0,
                # T110: the keys and values of an int8 model may be float16 (forward.js uses that on a shared memory)
                "half_kv": bool(int8) and "kv16" not in disable}
        engine = external.start(plan)
        self.backend = str(engine.backend)
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        engine.bind(logits)
        self._external = (engine, logits)  # keep both alive: JS writes into the array
        run = engine.forward
        # T108: a prompt's tokens go through the layers several at a time, when forward.js offers that
        many = getattr(engine, "forwardMany", None)
        if many is not None:
            self.forward_many = lambda tokens, pos: many(list(tokens), pos)
            if getattr(engine, "promptBlock", None) is not None:
                self.prompt_block = lambda: int(engine.promptBlock)
        # T152: the steps of generate() on the GPU, where forward.js offers that
        on_gpu = getattr(engine, "generateMany", None)
        if on_gpu is not None:
            def generate_many(token, pos, history, count, temperature, topp, penalty, randoms, stops):
                ids = on_gpu(token, pos, list(history[-REPETITION_WINDOW:]), len(history), count, temperature, topp,
                            penalty, list(randoms), list(stops))
                return None if ids is None else [int(i) for i in ids]

            self.generate_many = generate_many
            self.token_block = lambda: int(engine.tokenBlock)

        def forward(token, pos, need_logits=True):
            run(token, pos, need_logits)
            return logits if need_logits else None

        return forward

    def release(self):
        """Let go of what JavaScript holds for this engine (the forward pass of forward.js and the array it fills).
        The worker calls it before it drops a model (T93). T205: what forward.js answers (a promise, settled once the
        GPU's worker let go of the device), for the worker to wait on before it reads the next model; else None."""
        external = getattr(self, "_external", None)
        if external is None:
            return None
        self._external = None
        return external[0].release()

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
        if self.linear is not None or self.convolution is not None:
            self.follow(pos)
        # GPT-2 and GPT-NeoX normalize by the mean as well, and have a bias on every projection
        eps = self.rms_norm_eps
        norm = (lambda v, w, b: layernorm(v, w, b)) if layer_norm else (lambda v, w, b: rmsnorm(v, w, eps))
        heads_of = lambda v, c, s: v.reshape(-1, head_size)  # (q or k as they are)
        if gpt2:
            turn = heads_of
        elif neox or self.linear is not None:
            # only the first self.rotary of every head are rotated, the rest go through untouched
            turn = lambda v, c, s: partial_rope(v.reshape(-1, head_size), c, s, self.rotary)
        else:
            turn = rope
        # T255: a layer RoPE leaves alone, where the model has such
        turns = [heads_of if l in self.unturned else turn for l in range(self.n_layers)] if self.unturned else None

        # Copy the token embedding into x, and (GPT-2) the row of this position
        x = self.embedding(token)
        if gpt2:
            x = x + self.positions[pos]
        turned = self.turned
        if self.rotated is not None:  # T237: the table holds rotated rows
            x = unrotate(x, self.rotated["signs"][self.dim], self.rotated["block"])

        # Forward all the layers
        for l, (lines, a) in enumerate(self.slots):
            xb = norm(x, self.rms_att_weight[l], self.ln_att_bias[l] if layer_norm else None)
            if lines:  # T229: a linear-attention layer, the a-th of them; T260: or an LFM2's convolution layer
                attended = self.linear_attention(a, xb) if self.linear is not None else self.short_convolution(a, xb)
            else:
                # QKV matmuls for this position, RoPE on q and k, k and v go to the kv cache (a: the layer's place
                # among the attending layers, which is l where all of them attend)
                # (xr: what the matrices read of xb, itself but in a rotated basis, T237; turned once for all of them)
                xr = turned(xb)
                qv, kv, vv = self.wq[a] @ xr, self.wk[a] @ xr, self.wv[a] @ xr
                if self.bq is not None:  # Qwen2 and GPT-2 add a bias to q, k and v
                    qv, kv, vv = qv + self.bq[a], kv + self.bk[a], vv + self.bv[a]
                if self.q_norm is not None:  # Qwen3 normalizes every head of q and k
                    qv, kv = head_norm(qv, self.q_norm[a], eps), head_norm(kv, self.k_norm[a], eps)
                if turns:
                    turn = turns[l]
                q = turn(qv, cos, sin).reshape(n_kv_heads, kv_mul, head_size)
                self.key_cache[a, :, pos] = turn(kv, cos, sin)
                self.value_cache[a, :, pos] = vv.reshape(n_kv_heads, head_size)

                # Multihead attention over all timesteps so far, all heads at once
                keys = self.key_cache[a, :, :pos + 1]      # (n_kv_heads, pos + 1, head_size)
                values = self.value_cache[a, :, :pos + 1]
                att = (q @ keys.transpose(0, 2, 1)) * scale  # (n_kv_heads, kv_mul, pos + 1)
                att = np.exp(att - att.max(axis=-1, keepdims=True))
                att /= att.sum(axis=-1, keepdims=True)
                attended = (att @ values).reshape(self.q_dim)
                if self.wg is not None:  # Qwen3.5 gates what the attention read
                    attended = attended / (1.0 + np.exp(-(self.wg[a] @ xr)))
                # Output projection and residual connection
                attended = self.wo[a] @ turned(attended)
                if layer_norm:
                    attended = attended + self.bo[a]
            # GPT-NeoX with use_parallel_residual: both branches read the x this layer began with
            before = x
            x = x + attended

            # FFN: w2(silu(w1(x)) * w3(x)), or GPT-2's w2(gelu(w1(x))), and residual connection
            xb = norm(before if self.parallel_residual else x, self.rms_ffn_weight[l],
                      self.ln_ffn_bias[l] if layer_norm else None)
            xr = turned(xb)
            hb = self.w1[l] @ xr
            if layer_norm:
                x = x + self.w2[l] @ gelu(hb + self.b1[l]) + self.b2[l]
            else:
                hb = hb / (1.0 + np.exp(-hb)) * (self.w3[l] @ xr)
                x = x + self.w2[l] @ turned(hb)

        if not need_logits:
            return None
        # Final norm, then the classifier into logits (60% of all the multiply-adds of stories15M)
        return self.wcls @ turned(norm(x, self.rms_final_weight, self.ln_final_bias))

    def follow(self, pos):
        """T229: the linear-attention layers' state is what the tokens before this position left, so position 0 clears
        it and every other position has to be the one after the last. Keys and values could be written again at any
        position; a state cannot, and a token out of turn would compute on the wrong one without a word. T260: an
        LFM2's convolution layers' state the same."""
        if pos == 0:
            if self.linear is not None:
                self.delta_state.fill(0.0)
            self.conv_state.fill(0.0)
        elif pos != self.state_at:
            raise ValueError(f"This model keeps a state from token to token: position {self.state_at} comes next "
                             f"(or 0, to begin again), not {pos}.")
        self.state_at = pos + 1

    def linear_attention(self, a, xb):
        """One token through the a-th Gated DeltaNet layer (the comment on Qwen3.5 in engine/layers.py has the rule): what the
        layer adds to x, with the layer's state moved on by this token."""
        linear, eps = self.linear, np.float32(self.rms_norm_eps)
        key_heads, value_heads, key_dim = linear["key_heads"], linear["value_heads"], linear["key_dim"]
        _, keys, _ = linear_widths(linear)
        xr = self.turned(xb)  # T237: what the two large matrices read (the gates' small ones read xb itself)
        convolved = silu(self.convolved(a, self.wqkv[a] @ xr))
        # every value head reads the key head it belongs to
        q = np.repeat(l2_heads(convolved[:keys], key_heads) * np.float32(1.0 / math.sqrt(key_dim)), value_heads // key_heads, axis=0)
        k = np.repeat(l2_heads(convolved[keys:2 * keys], key_heads), value_heads // key_heads, axis=0)
        v = convolved[2 * keys:].reshape(value_heads, -1)
        beta = 1.0 / (1.0 + np.exp(-(self.wb[a] @ xb)))
        decay = np.exp(self.decay[a] * softplus(self.wa[a] @ xb + self.dt_bias[a]))
        read = delta_rule(self.delta_state[a], q, k, v, beta.astype(np.float32), decay.astype(np.float32))
        read = self.delta_norm[a] * read / np.sqrt((read * read).mean(axis=1, keepdims=True) + eps)
        return self.wout[a] @ self.turned((read * silu((self.wz[a] @ xr).reshape(value_heads, -1))).reshape(-1))

    def convolved(self, a, values):
        """The causal convolution of the a-th layer that has one (a linear-attention layer's, an LFM2's convolution
        layer's): each channel of values with its own taps over this token and the taps - 1 before it, the oldest tap
        first, and the layer's state moved on by this token."""
        taps, before = self.conv[a], self.conv_state[a]
        out = (taps[:-1] * before).sum(axis=0) + taps[-1] * values
        before[:-1] = before[1:]
        before[-1] = values
        return out

    def short_convolution(self, a, xb):
        """One token through the a-th convolution layer of an LFM2 (the comment on LFM2 in engine/layers.py has the rule):
        what the layer adds to x, with the layer's state moved on by this token."""
        dim = self.dim
        mixed = self.win[a] @ xb  # B, C and what B multiplies, dim values each
        return self.wout[a] @ (mixed[dim:2 * dim] * self.convolved(a, mixed[:dim] * mixed[2 * dim:]))


