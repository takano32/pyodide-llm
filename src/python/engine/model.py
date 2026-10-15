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
from engine.layout import (ATTENDING, DOUBLE, GELU, NARROW, NARROW_KEPT, PARTLY, PROPORTIONAL, WIDE, WIDE_KEPT, Dims, file_size, layout_of, linear_form, linear_widths, local_layers,
                           stateful_kinds, unturned_layers)
from engine.layers import (RMS_EPS, delta_rule, gelu, head_norm, l2_heads, layernorm, partial_rope, rmsnorm, rope,
                           rope_frequencies, rope_magnitude, rotate, rotated_form, rotated_widths, silu, softplus,
                           unrotate)
from engine.kernels import load_kernels
from engine.dtypes import QUANTIZED, dtype_of
from engine.tensors import local_tables, read_rows, rope_tables
from engine.sampler import KernelSampler, NumpySampler, greedy
from engine.external import STEPS, ExternalForward
from engine.plan import Settings, forward_plan
from engine import generation

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


# The NumPy forward pass's part of each kind of layer that keeps a state (engine/layout.py's Stateful, by its name):
# the method of Llama that takes a token through one such layer, (a, xb) -> what the layer adds to x, and its states,
# (the kind's numbers, how many such layers, dim) -> {the attribute: the array, zero before the first token}. A model
# has them under those names, and follow() clears them at position 0.
def linear_states(linear, layers, dim):
    """A matrix for every value head, and the last conv - 1 tokens' q, k and v before the convolution (the oldest
    first): of a size the context does not change."""
    return {"delta_state": np.zeros((layers, linear["value_heads"], linear["key_dim"], linear["value_dim"]), dtype=np.float32),
            "conv_state": np.zeros((layers, linear["conv"] - 1, linear_widths(linear)[0]), dtype=np.float32)}


def convolution_states(convolution, layers, dim):
    """T260: the last taps - 1 tokens' values before the convolution alone (the oldest first)."""
    return {"conv_state": np.zeros((layers, convolution["taps"] - 1, dim), dtype=np.float32)}


STATEFUL_LAYERS = {"linear": ("linear_attention", linear_states), "convolution": ("short_convolution", convolution_states)}


class Llama:
    """A model: what its checkpoint holds, each row an attribute of its name, the tokenizer, and the parts it writes
    with. The forward pass is one of two: the reference in NumPy, which is this class's own (forward() and the
    methods after it, over the rows as arrays and the keys, values and states kept here), or public/forward.js's
    (engine/external.py, with external: what the page runs). The sampler is one of two as well (engine/sampler.py):
    NumPy's or the kernels'. generate() (engine/generation.py) writes with whichever the model has.

    The names of the forward pass's steps are attributes here, and that is how it is chosen: the class has the
    reference's (forward, and no blocks or GPU steps), and an instance whose weights are outside has that part's own
    functions under the same names, set once when it is made. The sampler is one attribute, sampler: the class's is
    the reference, an instance with the kernels has its own. A step of generate() then calls the part itself, with no
    call in between. Tests and tools put their own in these places, on a model or on the class.
    """
    # how a model writes (engine/generation.py)
    generate = generation.generate
    # the reference sampler, NumPy's (it keeps nothing, so one is every model's): an instance with the kernels has a
    # KernelSampler in its place. A generation asks it once how it draws (drawing(), by the settings: a Sampling)
    sampler = NumpySampler()
    greedy = staticmethod(greedy)
    external_forward = None  # the ExternalForward of a model whose weights are outside Python
    _external = None  # (its engine and the array it fills, until release(): tools read the engine off it)
    forward_many = None  # T108: forward.js's forwardMany(tokens, pos) for a prompt, where there is one
    prompt_block = staticmethod(lambda: PROMPT_BLOCK)  # T147: how many tokens forward_many() takes at once, now
    # T152: gpu_steps(sampling, rng) is, for one generation, what runs count steps of it on forward.js's generateMany()
    # (the forward pass and the sampling on the GPU, with the random numbers drawn here) and returns the tokens it
    # sampled (a stop token last), or None where it did not (the CPU then takes the step); None itself where the GPU's
    # sampler has not all the settings. token_block(): how many steps it takes at once now, 0 where the CPU is
    # faster (or there is no GPU)
    gpu_steps = None
    token_block = staticmethod(lambda: 0)
    tiled = False  # T363: FORM's "tiled", of a model made with it

    def __init__(self, checkpoint, tokenizer, dtype="float32", rope_theta=10000.0,
                 tokenizer_kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", bias=False, arch="llama",
                 rotary=0, parallel_residual=False, bos=BOS, stop_tokens=(BOS,), kernels=None, specials=(),
                 disable=(), external=None, rope_scaling=None, ignore_merges=False, collapse=False,
                 unknown=None, qk_norm=False, head_dim=0, rms_norm_eps=RMS_EPS, linear=None, rotated=None,
                 convolution=None, unturned=(), kinds=None, lowercase=False, prefixed=False, tiled=None,
                 local=None, wide=None, double=None, softcap=0.0, kv_shared=None, per_layer=0):
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
        arch="gemma3" (T261): Google's Gemma 3, see the comment on it in engine/layers.py: a Qwen3 that normalizes
        what its attention and its FFN give as well, whose FFN's activation is a GELU, and most of whose layers
        are local ones. local: which those are, the window they attend and the base RoPE turns them by
        (engine/layout.py's local_layers(); the file is the same whichever they are).
        arch="gemma4" (T331): Google's Gemma 4, a Gemma 3 whose norms are of the weight itself (not 1 + it), that
        normalizes each head of v too (without a weight), whose layers end by multiplying the stream by a number of
        their own (layer_scale), and whose full-attention layers turn the first rotary values of a head by the angles
        of its whole width (proportional RoPE). wide: the layers whose heads are of a size of their own (FORM's: the
        full-attention layers' 512), double: those whose FFN is twice as wide (FORM's), softcap: the logits are
        softcap * tanh(logits / softcap) (0: as they are). kv_shared (T330): the layers that read the keys and values of
        a layer before them in place of their own, and which each reads (FORM's). per_layer (T329): the width of the
        input every layer takes of a table of its own, and of a projection of the token's row (FORM's; 0: none).
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
        kinds (T359): the rows the file holds in a kind of their own, {the row's name: "int6", ...} (FORM's, the file
        cannot say them): every other row is stored the way dtype stores a row of its role. T363: a list gives each
        layer of a stack one, and llama.cpp's blocks (engine/dtypes.py's BLOCKS) are kinds: NumPy reads them by the
        converter's readers, and forward.js (external) multiplies by them as they are (T363.5).
        tiled (T363): wout's columns are in llama.cpp's order of the value heads (FORM's), as a GGUF's blocks have them.
        rope_scaling: config.json's, for the RoPE tables that are not in the file (int8, float16); see
        rope_frequencies(). ignore_merges: a byte-level BPE takes a pre-tokenized piece that is in the vocabulary
        whole (Llama 3). lowercase (T265) and prefixed (T308): the tokenizer's, see Tokenizer.
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
        # how the text is read: the tokenizer's own settings (the tokenizer is made last, below)
        reading = dict(kind=tokenizer_kind, nfkc=nfkc, nfc=nfc, pretokenizer=pretokenizer, ignore_merges=ignore_merges,
                       collapse=collapse, unknown=unknown, lowercase=lowercase, prefixed=prefixed)
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
        # the file's dtype by its name (engine/dtypes.py). int6 (T98) and ternary (T230) are int8 with the values
        # packed: past where the bytes are read, a quantized file is int8
        stored = dtype_of(dtype)
        quantized = stored in QUANTIZED
        # the architecture's facts (engine/layout.py's Layout), which is all of the architecture that is read below
        layout = layout_of(arch)
        # The int8 kernels work on groups of 32 only
        # (the numbers of linear-attention layers, read here for what is sized by them below: the Dims, further down,
        # holds them to the architecture)
        self.linear = linear_form(linear)
        # (T229: and a linear-attention layer's output matrix, whose rows are as long as its value heads together)
        suitable = not quantized or (dim % 32 == 0 and self.q_dim % 32 == 0 and kv_dim % 32 == 0 and hidden_dim % 32 == 0
                                        and (self.linear is None or linear_widths(self.linear)[2] % 32 == 0))
        kernels = load_kernels(kernels, "relaxed" in disable) if kernels and "kernels" not in disable and \
            (suitable or external is not None) else None
        # int8 kernels compute on the int8 weights directly: they are never widened, a quarter of the memory
        # (only forward.js computes on them: the NumPy forward widens every matrix)
        keep_int8 = external is not None and suitable and quantized and "int8" not in disable

        self.arch, self.parallel_residual = arch, parallel_residual
        # T237: a rotated basis turns what every matrix reads (turned), and the embedding's row back
        self.rotated = rotated_form(rotated, rotated_widths(dim, self.q_dim, hidden_dim, self.linear))
        if self.rotated is not None and not layout.rotatable:
            raise ValueError("A rotated basis is a Llama's, a Qwen's or a Qwen3.5's: no GPT-2, GPT-NeoX or LFM2 has one.")
        # (the signs and the block themselves, not this model: a function that held the model would be a cycle of
        # references, and the weights would then stay until a collection)
        signs, block = (self.rotated["signs"], self.rotated["block"]) if self.rotated else (None, None)
        self.turned = (lambda v: v) if self.rotated is None else (lambda v: rotate(v, signs[v.size], block))
        self.rms_norm_eps = float(rms_norm_eps)
        # how many values of each head RoPE turns: all of them unless the model says otherwise
        self.rotary = int(rotary) if rotary else self.head_size
        # T255: the layers whose q and k go into the scores as their matrices (and norms) leave them
        self.unturned = unturned_layers(arch, n_layers, unturned)
        # T261: the layers that attend a window of the last positions and have RoPE tables of their own (None: none)
        self.local = local_layers(arch, n_layers, local)
        # what the file cannot say of itself (engine/layout.py's FORM; the rotated basis moves no tensor), and with it
        # for every layer (does it keep a state: a linear-attention layer or a convolution one, its place in the stacks
        # of its kind's tensors). The form is held to its architecture there: one without the layers its architecture
        # has, or with another's, is refused
        form = {"arch": arch, "bias": bias, "qk_norm": qk_norm, "head_dim": head_dim, "linear": linear,
                "convolution": convolution, "kinds": kinds, "wide": wide, "double": double, "kv_shared": kv_shared,
                "per_layer": per_layer}
        dims = Dims((dim, hidden_dim, n_layers, self.n_heads, self.n_kv_heads, vocab_size, self.seq_len), form)
        self.slots = dims.slots
        # T331: for every layer that attends, by its place among them: the names of its attention's tensors are those
        # of its width (WIDE's "_wide"), its place in their stacks, and the size of its heads; T330: its place in the
        # stacks of k and v (None: it reads another's keys and values) and the place among the attending layers whose
        # keys and values it reads. And for every layer, the name of its FFN's (DOUBLE's "_double") and its place in
        # theirs. Every layer of a model without such is ("", its place, head_size, its place, its place) and ("", l)
        self.wide, self.double, self.shared = dims.wide, dims.double, dims.sharing
        attending, sources = dims.layers(ATTENDING), dict(zip(*dims.sharing.values())) if dims.sharing else {}
        stacks = {per: dims.layers(per) for per in (NARROW, WIDE, NARROW_KEPT, WIDE_KEPT)}
        self.heads = []
        for l in attending:
            named, per, kept, size = ("_wide", WIDE, WIDE_KEPT, dims.wide_head) if l in stacks[WIDE] else ("", NARROW, NARROW_KEPT, self.head_size)
            self.heads.append((named, stacks[per].index(l), size, stacks[kept].index(l) if l in stacks[kept] else None,
                               attending.index(sources.get(l, l))))
        doubled = dims.layers(DOUBLE)
        self.ffns = [("_double", doubled.index(l)) if l in doubled else ("", l - sum(d < l for d in doubled))
                     for l in range(n_layers)]
        self.softcap = float(softcap or 0.0)
        # the numbers of its layers that keep a state, under their kind's name: linear (T229, a Qwen3.5's
        # linear-attention layers), convolution (T260, an LFM2's convolution layers); None, of a model without
        for kind in stateful_kinds():
            setattr(self, kind, getattr(dims, kind))
        # what of the forward pass differs by the architecture, as the reference reads it at every token: the kind of
        # its norms and of its FFN, and whether RoPE turns the heads of q and k whole, in part or not at all
        self.layer_norm, self.gated_ffn = layout.layer_norm, layout.gated_ffn
        # (the activation of the FFN: a SiLU, or a GELU)
        self.activation = gelu if layout.activation == GELU else silu
        self.turning, self.partly = layout.rope is not None, layout.rope in (PARTLY, PROPORTIONAL)
        # T331: what the attention reads of v is normalized head by head, without a weight
        self.value_norm = layout.value_norm
        # (and of its layers that keep a state: the name of the reference's step that is theirs, and of their states;
        # none, unless NumPy computes such layers, below)
        self.stateful, self.states = None, ()
        self.positions = None
        self.q_norm = self.k_norm = self.wg = None
        self.q_norm_wide = self.k_norm_wide = None  # (T331: of the wide layers)
        self.rms_att_post_weight = self.rms_ffn_post_weight = None
        self.layer_scale = None  # T331
        self.per_layer_table = self.ple_gate = None  # T329
        self.per_layer = dims.per_layer
        # what the file holds: its rows
        self.rows = dims.rows()
        if tiled:
            self.tiled = True
        self.ln_att_bias = self.ln_ffn_bias = self.ln_final_bias = None
        self.bq = self.bk = self.bv = self.bo = self.b1 = self.b2 = None
        self.w3 = None
        # a dict from Python, or a JavaScript object from the worker
        rope_scaling = rope_scaling.to_py() if hasattr(rope_scaling, "to_py") else rope_scaling
        frequencies = lambda width: rope_frequencies(width, rope_theta, rope_scaling)
        self.rope_magnitude = rope_magnitude(rope_scaling)
        # each row as an attribute of its name: where it is, with the weights outside Python (public/forward.js reads
        # them, and widens what has to be widened, itself), or an array
        self.file_tensors(checkpoint, self.rows, stored, shared_weights, external is not None)
        # (T331: the tables of the layers that attend over all the positions, whose heads are the wide ones where the
        # model has such)
        tables = rope_tables(arch, self.seq_len, dims.wide_head or self.head_size, self.rotary, self.rope_magnitude,
                             stored, frequencies)
        if tables is not None:
            self.freq_cis_real, self.freq_cis_imag = tables
        # T261: and those of the local layers, which no file has
        self.local_cis = local_tables(self.seq_len, self.head_size, self.local["rope_theta"], rope_frequencies) \
            if self.local else None
        self.backend = "NumPy"
        if external is not None:
            if file_size(self.rows, stored) != int(external.size):
                raise ValueError(f"The checkpoint has {int(external.size)} bytes, and its header asks for "
                                 f"{file_size(self.rows, stored)} as {stored}.")
            # forward.js's forward pass, and what else its engine offers, in the reference's places
            # (the plan is made here and handed over: nothing in Python keeps it)
            outside = self.external_forward = ExternalForward(forward_plan(dims, stored, external.read, Settings(
                int8=keep_int8, disable=disable, kv_start=KV_START, rotary=self.rotary,
                parallel_residual=parallel_residual, rms_norm_eps=self.rms_norm_eps, unturned=self.unturned,
                rotated=self.rotated, tables=tables, tiled=self.tiled, local=self.local,
                local_tables=self.local_cis)), external)
            self.backend, self._external = outside.backend, (outside.engine, outside.logits)
            for name in STEPS:
                if hasattr(outside, name):
                    setattr(self, name, getattr(outside, name))
        else:
            # NumPy's forward pass; the forward pass on the kernels is forward.js's (external), since T93
            # (the linear-attention layers and the convolution layers have no keys and values)
            attending = sum(not lines for lines, _ in self.slots)
            if self.wide is None and self.shared is None:
                self.key_cache = np.zeros((attending, self.n_kv_heads, min(KV_START, self.seq_len), self.head_size), dtype=np.float32)
                self.value_cache = np.zeros_like(self.key_cache)
            else:
                # T331: a layer's keys and values are of the size of its heads: one array for each; T330: none for a
                # layer that reads another's
                self.key_cache = [np.zeros((self.n_kv_heads, min(KV_START, self.seq_len), size), dtype=np.float32)
                                  if kept is not None else None for _, _, size, kept, _ in self.heads]
                self.value_cache = [None if keys is None else np.zeros_like(keys) for keys in self.key_cache]
            if dims.stateful_kind is not None:
                # their states (STATEFUL_LAYERS), each an attribute of its name. state_at: the next position
                self.stateful, states = STATEFUL_LAYERS[dims.stateful_kind]
                states = states(getattr(dims, dims.stateful_kind), n_layers - attending, dim)
                self.states = tuple(states)
                for name, state in states.items():
                    setattr(self, name, state)
                self.state_at = 0
        # the sampler: the kernels' where there are kernels, in the reference's place
        if kernels and "sampler" not in disable:
            self.sampler = KernelSampler(kernels, self.vocab_size)
        if (kernels or external is not None) and "sampler" in disable:
            self.backend += ", NumPy sampling"
        if disable:
            # the line has to say what the numbers are the numbers of
            self.backend += " (without " + ", ".join(name for name in SWITCHES if name in disable) + ")"
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, **reading)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.specials = tuple(str(special) for special in specials)  # see Tokenizer.encode()
        self.stats = {}
        self._run = 0

    def embedding(self, token, table=None):
        """A token's row of the embedding (or of another table read a row at a time: T329's), in float32."""
        table = self.token_embedding_table if table is None else table
        if isinstance(table, tuple):
            values, scales = table
            return (values[token] * scales[token]).reshape(-1)
        return table[token].astype(np.float32)

    def file_tensors(self, checkpoint, rows, dtype, shared_weights, external):
        """Takes every row of a file of this dtype as an attribute of its name (engine/tensors.py's read_rows()). A
        method, for a model that has its rows from somewhere else to put its own here (tests/reference_27b.py's, over
        a GGUF)."""
        for name, tensor in read_rows(checkpoint, rows, dtype, shared_weights, external).items():
            setattr(self, name, tensor)

    def release(self):
        """Let go of what JavaScript holds for this engine (the forward pass of forward.js and the array it fills).
        The worker calls it before it drops a model (T93). T205: what forward.js answers (a promise, settled once the
        GPU's worker let go of the device), for the worker to wait on before it reads the next model; else None."""
        self._external = None
        return None if self.external_forward is None else self.external_forward.release()

    def forward(self, token, pos, need_logits=True):
        """The forward pass in NumPy, the reference: the logits after this token at this position (None where they
        are not needed), with its keys and values, or its layers' states, kept for the tokens after it."""
        n_kv_heads, head_size = self.n_kv_heads, self.head_size
        kv_mul = self.n_heads // n_kv_heads  # >1 with grouped-query attention
        if pos >= self.key_cache[0].shape[-2]:
            # room for twice as many positions (see KV_START); T331: of each layer's array, where each has one
            positions = min(max(2 * self.key_cache[0].shape[-2], pos + 1), self.seq_len)
            grown = lambda cache: np.concatenate([cache, np.zeros((*cache.shape[:-2], positions - cache.shape[-2],
                                                                   cache.shape[-1]), dtype=np.float32)], axis=-2)
            for name in ("key_cache", "value_cache"):
                cache = getattr(self, name)
                setattr(self, name, [layer if layer is None else grown(layer) for layer in cache] if isinstance(cache, list)
                        else grown(cache))
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]
        # T261: the local layers, where the model has such: their window and their own angles of this position
        local = self.local and set(self.local["layers"])
        tables = (cos, sin)
        # (what the architecture is, as its Layout said when the model was made)
        layer_norm, gated_ffn, activation = self.layer_norm, self.gated_ffn, self.activation
        # the step of the layers that keep a state, where there are such (read off the model at every token, as a
        # method is: a tool puts its own there)
        stateful = None
        if self.stateful is not None:
            self.follow(pos)
            stateful = getattr(self, self.stateful)
        # GPT-2 and GPT-NeoX normalize by the mean as well, and have a bias on every projection
        eps = self.rms_norm_eps
        norm = (lambda v, w, b: layernorm(v, w, b)) if layer_norm else (lambda v, w, b: rmsnorm(v, w, eps))
        # (size: the layer's heads', T331)
        heads_of = lambda v, c, s, size: v.reshape(-1, size)  # (q or k as they are)
        whole = lambda v, c, s, size: rope(v, c, s)
        if not self.turning:
            turn = heads_of
        elif self.partly:
            # only the first self.rotary of every head are rotated, the rest go through untouched
            turn = lambda v, c, s, size: partial_rope(v.reshape(-1, size), c, s, self.rotary)
        else:
            turn = whole
        # T255: a layer RoPE leaves alone, where the model has such
        turns = [heads_of if l in self.unturned else turn for l in range(self.n_layers)] if self.unturned else None

        # Copy the token embedding into x, and (GPT-2) the row of this position
        x = self.embedding(token)
        inputs = None
        if self.per_layer_table is not None:
            # T329: every layer's input of its own: the projection of the row, normalized P values a layer, and the
            # token's row of the table of every layer (the converter folded the model's scales into the three)
            projected = head_norm(self.per_layer_projection @ x, self.per_layer_norm, eps)
            inputs = (projected + self.embedding(token, self.per_layer_table)).reshape(self.n_layers, -1)
        if self.positions is not None:
            x = x + self.positions[pos]
        turned = self.turned
        if self.rotated is not None:  # T237: the table holds rotated rows
            x = unrotate(x, self.rotated["signs"][self.dim], self.rotated["block"])

        # Forward all the layers
        for l, (lines, a) in enumerate(self.slots):
            xb = norm(x, self.rms_att_weight[l], self.ln_att_bias[l] if layer_norm else None)
            if lines:  # T229: a linear-attention layer, the a-th of them; T260: or an LFM2's convolution layer
                attended = stateful(a, xb)
            else:
                # QKV matmuls for this position, RoPE on q and k, k and v go to the kv cache (a: the layer's place
                # among the attending layers, which is l where all of them attend)
                # (xr: what the matrices read of xb, itself but in a rotated basis, T237; turned once for all of them)
                xr = turned(xb)
                # (T331: the tensors of the layer's width, its place in their stacks and the size of its heads)
                # (T330: and its place in the stacks of k and v, None where it reads the keys and values at held)
                named, at, size, kept, held = self.heads[a]
                wq, wo = getattr(self, "wq" + named), getattr(self, "wo" + named)
                qv = wq[at] @ xr
                if kept is not None:
                    kv, vv = getattr(self, "wk" + named)[kept] @ xr, getattr(self, "wv" + named)[kept] @ xr
                if self.bq is not None:  # Qwen2 and GPT-2 add a bias to q, k and v
                    qv, kv, vv = qv + self.bq[a], kv + self.bk[a], vv + self.bv[a]
                q_norm = getattr(self, "q_norm" + named)
                if q_norm is not None:  # Qwen3 normalizes every head of q and k
                    qv = head_norm(qv, q_norm[at], eps)
                    if kept is not None:
                        kv = head_norm(kv, getattr(self, "k_norm" + named)[kept], eps)
                if self.value_norm and kept is not None:  # T331: a Gemma 4 normalizes every head of v, without a weight
                    vv = head_norm(vv, np.ones(size, dtype=np.float32), eps)
                if turns:
                    turn = turns[l]
                first, turning = 0, turn
                if local:
                    # a local layer attends the last window positions, this one among them, and turns by its own base
                    here = l in local
                    cos, sin = (self.local_cis[0][pos], self.local_cis[1][pos]) if here else tables
                    first = max(0, pos + 1 - self.local["window"]) if here else 0
                    # (T331: and turns its whole head, where the others turn the model's part of theirs)
                    if here:
                        turning = whole
                q = turning(qv, cos, sin, size).reshape(n_kv_heads, kv_mul, size)
                if kept is not None:
                    self.key_cache[held][:, pos] = turning(kv, cos, sin, size)
                    self.value_cache[held][:, pos] = vv.reshape(n_kv_heads, size)

                # Multihead attention over all timesteps so far (or the last of them), all heads at once
                keys = self.key_cache[held][:, first:pos + 1]      # (n_kv_heads, pos + 1 - first, size)
                values = self.value_cache[held][:, first:pos + 1]
                att = (q @ keys.transpose(0, 2, 1)) * np.float32(1.0 / math.sqrt(size))  # (n_kv_heads, kv_mul, pos + 1)
                att = np.exp(att - att.max(axis=-1, keepdims=True))
                att /= att.sum(axis=-1, keepdims=True)
                attended = (att @ values).reshape(-1)
                if self.wg is not None:  # Qwen3.5 gates what the attention read
                    attended = attended / (1.0 + np.exp(-(self.wg[a] @ xr)))
                # Output projection and residual connection
                attended = wo[at] @ turned(attended)
                if self.bo is not None:  # GPT-2 and GPT-NeoX, after every projection
                    attended = attended + self.bo[a]
            if self.rms_att_post_weight is not None:  # T261: a Gemma normalizes what joins the stream too
                attended = rmsnorm(attended, self.rms_att_post_weight[l], eps)
            # GPT-NeoX with use_parallel_residual: both branches read the x this layer began with
            before = x
            x = x + attended

            # FFN: w2(act(w1(x)) * w3(x)), or GPT-2's w2(act(w1(x) + b1) + b2, and residual connection
            xb = norm(before if self.parallel_residual else x, self.rms_ffn_weight[l],
                      self.ln_ffn_bias[l] if layer_norm else None)
            xr = turned(xb)
            # (T331: the FFN of the layer's width, and its place in their stacks)
            named, at = self.ffns[l]
            w1, w2, w3 = (getattr(self, name + named) for name in ("w1", "w2", "w3")) if named else (self.w1, self.w2, self.w3)
            hb = w1[at] @ xr
            if gated_ffn:
                hb = w2[at] @ turned(activation(hb) * (w3[at] @ xr))
                if self.rms_ffn_post_weight is not None:
                    hb = rmsnorm(hb, self.rms_ffn_post_weight[l], eps)
                x = x + hb
            else:
                x = x + w2[at] @ activation(hb + self.b1[l]) + self.b2[l]
            if inputs is not None:  # T329: what the layer makes of its input joins the stream, normalized
                x = x + rmsnorm(self.ple_proj[l] @ (activation(self.ple_gate[l] @ x) * inputs[l]), self.ple_post[l], eps)
            if self.layer_scale is not None:  # T331: a Gemma 4's layer ends by scaling the stream
                x = x * self.layer_scale[l]

        if not need_logits:
            return None
        # Final norm, then the classifier into logits (60% of all the multiply-adds of stories15M)
        logits = self.wcls @ turned(norm(x, self.rms_final_weight, self.ln_final_bias))
        if self.softcap:  # T331: a Gemma 4 caps its logits
            logits = (np.float32(self.softcap) * np.tanh(logits / np.float32(self.softcap))).astype(np.float32)
        return logits

    def follow(self, pos):
        """T229: the linear-attention layers' state is what the tokens before this position left, so position 0 clears
        it and every other position has to be the one after the last. Keys and values could be written again at any
        position; a state cannot, and a token out of turn would compute on the wrong one without a word. T260: an
        LFM2's convolution layers' state the same."""
        if pos == 0:
            for name in self.states:
                getattr(self, name).fill(0.0)
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
        read = read * silu((self.wz[a] @ xr).reshape(value_heads, -1))
        if self.tiled:  # T363: the matrix's columns are every key head's first value head, then every second, ...
            read = read.reshape(key_heads, value_heads // key_heads, -1).swapaxes(0, 1)
        return self.wout[a] @ self.turned(read.reshape(-1))

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


