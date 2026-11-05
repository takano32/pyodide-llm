# T648.2: Firefox Translations' models (Marian's transformer encoder and SSRU decoder, read by engine/marian.py)
# translating in NumPy, greedily, so that each step comes to the same numbers as fxtranslate's int8 engine. It is the
# reference the kernel (T648.3: affine_q8, the same numbers to the bit) and the page (T648.4) are held to.
#
#     translator = Translator(read_marian(data), source, target)   source, target: engine.tokenizer.Tokenizer
#     Translator(read_marian(data), source, target, kernels=load_kernels(path))   every affine on the kernel (Pyodide)
#     translator.translate("こんにちは。")                           the text, as one unit (no sentences are cut)
#     translator.greedy(translator.source_ids(text))                the target's ids, without </s>
#     for logits in translator.steps(ids, forced): ...              each step's logits, the tokens being forced's
#
# The steps were read from fxtranslate (https://github.com/gregtatum/translations, inference-rs/, commit f2f77cd3,
# MPL-2.0) and their form taken; no line of it is copied here. Where each is (crates/fxtranslate/src/):
#     the encoder's layer: engine.rs 889–899; the attention: multihead() 1531–1599; the FFN: 1778–1797; post-norm:
#     1800–1808; the embedding and the positions: embed_into() 1827–1878, the table in Engine::new() 285–310
#     the decoder's step: decode_step() 947–990 (the SSRU's highway: ops.rs 103–119); greedy(): 454–499
#     the output layer: weights.rs full_logits_batch_into() 615–654 (lean-embed: int8) and argmax() 1958–1968
#     LayerNorm: ops.rs 27–74; softmax: 140–155; the int8 affine: prepare_a 464–472, prepare_bias 492–510,
#     intgemm_affine 532–559 and weights.rs affine() 794–863 (the scalar path, which this machine's wheel runs)
# What the model is, and how fxtranslate's int8 is intgemm's int8shiftAlphaAll, is in
# docs/notes/t648-firefox-translations-engines-2026-11-05.md.
#
# To come to the same numbers, each float32 operation is the one fxtranslate does, in its order:
#     a sum is added one by one from the first (Rust's iter().sum() and its loops), not in NumPy's pairs: total()
#     an affine's activation is x · qA rounded to even, kept to ±127 and moved up by 127; its sum over k is the
#     exact integer, turned into a float32 once; then unquant · sum + the bias that takes the 127 back off
#     exp, sin and the positions' pow are glibc's expf, sinf and powf, which fxtranslate calls: FLOAT_FUNCTIONS
import math
from typing import NamedTuple

import numpy as np

F = np.float32
EPS = F(1e-6)  # LayerNorm's (Marian's layers/generic.h, fxtranslate's engine.rs EPS)
ALPHA = "_QuantMultA"
EXACT = 1 << 24  # a float32 holds every integer up to this


def total(values, axis=-1):
    """The float32 sum along axis, added one after another from the first, as a loop adds them: NumPy's sum adds in
    pairs, which rounds otherwise. (np.cumsum is the running sum, so its last is that.)"""
    return np.cumsum(values, axis=axis, dtype=F).take(-1, axis=axis)


def rounded(function):
    """function on float32 values as float64, rounded once to float32: the correctly rounded value, which glibc's
    float functions give for most arguments (not for all: FLOAT_FUNCTIONS)."""
    return lambda values: function(np.asarray(values, np.float64)).astype(F)


class FloatFunctions(NamedTuple):
    """The float32 functions the steps call: exp (the softmax and the SSRU's gate), sin (the positions) and pow (the
    positions' frequencies, once). fxtranslate calls glibc's expf, sinf and powf; NumPy's are other roundings."""
    exp: object
    sin: object
    pow: object


# the correctly rounded ones: glibc's sinf rounds otherwise for about 1 argument in 80 of the positions' range, and its
# expf for about 1 in 2,000 (measured against glibc 2.43); powf agreed on all of them
FLOAT_FUNCTIONS = FloatFunctions(rounded(np.exp), rounded(np.sin), lambda x, y: rounded(lambda y: np.power(np.float64(x), y))(y))


class Affine:
    """y = x W + b with W intgemm8 (int8 and one quantMult qB) and the activation's fixed quantMult qA, as fxtranslate's
    scalar int8 affine: q = clamp(round_even(x · qA), −127, 127) + 127, the exact integer Σ q W, then
    unquant · Σ + (b − 127 · unquant · Σ_k W) with unquant = 1 / (qA · qB), each in float32."""

    def __init__(self, weight, alpha, bias=None):
        rows = self.prepared(weight, alpha, bias)
        k = rows.shape[1]
        # the products' sums are integers of at most 254 · 127 · k: float32 adds them exactly in any order while that
        # is below 2**24 (k up to 520), float64 past it
        self.weight = np.ascontiguousarray(rows.T, F if 254 * 127 * k < EXACT else np.float64)  # [K, N]

    def prepared(self, weight, alpha, bias):
        """alpha, unquant and the bias with the shift taken off, which both affines use; the int8 rows, [N, K]."""
        rows = weight.values  # [N, K]: out rows of in (an intgemm8 matrix's file order, and an embedding's)
        n = rows.shape[0]
        self.alpha = F(alpha)
        self.unquant = F(1) / (self.alpha * F(weight.quant_mult))
        self.sums = rows.sum(axis=1, dtype=np.int64)
        raw = np.zeros(n, F) if bias is None else np.asarray(bias, F).reshape(n)
        self.bias = raw - F(127) * self.unquant * self.sums.astype(F)  # (|Σ| ≤ 127 k: exact in float32)
        return rows

    def __call__(self, x):
        q = np.clip(np.rint(x * self.alpha), -127, 127) + F(127)
        sums = (q.astype(self.weight.dtype) @ self.weight).astype(F)
        return self.unquant * sums + self.bias


class KernelAffine(Affine):
    """Affine on the SIMD kernel affine_q8 (T648.3): the same numbers to the bit, in one call a matrix. The kernel sums
    in int32, exact while 254 · 128 · k is below 2**31 (k up to 66,051: Firefox's models are 2048 at most); the 127 of
    the activations' shift comes back as the integer 127 Σ_k W of each row (shifts)."""

    def __init__(self, weight, alpha, bias, kernel):
        rows = self.prepared(weight, alpha, bias)
        n, k = rows.shape
        if 254 * 128 * k >= 1 << 31:
            raise ValueError(f"An affine of {k} inputs is past the int32 sums of the kernel.")
        self.kernel, self.n, self.k = kernel, n, k
        self.rows = np.ascontiguousarray(rows, np.int8)
        self.shifts = (127 * self.sums).astype(np.int32)
        self.bias = np.ascontiguousarray(self.bias, F)

    def __call__(self, x):
        x = np.ascontiguousarray(x, F)
        m = x.shape[0]
        out, xq = np.empty((m, self.n), F), np.empty(m * self.k, np.int8)
        self.kernel(out.ctypes.data, x.ctypes.data, xq.ctypes.data, m, self.k, self.rows.ctypes.data, self.n, self.alpha,
                    self.shifts.ctypes.data, self.unquant, self.bias.ctypes.data)
        return out


def layer_norm(x, scale, bias):
    """LayerNorm over the last axis: γ ⊙ (x − mean) / sqrt(Σ (x − mean)² / D + eps) + β, the sums one by one."""
    cols = F(x.shape[-1])
    mean = total(x) / cols
    centred = x - mean[..., None]
    sigma = np.sqrt(total(centred * centred) / cols + EPS)
    return scale * (centred / sigma[..., None]) + bias


class Attention(NamedTuple):
    q: Affine
    k: Affine
    v: Affine
    o: Affine


class Translator:
    """A Marian transformer of Firefox Translations: a post-norm encoder of self-attention and ReLU FFNs, and a
    decoder of SSRU, attention on the source and FFN layers; greedy, without a shortlist."""

    def __init__(self, model, source, target, functions=FLOAT_FUNCTIONS, kernels=None):
        """kernels: engine.kernels.load_kernels()'s, whose affine_q8 then does every affine (T648.3: the same numbers),
        or None for NumPy's."""
        self.source, self.target, self.functions = source, target, functions
        product = (lambda *made: KernelAffine(*made, kernels["affine_q8"])) if kernels else Affine
        items, config = model.items, model.config
        alphas = {name[:-len(ALPHA)]: item for name, item in items.items() if name.endswith(ALPHA)}

        def floats(name):
            return np.asarray(items[name].values, F).reshape(-1)

        def affine(name, bias=None):
            return product(items[name], floats(name + ALPHA)[0], None if bias is None else floats(bias))

        def norm(name):
            return floats(name + "_ln_scale"), floats(name + "_ln_bias")

        def attention(prefix):
            return Attention(*(affine(f"{prefix}_W{x}", f"{prefix}_b{x}") for x in "qkvo"))

        def ffn(prefix):
            return affine(f"{prefix}_W1", f"{prefix}_b1"), affine(f"{prefix}_W2", f"{prefix}_b2"), norm(f"{prefix}_ffn")

        shared = "Wemb" in items  # tied-embeddings-all (ja-en); en-ja has encoder_Wemb and decoder_Wemb
        source_table, target_table = (items["Wemb"],) * 2 if shared else (items["encoder_Wemb"], items["decoder_Wemb"])
        self.heads = int(config["transformer-heads"])
        self.dim = target_table.values.shape[1]
        if self.dim % self.heads or source_table.values.shape[1] != self.dim:
            raise ValueError(f"This Marian model's embeddings are {self.dim} wide, which its {self.heads} heads do not divide.")
        self.tables = [(table.values, F(1) / F(table.quant_mult)) for table in (source_table, target_table)]
        self.encoder = [(attention(f"encoder_l{l}_self"), norm(f"encoder_l{l}_self_Wo"), ffn(f"encoder_l{l}_ffn"))
                        for l in range(1, int(config["enc-depth"]) + 1)]
        self.decoder = [(affine(f"decoder_l{l}_rnn_W"), affine(f"decoder_l{l}_rnn_Wf", f"decoder_l{l}_rnn_bf"),
                         norm(f"decoder_l{l}_rnn_ffn"), attention(f"decoder_l{l}_context"),
                         norm(f"decoder_l{l}_context_Wo"), ffn(f"decoder_l{l}_ffn"))
                        for l in range(1, int(config["dec-depth"]) + 1)]
        # the output layer is the target's embedding, its activation's quantMult none_QuantMultA where there is one
        # (ja-en's) and else decoder_Wemb_QuantMultA, an intgemm8 [1, 1] that is int8 / its quantMult (en-ja's)
        out = alphas["none"] if "none" in alphas else alphas["decoder_Wemb"]
        out_alpha = out.values[0, 0] if out.kind == "float32" else F(out.values[0, 0]) / F(out.quant_mult)
        self.output = product(target_table, out_alpha, items["decoder_ff_logit_out_b"].values if "decoder_ff_logit_out_b" in items else None)
        # the positions: PE(t)[c] = sin(t f_c + o_c), f_c = 1e-4 ^ ((c mod D/2) / (D/2 − 1)), o_c = ⌊c / (D/2)⌋ π/2
        half = self.dim // 2
        column = np.arange(self.dim)
        self.frequencies = functions.pow(F(1e-4), (column % half).astype(F) / (F(half) - F(1)))
        self.offsets = (column // half).astype(F) * F(math.pi / 2)
        self.scale = np.sqrt(F(self.dim))
        self.eos = self.end_of(source), self.end_of(target)

    @staticmethod
    def end_of(tokenizer):
        """</s>, Marian's end of a sentence (id 0 in Firefox's vocabularies), which the source ends with and the
        target stops at."""
        eos = tokenizer.index.get(b"</s>")
        if eos is None:
            raise ValueError("This vocabulary has no </s>, which a Marian model ends its sentences with.")
        return eos

    # ---- the parts

    def embedded(self, side, ids, start):
        """√D · E[id] + PE(position) for ids from position start; the decoder's position 0 is PE(0) alone (Marian
        shifts the target right and puts 0s in the first place)."""
        table, inverse = self.tables[side]
        rows = table[np.asarray(ids, np.int64)].astype(F) * inverse
        if side == 1 and start == 0:
            rows[0] = 0
        positions = np.arange(start, start + len(ids)).astype(F)[:, None]
        return self.scale * rows + self.functions.sin(positions * self.frequencies + self.offsets)

    def attend(self, attention, x, keys, values):
        """Multi-head attention of x's queries on keys and values already made: softmax(q kᵀ / √dk) v by head, then
        Wo. The scores and the weighted sums are added one by one, over the head's width and over the positions."""
        heads, dim = self.heads, self.dim
        width = dim // heads
        q = attention.q(x).reshape(len(x), heads, width).transpose(1, 0, 2)  # [H, Sq, dk]
        k = keys.reshape(len(keys), heads, width).transpose(1, 0, 2)  # [H, Sk, dk]
        v = values.reshape(len(values), heads, width).transpose(1, 0, 2)
        scores = total(q[:, :, None, :] * k[:, None, :, :]) * (F(1) / np.sqrt(F(width)))  # [H, Sq, Sk]
        weights = self.functions.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / total(weights)[..., None]
        joined = total(weights[..., None] * v[:, None, :, :], axis=2)  # [H, Sq, dk]
        return attention.o(joined.transpose(1, 0, 2).reshape(len(x), dim))

    @staticmethod
    def ffn(ffn, x):
        first, second, (scale, bias) = ffn
        return layer_norm(second(np.maximum(first(x), F(0))) + x, scale, bias)

    def encode(self, ids):
        """The encoder's output C, [S, D], for the source's ids (with their </s>)."""
        x = self.embedded(0, ids, 0)
        for attention, (scale, bias), ffn in self.encoder:
            x = layer_norm(self.attend(attention, x, attention.k(x), attention.v(x)) + x, scale, bias)
            x = self.ffn(ffn, x)
        return x

    def step(self, token, position, context, cells):
        """One decoder step for the token before (at position): each layer's SSRU, c = σ(f) c' + (1 − σ(f)) x̃ with
        x̃ = u W and f = u Wf + bf, then LN(u + relu(c)), the attention on the source and the FFN. cells (one [1, D]
        state a layer) are written; the top is returned."""
        exp = self.functions.exp
        x = self.embedded(1, [token], position)
        for l, (candidate, forget, (scale, bias), attention, (scale2, bias2), ffn) in enumerate(self.decoder):
            gate = F(1) / (F(1) + exp(-forget(x)))
            cells[l] = gate * cells[l] + (F(1) - gate) * candidate(x)
            x = layer_norm(np.maximum(cells[l], F(0)) + x, scale, bias)
            keys, values = context[l]
            x = layer_norm(self.attend(attention, x, keys, values) + x, scale2, bias2)
            x = self.ffn(ffn, x)
        return x

    # ---- translating

    def source_ids(self, text):
        """The source's ids for text: its pieces and </s> (Marian puts no beginning before them)."""
        return [*self.source.encode(text), self.eos[0]]

    def steps(self, ids, forced=None):
        """Each step's logits (the whole target vocabulary's, float32) for the source's ids. The token each step
        writes is the logits' first largest, or forced's (a list of the target's ids) while it lasts; it stops after
        </s> or at the length's bound, min(⌈2 S⌉ + 4, 256) steps with S the source's ids."""
        context = self.encode(ids)
        context = [(attention.k(context), attention.v(context)) for _, _, _, attention, _, _ in self.decoder]
        cells = [np.zeros((1, self.dim), F) for _ in self.decoder]
        token = self.eos[1]  # (position 0 embeds nothing: it only stands in)
        for position in range(min(2 * len(ids) + 4, 256)):
            logits = self.output(self.step(token, position, context, cells))[0]
            yield logits
            token = forced[position] if forced is not None and position < len(forced) else int(np.argmax(logits))
            if token == self.eos[1]:
                return

    def greedy(self, ids):
        """The target's ids the greedy search writes, without its </s>."""
        written = []
        for logits in self.steps(ids):
            token = int(np.argmax(logits))  # (the first of equal largest, as fxtranslate's argmax)
            if token == self.eos[1]:
                break
            written.append(token)
        return written

    def text(self, ids):
        """The text of the target's ids: their pieces one after another, a byte piece as its byte, then UTF-8; the
        unknown piece and </s> write nothing, and the text's first spaces are taken off."""
        vocab, unknown = self.target.vocab, self.target.index.get(b"<unk>")
        pieces = bytearray()
        for token in ids:
            if token in (self.eos[1], unknown):
                continue
            piece = vocab[token]
            if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
                piece = bytes([int(piece[3:5], 16)])
            pieces += piece
        return pieces.decode("utf-8", "replace").lstrip()

    def translate(self, text):
        """text translated as one unit, greedily (the sentences are not cut apart: that is the page's, T648.4)."""
        return self.text(self.greedy(self.source_ids(text)))
