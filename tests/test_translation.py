# T648.2: engine/translation.py, Firefox Translations' Marian models translating in NumPy as fxtranslate's int8 engine
# does. A small made-up Marian model (written as tests/test_marian.py writes one) is translated here, and each part is
# held to the plain formula: the int8 affine to the exact integers, the sums to a loop, LayerNorm and the attention to
# float64, and every step's logits to a whole forward written out again in float64. Whether the real models write
# fxtranslate's translations is tests/translation_check.py's (it needs the models and fxtranslate: not in CI).
import math

import numpy as np
import pytest

from convert.tokenizer import tokenizer_bin
from engine.marian import read_marian
from engine.tokenizer import Tokenizer
from engine.translation import FLOAT_FUNCTIONS, Affine, Translator, layer_norm, total
from test_marian import floats, intgemm, marian

F = np.float32
D, HEADS, FFN, VOCAB, ENCODER, DECODER = 8, 2, 12, 16, 2, 2
# the made-up vocabulary: Marian's </s> and <unk>, a byte piece and a few words
PIECES = ["</s>", "<unk>", "<0xE2>", "<0x98>", "<0x95>", "▁a", "▁b", "▁c", "a", "b", "c", "▁ab", "▁ba", "▁cab", "▁", "ca"]
SETTINGS = f"""dec-depth: {DECODER}
enc-depth: {ENCODER}
dim-emb: {D}
transformer-heads: {HEADS}
type: transformer
"""


def tokenizer():
    rows = [(text, -float(i), i >= 5) for i, text in enumerate(PIECES)]
    return Tokenizer(tokenizer_bin(rows, VOCAB), VOCAB, kind="unigram")


def made_up(seed=648, eos_bias=0.0, shared=True):
    """A made-up Marian model's items, laid out as the real ones: intgemm8 matrices (the file holding [out, in]),
    float32 biases and LayerNorms, a fixed quantMult for each matrix's activation."""
    rng = np.random.default_rng(seed)
    items = []

    def matrix(name, k, n, alpha=None):
        rows = rng.integers(-127, 128, (n, k), dtype=np.int8)
        items.append((name, 0x4101, (k, n), intgemm(rows, float(rng.uniform(200, 400)))))
        items.append((name + "_QuantMultA", 0x0404, (1, 1), floats([[alpha or rng.uniform(20, 40)]])))

    def vector(name, n, around=0.0, spread=0.3):
        items.append((name, 0x0404, (1, n), floats(around + spread * rng.standard_normal((1, n)))))

    def norm(name):
        vector(name + "_ln_bias", D, 0.0, 0.1)
        vector(name + "_ln_scale", D, 1.0, 0.1)

    def attention(prefix):
        for x in "qkvo":
            matrix(f"{prefix}_W{x}", D, D)
            vector(f"{prefix}_b{x}", D)
        norm(f"{prefix}_Wo")

    def ffn(prefix):
        matrix(f"{prefix}_W1", D, FFN)
        vector(f"{prefix}_b1", FFN)
        matrix(f"{prefix}_W2", FFN, D)
        vector(f"{prefix}_b2", D)
        norm(f"{prefix}_ffn")

    tables = ["Wemb"] if shared else ["encoder_Wemb", "decoder_Wemb"]
    for name in tables:
        items.append((name, 0x4101, (VOCAB, D), intgemm(rng.integers(-127, 128, (VOCAB, D)), float(rng.uniform(80, 120)))))
    if shared:
        items.append(("none_QuantMultA", 0x0404, (1, 1), floats([[12.5]])))
    else:  # en-ja's: an intgemm8 [1, 1], 127 and its quantMult
        items.append(("decoder_Wemb_QuantMultA", 0x4101, (1, 1), intgemm([127], 10.2)))
    for l in range(1, ENCODER + 1):
        attention(f"encoder_l{l}_self")
        ffn(f"encoder_l{l}_ffn")
    for l in range(1, DECODER + 1):
        matrix(f"decoder_l{l}_rnn_W", D, D)
        matrix(f"decoder_l{l}_rnn_Wf", D, D)
        vector(f"decoder_l{l}_rnn_bf", D)
        norm(f"decoder_l{l}_rnn_ffn")
        attention(f"decoder_l{l}_context")
        ffn(f"decoder_l{l}_ffn")
    bias = 0.5 * rng.standard_normal((1, VOCAB))
    bias[0, 0] += eos_bias
    items.append(("decoder_ff_logit_out_b", 0x0404, (1, VOCAB), floats(bias)))
    items.append(("special:model.yml", 0x0101, (len(SETTINGS) + 1,), SETTINGS.encode() + b"\0"))
    return read_marian(marian(items))


def translator(**options):
    return Translator(made_up(**options), tokenizer(), tokenizer())


# ---- the parts, against their plain formulas

def test_total_adds_one_by_one():
    values = np.asarray([1e8] + [1] * 200 + [-1e8], F)  # one by one, each 1 is lost on 1e8 (float32 steps of 8 there)
    loop = _loop(values)
    assert total(values) == loop == F(0)
    assert np.sum(values) != loop  # (NumPy's pairs round otherwise: the reason for total)
    rows = np.random.default_rng(1).standard_normal((3, 50)).astype(F)
    assert np.array_equal(total(rows), [_loop(row) for row in rows])
    assert np.array_equal(total(rows.T, axis=0), total(rows))


def _loop(row):
    sum_ = F(0)
    for value in row:
        sum_ = F(sum_ + value)
    return sum_


@pytest.mark.parametrize("k", [520, 521, 1536])
def test_the_affine_is_fxtranslates_int8_to_the_bit(k):
    """unquant · Σ (q W) + (b − 127 · unquant · Σ W), with the integer sums exact: k past 520 takes float64 (a sum of
    254 · 127 · k passes 2**24 there), and the largest activations and weights reach that sum."""
    rng = np.random.default_rng(k)
    n = 5
    rows = rng.integers(-127, 128, (n, k)).astype(np.int8)
    rows[0] = 127  # with q = 254 everywhere, the largest sum there is
    item = read_marian(marian([("w", 0x4101, (k, n), intgemm(rows, 71.25)),
                               ("special:model.yml", 0x0101, (5,), b"a: 1\0")])).items["w"]
    bias = rng.standard_normal(n).astype(F)
    affine = Affine(item, 4.0, bias)
    x = (40 * rng.standard_normal((3, k))).astype(F)
    x[0] = 1000  # past 127 / qA: kept to 254
    x[1, :4] = [0.125, 0.375, -0.125, -0.625]  # 0.5, 1.5, −0.5, −2.5: to the even
    got = affine(x)
    alpha, unquant = F(4.0), F(1) / (F(4.0) * F(71.25))
    for m in range(len(x)):
        q = [min(max(int(np.rint(F(value * alpha))), -127), 127) + 127 for value in x[m]]
        assert q[:4] == ([254] * 4 if m == 0 else [127, 129, 127, 125] if m == 1 else q[:4])
        for j in range(n):
            exact = sum(a * int(w) for a, w in zip(q, rows[j]))
            prepared = F(bias[j] - F(F(127) * unquant) * F(int(rows[j].astype(np.int64).sum())))
            assert got[m, j] == F(F(unquant * F(exact)) + prepared), (m, j)
    assert affine.weight.dtype == (np.float32 if k <= 520 else np.float64)


def test_layer_norm_and_attention_are_their_formulas():
    rng = np.random.default_rng(3)
    x = (2 * rng.standard_normal((4, D)) + 1).astype(F)
    scale, bias = rng.standard_normal(D).astype(F), rng.standard_normal(D).astype(F)
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=1, keepdims=True)
    plain = scale * (x64 - mean) / np.sqrt(((x64 - mean) ** 2).mean(axis=1, keepdims=True) + 1e-6) + bias
    assert np.allclose(layer_norm(x, scale, bias), plain, rtol=1e-5, atol=1e-5)

    # a row of small spread, where eps (1e-6, not 1e-5) is a good part of the variance
    flat = (1 + 1e-3 * rng.standard_normal((2, D))).astype(F)
    f64 = flat.astype(np.float64)
    centred = f64 - f64.mean(axis=1, keepdims=True)
    assert np.allclose(layer_norm(flat, scale, bias), scale * centred / np.sqrt((centred ** 2).mean(axis=1, keepdims=True) + 1e-6) + bias,
                       rtol=1e-5, atol=1e-5)

    t = translator()
    attention = t.encoder[0][0]
    q, k, v = attention.q(x).astype(np.float64), attention.k(x).astype(np.float64), attention.v(x).astype(np.float64)
    width = D // HEADS
    heads = []
    for h in range(HEADS):
        part = slice(h * width, (h + 1) * width)
        scores = q[:, part] @ k[:, part].T / math.sqrt(width)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        heads.append((weights / weights.sum(axis=1, keepdims=True)) @ v[:, part])
    joined = np.concatenate(heads, axis=1).astype(F)
    assert np.allclose(t.attend(attention, x, attention.k(x), attention.v(x)), attention.o(joined), rtol=1e-4, atol=1e-3)


def test_the_embedding_and_the_positions():
    t = translator()
    table, inverse = t.tables[1]
    half = D // 2
    for position in (0, 1, 7):
        row = t.embedded(1, [9], position)[0]
        angles = [position * 1e-4 ** ((c % half) / (half - 1)) + (c // half) * math.pi / 2 for c in range(D)]
        word = 0 if position == 0 else table[9].astype(np.float64) * float(inverse)
        assert np.allclose(row, math.sqrt(D) * np.asarray(word) + np.sin(angles), atol=1e-5), position
    # the decoder's position 0 embeds nothing (Marian shifts the target right); the encoder's embeds its token
    assert np.allclose(t.embedded(1, [9], 0), t.embedded(1, [3], 0))
    assert not np.allclose(t.embedded(0, [9], 0), t.embedded(0, [3], 0))


# ---- the whole, against a forward written out again in float64

def plain_logits(model, ids, forced):
    """The steps' logits from the formulas of docs/notes/t648-firefox-translations-engines-2026-11-05.md, in float64,
    the int8 affines as there (q from x · qA rounded to even, the integer sums, the shift taken back)."""
    items = model.items

    def v(name):
        return items[name].values.astype(np.float64).reshape(-1)

    def affine(name, x, bias=None, alpha=None):
        w = items[name].values.astype(np.int64)  # [N, K]
        a = v(name + "_QuantMultA")[0] if alpha is None else alpha
        q = np.clip(np.rint(x * a), -127, 127) + 127
        un = 1 / (a * items[name].quant_mult)
        return un * (q @ w.T) - 127 * un * w.sum(axis=1) + (0 if bias is None else v(bias))

    def ln(x, name):
        mean = x.mean(axis=1, keepdims=True)
        return v(name + "_ln_scale") * (x - mean) / np.sqrt(((x - mean) ** 2).mean(axis=1, keepdims=True) + 1e-6) + v(name + "_ln_bias")

    def attn(prefix, x, c):
        q, k, vv = (affine(f"{prefix}_W{n}", y, f"{prefix}_b{n}") for n, y in (("q", x), ("k", c), ("v", c)))
        width, out = D // HEADS, []
        for h in range(HEADS):
            part = slice(h * width, (h + 1) * width)
            s = q[:, part] @ k[:, part].T / math.sqrt(width)
            e = np.exp(s - s.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True) @ vv[:, part])
        return affine(f"{prefix}_Wo", np.concatenate(out, axis=1), f"{prefix}_bo")

    def ffn(prefix, x):
        return ln(affine(f"{prefix}_W2", np.maximum(affine(f"{prefix}_W1", x, f"{prefix}_b1"), 0), f"{prefix}_b2") + x, f"{prefix}_ffn")

    def embed(table, ids, start, zero_first=False):
        half = D // 2
        c = np.arange(D)
        angles = np.arange(start, start + len(ids))[:, None] * 1e-4 ** ((c % half) / (half - 1)) + (c // half) * math.pi / 2
        rows = items[table].values[ids].astype(np.float64) / items[table].quant_mult
        if zero_first and start == 0:
            rows[0] = 0
        return math.sqrt(D) * rows + np.sin(angles)

    x = embed("Wemb", ids, 0)
    for l in range(1, ENCODER + 1):
        x = ln(attn(f"encoder_l{l}_self", x, x) + x, f"encoder_l{l}_self_Wo")
        x = ffn(f"encoder_l{l}_ffn", x)
    cells, token, logits = [np.zeros((1, D)) for _ in range(DECODER)], 0, []
    for position in range(len(forced)):
        y = embed("Wemb", [token], position, zero_first=True)
        for l in range(1, DECODER + 1):
            p = f"decoder_l{l}"
            g = 1 / (1 + np.exp(-affine(f"{p}_rnn_Wf", y, f"{p}_rnn_bf")))
            cells[l - 1] = g * cells[l - 1] + (1 - g) * affine(f"{p}_rnn_W", y)
            y = ln(np.maximum(cells[l - 1], 0) + y, f"{p}_rnn_ffn")
            y = ln(attn(f"{p}_context", y, x) + y, f"{p}_context_Wo")
            y = ffn(f"{p}_ffn", y)
        logits.append(affine("Wemb", y, "decoder_ff_logit_out_b", alpha=12.5)[0])
        token = forced[position]
    return logits


def test_every_step_is_the_plain_forward():
    t = translator(eos_bias=-50.0)  # (it does not stop: every step up to the bound is compared)
    ids = [5, 9, 11, 7, 0]
    written = t.greedy(ids)
    assert len(written) == 2 * len(ids) + 4 and 0 not in written
    steps = list(t.steps(ids))
    plain = plain_logits(made_up(eos_bias=-50.0), ids, written)
    assert len(steps) == len(plain) == len(written)
    for position, (ours, theirs) in enumerate(zip(steps, plain)):
        # (a float64 activation can round to the other int8 of a float32 one: a little, now and then)
        assert np.allclose(ours, theirs, atol=0.05 * np.ptp(theirs)), position
        assert written[position] == int(np.argmax(ours))
    # forced: the tokens are forced's, the logits the same steps'
    again = list(t.steps(ids, forced=written))
    assert all(np.array_equal(a, b) for a, b in zip(again, steps))


def test_it_stops_at_the_end_and_at_the_bound():
    t = translator(eos_bias=50.0)
    assert t.greedy([5, 0]) == [] and len(list(t.steps([5, 0]))) == 1
    long = translator(eos_bias=-50.0)
    assert len(long.greedy([5] * 200 + [0])) == 256  # min(⌈2 S⌉ + 4, 256)
    assert len(long.greedy([5, 6, 0])) == 10


def test_split_vocabularies_and_the_output_alpha():
    t = Translator(made_up(shared=False), tokenizer(), tokenizer())
    assert t.output.alpha == F(127) / F(10.2)  # (decoder_Wemb_QuantMultA: int8 / its quantMult, as fxtranslate)
    assert t.tables[0][0] is not t.tables[1][0]
    assert len(t.greedy([5, 0])) <= 6


def test_ids_and_text():
    t = translator()
    assert t.source_ids("a b") == [*t.source.encode("a b"), 0]
    # a byte piece is its byte, </s> and <unk> write nothing, the first space goes
    assert t.text([5, 2, 3, 4, 1, 6, 0]) == "a☕ b"
    assert t.text([14, 11, 12]) == "ab ba"
    with pytest.raises(ValueError, match="has no </s>"):
        Translator(made_up(), Tokenizer(tokenizer_bin([("x", 0.0, True)] * VOCAB, VOCAB), VOCAB, kind="unigram"), tokenizer())


def test_the_float_functions_round_once():
    x = np.asarray([0.5, -3.25, 80.0, 1e-3], F)
    assert np.array_equal(FLOAT_FUNCTIONS.exp(x), np.exp(x.astype(np.float64)).astype(F))
    assert np.array_equal(FLOAT_FUNCTIONS.sin(x), np.sin(x.astype(np.float64)).astype(F))
    assert FLOAT_FUNCTIONS.exp(x).dtype == F
