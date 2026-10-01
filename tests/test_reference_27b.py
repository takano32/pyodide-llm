"""T238: the parts of tests/reference_27b.py that can be checked without the 27B: how a PTQ1_0 block is read, and the
one pass that serves every position (Conductor) against the engine's forward pass a position at a time."""
import numpy as np
from conftest import qwen35_model
from reference_27b import Conductor, Matrix, Stored, as_bf16, as_q8_0, grouped, pack_ptq1_0, widen_ptq1_0
from test_qwen35 import engine

TOKENS = [1, 5, 7, 9, 11, 5, 5, 300, 2]
MATRICES = ("wq", "wg", "wk", "wv", "wo", "wqkv", "wz", "wout", "w1", "w2", "w3")


def test_a_ptq1_0_block_is_read_as_the_fork_packs_it():
    """Every byte of five digits there is, random blocks, and a weight alone at each kind of place: the weights are
    in the fork's order (the digits of 16 bytes, of 8 and of 2 run across the bytes first)."""
    rng = np.random.default_rng(3)
    digits = rng.integers(0, 3, (700, 128))
    digits[:243, 0:80:16] = np.array([[n // 3 ** i % 3 for i in range(5)] for n in range(243)])  # byte 0's five digits
    scales = rng.uniform(0.004, 0.03, 700).astype(np.float16)
    got = widen_ptq1_0(np.frombuffer(pack_ptq1_0(digits, scales), dtype=np.uint8))
    assert np.array_equal(got, (digits - 1).astype(np.float32) * scales.astype(np.float32)[:, None])
    for place in range(128):
        one = np.ones((1, 128), dtype=np.int64)
        one[0, place] = 2
        got = widen_ptq1_0(np.frombuffer(pack_ptq1_0(one, [0.5]), dtype=np.uint8))
        assert got[0, place] == 0.5 and np.count_nonzero(got) == 1, place


def test_the_bytes_of_a_block_hold_the_weights_the_survey_found():
    """The places as T228 read them from the fork: weights 0 to 15 are the first digits of qs[0..15], 16 to 31 their
    second digits; 80 to 87 the first digits of qs[16..23]; 120 and 121 the first digits of qh[0] and qh[1]."""
    def block(byte_index, digit):
        # the bytes of zeros alone: five digits 1 are 121, ceil(121 * 256 / 243) = 128; four, shifted up, 120: 127
        raw = np.array([128] * 24 + [127] * 2 + [0, 0], dtype=np.uint8)
        digits = [1, 1, 1, 1, 1]
        digits[digit] = 2
        raw[byte_index] = -(-sum(d * 3 ** (4 - i) for i, d in enumerate(digits)) * 256 // 243)
        raw[26:28] = np.float16(1.0).reshape(1).view(np.uint8)
        return np.flatnonzero(widen_ptq1_0(raw) == 1.0).tolist()

    assert block(0, 0) == [0] and block(15, 0) == [15] and block(0, 1) == [16] and block(3, 4) == [67]
    assert block(16, 0) == [80] and block(23, 4) == [119]
    assert block(24, 0) == [120] and block(25, 0) == [121] and block(25, 3) == [127]


def test_the_fork_rounding_of_an_activation():
    x = np.linspace(-3, 5, 64, dtype=np.float32)
    q = as_q8_0(x)
    step = np.float32(np.float16(5 / 127))
    assert np.abs(q - x).max() <= step / 2 * 1.01 and np.allclose(q[-1], 5, rtol=1e-3)
    assert np.array_equal(as_q8_0(np.zeros(32, dtype=np.float32)), np.zeros(32, dtype=np.float32))
    b = as_bf16(x)
    assert np.all(b.view(np.uint32) & 0xFFFF == 0) and np.abs(b - x).max() <= np.abs(x).max() * 2.0 ** -8


def test_the_heads_of_hugging_faces_order_in_a_gguf():
    # 2 key heads of 3 value heads each: Hugging Face has (k0 v0, k0 v1, k0 v2, k1 v0, ...), the GGUF every key
    # head's first, then every key head's second: (k0 v0, k1 v0, k0 v1, k1 v1, k0 v2, k1 v2)
    assert grouped(2, 6).tolist() == [0, 2, 4, 1, 3, 5]
    assert grouped(2, 4, 2).tolist() == [0, 1, 4, 5, 2, 3, 6, 7]
    assert grouped(4, 4).tolist() == [0, 1, 2, 3]


def test_one_pass_for_all_positions_is_the_forward_pass_a_position_at_a_time():
    """Two runs of a made-up Qwen3.5 (state and keys from position to position) through the conductor, every matrix
    multiplied once for all of them, against the same engine called in order."""
    tensors, config = qwen35_model(n_layers=8, every=4, shared=False)
    plain = engine(tensors, config)[0]
    want = [plain.forward(token, pos).copy() for pos, token in enumerate(TOKENS)]
    conductor, read, stores, models = Conductor(), [], {}, []

    def stored(name, w):
        def rows(first, last):
            read.append((name, first, last))
            return w[first:last]
        return stores.setdefault(name, Stored(name, *w.shape, rows))

    for _ in range(2):
        model = engine(tensors, config)[0]
        for name in MATRICES:
            setattr(model, name, [Matrix(conductor, stored(f"{name}.{a}", w)) for a, w in enumerate(getattr(model, name))])
        model.wcls = Matrix(conductor, stored("wcls", model.wcls))
        models.append(model)
    jobs = [(lambda model=model, token=token, pos=pos: np.array(model.forward(token, pos)))
            for model in models for pos, token in enumerate(TOKENS)]
    got = conductor.run(jobs)
    for index, logits in enumerate(got):
        # (a matrix times all the vectors at once sums in another order than times one: float32 rounding)
        assert np.allclose(logits, want[index % len(TOKENS)], rtol=2e-4, atol=2e-4), index
    # every matrix was read once for the 18 positions, not once a position
    assert len(read) == len(set(read)) == len(stores)


def test_a_job_that_fails_ends_the_pass():
    conductor = Conductor()
    try:
        conductor.run([lambda: 1, lambda: 1 / 0])
    except ZeroDivisionError:
        return
    raise AssertionError("the error of a job was lost")


# ------------------------------------------------------------------------------------------- the review of T237
def test_the_roundings_of_the_page_are_what_the_converter_makes_of_a_row():
    """The page's 8-bit rounding (Safari) is llama2_convert.quantize() on the activation (which quantize_x is to the bit,
    smoke.mjs); the 7-bit one (relaxed SIMD) is the same with 63."""
    import llama2_convert
    from reference_27b import as_page
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(256) * 10.0 ** rng.integers(-3, 3, 256)).astype(np.float32)
    x[:32] = 0.0  # a group of zeros: no scale
    q, scales = llama2_convert.quantize(x)
    want = (q.astype(np.float32) * scales[:, None]).reshape(-1)
    assert np.array_equal(as_page(127)(x), want)
    seven = as_page(63)(x).reshape(-1, 32)
    scale7 = np.abs(x.reshape(-1, 32)).max(axis=1) / np.float32(63)
    assert np.all(np.abs(seven - x.reshape(-1, 32)) <= scale7[:, None] * 0.5 * (1 + 1e-6))
    assert np.array_equal(seven[0], np.zeros(32, dtype=np.float32))
    # the integers it stands for are in -63..63
    steps = np.divide(seven, scale7[:, None], out=np.zeros_like(seven), where=scale7[:, None] > 0)
    assert np.abs(np.rint(steps)).max() <= 63


def test_a_distance_is_the_largest_difference_the_agreement_and_the_kl():
    from reference_27b import Distance, kl_of
    rng = np.random.default_rng(1)
    theirs = rng.standard_normal((3, 50)).astype(np.float32) * 4
    ours = theirs.copy()
    ours[1, 7] += 0.5
    ours[2] = theirs[2][::-1]
    same = Distance(ours[:2], theirs[:2])
    assert same.count == 2 and same.same == 2 and abs(same.worst - 0.5) < 1e-6 and same.where == 1
    assert same.kls[0] == 0.0 and same.kls[1] > 0 and same.kl_worst == same.kls[1]
    assert Distance(ours, theirs[:2]).count == 2  # the positions both have
    other = Distance(ours, theirs)
    assert other.same < 3 and len(other.close) == 3 - other.same
    # KL(P || Q) against the definition, in float64
    t, o = theirs[0].astype(np.float64), ours[2].astype(np.float64)
    p, q = np.exp(t - t.max()), np.exp(o - o.max())
    p, q = p / p.sum(), q / q.sum()
    assert abs(kl_of(ours[2], theirs[0]) - float((p * np.log(p / q)).sum())) < 1e-9
    assert kl_of(theirs[0], theirs[0]) == 0.0


def test_the_signs_a_broken_run_is_given_differ_from_the_files_where_it_says():
    import llama2_numpy
    from reference_27b import BREAKS, SIGN_BREAKS, broken_basis, signs_of
    rng = np.random.default_rng(2)
    widths = (5120, 6144, 17408)
    basis = {"block": 1024, "signs": {str(w): llama2_numpy.sign_bits(rng.choice([-1, 1], w)) for w in widths}}
    assert all(name in BREAKS for name in SIGN_BREAKS)
    for width in widths:  # the text of a width reads back as the signs it was made of
        assert llama2_numpy.sign_bits(signs_of(basis, width)) == basis["signs"][str(width)]
    for name, (width, places) in SIGN_BREAKS.items():
        broken = broken_basis(basis, name)
        flipped = np.flatnonzero(signs_of(basis, width) != signs_of(broken, width))
        assert flipped.tolist() == list(range(width))[places], name
        assert all(broken["signs"][str(w)] == basis["signs"][str(w)] for w in widths if w != width), name
    wide = broken_basis(basis, "6144 with 5120's signs")
    assert np.array_equal(signs_of(wide, 6144)[:5120], signs_of(basis, 5120))
    assert np.array_equal(signs_of(wide, 6144)[5120:], signs_of(basis, 6144)[5120:])
    last = broken_basis(basis, "last block of 5120 as the first")
    assert np.array_equal(signs_of(last, 5120)[4096:], signs_of(basis, 5120)[:1024])
    assert np.array_equal(signs_of(last, 5120)[:4096], signs_of(basis, 5120)[:4096])
    assert broken_basis(basis, "no rotation") is None
    assert not np.any(signs_of(broken_basis(basis, "no signs"), 5120) == -1)
    for unchanged in ("tiled", "embedding", "halves", "gates rotated", "output normalized twice", "epsilon 1e-5"):
        assert broken_basis(basis, unchanged) == basis
