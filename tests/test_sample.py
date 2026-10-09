# The NumPy sampler: greedy, the nucleus, and the repetition penalty. (The kernels repeat these in WASM.)
import numpy as np
import pytest

from conftest import pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from llama2_numpy import NOT_FINITE, REPETITION_WINDOW, Llama


@pytest.fixture(scope="module")
def llama():
    config, weights = synthetic_weights()
    return Llama(pack_checkpoint(config, weights), pack_tokenizer(tiny_vocab(config["vocab_size"])))


class FixedRng:
    """Stands in for numpy's Generator: sample() draws exactly one number per call."""

    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


def softmax(logits, temperature=1.0):
    probabilities = np.exp((logits - logits.max()) / temperature)
    return probabilities / probabilities.sum()


def reference_nucleus(logits, temperature, topp):
    """The smallest set of most probable tokens whose probabilities reach topp."""
    probabilities = softmax(logits, temperature)
    order = np.argsort(-probabilities)
    cumulative = np.cumsum(probabilities[order])
    return set(order[:int(np.searchsorted(cumulative, topp)) + 1].tolist())


def test_greedy_is_argmax(llama):
    logits = np.random.default_rng(0).standard_normal(200).astype(np.float32)
    assert llama.sample(logits, 0.0, 0.9, FixedRng([])) == int(np.argmax(logits))


def test_nucleus_is_exactly_the_top_p_set(llama):
    logits = (np.random.default_rng(1).standard_normal(200) * 3.0).astype(np.float32)
    temperature, topp = 1.0, 0.9
    # every random number in [0, 1) leads to one token: together they are the set that can be drawn
    drawn = {llama.sample(logits, temperature, topp, FixedRng([u])) for u in np.linspace(0.0, 1.0, 1001)[:-1]}
    assert drawn == reference_nucleus(logits, temperature, topp)


def test_a_peaked_distribution_leaves_one_candidate(llama):
    logits = np.full(200, -20.0, dtype=np.float32)
    logits[42] = 20.0
    assert reference_nucleus(logits, 1.0, 0.9) == {42}
    assert {llama.sample(logits, 1.0, 0.9, FixedRng([u])) for u in (0.0, 0.5, 0.999)} == {42}


def test_frequencies_follow_the_distribution(llama):
    logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32)
    expected = softmax(logits, 1.0)
    draws = 4000
    rng = np.random.default_rng(3)
    counts = np.bincount([llama.sample(logits, 1.0, 1.0, rng) for _ in range(draws)], minlength=logits.size)
    sigma = np.sqrt(draws * expected * (1 - expected))
    assert (np.abs(counts - draws * expected) < 5 * sigma).all(), counts


def test_a_low_temperature_sharpens(llama):
    logits = np.array([2.0, 1.9, 0.0], dtype=np.float32)
    cold = [llama.sample(logits, 0.01, 1.0, FixedRng([u])) for u in (0.1, 0.5, 0.9)]
    assert cold == [0, 0, 0]


def test_penalize_divides_positive_and_multiplies_negative_logits(llama):
    logits = np.array([2.0, -2.0, 5.0, -5.0], dtype=np.float32)
    llama.penalize(logits, [0, 1, 1], 2.0)
    assert logits == pytest.approx([1.0, -4.0, 5.0, -5.0])


def test_penalize_looks_only_at_the_last_tokens(llama):
    logits = np.ones(4, dtype=np.float32)
    history = [3] + [0] * REPETITION_WINDOW
    llama.penalize(logits, history, 2.0)
    assert logits == pytest.approx([0.5, 1.0, 1.0, 1.0])  # token 3 fell out of the window


@pytest.mark.parametrize("topp", [0.05, 0.1, 0.2, 0.45])
@pytest.mark.parametrize("above", [1, 2, 3, 5])
def test_a_low_top_p_over_few_likely_tokens(llama, topp, above):
    """T178: with a few tokens above the floor and n * topp < 1, llama2.c's cutoff (1 - topp) / (n - 1) was above all
    of them, nothing was left and sample() read past the end (IndexError). The nucleus is still the top-p set."""
    rng = np.random.default_rng(above)
    for spread in (0.0, 0.05, 0.5):
        logits = np.full(200, -100.0, dtype=np.float32)
        logits[rng.choice(200, above, replace=False)] = (5.0 + rng.standard_normal(above) * spread).astype(np.float32)
        for temperature in (0.1, 0.2, 1.0):
            drawn = {llama.sample(logits, temperature, topp, FixedRng([u])) for u in np.linspace(0.0, 1.0, 101)[:-1]}
            nucleus = reference_nucleus(logits, temperature, topp)
            # equal logits: any of them is the most probable one (the sort decides which)
            assert drawn <= set(np.flatnonzero(logits == logits.max()).tolist()) if spread == 0.0 else drawn == nucleus


@pytest.mark.parametrize("where", [0, 7, 199])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("temperature, topp", [(1.0, 0.9), (0.7, 1.0), (0.0, 0.9)])
def test_logits_that_are_not_finite_stop(llama, where, bad, temperature, topp):
    """T195: a NaN or +inf anywhere (and all -inf) leaves no distribution to draw from: a broken model or an overflow.
    sample() says so, as the kernel's does (tests/smoke.mjs), rather than draw a token."""
    logits = np.random.default_rng(where).standard_normal(200).astype(np.float32)
    logits[where] = bad
    for broken in (logits, np.full(200, -np.inf, dtype=np.float32)):
        with pytest.raises(ValueError) as refused:
            llama.sample(broken, temperature, topp, FixedRng([0.5]))
        assert str(refused.value) == NOT_FINITE


def test_some_minus_infinity_is_no_fault(llama):
    """T195: tokens at -inf cannot be drawn; the others are drawn as before."""
    logits = np.random.default_rng(3).standard_normal(200).astype(np.float32)
    logits[::3] = -np.inf
    for topp in (0.9, 1.0):
        drawn = {llama.sample(logits, 1.0, topp, FixedRng([u])) for u in np.linspace(0.0, 1.0, 51)[:-1]}
        assert all(np.isfinite(logits[token]) for token in drawn)
    assert llama.sample(logits, 0.0, 0.9, FixedRng([])) == int(np.argmax(logits))


def reference_narrowed(logits, temperature, topp, top_k, min_p):
    """T274: the tokens left by a top-k, then a nucleus of those, then a min-p: each step on the probabilities the one
    before left, adding up to one again (llama.cpp's sampler chain, transformers' logits warpers), written plainly."""
    probabilities = softmax(logits.astype(np.float64), temperature)
    order = np.argsort(-probabilities, kind="stable")
    if top_k > 0:
        order = order[:top_k]
    left = probabilities[order] / probabilities[order].sum()
    if 0.0 < topp < 1.0:
        order = order[:int(np.searchsorted(np.cumsum(left), topp)) + 1]
        left = probabilities[order] / probabilities[order].sum()
    if min_p > 0.0:
        order = order[left >= min_p * left.max()]
    return order


@pytest.mark.parametrize("top_k", [1, 2, 20, 199, 200, 500])
@pytest.mark.parametrize("topp", [1.0, 0.95, 0.5])
@pytest.mark.parametrize("min_p", [0.0, 0.05, 0.5, 1.0])
def test_top_k_then_the_nucleus_then_min_p(llama, top_k, topp, min_p):
    logits = (np.random.default_rng(top_k).standard_normal(200) * 2.0).astype(np.float32)
    for temperature in (0.6, 1.0):
        left = reference_narrowed(logits, temperature, topp, top_k, min_p)
        draws = [llama.sample(logits, temperature, topp, FixedRng([u]), top_k, min_p) for u in np.linspace(0.0, 1.0, 2001)[:-1]]
        # each as often as its share of what is left, and none that is not left (the least likely of 200 fall
        # between the 2000 numbers tried)
        share = softmax(logits.astype(np.float64), temperature)[left]
        share /= share.sum()
        assert set(draws) <= set(left.tolist()), (temperature, sorted(set(draws)), sorted(left.tolist()))
        assert set(draws) >= set(left[share > 1e-3].tolist()), (temperature, sorted(set(draws)), sorted(left.tolist()))
        counts = np.array([draws.count(int(token)) for token in left]) / len(draws)
        assert np.abs(counts - share).max() < 2e-3, (temperature, counts, share)


def test_min_p_alone_keeps_the_order_of_the_index(llama):
    logits = np.array([0.0, 3.0, -9.0, 2.9, 1.0], dtype=np.float32)
    # 0.5 of the most probable: tokens 1 and 3, token 1 first (the walk is the index's without a top-k or a nucleus)
    assert [llama.sample(logits, 1.0, 1.0, FixedRng([u]), 0, 0.5) for u in (0.0, 0.4, 0.6, 0.999)] == [1, 1, 3, 3]


def test_greedy_ignores_top_k_and_min_p(llama):
    logits = np.random.default_rng(5).standard_normal(200).astype(np.float32)
    assert llama.sample(logits, 0.0, 0.9, FixedRng([]), 3, 0.5) == int(np.argmax(logits))


def test_a_presence_penalty_is_taken_off_once(llama):
    logits = np.array([2.0, -2.0, 5.0, -5.0], dtype=np.float32)
    llama.penalize(logits, [0, 1, 1, 1], 1.0, 1.5)
    assert logits == pytest.approx([0.5, -3.5, 5.0, -5.0])
    llama.penalize(logits, [0], 2.0, 1.5)  # after the repetition penalty
    assert logits == pytest.approx([-1.25, -3.5, 5.0, -5.0])
    llama.penalize(logits, [3] + [2] * REPETITION_WINDOW, 1.0, 1.0)  # token 3 fell out of the window
    assert logits == pytest.approx([-1.25, -3.5, 4.0, -5.0])


@pytest.mark.parametrize("settings", [dict(top_k=-1), dict(min_p=1.5), dict(min_p=-0.1), dict(presence_penalty=-1.0)])
def test_generate_refuses_settings_out_of_range(llama, settings):
    with pytest.raises(ValueError):
        list(llama.generate("a", steps=4, temperature=1.0, **settings))


def test_generate_with_the_three_reproduces_and_differs(llama):
    """T274: a seed reproduces with them, a top-k of 1 is greedy's text, and a presence penalty changes a greedy one."""
    def text(**settings):
        return "".join(llama.generate("a", steps=24, seed=7, **settings))
    narrowed = dict(temperature=1.0, topp=0.95, top_k=20, min_p=0.05, presence_penalty=1.5)
    assert text(**narrowed) == text(**narrowed)
    assert text(temperature=1.0, topp=1.0, top_k=1) == text(temperature=0.0)
    assert text(temperature=0.0, presence_penalty=100.0) != text(temperature=0.0)
    # (greedy with a large presence penalty writes no token of the last 64 twice)
    ids = llama.tokenizer.encode(text(temperature=0.0, presence_penalty=100.0), llama.specials)
    assert len(ids) > 4


@pytest.mark.parametrize("settings, on_the_gpu", [
    (dict(), True),
    (dict(top_k=20), False),
    (dict(min_p=0.05), False),
    (dict(presence_penalty=1.5), False),
    (dict(repetition_penalty=1.3), True),
    # greedy takes the largest logit, which a top-k and a min-p leave in: the GPU may still take its steps
    (dict(temperature=0.0, top_k=20, min_p=0.05), True),
    # ... but a presence penalty moves the largest logit
    (dict(temperature=0.0, presence_penalty=1.5), False),
])
def test_only_a_step_without_the_three_may_go_to_the_gpu(llama, monkeypatch, settings, on_the_gpu):
    """T274 (the review): the GPU's SAMPLE has no top-k, min-p or presence penalty, so a step with any of them must
    not be offered to it (it would draw without them, and nothing says so). Nothing else caught that: the GPU's steps
    are forward.js's, and pytest has no GPU (a presence penalty alone passed both pytest and smoke with on_cpu left
    out of it)."""
    asked = []

    def offered(token, pos, history, count, temperature, topp, penalty, randoms, stops):
        asked.append(count)
        return None  # "the CPU takes the step"

    monkeypatch.setattr(llama, "token_block", lambda: 4)
    monkeypatch.setattr(llama, "generate_many", offered)
    list(llama.generate("a", steps=12, **{"temperature": 1.0, "seed": 3, **settings}))
    assert bool(asked) == on_the_gpu, (settings, asked)
