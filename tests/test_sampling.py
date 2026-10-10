"""The settings of a generation's sampling as one value (T359.7): Sampling alone, with no model and no sampler; each
sampler with that value and logits and nothing else (the kernels' with stand-ins for its two kernels); the GPU's steps
with the value and a stand-in for forward.js's engine."""
import dataclasses
import types

import numpy as np
import pytest

from engine.external import ON_GPU
from engine.sampler import NARROWING
from llama2_numpy import REPETITION_WINDOW, ExternalForward, KernelSampler, NumpySampler, Sampling
from test_parts import VOCAB, Numbers, kernels_in_numpy, some_logits

# the page's names (src/page/settings.ts sends them, the URL has them) and what each is when nobody says: a second
# copy, on purpose. They are generate()'s keywords
DEFAULTS = dict(temperature=0.0, topp=0.9, repetition_penalty=1.0, top_k=0, min_p=0.0, presence_penalty=0.0)
REFUSAL = "top_k and presence_penalty are 0 or more, and min_p is from 0 to 1."


# ---------------------------------------------------------------------------------------------- the value alone
def test_the_settings_are_the_pages_names_with_generates_defaults():
    assert dataclasses.asdict(Sampling()) == DEFAULTS
    assert list(DEFAULTS) == [field.name for field in dataclasses.fields(Sampling)]
    said = dict(temperature=0.7, topp=0.8, repetition_penalty=1.3, top_k=20, min_p=0.05, presence_penalty=1.5)
    assert dataclasses.asdict(Sampling(**said)) == said


def test_a_value_cannot_be_changed_and_two_of_the_same_settings_are_one():
    sampling = Sampling(temperature=0.7, top_k=20)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sampling.temperature = 1.0
    assert sampling == Sampling(top_k=20, temperature=0.7) and sampling != Sampling(temperature=0.7)
    assert len({sampling, Sampling(top_k=20, temperature=0.7), Sampling()}) == 2


@pytest.mark.parametrize("settings", [dict(top_k=-1), dict(min_p=1.5), dict(min_p=-0.1), dict(presence_penalty=-1.0),
                                      dict(temperature=0.0, top_k=-3), dict(top_k=4, min_p=1.0001)])
def test_what_a_setting_cannot_be_is_refused_where_the_value_is_made(settings):
    with pytest.raises(ValueError) as refused:
        Sampling(**settings)
    assert str(refused.value) == REFUSAL


def test_the_edges_are_taken():
    for settings in (dict(top_k=0), dict(min_p=0.0), dict(min_p=1.0), dict(presence_penalty=0.0), dict(top_k=10 ** 6)):
        Sampling(**settings)


def test_a_top_k_is_a_whole_number_whatever_it_was_said_as():
    """(the kernel's argument is an int: a float from JavaScript would be refused at the first token)"""
    for said in (20.0, np.int32(20), np.float64(20.9)):
        top_k = Sampling(top_k=said).top_k
        assert top_k == 20 and type(top_k) is int


def test_a_setting_nobody_has_is_refused():
    with pytest.raises(TypeError, match="typical_p"):
        Sampling(typical_p=0.9)


@pytest.mark.parametrize("settings, beyond", [
    (dict(), []),
    (dict(temperature=1.0, topp=0.5, repetition_penalty=1.3), []),
    (dict(temperature=1.0, top_k=20), ["top_k"]),
    (dict(temperature=1.0, min_p=0.05), ["min_p"]),
    (dict(temperature=1.0, presence_penalty=1.5), ["presence_penalty"]),
    (dict(temperature=1.0, top_k=20, min_p=0.05, presence_penalty=1.5), ["top_k", "min_p", "presence_penalty"]),
    # greedy takes the largest logit, which a top-k and a min-p leave in
    (dict(temperature=0.0, top_k=20, min_p=0.05), []),
    # ... but a presence penalty moves the largest logit
    (dict(temperature=0.0, top_k=20, presence_penalty=1.5), ["presence_penalty"]),
])
def test_what_is_beyond_a_sampler_that_knows_three(settings, beyond):
    assert Sampling(**settings).beyond(ON_GPU) == beyond


def test_beyond_is_by_the_names_it_is_told_and_by_the_defaults():
    assert ON_GPU == ("temperature", "topp", "repetition_penalty") and NARROWING == ("topp", "top_k", "min_p")
    warm = Sampling(temperature=0.8, topp=0.5, repetition_penalty=1.2, top_k=3)
    assert warm.beyond(()) == ["temperature", "topp", "repetition_penalty", "top_k"]
    assert warm.beyond(("temperature", "top_k")) == ["topp", "repetition_penalty"]
    assert Sampling().beyond(()) == [] and Sampling(topp=0.5).beyond(()) == []  # (greedy: no nucleus)
    assert Sampling(repetition_penalty=1.2).beyond(()) == ["repetition_penalty"]  # (a penalty is greedy's too)
    # a setting said as its default is not beyond anybody
    assert Sampling(temperature=1.0, top_k=0, min_p=0.0, presence_penalty=0.0).beyond(ON_GPU) == []


@dataclasses.dataclass(frozen=True)
class WithARuleMore(Sampling):
    """a rule nobody was told of yet: a setting, whose default leaves it out"""
    typical_p: float = 1.0


def test_a_setting_a_sampler_was_never_told_of_is_beyond_it():
    """What keeps a new rule from the GPU's sampler, which has no stage for it: nobody has to remember to say so."""
    assert WithARuleMore(temperature=1.0).beyond(ON_GPU) == []
    assert WithARuleMore(temperature=1.0, typical_p=0.9).beyond(ON_GPU) == ["typical_p"]
    assert WithARuleMore(temperature=0.0, typical_p=0.9).beyond(ON_GPU) == ["typical_p"]  # (not known to narrow)
    with pytest.raises(ValueError):
        WithARuleMore(typical_p=0.9, min_p=2.0)  # (the checks are the value's, a longer one's too)


# ------------------------------------------------------------- a sampler, with the value and logits and nothing else
class Counted:
    """A Sampling whose settings say how often each was read."""

    def __init__(self, **settings):
        vars(self)["of"], vars(self)["reads"] = Sampling(**settings), []

    def __getattr__(self, name):
        self.reads.append(name)
        return getattr(self.of, name)


def samplers(calls=None):
    return {"NumPy": NumpySampler(), "the kernels": KernelSampler(kernels_in_numpy(calls), VOCAB)}


@pytest.mark.parametrize("which", ["NumPy", "the kernels"])
def test_a_sampler_reads_the_settings_once_a_generation(which):
    """Not at every token (T359.7): drawing() takes what it needs of the value, and the draws read none of it."""
    if which == "NumPy" and NumpySampler.drawing.__module__ != "engine.sampler":
        pytest.skip("NumpySampler.drawing is wrapped, and the wrapper reads two settings itself (the recorder of tests/unchanged.mjs)")
    sampling = Counted(temperature=0.8, topp=0.9, repetition_penalty=1.2, top_k=20, min_p=0.05, presence_penalty=0.5)
    draw = samplers()[which].drawing(sampling, np.random.default_rng(3))
    assert sorted(sampling.reads) == sorted(DEFAULTS)  # each once
    history, written = [1, 5], []
    for step in range(REPETITION_WINDOW + 3):
        written.append(draw(some_logits(step), history, written))
        history.append(written[-1])
    assert len(sampling.reads) == len(DEFAULTS) and len(set(written)) > 3


def recording_kernels():
    """The two kernels as what they were called with (and the largest logit for the token)."""
    calls = []

    def penalize(logits, recent, count, penalty, presence):
        calls.append(("penalize", count, penalty, presence))

    def sample(logits, size, temperature, topp, random, probabilities, order, top_k, min_p):
        calls.append(("sample", size, temperature, topp, random, top_k, min_p))
        return 7

    return calls, {"penalize": penalize, "sample": sample}


def test_the_kernels_are_handed_each_setting_in_its_place():
    """KernelSampler's own: which setting goes where in a kernel's arguments (the kernels themselves are held to NumPy
    in tests/smoke.mjs). The repetition penalty over the history and with no presence, the presence penalty over what
    was written and with a penalty of 1, the rest to sample()."""
    calls, kernels = recording_kernels()
    sampler = KernelSampler(kernels, VOCAB)
    sampling = Sampling(temperature=0.8, topp=0.6, repetition_penalty=1.2, top_k=20, min_p=0.05, presence_penalty=0.5)
    draw = sampler.drawing(sampling, Numbers([0.25, 0.75]))
    assert draw(some_logits(0), [1, 2, 3], []) == 7  # nothing written yet: no presence penalty
    assert draw(some_logits(1), [1, 2, 3, 7], [7]) == 7
    assert calls == [("penalize", 3, 1.2, 0.0), ("sample", VOCAB, 0.8, 0.6, 0.25, 20, 0.05),
                     ("penalize", 4, 1.2, 0.0), ("penalize", 1, 1.0, 0.5), ("sample", VOCAB, 0.8, 0.6, 0.75, 20, 0.05)]
    # what is left out is not asked of a kernel: no penalty, and greedy is the largest logit with no number drawn
    calls.clear()
    rng = Numbers([])
    logits = some_logits(2)
    assert sampler.drawing(Sampling(top_k=20, min_p=0.05), rng)(logits, [1, 2, 3], [3]) == int(np.argmax(logits))
    assert calls == [] and rng.drawn == 0
    assert sampler.drawing(Sampling(repetition_penalty=1.2, presence_penalty=0.5), rng)(logits, [1, 2, 3], [3]) == int(np.argmax(logits))
    assert calls == [("penalize", 3, 1.2, 0.0), ("penalize", 1, 1.0, 0.5)] and rng.drawn == 0


@pytest.mark.parametrize("which", ["NumPy", "the kernels"])
def test_the_repetition_penalty_is_the_historys_and_the_presence_penalty_what_was_written(which):
    """Each penalty on its own tokens, the latest REPETITION_WINDOW of them, and the presence penalty after the other."""
    sampler = samplers()[which]
    plain = np.linspace(-3.0, 3.0, VOCAB).astype(np.float32)
    history, written = [0, 10, VOCAB - 1, VOCAB - 2], [VOCAB - 2, 20]
    logits = plain.copy()
    sampler.drawing(Sampling(repetition_penalty=2.0, presence_penalty=0.5), None)(logits, history, written)
    expected = plain.copy()
    expected[[0, 10]] *= 2.0  # negative: multiplied
    expected[[VOCAB - 1, VOCAB - 2]] /= 2.0  # positive: divided
    expected[[VOCAB - 2, 20]] -= 0.5  # once each, after the repetition penalty
    assert np.allclose(logits, expected, rtol=1e-6) and not np.array_equal(logits, plain)
    # nothing written: no presence penalty; a penalty of 1: none
    logits = plain.copy()
    sampler.drawing(Sampling(repetition_penalty=1.0, presence_penalty=0.5), None)(logits, history, [])
    assert np.array_equal(logits, plain)
    # a token that left the window is not penalized, by either
    old = [5] + [0] * REPETITION_WINDOW
    logits = plain.copy()
    sampler.drawing(Sampling(repetition_penalty=2.0, presence_penalty=0.5), None)(logits, old, old)
    assert logits[5] == plain[5] and logits[0] == pytest.approx(plain[0] * 2.0 - 0.5)


def softmax(logits, temperature):
    probabilities = np.exp((logits.astype(np.float64) - logits.max()) / temperature)
    return probabilities / probabilities.sum()


def narrowed(logits, sampling):
    """The tokens a top-k, then a nucleus of those, then a min-p leave, each on the probabilities the one before left
    (llama.cpp's sampler chain, transformers' logits warpers), written plainly."""
    probabilities = softmax(logits, sampling.temperature)
    order = np.argsort(-probabilities, kind="stable")
    if sampling.top_k > 0:
        order = order[:sampling.top_k]
    left = probabilities[order] / probabilities[order].sum()
    if 0.0 < sampling.topp < 1.0:
        order = order[:int(np.searchsorted(np.cumsum(left), sampling.topp)) + 1]
        left = probabilities[order] / probabilities[order].sum()
    if sampling.min_p > 0.0:
        order = order[left >= sampling.min_p * left.max()]
    return set(order.tolist())


@pytest.mark.parametrize("which", ["NumPy", "the kernels"])
@pytest.mark.parametrize("settings", [dict(top_k=5, topp=0.6, min_p=0.3), dict(top_k=40, topp=0.9, min_p=0.02),
                                      dict(top_k=3, topp=1.0, min_p=0.5), dict(top_k=0, topp=0.7, min_p=0.2)])
def test_a_top_k_then_the_nucleus_then_a_min_p(which, settings):
    """The order of the three: with these logits each other order leaves another set of tokens."""
    sampling = Sampling(temperature=1.0, **settings)
    logits = (np.random.default_rng(8).standard_normal(VOCAB) * 1.5).astype(np.float32)
    sampler = samplers()[which]
    drawn = {sampler.drawing(sampling, Numbers([value]))(logits, [], []) for value in np.linspace(0.0, 1.0, 801)[:-1]}
    assert drawn == narrowed(logits, sampling)


def test_the_orders_of_the_three_differ_on_these_logits():
    """(what the test above stands on: the nucleus before the top-k, or the min-p before either, is another set)"""
    logits = (np.random.default_rng(8).standard_normal(VOCAB) * 1.5).astype(np.float32)
    probabilities = softmax(logits, 1.0)
    order = np.argsort(-probabilities, kind="stable")
    sampling = Sampling(temperature=1.0, top_k=5, topp=0.6, min_p=0.3)
    nucleus_first = order[:int(np.searchsorted(np.cumsum(probabilities[order]), 0.6)) + 1][:5]
    min_p_first = order[probabilities[order] >= 0.3 * probabilities.max()][:5]
    assert set(nucleus_first.tolist()) != narrowed(logits, sampling)
    assert set(min_p_first.tolist()) != narrowed(logits, sampling)


# --------------------------------------------------------------------- the GPU's steps, with the value and an engine
class Engine:
    backend, tokenBlock, forward, answer = "a stand-in", 4, None, None

    def __init__(self):
        self.asked = []

    def bind(self, logits):
        pass

    def generateMany(self, token, pos, recent, length, count, temperature, topp, penalty, randoms, stops):
        self.asked.append((count, temperature, topp, penalty, randoms))
        return self.answer


def part_with_steps():
    engine = Engine()
    return ExternalForward({"vocab_size": 16}, types.SimpleNamespace(start=lambda plan: engine)), engine


@pytest.mark.parametrize("settings, takes", [
    (dict(temperature=0.8, topp=0.5, repetition_penalty=1.3), True),
    (dict(temperature=0.8, top_k=20), False), (dict(temperature=0.8, min_p=0.05), False),
    (dict(temperature=0.8, presence_penalty=1.5), False),
    (dict(temperature=0.0, top_k=20, min_p=0.05), True), (dict(temperature=0.0, presence_penalty=1.5), False)])
def test_the_gpu_is_offered_a_generation_its_sampler_can_draw(settings, takes):
    """T274: the GPU's SAMPLE has a temperature, a nucleus and a repetition penalty. A generation with anything more
    gets no steps of it (it would draw without, and nothing would say so)."""
    part, engine = part_with_steps()
    steps = part.gpu_steps(Sampling(**settings), None)
    assert (steps is not None) == takes and engine.asked == []


def test_a_generation_with_a_rule_the_gpu_was_never_told_of_gets_no_steps():
    part, _ = part_with_steps()
    assert part.gpu_steps(WithARuleMore(temperature=0.8, typical_p=0.9), None) is None
    assert part.gpu_steps(WithARuleMore(temperature=0.8), None) is not None


def test_the_gpus_steps_read_the_settings_once_and_draw_a_number_a_step():
    part, engine = part_with_steps()
    sampling = Counted(temperature=0.8, topp=0.6, repetition_penalty=1.2)
    rng = Numbers([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    steps = part.gpu_steps(sampling, rng)
    read = len(sampling.reads)
    assert steps(1, 0, [1], 4, [0]) is None and steps(1, 0, [1], 3, [0]) is None
    # each setting in its place, the numbers in the order drawn, one for every step (before the GPU answered)
    assert engine.asked == [(4, 0.8, 0.6, 1.2, [0.1, 0.2, 0.3, 0.4]), (3, 0.8, 0.6, 1.2, [0.5, 0.6, 0.7])]
    assert len(sampling.reads) == read and rng.drawn == 7
    # greedy draws none
    engine.asked.clear()
    rng = Numbers([])
    assert part.gpu_steps(Sampling(repetition_penalty=1.2), rng)(1, 0, [1], 4, [0]) is None
    assert engine.asked == [(4, 0.0, 0.9, 1.2, [])] and rng.drawn == 0
