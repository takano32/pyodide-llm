"""T668: DRY's penalty (engine/sampler.py's dry_penalties(), Sampling's dry_* settings): llama.cpp's own examples
(tests/test-sampling.cpp's test_dry() at b1b770d9) and runs counted by hand, the sequence breakers, the cap of the
exponent, the Z-algorithm against the plain definition, the two samplers (the kernels' with a stand-in for its kernel:
tests/smoke.ts holds the WebAssembly one and shaders/likecpu.ts's to NumPy), and generate() handing the tokenizer's
breakers to the sampler where DRY is on and nothing new where it is off."""
import ctypes
import math
import types

import numpy as np
import pytest

from engine import document, generation
from engine.external import ON_GPU
from engine.sampler import DRY_REFUSAL, dry_cap, dry_penalties
from llama2_numpy import KernelSampler, NumpySampler, Sampling
from conftest import tiny_tokenizer
from made_up import tiny_vocab
from test_parts import VOCAB, at, kernels_in_numpy, some_logits, stand_in_model
from test_sampling import part_with_steps

F32 = lambda value: float(np.float32(value))  # noqa: E731


# --------------------------------------------------------------------------------------------- the penalty alone
@pytest.mark.parametrize("probabilities, tokens, expected, multiplier, base, allowed, last, breakers", [
    # llama.cpp's five, as test-sampling.cpp writes them (the breakers there are sequences of one token each)
    ([0.25] * 4, [0, 1], [0.25] * 4, 1.0, 1.1, 2, 4, []),
    ([0.25] * 4, [0, 1, 2, 0, 1], [0.296923, 0.296923, 0.109232, 0.296923], 1.0, 1.1, 2, 5, []),
    ([0.2] * 5, [0, 1, 3, 4, 0, 1], [0.2] * 5, 1.0, 1.1, 2, 6, [3]),
    ([0.2] * 5, [0, 1, 2, 0, 1], [0.241818, 0.241818, 0.032727, 0.241818, 0.241818], 2.0, 1.1, 2, 5, []),
    ([0.2] * 5, [0, 1, 2, 3, 4, 0, 1], [0.2] * 5, 1.0, 1.1, 4, 7, [])])
def test_llama_cpps_examples(probabilities, tokens, expected, multiplier, base, allowed, last, breakers):
    """test_dry(): logits of log(p), the penalty, then softmax (llama.cpp's tester checks to 1e-5)"""
    logits = np.log(np.array(probabilities, dtype=np.float32))
    for token, penalty in dry_penalties(tokens[-last:], frozenset(breakers), F32(multiplier), F32(base), allowed).items():
        logits[token] -= penalty
    drawn = np.exp(logits.astype(np.float64) - logits.max())
    assert np.allclose(drawn / drawn.sum(), expected, atol=1e-5)


RUNS = [5, 6, 7, 8, 5, 6, 7, 9, 5, 6, 7]


@pytest.mark.parametrize("breakers, expected", [
    # "5 6 7", the end, came twice before: then 8 and then 9 carry it on, a run of 3: 1 * 2 ** (3 - 2) each. The
    # shorter runs ("6 7", "7") end at the same places, and the longest counts
    ((), {8: 2.0, 9: 2.0}),
    # a breaker is not penalized, and the run after the latest one (7 tokens) is longer than 3
    ((8,), {9: 2.0}), ((9,), {8: 2.0}),
    # the latest breaker is a token before the end: no run of 2 after it
    ((6,), {}),
    # ... the end itself
    ((7,), {})])
def test_runs_counted_by_hand(breakers, expected):
    assert dry_penalties(RUNS, frozenset(breakers), 1.0, 2.0, 2) == expected


def test_how_the_penalty_grows_and_where_it_is_capped():
    # a run of 5 ("1 2 3 4 5" then 9 once): 0.5 * 1.75 ** 3 in float32
    tokens = [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5]
    assert dry_penalties(tokens, frozenset(), F32(0.5), F32(1.75), 2) == {9: np.float32(F32(0.5) * F32(1.75) ** 3)}
    # allowed 0: every token that came after any other (a run of 0) is penalized by the multiplier alone, and "1 2 3",
    # whose 3 ** 1e30 would be no float32, by the exponent llama.cpp caps it at: int(88.72 / ln 1e30) = 1
    capped = dry_penalties([1, 2, 3, 1, 2, 3], frozenset(), 1.0, F32(1e30), 0)
    assert capped == {1: np.float32(F32(1e30)), 2: np.float32(1.0), 3: np.float32(1.0)}
    assert int(np.float32(88.7228391) / np.float32(math.log(F32(1e30)))) == 1
    # a base of 1.000001 or less is not capped (llama.cpp's max_exponent of 0): 1.0 ** 40 is 1
    assert dry_penalties([4] * 42, frozenset(), 1.0, 1.0, 0) == {4: np.float32(1.0)}
    # as many tokens as allowed or fewer: none
    assert dry_penalties([3, 3], frozenset(), 1.0, 2.0, 2) == {} and dry_penalties([], frozenset(), 1.0, 2.0, 0) == {}


def plainly(tokens, breakers, multiplier, base, allowed):
    """The definition, walked plainly: for each earlier place, the longest run up to it that is also the end of the
    tokens (no further back than the latest breaker), and the token after it penalized by its longest"""
    count = len(tokens)
    if count <= allowed:
        return {}
    limit = next((j for j in range(count) if tokens[count - 1 - j] in breakers), count)
    if limit < allowed:
        return {}
    longest = {}
    for place in range(count - 1):  # a run ending at tokens[place], carried on by tokens[place + 1]
        run = 0
        while run <= place and tokens[place - run] == tokens[count - 1 - run]:
            run += 1
        run = min(run, limit)
        if run >= allowed:
            longest[tokens[place + 1]] = max(longest.get(tokens[place + 1], -1), run)
    most = int(np.float32(88.7228391) / np.float32(math.log(base))) if base > F32(1.000001) else 0
    return {token: np.float32(multiplier * base ** (min(run - allowed, most) if most else run - allowed))
            for token, run in longest.items() if token not in breakers}


def test_the_z_algorithm_is_the_definition():
    """Random runs of few tokens (many repeats, boxes inside boxes), breakers among them or not, every allowed length
    from 0 to 4: the same tokens and the same float32 (the power multiplied out against Python's ** here: equal for these
    small exponents)"""
    rng = np.random.default_rng(668)
    for case in range(400):
        tokens = rng.integers(0, 2 + case % 5, size=int(rng.integers(0, 60))).tolist()
        breakers = frozenset(rng.choice(8, size=case % 3, replace=False).tolist())
        for allowed in range(5):
            ours = dry_penalties(tokens, breakers, F32(0.8), F32(1.75), allowed)
            assert ours == plainly(tokens, breakers, F32(0.8), F32(1.75), allowed), (tokens, breakers, allowed)


# ------------------------------------------------------------------------------------------------- the settings
DRY_DEFAULTS = dict(dry_multiplier=0.0, dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=64,
                    dry_sequence_breakers=("\n", ":", '"', "*"))


def test_the_defaults_are_llama_cpps_and_dry_is_off():
    assert {name: getattr(Sampling(), name) for name in DRY_DEFAULTS} == DRY_DEFAULTS and not Sampling().dry
    assert Sampling(dry_multiplier=0.8).dry and not Sampling(dry_multiplier=0.8, dry_penalty_last_n=0).dry
    said = Sampling(dry_multiplier=0.8, dry_allowed_length=np.int32(3), dry_penalty_last_n=-1, dry_sequence_breakers=["\n"])
    assert said.dry_sequence_breakers == ("\n",) and type(said.dry_allowed_length) is int and said.dry_penalty_last_n == -1


@pytest.mark.parametrize("settings", [
    dict(dry_multiplier=-0.1), dict(dry_multiplier=float("nan")), dict(dry_multiplier=float("inf")),
    dict(dry_base=0.99), dict(dry_base=float("nan")), dict(dry_base=float("inf")), dict(dry_allowed_length=-1),
    dict(dry_allowed_length=2.0), dict(dry_allowed_length=True), dict(dry_penalty_last_n=-2), dict(dry_penalty_last_n=1.5),
    dict(dry_sequence_breakers="\n:"), dict(dry_sequence_breakers=("\n\n",)), dict(dry_sequence_breakers=("。",)),
    dict(dry_sequence_breakers=("",)), dict(dry_sequence_breakers=(10,))])
def test_what_a_dry_setting_cannot_be_is_refused(settings):
    with pytest.raises(ValueError) as refused:
        Sampling(**settings)
    assert str(refused.value) == DRY_REFUSAL


def test_dry_keeps_a_generation_from_the_gpu_and_says_nothing_while_off():
    """No shader has DRY (its text would move every device's key): a generation with it takes its steps on the CPU"""
    assert Sampling(temperature=0.8, dry_multiplier=0.8).beyond(ON_GPU) == ["dry_multiplier"]
    assert Sampling(dry_multiplier=0.8, dry_base=2.0).beyond(ON_GPU) == ["dry_multiplier", "dry_base"]
    # off: its other settings are nobody's business
    assert Sampling(dry_base=2.0, dry_allowed_length=5, dry_sequence_breakers=()).beyond(ON_GPU) == []
    assert Sampling(dry_multiplier=0.8, dry_penalty_last_n=0).beyond(ON_GPU) == []
    part, _ = part_with_steps()
    assert part.gpu_steps(Sampling(temperature=0.8, dry_multiplier=0.8), None) is None
    assert part.gpu_steps(Sampling(temperature=0.8, dry_base=2.0), None) is not None


# ------------------------------------------------------------------------------------------------- the samplers
def kernels_with_dry(calls=None):
    """test_parts.py's kernels_in_numpy() and DRY's kernel as simdkernel.so has it (kernels/kernel/sample.ts's dry():
    addresses and counts in, the logits written), computed by dry_penalties(): what is left to differ is KernelSampler's
    own, the buffer of the written tokens, the breakers' bytes and the addresses"""
    kernels = kernels_in_numpy()

    def dry(logits, tokens, count, breaking, multiplier, base, allowed, most, repeats, longest):
        # (the window as the kernel reads it, the breakers' bytes, the float32 a c_float makes of each)
        window = at(tokens, count, ctypes.c_int32).tolist() if count else []
        breakers = frozenset(np.flatnonzero(at(breaking, VOCAB, ctypes.c_uint8)).tolist())
        if calls is not None:
            calls.append(window)
        assert not at(longest, VOCAB, ctypes.c_int32).any()  # (the kernel's longest runs: all 0 when it is called)
        assert most == dry_cap(F32(base))
        view = at(logits, VOCAB, ctypes.c_float)
        for token, penalty in dry_penalties(window, breakers, F32(multiplier), F32(base), allowed).items():
            view[token] -= penalty

    return {**kernels, "dry": dry}


def samplers(calls=None):
    return {"NumPy": NumpySampler(), "the kernels": KernelSampler(kernels_with_dry(calls), VOCAB)}


@pytest.mark.parametrize("which", ["NumPy", "the kernels"])
def test_a_sampler_takes_the_penalty_off_after_the_others_over_what_was_written(which):
    """The written tokens alone (not the history's prompt), the latest dry_penalty_last_n of them, after the repetition
    and presence penalties (llama.cpp's chain), and then greedy picks among what is left"""
    plain = np.linspace(-1.0, 1.0, VOCAB).astype(np.float32)
    sampling = Sampling(dry_multiplier=1.0, dry_base=2.0, presence_penalty=0.25)
    # (the prompt's "5 6 7 4" would have 4 penalized too)
    history, written = [1, 5, 6, 7, 4] + RUNS, list(RUNS)
    logits = plain.copy()
    samplers()[which].drawing(sampling, None)(logits, history, written)
    expected = plain.copy()
    expected[sorted(set(RUNS))] -= np.float32(0.25)
    expected[[8, 9]] -= np.float32(2.0)
    assert np.array_equal(logits, expected)
    # the window: the latest 6 hold "6 7" once before the end, carried on by 9 alone (the latest 7 would hold "5 6 7")
    logits = plain.copy()
    samplers()[which].drawing(Sampling(dry_multiplier=1.0, dry_base=2.0, dry_penalty_last_n=6), None)(logits, history, written)
    expected = plain.copy()
    expected[9] -= np.float32(1.0)
    assert np.array_equal(logits, expected)
    # the breakers the sampler is told of
    logits = plain.copy()
    samplers()[which].drawing(Sampling(dry_multiplier=1.0, dry_base=2.0), None, frozenset({8}))(logits, history, written)
    expected[9] -= np.float32(1.0)
    assert np.array_equal(logits, expected)


def test_the_two_samplers_draw_the_same_tokens_with_dry():
    """A run of steps as generate() takes them: the written tokens growing by the token drawn (the kernels' buffer of
    them written one number a step, and grown past its 256), random logits of few tokens so that runs come"""
    numbers = np.random.default_rng(5).random(600).tolist()
    drawn = []
    for which, sampler in samplers().items():
        rng = iter(numbers)
        sampling = Sampling(temperature=0.9, dry_multiplier=0.8, dry_penalty_last_n=-1)
        draw = sampler.drawing(sampling, types.SimpleNamespace(random=lambda: next(rng)), frozenset({3}))
        history, written = [1], []
        for step in range(300):
            logits = some_logits(step % 7)  # (seven logits over and over: a model that would loop)
            token = draw(logits, history, written)
            history.append(token)
            written.append(token)
        drawn.append(written)
    assert drawn[0] == drawn[1] and len(set(drawn[0])) > 3


def test_the_kernels_are_handed_the_written_tokens_in_one_buffer():
    """KernelSampler's own: the window is the latest of the written tokens, in their order, wherever the list was
    read from before (another list, one that grew by more than one)"""
    calls = []
    draw = KernelSampler(kernels_with_dry(calls), VOCAB).drawing(Sampling(dry_multiplier=1.0, dry_penalty_last_n=5), None)
    written = [7, 8]
    draw(some_logits(0), [1], written)
    for token in range(20, 300):
        written.append(token)
        draw(some_logits(0), [1], written)
    written.extend([1, 2, 3])
    draw(some_logits(0), [1], written)
    draw(some_logits(0), [1], [4, 5])
    windows = calls
    assert windows[0] == [7, 8] and windows[-3] == list(range(295, 300)) and windows[-2] == [298, 299, 1, 2, 3]
    assert windows[-1] == [4, 5] and all(len(window) <= 5 for window in windows)


# ------------------------------------------------------------------------------------------------- generate()
class Spelled:
    """A tokenizer's stand-in that spells each token as a character, and says which hold the breakers' characters"""
    replaced = ()

    def __init__(self):
        self.asked = []

    def encode(self, text, specials=(), after=None):
        return [ord(character) for character in text]

    def decode(self, previous, token, bos=1):
        return chr(token).encode()

    def containing(self, characters):
        self.asked.append(characters)
        return frozenset(ord(character) for character in characters)


def test_generate_hands_the_breakers_where_dry_is_on():
    log = []
    model = stand_in_model(log, [ord("x"), 0], tokenizer=Spelled())
    asked = []
    model.sampler.drawing = lambda sampling, rng, breaking=None: asked.append(breaking) or (lambda *step: 0)
    list(generation.generate(model, "ab", dry_multiplier=0.8, dry_sequence_breakers=("\n", ":")))
    assert asked == [frozenset({10, 58})] and model.tokenizer.asked == [("\n", ":")]
    # off: the sampler asked as before, and the vocabulary not walked
    asked.clear()
    model.tokenizer.asked.clear()
    list(generation.generate(model, "ab", dry_base=3.0))
    assert asked == [None] and model.tokenizer.asked == []


def test_a_tokenizer_says_which_tokens_hold_a_breaker():
    """Tokenizer.containing(): each token decoded alone (a byte token as its byte), once for a tuple"""
    tokenizer = tiny_tokenizer()
    spelled = [piece for _, piece in tiny_vocab(320)]
    held = tokenizer.containing(("\n", ":"))
    assert held == {3 + 10, 3 + 58, spelled.index(b" \n"), spelled.index(b"\n")}
    assert tokenizer.containing(("\n", ":")) is held and tokenizer.containing(()) == frozenset()


def test_the_last_try_of_a_document_is_llama_cpps_dry_as_tanakhs():
    assert document.PENALIZED is not None and Sampling(**document.PENALIZED).dry
    assert Sampling(**document.PENALIZED) == Sampling(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=-1)


def test_the_kernels_without_dry_are_enough_where_it_is_off():
    """(the stand-ins of the two other kernels, as test_parts.py's: DRY's is asked for where DRY is on)"""
    sampler = KernelSampler(kernels_in_numpy(), VOCAB)
    assert sampler.drawing(Sampling(), None)(some_logits(0), [1], [2]) == int(np.argmax(some_logits(0)))
    with pytest.raises(KeyError, match="dry"):
        sampler.drawing(Sampling(dry_multiplier=0.8), None)
