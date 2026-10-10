# The parts a Llama writes with (T359): the two samplers against each other, the forward pass outside Python and what
# it refuses, how a model is given its parts (once, with no call between generate() and a part), and that a model lets
# go of everything when it is dropped.
import ctypes
import gc
import sys
import types
import weakref
from unittest import mock

import numpy as np
import pytest

import engine.model
from conftest import pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from engine import generation
from llama2_numpy import (NOT_FINITE, REPETITION_WINDOW, SEVERAL_KINDS, ExternalForward, KernelSampler, Llama,
                          NumpySampler, Sampling, greedy)

VOCAB = 300


class Numbers:
    """Stands in for NumPy's Generator: the numbers a sampler draws, one for each call, and how many it drew."""

    def __init__(self, values):
        self.values, self.drawn = list(values), 0

    def random(self):
        self.drawn += 1
        return self.values.pop(0)


def at(address, count, kind):
    return np.ctypeslib.as_array(ctypes.cast(address, ctypes.POINTER(kind)), (count,))


def kernels_in_numpy(calls=None, vocab=VOCAB):
    """The two sampling kernels as simdkernel.so has them (kernels/kernel/sample.ts: addresses and counts in, a token
    or -1 out), computed by NumpySampler: ctypes cannot load the WebAssembly ones here (tests/smoke.mjs holds those to
    NumPy in Pyodide). What is left to differ is KernelSampler's own: the ring of the latest tokens, the addresses,
    the random numbers, greedy and the refusal."""
    reference = NumpySampler()

    def penalize(logits, recent, count, penalty, presence):
        # (the ring holds the latest tokens in no order of time, and the penalty asks for none)
        tokens = at(recent, REPETITION_WINDOW, ctypes.c_int32)[:count].tolist()
        if calls is not None:
            calls.append(sorted(tokens))
        penalties = reference.drawing(Sampling(repetition_penalty=penalty, presence_penalty=presence), None)
        try:
            penalties(at(logits, vocab, ctypes.c_float), tokens, tokens)
        except ValueError:  # (greedy's, after the penalties: the kernel draws nothing)
            pass

    def sample(logits, size, temperature, topp, random, probabilities, order, top_k, min_p):
        settings = Sampling(temperature=temperature, topp=topp, top_k=top_k, min_p=min_p)
        try:
            return reference.drawing(settings, Numbers([random]))(at(logits, size, ctypes.c_float), [], [])
        except ValueError:
            return -1

    return {"penalize": penalize, "sample": sample}


def some_logits(seed):
    return (np.random.default_rng(seed).standard_normal(VOCAB) * 3.0).astype(np.float32)


@pytest.mark.parametrize("settings", [dict(temperature=0.0), dict(temperature=0.8, topp=0.9), dict(temperature=1.0, topp=1.0),
                                      dict(temperature=0.7, topp=0.8, top_k=20), dict(temperature=1.0, topp=0.95, min_p=0.05),
                                      dict(temperature=0.9, topp=0.5, top_k=7, min_p=0.1)])
def test_the_two_samplers_draw_the_same_token_from_the_same_numbers(settings):
    """A run of steps as generate() takes them: one history that grows by the token drawn, past the window of the
    repetition penalty, both penalties, and one random number a step (none when greedy)."""
    in_numpy, on_kernels = NumpySampler(), KernelSampler(kernels_in_numpy(), VOCAB)
    numbers = np.random.default_rng(11).random(3 * REPETITION_WINDOW).tolist()
    drawn = []
    for sampler in (in_numpy, on_kernels):
        rng, history, written, tokens = Numbers(numbers), [1, 5, 9], [], []
        draw = sampler.drawing(Sampling(repetition_penalty=1.3, presence_penalty=1.5, **settings), rng)
        for step in range(2 * REPETITION_WINDOW + 5):
            token = draw(some_logits(step), history, written)
            tokens.append(token)
            history.append(token)
            written.append(token)
        assert rng.drawn == (0 if settings["temperature"] == 0.0 else len(tokens))
        drawn.append(tokens)
    assert drawn[0] == drawn[1]
    assert len(set(drawn[0])) > 3  # (not one token over and over: the penalties and the numbers did something)


def test_the_kernel_sampler_penalizes_the_latest_tokens_whatever_the_history_was():
    """Its ring follows a history that grew by one token; any other history (another list, one that grew by more, one
    that shrank) is read again, the latest REPETITION_WINDOW of it."""
    calls = []
    sampler = KernelSampler(kernels_in_numpy(calls), VOCAB)
    logits = some_logits(0)
    history = list(range(10))
    histories = [history]
    draw = sampler.drawing(Sampling(repetition_penalty=1.2), None)
    draw(logits, history, [])
    for token in range(10, 10 + REPETITION_WINDOW + 3):  # one more a step, past the window
        history.append(token)
        draw(logits, history, [])
        histories.append(list(history))
    history.extend([200, 201])  # two at once
    draw(logits, history, [])
    histories.append(list(history))
    other = [7, 8, 9]  # another list, shorter
    draw(logits, other, [])
    histories.append(other)
    same_length = list(range(100, 100 + len(history)))  # another list as long as the last but one
    draw(logits, same_length, [])
    histories.append(same_length)
    assert calls == [sorted(h[-REPETITION_WINDOW:]) for h in [list(range(10))] + histories[1:]]


def test_the_kernel_sampler_refuses_what_the_kernel_refuses_and_what_it_cannot_address():
    sampler = KernelSampler(kernels_in_numpy(), VOCAB)
    bad = some_logits(1)
    bad[17] = np.nan
    for temperature in (0.0, 0.8):
        with pytest.raises(ValueError) as refused:
            sampler.drawing(Sampling(temperature=temperature), Numbers([0.5]))(bad, [], [])
        assert str(refused.value) == NOT_FINITE
    with pytest.raises(ValueError) as refused:
        greedy(bad)
    assert str(refused.value) == NOT_FINITE
    with pytest.raises(TypeError, match="contiguous float32"):
        sampler.drawing(Sampling(temperature=0.8), Numbers([0.5]))(some_logits(2).astype(np.float64), [], [])
    with pytest.raises(TypeError, match="contiguous float32"):
        sampler.drawing(Sampling(repetition_penalty=1.3), None)(some_logits(2)[::2], [1, 2], [])
    # another array of logits after one is addressed anew (forward() returns the same one every time, a test may not)
    first, second = some_logits(3), some_logits(4)
    assert sampler.drawing(Sampling(), Numbers([]))(first, [], []) == int(np.argmax(first))
    warm = Sampling(temperature=0.8)
    assert sampler.drawing(warm, Numbers([0.3]))(second, [], []) == NumpySampler().drawing(warm, Numbers([0.3]))(second, [], [])


# ---------------------------------------------------------------------------------------- the forward pass outside
class Engine:
    """What forward.js's engine is to Python: the forward pass alone. (What an engine offers is in its class, not set
    on it: a method kept on its own object would be a cycle of this test's making.)"""
    backend = "outside"

    def __init__(self):
        self.released, self.asked = 0, []

    def bind(self, logits):
        self.logits = logits

    def forward(self, token, pos, need_logits):
        self.asked.append(("forward", token, pos, need_logits))
        self.logits[:] = 0.0
        self.logits[(token + pos) % self.logits.size] = 1.0

    def release(self):
        self.released += 1
        return f"released {self.released}"


class Blocks:
    def forwardMany(self, tokens, pos):
        self.asked.append(("many", tokens, pos))


class SizedBlocks(Blocks):
    promptBlock = 7


class Steps:
    tokenBlock, answer = 4, None

    def generateMany(self, token, pos, recent, length, count, temperature, topp, penalty, randoms, stops):
        self.asked.append(("steps", token, pos, recent, length, count, temperature, topp, penalty, randoms, stops))
        return self.answer


OFFERS = {"blocks": SizedBlocks, "blocks without a size": Blocks, "steps": Steps}


class Outside:
    def __init__(self, data, offers=()):
        self.data, self.size, self.plan = data, len(data), None
        self.engine = type("Engine", (*(OFFERS[offer] for offer in offers), Engine), {})()

    def read(self, offset, length):
        return self.data[offset:offset + length]

    def start(self, plan):
        self.plan = plan
        return self.engine


MODEL_VOCAB = synthetic_weights()[0]["vocab_size"]


def files():
    config, weights = synthetic_weights()
    return pack_checkpoint(config, weights), pack_tokenizer(tiny_vocab(config["vocab_size"]))


def test_a_model_outside_runs_by_what_its_engine_offers():
    checkpoint, tokenizer = files()
    reference = Llama(checkpoint, tokenizer)
    assert reference.external_forward is None and reference.forward_many is None and reference.gpu_steps is None
    assert reference.prompt_block() == engine.model.PROMPT_BLOCK and reference.token_block() == 0
    # an engine that offers the forward pass alone
    outside = Outside(checkpoint)
    llama = Llama(None, tokenizer, external=outside)
    part = llama.external_forward
    assert isinstance(part, ExternalForward) and part.engine is outside.engine and llama.backend == "outside"
    assert vars(llama)["forward"] is part.forward and llama.forward_many is None and llama.gpu_steps is None
    assert llama.prompt_block() == engine.model.PROMPT_BLOCK and llama.token_block() == 0
    logits = llama.forward(5, 3)
    assert logits is part.logits is outside.engine.logits and logits[8] == 1.0 and logits.dtype == np.float32
    assert llama.forward(5, 4, need_logits=False) is None
    assert outside.engine.asked == [("forward", 5, 3, True), ("forward", 5, 4, False)]
    # one that takes a prompt's blocks, with and without a size of its own
    outside = Outside(checkpoint, ("blocks",))
    llama = Llama(None, tokenizer, external=outside)
    llama.forward_many((1, 2, 3), 9)
    assert outside.engine.asked == [("many", [1, 2, 3], 9)] and llama.prompt_block() == 7
    llama = Llama(None, tokenizer, external=Outside(checkpoint, ("blocks without a size",)))
    assert llama.forward_many is not None and llama.prompt_block() == engine.model.PROMPT_BLOCK
    # one that takes steps on the GPU: the window of the history and its length, and its ids as Python's integers
    outside = Outside(checkpoint, ("steps",))
    llama = Llama(None, tokenizer, external=outside)
    assert llama.token_block() == 4 and llama.forward_many is None
    history = list(range(100))
    outside.engine.answer = np.array([4, 5, 6], dtype=np.int32)
    steps = llama.gpu_steps(Sampling(temperature=0.8, topp=0.9, repetition_penalty=1.2), Numbers([0.1, 0.2, 0.3, 0.4]))
    ids = steps(99, 100, history, 4, [1, 2])
    assert ids == [4, 5, 6] and all(type(i) is int for i in ids)
    assert outside.engine.asked == [("steps", 99, 100, history[-REPETITION_WINDOW:], 100, 4, 0.8, 0.9, 1.2,
                                     [0.1, 0.2, 0.3, 0.4], [1, 2])]


def test_a_step_the_gpu_gives_back_is_none_and_the_cpu_takes_it():
    """T152: generateMany() answers nothing where the GPU did not take the steps; generate() then runs that step by
    forward() and the sampler."""
    checkpoint, tokenizer = files()
    outside = Outside(checkpoint, ("steps",))
    llama = Llama(None, tokenizer, external=outside)
    llama.stop_tokens = {-1}
    assert llama.gpu_steps(Sampling(), None)(1, 0, [1], 4, [1]) is None
    outside.engine.asked.clear()
    list(llama.generate("", steps=3, temperature=0.0))
    kinds = [asked[0] for asked in outside.engine.asked]
    assert kinds == ["steps", "forward"] * 3 and llama.stats["sampled"] == 3


def test_what_cannot_run_outside_is_refused():
    checkpoint, tokenizer = files()
    with pytest.raises(ValueError, match="in Python"):
        Llama(None, tokenizer, external=Outside(checkpoint), disable=("kernels",))
    with pytest.raises(ValueError, match="asks for"):
        Llama(None, tokenizer, external=Outside(checkpoint + b"\0\0\0\0"))
    with pytest.raises(ValueError) as refused:
        Llama(None, tokenizer, external=Outside(checkpoint), kinds={"wq": "float16"})
    assert str(refused.value) == SEVERAL_KINDS
    # (and the plan was asked of none of them)
    outside = Outside(checkpoint + b"\0")
    with pytest.raises(ValueError):
        Llama(None, tokenizer, external=outside)
    assert outside.plan is None


def test_release_lets_go_of_the_engine_once():
    checkpoint, tokenizer = files()
    outside = Outside(checkpoint)
    llama = Llama(None, tokenizer, external=outside)
    assert llama._external[0] is outside.engine and llama._external[1] is llama.external_forward.logits
    assert llama.release() == "released 1"  # (forward.js's answer: a promise to wait on, T205)
    assert llama._external is None and llama.external_forward.engine is None and llama.external_forward.logits is None
    assert llama.release() is None and outside.engine.released == 1
    assert Llama(checkpoint, tokenizer).release() is None  # nothing outside: nothing to let go of


# -------------------------------------------------------------------------------- how a model is given its parts
def unwrapped():
    """tests/unchanged_recorder.py puts its own functions around the class's names to write down what they return: what
    these two tests look at (which function a name is, whose frame calls it) is then the recorder's."""
    if Llama.generate is not generation.generate:
        pytest.skip("the names of the class are wrapped (the recorder of tests/unchanged.mjs)")


def test_the_class_has_the_reference_and_a_model_is_given_its_parts_once(monkeypatch):
    unwrapped()
    checkpoint, tokenizer = files()
    assert isinstance(Llama.sampler, NumpySampler)
    assert Llama.generate is generation.generate and Llama.greedy is greedy
    reference = Llama(checkpoint, tokenizer)
    assert reference.sampler is Llama.sampler
    assert not {"forward", "sampler", "generate", "forward_many", "gpu_steps"} & set(vars(reference))
    # with the kernels: their sampler, on the model itself
    monkeypatch.setattr(engine.model, "load_kernels", lambda path, relaxed: kernels_in_numpy(vocab=MODEL_VOCAB))
    for external in (None, Outside(checkpoint)):
        llama = Llama(None if external else checkpoint, tokenizer, kernels="simdkernel.so", external=external)
        assert isinstance(vars(llama)["sampler"], KernelSampler)
        without = Llama(None if external else checkpoint, tokenizer, kernels="simdkernel.so", disable=("sampler",),
                        external=Outside(checkpoint) if external else None)
        assert without.sampler is Llama.sampler and "sampler" not in vars(without)
        assert "NumPy sampling" in without.backend


@pytest.mark.parametrize("outside, kernels", [(False, False), (False, True), (True, False), (True, True)])
def test_a_step_of_generate_calls_the_parts_themselves(monkeypatch, outside, kernels):
    """T359: nothing is called between generate() and a part (the forward pass, the sampler's draw), nor between a
    part and what it runs (forward.js's engine, a kernel): the page runs these once a token, and a call that only hands
    on to another is a cost for every token it writes. Each part is called by generate()'s own frame, and the engine
    and the kernels by a frame that generate() called."""
    unwrapped()
    checkpoint, tokenizer = files()
    kernels_made = []
    monkeypatch.setattr(engine.model, "load_kernels",
                        lambda path, relaxed: kernels_made.append(kernels_in_numpy(vocab=MODEL_VOCAB)) or kernels_made[-1])
    started = Outside(checkpoint) if outside else None
    outside_engine = started and started.engine
    llama = Llama(None if outside else checkpoint, tokenizer, external=started, kernels="simdkernel.so" if kernels else None)
    llama.stop_tokens = {-1}
    calls = []

    def profile(frame, event, arg):
        if event == "call":
            caller = frame.f_back
            calls.append((frame.f_code, caller.f_code if caller else None, caller.f_back.f_code if caller and caller.f_back else None))

    generator = llama.generate("", steps=6, temperature=0.8, repetition_penalty=1.2, seed=1)
    sys.setprofile(profile)
    try:
        list(generator)
    finally:
        sys.setprofile(None)
    written = generation.generate.__code__
    # (every draw a sampler makes is one function's: its code is what a step runs)
    parts = {"forward": llama.external_forward.forward if outside else Llama.forward,
             "draw": llama.sampler.drawing(Sampling(), None)}
    for name, part in parts.items():
        callers = [caller for code, caller, _ in calls if code is part.__code__]
        assert callers and all(caller is written for caller in callers), name
    # what the parts run: the engine outside one call below generate(), a kernel one below too (the stand-ins of both
    # are Python's here, so they are seen)
    below = ([outside_engine.forward.__code__] if outside else []) + \
        ([kernel.__code__ for kernel in kernels_made[-1].values()] if kernels else [])
    for code in below:
        through = [(caller, further) for called, caller, further in calls if called is code]
        assert through and all(further is written for _, further in through), code.co_qualname
    # and generate() calls nothing of the engine's but those, the tokenizer, and the sampler once for how it draws
    ours = {code.co_qualname for code, caller, _ in calls if caller is written and "/engine/" in code.co_filename}
    assert ours == {parts["forward"].__qualname__, parts["draw"].__qualname__, llama.sampler.drawing.__qualname__,
                    "Tokenizer.decode"}, ours
    assert [code for code, caller, _ in calls if caller is written].count(llama.sampler.drawing.__code__) == 1


# ------------------------------------------------------------------------------------- a dropped model is gone
def rotated_model():
    """a made-up Qwen3.5 folded into a rotated basis: its turned() is a function made for the model"""
    from conftest import basis, folded, qwen35_model
    from test_rotated import conversion, widths_of
    tensors, config = qwen35_model()
    said, signs = basis(8, widths_of(config))
    made = conversion(folded(tensors, 8, signs), config, said, vocab_size=config["text_config"]["vocab_size"])
    return Llama(bytes(made.checkpoint), made.tokenizer, **{key: value for key, value in made.options.items() if key != "template"})


@pytest.mark.parametrize("kind", ["NumPy", "NumPy with the kernels' sampler", "outside", "outside with the kernels' sampler",
                                  "NumPy, a rotated basis"])
def test_a_dropped_model_is_freed_without_a_collection(monkeypatch, kind):
    """A Llama holds the weights (NumPy) or the engine and the array JavaScript fills (outside). Nothing it holds may
    hold it back: a cycle of references keeps all of it until Python's collector runs, which in the page is a model's
    worth of memory at the moment the next one is read (AGENTS.md: the worker collects, and should not have to)."""
    checkpoint, tokenizer = files()
    monkeypatch.setattr(engine.model, "load_kernels", lambda path, relaxed: kernels_in_numpy(vocab=MODEL_VOCAB))
    kernels = "simdkernel.so" if "sampler" in kind else None
    gc.collect()
    gc.disable()
    try:
        if "rotated" in kind:
            llama = rotated_model()
            assert llama.rotated is not None
            for pos, token in enumerate((1, 5, 9)):
                llama.forward(token, pos)
            watched = [llama, llama.wq, llama.delta_state]
        elif kind.startswith("outside"):
            outside = Outside(checkpoint, ("blocks", "steps"))
            llama = Llama(None, tokenizer, external=outside, kernels=kernels)
            llama.stop_tokens = {-1}
            list(llama.generate("ab", steps=8, temperature=0.8, repetition_penalty=1.2, seed=2))
            watched = [llama, llama.external_forward, outside.engine, llama.external_forward.logits]
            assert llama.release() == "released 1"
            del outside
        else:
            llama = Llama(checkpoint, tokenizer, kernels=kernels)
            llama.stop_tokens = {-1}
            list(llama.generate("ab", steps=8, temperature=0.8, repetition_penalty=1.2, seed=2))
            watched = [llama, llama.key_cache, llama.wq]
        if kernels:
            watched += [llama.sampler, llama.sampler.buffers[0]]
        gone = [weakref.ref(thing) for thing in watched]
        del llama, watched
        assert [ref() for ref in gone] == [None] * len(gone)
    finally:
        gc.enable()


# ------------------------------------------------------------------------- each part alone, with stand-ins for the rest
class Letters:
    """A tokenizer's stand-in: a token a character, by its code."""

    def encode(self, text, specials=()):
        return [ord(character) for character in text]

    def decode(self, previous, token, bos=1):
        return chr(token).encode()


def stand_in_model(log, answers, stop_tokens=(0,), **more):
    """What generate() writes with, each part a stand-in that says it was called: no Llama, no weights, no kernels.
    answers: the token the sampler draws at each step."""
    answers = list(answers)

    def forward(token, pos, need_logits=True):
        log.append(("forward", token, pos, need_logits))
        return np.full(4, float(pos), dtype=np.float32) if need_logits else None

    def drawing(sampling, rng):
        log.append(("drawing", sampling, type(rng).__name__))

        def draw(logits, history, written):
            log.append(("draw", float(logits[0]), list(history), list(written)))
            return answers.pop(0)

        return draw

    return types.SimpleNamespace(**{**dict(
        tokenizer=Letters(), specials=(), bos=1, stop_tokens=set(stop_tokens), seq_len=64, stats={}, _run=0, forward=forward,
        sampler=types.SimpleNamespace(drawing=drawing), forward_many=None, prompt_block=lambda: 16, gpu_steps=None,
        token_block=lambda: 0), **more})


def test_generate_writes_with_stand_ins_for_every_part():
    """generate() alone: the sampler asked once how it draws, by the settings as one value; the prompt forced without
    logits, then forward and a draw a step, with every token so far and the sampled ones, until a stop token."""
    log = []
    model = stand_in_model(log, [ord("x"), ord("y"), 0, ord("z")])
    settings = dict(temperature=0.5, topp=0.8, repetition_penalty=1.2, top_k=3, min_p=0.1, presence_penalty=0.5)
    pieces = list(generation.generate(model, "ab", steps=20, seed=4, **settings))
    a, b, x, y = ord("a"), ord("b"), ord("x"), ord("y")
    assert "".join(pieces) == "abxy" and list(generation.generate(stand_in_model([], [x, 0]), "ab", echo=False, temperature=1.0)) == ["x"]
    assert log == [
        ("drawing", Sampling(**settings), "Generator"), ("forward", 1, 0, False), ("forward", a, 1, False),
        ("forward", b, 2, True), ("draw", 2.0, [1, a, b], []),
        ("forward", x, 3, True), ("draw", 3.0, [1, a, b, x], [x]),
        ("forward", y, 4, True), ("draw", 4.0, [1, a, b, x, y], [x, y])]
    assert (model.stats["tokens"], model.stats["sampled"], model.stats["prompt_tokens"]) == (4, 3, 2)
    with pytest.raises(ValueError, match="only 1 fit"):
        list(generation.generate(stand_in_model([], []), "ab", steps=2))
    with pytest.raises(ValueError, match="top_k"):
        list(generation.generate(stand_in_model([], []), "ab", min_p=2.0))


def test_generate_hands_blocks_and_steps_to_stand_ins():
    """generate() alone, with a forward pass that takes the prompt in blocks and steps several at a time: the blocks
    at their positions, the steps asked once for the settings and the random numbers, a block given back taken by
    forward and the sampler's draw, a stop token inside a block ending the run."""
    log = []
    answers = iter([[ord("p"), ord("q")], None, [ord("r"), 0, ord("s")]])

    def many(tokens, pos):
        log.append(("many", list(tokens), pos))

    def gpu_steps(sampling, rng):
        log.append(("gpu steps", sampling, type(rng).__name__))

        def steps(token, pos, history, count, stops):
            log.append(("steps", token, pos, list(history), count, list(stops)))
            return next(answers)

        return steps

    model = stand_in_model(log, [ord("c")], stop_tokens=(0, 7), forward_many=many, prompt_block=lambda: 2,
                           gpu_steps=gpu_steps, token_block=lambda: 2)
    assert "".join(generation.generate(model, "abc", steps=30, temperature=0.9, seed=1)) == "abcpqcr"
    a, b, c, p, q, r = (ord(letter) for letter in "abcpqr")
    assert log == [
        ("many", [1, a], 0), ("many", [b], 2),
        ("drawing", Sampling(temperature=0.9), "Generator"), ("gpu steps", Sampling(temperature=0.9), "Generator"),
        ("steps", c, 3, [1, a, b, c], 2, [0, 7]),
        ("steps", q, 5, [1, a, b, c, p, q], 2, [0, 7]),  # given back:
        ("forward", q, 5, True), ("draw", 5.0, [1, a, b, c, p, q], [p, q]),
        ("steps", c, 6, [1, a, b, c, p, q, c], 2, [0, 7])]
    assert model.stats["sampled"] == 5 and model.stats["prompt_tokens"] == 3


@pytest.mark.parametrize("nothing", [[], (), None])
def test_steps_that_answer_no_tokens_are_the_cpus(nothing):
    """T390: an empty answer of the GPU's steps is no token to go on from, and generate() asked for the same step
    again, for ever. It is "not taken", as None is: the CPU takes the step."""
    log = []

    def gpu_steps(sampling, rng):
        def steps(token, pos, history, count, stops):
            log.append(("steps", pos, count))
            assert len(log) < 40, "generate() asks for the same step again and again"
            return nothing

        return steps

    model = stand_in_model(log, [ord("x"), ord("y"), 0], gpu_steps=gpu_steps, token_block=lambda: 4)
    assert "".join(generation.generate(model, "a", steps=20, temperature=0.9, seed=1)) == "axy"
    assert model.stats["sampled"] == 3
    assert [entry[:2] for entry in log if entry[0] in ("steps", "draw")] == [
        ("steps", 1), ("draw", 1.0), ("steps", 2), ("draw", 2.0), ("steps", 3), ("draw", 3.0)]


def test_the_numpy_forward_alone_computes_the_reference():
    """The forward pass in NumPy with no tokenizer and no sampler: a Llama made with a stand-in where it makes its
    tokenizer (Llama makes that itself, from its bytes: what the forward pass is still tied to)."""
    from conftest import naive_logits
    config, weights = synthetic_weights()
    with mock.patch.object(engine.model, "Tokenizer", lambda *arguments, **named: None):
        llama = Llama(pack_checkpoint(config, weights), None)
    assert llama.tokenizer is None
    tokens = [1, 5, 9, 5, 7]
    expected = naive_logits(config, weights, tokens)
    for pos, token in enumerate(tokens):
        assert np.allclose(llama.forward(token, pos), expected[pos], rtol=1e-4, atol=1e-4), pos
