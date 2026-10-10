# The parts a Llama writes with (T359): the two samplers against each other, the forward pass outside Python and what
# it refuses, how a model is given its parts (once, with no call between generate() and a part), and that a model lets
# go of everything when it is dropped.
import ctypes
import gc
import sys
import weakref

import numpy as np
import pytest

import engine.model
from conftest import pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab
from engine import generation
from llama2_numpy import (NOT_FINITE, REPETITION_WINDOW, SEVERAL_KINDS, ExternalForward, KernelSampler, Llama,
                          NumpySampler, greedy)

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
        if calls is not None:
            calls.append(sorted(at(recent, REPETITION_WINDOW, ctypes.c_int32)[:count].tolist()))
        # (the ring holds the latest tokens in no order of time, and the penalty asks for none)
        reference.penalize(at(logits, vocab, ctypes.c_float), at(recent, REPETITION_WINDOW, ctypes.c_int32)[:count].tolist(),
                           penalty, presence)

    def sample(logits, size, temperature, topp, random, probabilities, order, top_k, min_p):
        try:
            return reference.sample(at(logits, size, ctypes.c_float), temperature, topp, Numbers([random]), top_k, min_p)
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
        for step in range(2 * REPETITION_WINDOW + 5):
            logits = some_logits(step)
            sampler.penalize(logits, history, 1.3)
            if written:
                sampler.penalize(logits, written, 1.0, 1.5)
            token = sampler.sample(logits, settings["temperature"], settings.get("topp", 0.9), rng,
                                   settings.get("top_k", 0), settings.get("min_p", 0.0))
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
    sampler.penalize(logits, history, 1.2)
    for token in range(10, 10 + REPETITION_WINDOW + 3):  # one more a step, past the window
        history.append(token)
        sampler.penalize(logits, history, 1.2)
        histories.append(list(history))
    history.extend([200, 201])  # two at once
    sampler.penalize(logits, history, 1.2)
    histories.append(list(history))
    other = [7, 8, 9]  # another list, shorter
    sampler.penalize(logits, other, 1.2)
    histories.append(other)
    same_length = list(range(100, 100 + len(history)))  # another list as long as the last but one
    sampler.penalize(logits, same_length, 1.2)
    histories.append(same_length)
    assert calls == [sorted(h[-REPETITION_WINDOW:]) for h in [list(range(10))] + histories[1:]]


def test_the_kernel_sampler_refuses_what_the_kernel_refuses_and_what_it_cannot_address():
    sampler = KernelSampler(kernels_in_numpy(), VOCAB)
    bad = some_logits(1)
    bad[17] = np.nan
    for temperature in (0.0, 0.8):
        with pytest.raises(ValueError) as refused:
            sampler.sample(bad, temperature, 0.9, Numbers([0.5]))
        assert str(refused.value) == NOT_FINITE
    with pytest.raises(ValueError) as refused:
        greedy(bad)
    assert str(refused.value) == NOT_FINITE
    with pytest.raises(TypeError, match="contiguous float32"):
        sampler.sample(some_logits(2).astype(np.float64), 0.8, 0.9, Numbers([0.5]))
    with pytest.raises(TypeError, match="contiguous float32"):
        sampler.penalize(some_logits(2)[::2], [1, 2], 1.3)
    # another array of logits after one is addressed anew (forward() returns the same one every time, a test may not)
    first, second = some_logits(3), some_logits(4)
    assert sampler.sample(first, 0.0, 0.9, Numbers([])) == int(np.argmax(first))
    assert sampler.sample(second, 0.8, 0.9, Numbers([0.3])) == NumpySampler().sample(second, 0.8, 0.9, Numbers([0.3]))


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
    assert reference.external_forward is None and reference.forward_many is None and reference.generate_many is None
    assert reference.prompt_block() == engine.model.PROMPT_BLOCK and reference.token_block() == 0
    # an engine that offers the forward pass alone
    outside = Outside(checkpoint)
    llama = Llama(None, tokenizer, external=outside)
    part = llama.external_forward
    assert isinstance(part, ExternalForward) and part.engine is outside.engine and llama.backend == "outside"
    assert vars(llama)["forward"] is part.forward and llama.forward_many is None and llama.generate_many is None
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
    ids = llama.generate_many(99, 100, history, 4, 0.8, 0.9, 1.2, (0.1, 0.2, 0.3, 0.4), [1, 2])
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
    assert llama.generate_many(1, 0, [1], 4, 0.0, 0.9, 1.0, (), [1]) is None
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
def test_the_class_has_the_reference_and_a_model_is_given_its_parts_once(monkeypatch):
    checkpoint, tokenizer = files()
    assert Llama.penalize is NumpySampler.penalize and Llama.sample is NumpySampler.sample
    assert Llama.generate is generation.generate and Llama.greedy is greedy
    reference = Llama(checkpoint, tokenizer)
    assert isinstance(reference.sampler, NumpySampler)
    assert not {"forward", "penalize", "sample", "generate", "forward_many", "generate_many"} & set(vars(reference))
    # with the kernels: their sampler's two functions, on the model itself
    monkeypatch.setattr(engine.model, "load_kernels", lambda path, relaxed: kernels_in_numpy(vocab=MODEL_VOCAB))
    for external in (None, Outside(checkpoint)):
        llama = Llama(None if external else checkpoint, tokenizer, kernels="simdkernel.so", external=external)
        assert isinstance(llama.sampler, KernelSampler)
        assert vars(llama)["penalize"] is llama.sampler.penalize and vars(llama)["sample"] is llama.sampler.sample
        without = Llama(None if external else checkpoint, tokenizer, kernels="simdkernel.so", disable=("sampler",),
                        external=Outside(checkpoint) if external else None)
        assert isinstance(without.sampler, NumpySampler) and "sample" not in vars(without)
        assert "NumPy sampling" in without.backend


@pytest.mark.parametrize("outside, kernels", [(False, False), (False, True), (True, False), (True, True)])
def test_a_step_of_generate_calls_the_parts_themselves(monkeypatch, outside, kernels):
    """T359: nothing is called between generate() and a part (the forward pass, penalize(), sample()), nor between a
    part and what it runs (forward.js's engine, a kernel): the page runs these once a token, and a call that only hands
    on to another is a cost for every token it writes. Each part is called by generate()'s own frame, and the engine
    and the kernels by a frame that generate() called."""
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
    parts = {"forward": llama.external_forward.forward if outside else Llama.forward,
             "penalize": llama.sampler.penalize if kernels else NumpySampler.penalize,
             "sample": llama.sampler.sample if kernels else NumpySampler.sample}
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
    # and generate() calls nothing of the engine's but those, the tokenizer and what the model says of its blocks
    ours = {code.co_qualname for code, caller, _ in calls if caller is written and "/engine/" in code.co_filename}
    assert ours == {parts["forward"].__qualname__, parts["penalize"].__qualname__, parts["sample"].__qualname__,
                    "Tokenizer.decode", "Llama.<lambda>"}, ours


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
