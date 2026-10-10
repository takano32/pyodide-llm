"""engine/external.py (T359.5): ExternalForward with a plan made by hand and a stand-in for forward.js's engine. No
Llama, no checkpoint and no tokenizer is made in this file: the part is handed a plan and what makes the engine, and
that is all it reads."""
import gc
import types
import weakref

import numpy as np
import pytest

from engine.external import STEPS
from llama2_numpy import REPETITION_WINDOW, ExternalForward, Sampling


class Engine:
    """What forward.js's engine is to Python: the forward pass alone. (What an engine offers is in its class.)"""
    backend = "a stand-in"

    def __init__(self):
        self.released, self.asked = 0, []

    def bind(self, logits):
        self.asked.append(("bind", logits.size, str(logits.dtype)))
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
        return "taken"


class SizedBlocks(Blocks):
    promptBlock = 7.0  # (a number of JavaScript's)


class Steps:
    tokenBlock, answer = 4.0, None

    def generateMany(self, token, pos, recent, length, count, temperature, topp, penalty, randoms, stops):
        self.asked.append(("steps", token, pos, recent, length, count, temperature, topp, penalty, randoms, stops))
        return self.answer


class Outside:
    """What makes the engine: start(plan). It has nothing else (no size, no read): ExternalForward asks for neither."""

    def __init__(self, *offers):
        self.started = []
        self.engine = type("Engine", (*offers, Engine), {})()

    def start(self, plan):
        self.started.append(plan)
        return self.engine


class Plan(dict):
    """(a dict a weak reference can be taken to)"""


def test_it_is_made_of_a_plan_and_what_makes_the_engine():
    outside, plan = Outside(), {"vocab_size": 11}
    part = ExternalForward(plan, outside)
    # the plan is handed to start() as it is, once, and of the plan the part itself reads the vocabulary alone
    assert len(outside.started) == 1 and outside.started[0] is plan
    assert part.engine is outside.engine and part.backend == "a stand-in" and type(part.backend) is str
    assert part.logits is outside.engine.logits and part.logits.shape == (11,) and part.logits.dtype == np.float32
    assert outside.engine.asked == [("bind", 11, "float32")] and not part.logits.any()
    # an engine that offers the forward pass alone has that step alone
    assert [name for name in STEPS if hasattr(part, name)] == ["forward"]
    with pytest.raises(KeyError):
        ExternalForward({}, Outside())


def test_the_plan_is_handed_on_and_not_kept():
    """Its derived tables are megabytes of a model's context: Python keeps them only while the engine is made."""
    class Forgetful(Outside):
        def start(self, plan):
            return self.engine

    gc.disable()
    try:
        plan = Plan(vocab_size=5, derived={"freq_cis_real": bytes(64)})
        gone = weakref.ref(plan)
        part = ExternalForward(plan, Forgetful())
        del plan
        assert gone() is None and part.engine is not None
        assert all(not isinstance(value, dict) for value in vars(part).values())
    finally:
        gc.enable()


def test_forward_fills_the_one_array_and_returns_it_when_asked():
    outside = Outside()
    part = ExternalForward({"vocab_size": 16}, outside)
    logits = part.forward(5, 3)
    assert logits is part.logits and logits[8] == 1.0 and logits.sum() == 1.0
    assert part.forward(5, 4, need_logits=False) is None and part.logits[9] == 1.0  # (the engine ran all the same)
    assert part.forward(1, 1, True) is logits
    assert outside.engine.asked[1:] == [("forward", 5, 3, True), ("forward", 5, 4, False), ("forward", 1, 1, True)]
    # a step is a function of the part's own, not a method: a model takes it as it is
    assert "forward" in vars(part)


def test_a_prompts_blocks_where_the_engine_takes_them():
    outside = Outside(SizedBlocks)
    part = ExternalForward({"vocab_size": 16}, outside)
    assert [name for name in STEPS if hasattr(part, name)] == ["forward", "forward_many", "prompt_block"]
    assert part.forward_many((1, 2, 3), 9) == "taken" and part.forward_many(iter([4]), 12) == "taken"
    assert outside.engine.asked[1:] == [("many", [1, 2, 3], 9), ("many", [4], 12)]
    assert part.prompt_block() == 7 and type(part.prompt_block()) is int
    outside.engine.promptBlock = 64  # (forward.js changes it: the GPU takes more)
    assert part.prompt_block() == 64
    # an engine that takes blocks and says no size: the model's own size stays
    part = ExternalForward({"vocab_size": 16}, Outside(Blocks))
    assert [name for name in STEPS if hasattr(part, name)] == ["forward", "forward_many"]


def test_steps_on_the_gpu_return_ids_or_none():
    outside = Outside(Steps)
    part = ExternalForward({"vocab_size": 16}, outside)
    assert [name for name in STEPS if hasattr(part, name)] == ["forward", "gpu_steps", "token_block"]
    assert part.token_block() == 4 and type(part.token_block()) is int
    history = list(range(100))
    # the steps of one generation: its settings, and where its random numbers come from (one for every step, drawn
    # before the GPU is asked)
    numbers = iter((0.1, 0.2, 0.3, 0.4) * 2)
    steps = part.gpu_steps(Sampling(temperature=0.8, topp=0.9, repetition_penalty=1.2), types.SimpleNamespace(random=lambda: next(numbers)))
    # T152: the GPU did not take the steps: None, and the CPU takes them
    assert steps(99, 100, history, 4, {1}) is None
    # the ids it sampled, as Python's integers
    outside.engine.answer = np.array([4, 5, 6], dtype=np.int32)
    ids = steps(99, 100, history, 4, (1, 2))
    assert ids == [4, 5, 6] and all(type(i) is int for i in ids) and next(numbers, None) is None
    # the engine is handed the window of the history and its whole length, and lists
    assert outside.engine.asked[1:] == [
        ("steps", 99, 100, history[-REPETITION_WINDOW:], 100, 4, 0.8, 0.9, 1.2, [0.1, 0.2, 0.3, 0.4], [1]),
        ("steps", 99, 100, history[-REPETITION_WINDOW:], 100, 4, 0.8, 0.9, 1.2, [0.1, 0.2, 0.3, 0.4], [1, 2])]
    assert len(outside.engine.asked[1][3]) == REPETITION_WINDOW
    # (a history shorter than the window is handed whole)
    # (a history shorter than the window is handed whole; greedy draws no number)
    part.gpu_steps(Sampling(temperature=0.0, topp=1.0), None)(1, 2, [7, 8], 1, ())
    assert outside.engine.asked[-1][3:5] == ([7, 8], 2) and outside.engine.asked[-1][6:] == (0.0, 1.0, 1.0, [], [])
    outside.engine.tokenBlock = 0  # (the CPU is faster now)
    assert part.token_block() == 0


def test_an_engine_that_offers_everything():
    part = ExternalForward({"vocab_size": 4}, Outside(SizedBlocks, Steps))
    assert [name for name in STEPS if hasattr(part, name)] == list(STEPS)


def test_release_lets_go_once_and_answers_what_the_engine_answers():
    outside = Outside()
    part = ExternalForward({"vocab_size": 4}, outside)
    assert part.release() == "released 1"  # (forward.js's answer: a promise to wait on, T205)
    assert part.engine is None and part.logits is None
    assert part.release() is None and outside.engine.released == 1
