# T520: the budget of a thought in generate(), with the stand-ins of test_parts.py (no Llama, no weights).
import pytest

from engine import generation
from test_parts import Letters, stand_in_model


class Marked(Letters):
    """Letters, with the marks of a thought as one token each where they are specials (as T143 makes them)."""
    MARKS = {"<think>": 200, "</think>": 201}

    def encode(self, text, specials=()):
        tokens = []
        while text:
            mark = next((mark for mark in self.MARKS if mark in specials and text.startswith(mark)), None)
            tokens.append(self.MARKS[mark] if mark else ord(text[0]))
            text = text[len(mark):] if mark else text[1:]
        return tokens

    def decode(self, previous, token, bos=1):
        return {200: b"<think>", 201: b"</think>"}.get(token) or chr(token).encode()


def thinker(log, answers, **more):
    return stand_in_model(log, answers, **{**dict(tokenizer=Marked(), specials=("<think>", "</think>")), **more})


X, Y = ord("x"), ord("y")


@pytest.mark.parametrize("opened", [True, False])
def test_a_thought_at_its_budget_is_closed_and_the_answer_goes_on(opened):
    """A thought that reaches the budget gets its closing mark in place of the next draw, written with a forward pass
    that makes no logits; then the model draws its answer from after the mark. The format may open the thought (the
    prompt ends in <think>) or the model may: both are followed."""
    log = []
    answers = ([] if opened else [200]) + [X, X, X, Y, 0]
    model = thinker(log, answers)
    text = "".join(generation.generate(model, "a<think>\n" if opened else "a", steps=40, echo=False, thinking=3))
    assert text == ("" if opened else "<think>") + "xxx</think>y"
    assert (model.stats["thought"], model.stats["thought_cut"]) == (3, True)
    at = 6 if opened else 5  # where the mark is written
    steps = [entry for entry in log if entry[0] == "forward"]
    assert ("forward", X, at, False) in steps and ("forward", 201, at + 1, True) in steps
    assert len([entry for entry in log if entry[0] == "draw"]) == len(answers)  # the mark took no draw
    assert model.stats["sampled"] == len(answers) + 1  # every token, the mark among them (the last is the stop token)


def test_a_thought_that_ends_inside_its_budget_is_not_touched():
    """A thought shorter than the budget is the model's own, token for token, and the stats count it without its mark."""
    answers = [X, X, 201, Y, 0]
    model, unlimited = thinker([], list(answers)), thinker([], list(answers))
    text = "".join(generation.generate(model, "a<think>", steps=40, echo=False, thinking=3))
    assert text == "".join(generation.generate(unlimited, "a<think>", steps=40, echo=False)) == "xx</think>y"
    assert (model.stats["thought"], model.stats["thought_cut"]) == (2, False)
    assert (unlimited.stats["thought"], unlimited.stats["thought_cut"]) == (2, False)


def test_a_thought_that_never_closes_is_counted_whole():
    """Without a budget a thought that never closes runs to the end of the steps, and the stats count all of it."""
    model = thinker([], [X] * 50)
    assert "".join(generation.generate(model, "a<think>", steps=12, echo=False)) == "x" * 10
    assert (model.stats["thought"], model.stats["thought_cut"]) == (10, False)


def test_no_thought_where_the_model_has_no_marks_and_the_budget_is_checked():
    """A model without the marks among its specials is never cut, whatever it writes, and no thought is counted (the
    entries that do not think are written as before). A budget that is no number of tokens is refused."""
    model = stand_in_model([], [X] * 10 + [0])
    assert "".join(generation.generate(model, "a<think>", steps=40, echo=False, thinking=2)) == "x" * 10
    assert (model.stats["thought"], model.stats["thought_cut"]) == (None, False)
    for wrong in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="thinking"):
            list(generation.generate(stand_in_model([], []), "a", thinking=wrong))


def test_steps_on_the_gpu_stop_at_the_budget():
    """The GPU draws several steps at once (token_block()): no more are asked of it than the budget leaves, and the
    mark is the CPU's step after them; the answer goes back to the GPU's steps."""
    log = []

    def gpu_steps(sampling, rng):
        def steps(token, pos, history, count, stops):
            log.append(("steps", pos, count))
            return [X] * count if history[-1] != 201 else [Y, 0]

        return steps

    model = thinker(log, [], gpu_steps=gpu_steps, token_block=lambda: 4)
    text = "".join(generation.generate(model, "a<think>", steps=40, echo=False, temperature=0.9, seed=1, thinking=6))
    assert text == "xxxxxx</think>y"
    assert [entry for entry in log if entry[0] == "steps"] == [("steps", 2, 4), ("steps", 6, 2), ("steps", 9, 4)]
    assert ("forward", X, 8, False) in log and (model.stats["thought"], model.stats["thought_cut"]) == (6, True)


def test_the_marks_are_the_vocabularys_not_only_the_specials():
    """A DeepSeek-R1 distill's entry names other specials, and the model writes <think> and </think> as the tokens
    they are: the thought is followed all the same. A vocabulary without them as one piece has no marks."""
    model = thinker([], [X, X, X, Y, 0], specials=())
    assert "".join(generation.generate(model, "a<think>", steps=40, echo=False, thinking=2)) == "xx</think>xy"
    assert (model.stats["thought"], model.stats["thought_cut"]) == (2, True)
    assert (generation.mark_of(model, "</think>"), generation.mark_of(stand_in_model([], []), "</think>")) == (201, None)
