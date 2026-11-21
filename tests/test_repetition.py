"""T667: generate() telling where what it writes falls into a loop (loops=True; engine/repetition.py), held to the
definition counted the slow way (each stretch walked back from its end) on test_parts.py's stand-ins: a token a
character, so the characters worth keeping are the tokens. The numbers are tanakh/plamo-translate-desktop's
(engine::repetition's tests, and their own here)."""
import random
import sys

import pytest

from engine import generation
from engine.repetition import MAX_PERIOD, MIN_LOOP_SPAN, MIN_REPEATS, kept, least_span, longest_periodic_span
from test_parts import stand_in_model

BASE = 100  # (the tokens of the stand-ins: no stop token, no BOS)


def written(prompt, answers, loops=True, seq_len=4000):
    """what generate() writes after prompt (tokens) where the sampler draws answers, and its stats"""
    model = stand_in_model([], [BASE + token for token in answers] + [0], seq_len=seq_len)
    text = "".join(generation.generate(model, [BASE + token for token in prompt], steps=0, echo=False, loops=loops))
    return [ord(character) - BASE for character in text], model.stats


def slow_loop(tokens, least):
    """the first (index, period, start) where tokens end in a loop, by the definition: for a period of at most
    MAX_PERIOD, the tokens at the end each the one a period before (walked back), with the period, at least least and
    MIN_REPEATS periods (the shortest period first); None where they never do"""
    for end in range(len(tokens)):
        for period in range(1, min(MAX_PERIOD, end) + 1):
            run = 0
            while end - run - period >= 0 and tokens[end - run] == tokens[end - run - period]:
                run += 1
            if run + period >= least and run + period >= MIN_REPEATS * period:
                return end, period, end + 1 - run - period
    return None


def slow_span(tokens):
    longest = 0
    for end in range(len(tokens)):
        for period in range(1, min(MAX_PERIOD, end) + 1):
            run = 0
            while end - run - period >= 0 and tokens[end - run] == tokens[end - run - period]:
                run += 1
            if run + period >= MIN_REPEATS * period:
                longest = max(longest, run + period)
    return longest


def test_a_short_cycle_is_a_loop_once_it_is_64_tokens():
    cycle = list(range(1, 13))
    out, stats = written([], list(range(500, 510)) + cycle * 20)
    assert len(out) == 10 + MIN_LOOP_SPAN  # (ended at the 64th token of the cycle's stretch)
    assert stats["loop"] == 10 + 12  # (kept: to the end of the cycle's first time)
    assert stats["sampled"] == len(out)  # (no stop token)


@pytest.mark.parametrize("period", [1, 2, 7, 21, 22, 50, 64, 100, 127, 128])
def test_every_period_to_128_is_told_after_three_times(period):
    cycle = list(range(1, period + 1))
    out, stats = written([], [900] + cycle * (MIN_LOOP_SPAN // period + 4))
    assert len(out) == 1 + max(MIN_LOOP_SPAN, MIN_REPEATS * period)
    assert stats["loop"] == 1 + period


def test_a_period_past_128_is_not_told():
    cycle = list(range(1, MAX_PERIOD + 2))
    out, stats = written([], cycle * 5)
    assert len(out) == len(cycle) * 5 and stats["loop"] is None


def test_short_repetition_and_text_without_cycles_are_no_loop():
    out, stats = written([], [1, 2, 1, 2, 1, 2, 1, 2, 1, 3] + [7] * 40 + list(range(10, 20)))
    assert len(out) == 60 and stats["loop"] is None
    rng = random.Random(12345)
    out, stats = written([], [rng.randrange(50) for _ in range(3000)])
    assert len(out) == 3000 and stats["loop"] is None


def test_the_prompts_own_repetition_raises_the_line():
    prompt = [i % 4 for i in range(100)]
    assert longest_periodic_span(prompt) == 100 and least_span(prompt) == 200
    # the output may repeat as much as the prompt does, but not without end
    out, stats = written(prompt, [i % 4 for i in range(150)] + [i % 4 for i in range(150, 300)])
    assert len(out) == 200 and stats["loop"] is not None
    assert least_span([1, 2, 3]) == MIN_LOOP_SPAN and longest_periodic_span([1, 2, 3, 4, 5]) == 0


def test_a_cycle_of_the_prompt_is_kept_as_many_times_as_the_prompt_has_it():
    line = [10, 11, 12, 13, 14]
    prompt = [1] + line * 30 + [2]
    out, stats = written(prompt, [7] + line * 100)
    assert stats["loop"] == 1 + 5 * 30
    # a cycle the prompt does not have keeps its first time only
    out, stats = written(prompt, [3, 4, 5, 6] * 100)
    assert stats["loop"] == 4
    assert kept(prompt, [3, 4, 5, 6] * 3, 4, 0) == 4


@pytest.mark.parametrize("seed", range(40))
def test_the_loop_is_where_the_definition_says(seed):
    """random texts with a cycle planted at a random place (of a random period, as long as the line or not), some with
    a prompt that repeats itself: generate() ends where the slow count first finds a loop, keeping what kept() says"""
    rng = random.Random(seed)
    letters = rng.choice([3, 8, 60])
    prompt = [rng.randrange(letters) for _ in range(rng.randrange(0, 80))]
    if seed % 3 == 0:
        prompt += rng.sample(range(70, 90), rng.randrange(1, 8)) * rng.randrange(3, 30)
    period = rng.choice([1, 2, 3, rng.randrange(1, 140)])
    cycle = [rng.randrange(letters) for _ in range(period)]
    answers = [rng.randrange(letters) for _ in range(rng.randrange(0, 300))] + cycle * rng.randrange(1, 8) + \
        [rng.randrange(letters) for _ in range(100)]
    out, stats = written(prompt, answers)
    least = least_span(prompt)
    assert least == max(MIN_LOOP_SPAN, 2 * slow_span(prompt))
    found = slow_loop(answers, least)
    if found is None:
        assert out == answers and stats["loop"] is None
    else:
        end, period, start = found
        assert out == answers[:end + 1]
        assert stats["loop"] == kept(prompt, out, period, start)


@pytest.mark.parametrize("seed", range(20))
def test_the_longest_span_is_the_slow_one(seed):
    rng = random.Random(seed)
    tokens = [rng.randrange(rng.choice([2, 5, 40])) for _ in range(rng.randrange(0, 200))]
    at = rng.randrange(len(tokens) + 1)
    tokens[at:at] = rng.sample(range(50, 200), rng.randrange(1, 140)) * rng.randrange(1, 5)
    assert longest_periodic_span(tokens) == slow_span(tokens)


def test_without_loops_nothing_is_told_or_kept():
    out, stats = written([], [1] * 300, loops=False)
    assert len(out) == 300 and "loop" not in stats


def test_the_work_of_a_token_does_not_grow_with_the_text():
    """the lines generate() runs for a drawn token (its own frame's: the runs of each period a step on) are as many
    from the 1,500th token to the 3,000th as from the 300th to the 1,500th (past the first MAX_PERIOD, which have fewer
    periods): a constant, about MAX_PERIOD steps"""
    rng = random.Random(7)
    answers = [rng.randrange(1000 - BASE) for _ in range(3000)]
    code = generation.generate.__code__
    lines = []

    def trace(frame, event, arg):
        if frame.f_code is code:
            return lambda frame, event, arg: lines.append(1) if event == "line" else None
        return None

    counts = []
    for length in (300, 1500, 3000):
        lines.clear()
        sys.settrace(trace)
        try:
            out, _ = written([], answers[:length])
        finally:
            sys.settrace(None)
        assert len(out) == length
        counts.append(len(lines))
    early, late = (counts[1] - counts[0]) / 1200, (counts[2] - counts[1]) / 1500
    assert abs(late - early) < 1 and late < 4 * MAX_PERIOD + 80


class Pieces:
    """a tokenizer's stand-in whose tokens are not a character each (T667 review): token 5 is no text, 6 is 3 characters,
    7 and 8 the two halves of a character of 3 bytes (7 is no text, 8 is the character), any other a character"""
    replaced = ()

    def encode(self, text, specials=(), after=None):
        return [ord(character) for character in text]

    def decode(self, previous, token, bos=1):
        return {5: b"", 6: b"abc", 7: b"\xe3\x81", 8: b"\x82"}.get(token, chr(token).encode())


def test_the_characters_kept_are_those_of_the_tokens_written_not_a_token_a_character():
    """where tokens are not a character each (no text, several, half of one), the characters kept are the text of the
    drawn tokens to the end of the cycle's first time: the prompt's tokens (their text is not written) do not count,
    and half a character is no character until it is whole"""
    cycle = [6, 5, 7, 8, 40, 5, 6]  # (3 + 0 + 0 + 1 + 1 + 0 + 3 = 8 characters)
    for prompt in ([], [30, 31, 6, 5, 32], [6] * 9):
        model = stand_in_model([], [3, 4] + cycle * 40 + [0], seq_len=4000, tokenizer=Pieces())
        text = "".join(generation.generate(model, prompt, steps=0, echo=False, loops=True))
        assert model.stats["loop"] == 2 + 8 and len(text) >= model.stats["loop"], (prompt, model.stats["loop"])
        assert text[:model.stats["loop"]] == "\x03\x04abcあ(abc"
