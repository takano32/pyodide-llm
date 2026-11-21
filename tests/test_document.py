"""T666: engine/document.py, a long text cut into chunks of paragraphs and translated a chunk at a time with the
chunks before as its history. The cutting is counted in characters here (a token a character), where its order of
the ways to cut shows; the translation runs on a stand-in of the model's generate() and PLaMo's made-up vocabulary
(tests/test_pfml.py's), where the requests and their histories show. generate() taking the ids of a request is
held with test_parts.py's stand-ins."""
import random

import pytest

from engine import generation
from engine.document import (CHUNK_TOKENS, HISTORY_PAIR, HISTORY_TOKENS, LEAST, LONGER, MORE, ROOM, budgets, recent,
                             split, translate_document)
from engine.pfml import pfml_tokens
from engine.tokenizer import WHITE_SPACE
from test_parts import stand_in_model
from test_thinking import thinker
from test_pfml import made  # noqa: F401 (the fixture)


def joined(cut):
    leading, chunks = cut
    return leading + "".join(chunk + separator for chunk, separator in chunks)


def test_a_short_text_is_one_chunk_with_what_lies_around_it():
    assert split("Hello.\n\nWorld.\n", 100, len) == ("", [("Hello.\n\nWorld.", "\n")])
    assert split("  \n\n Hello.", 100, len) == ("  \n\n ", [("Hello.", "")])
    # a text of only whitespace is that whitespace, and no chunk
    assert split(" \n\t\n", 100, len) == (" \n\t\n", [])
    assert split("", 100, len) == ("", [])


def test_paragraphs_are_put_together_as_far_as_they_fit():
    text = "aaaa\n\nbbbb\n \t\ncccc\n\n\ndddd"
    assert split(text, 11, len) == ("", [("aaaa\n\nbbbb", "\n \t\n"), ("cccc\n\n\ndddd", "")])
    assert split(text, 4, len) == ("", [("aaaa", "\n\n"), ("bbbb", "\n \t\n"), ("cccc", "\n\n\n"), ("dddd", "")])
    # (a line of other whitespace is no empty line: one paragraph, cut at its newline as a line)
    assert split("aaaa\n　\nbbbb", 6, len) == ("", [("aaaa", "\n　\n"), ("bbbb", "")])


def test_a_paragraph_too_long_is_cut_at_a_line_that_ends_a_sentence_first():
    """a line that ends a sentence before a sentence in a line, and before a line that ends none"""
    assert split("aaaa. bbbb\ncccc.\ndddd", 12, len) == \
        ("", [("aaaa.", " "), ("bbbb\ncccc.", "\n"), ("dddd", "")])
    # where the line after a sentence is the cut that fits, the sentences inside it are not cut
    assert split("aa. bb.\ncc. dd.", 8, len) == ("", [("aa. bb.", "\n"), ("cc. dd.", "")])
    # (cut at sentences first, the parts would be put together across the line: "aaaaaa. b.\nc." fits)
    assert split("aaaaaa. b.\nc. dddddd.", 13, len) == ("", [("aaaaaa. b.", "\n"), ("c. dddddd.", "")])


def test_then_at_sentences_with_what_closes_after_them():
    assert split("今日は晴れ。「明日は？」と聞いた。それから帰った", 11, len) == \
        ("", [("今日は晴れ。", ""), ("「明日は？」と聞いた。", ""), ("それから帰った", "")])
    assert split("今日は晴れ。「明日は？」と聞いた。それから帰った", 10, len) == \
        ("", [("今日は晴れ。", ""), ("「明日は？」", ""), ("と聞いた。", ""), ("それから帰った", "")])
    assert split('He said "Stop." Then he left.', 16, len) == ("", [('He said "Stop."', " "), ("Then he left.", "")])
    # a Latin terminal ends a sentence only before whitespace: not in 3.14
    assert split("Pi is 3.14 or so. Yes it is.", 18, len) == ("", [("Pi is 3.14 or so.", " "), ("Yes it is.", "")])
    assert split("Pi is 3.14 or so. Yes it is.", 10, len) == ("", [("Pi is 3.14", " "), ("or so.", " "), ("Yes it is.", "")])


def test_then_at_lines_then_at_clauses_then_at_characters():
    # no sentence: lines
    assert split("aaaa bbbb\ncccc dddd", 10, len) == ("", [("aaaa bbbb", "\n"), ("cccc dddd", "")])
    # (cut at the spaces first, the parts would be put together across the line)
    assert split("aaaaaa b\nc dddddd", 10, len) == ("", [("aaaaaa b", "\n"), ("c dddddd", "")])
    # no line: after a comma and at whitespace
    assert split("aaaa、bbbb cccc,dddd", 9, len) == ("", [("aaaa、bbbb", " "), ("cccc,dddd", "")])
    assert split("あいうえお、かきくけこ、さしすせそ", 7, len) == \
        ("", [("あいうえお、", ""), ("かきくけこ、", ""), ("さしすせそ", "")])
    # nothing: characters, seven eighths of the budget each
    cut = split("x" * 30, 8, len)
    assert cut == ("", [("x" * 7, "")] * 4 + [("xx", "")])


@pytest.mark.parametrize("seed", range(40))
def test_the_chunks_are_the_text_again_and_fit(seed):
    """any text of words, sentences, lines and paragraphs, Japanese and English, with whitespace of every kind: the
    chunks and what lies between them are the text, a chunk fits, has no whitespace at its ends, and what lies
    between them is whitespace"""
    rng = random.Random(seed)
    words = ["word", "the", "日本語", "かな", "3.14", "e.g.", "「はい」", '"yes"', "(so)", "x" * 40, "あ" * 25]
    marks = ["", "", "", ".", "。", "!", "？", "、", ",", ";", ".”", "。」"]
    gaps = [" ", " ", "  ", "\n", "\n\n", "\n \n", "\t", "　", "\r\n", "\n\n\n "]
    text = "".join(rng.choice(" \n\t") * rng.randrange(3) for _ in range(1)) + "".join(
        rng.choice(words) + rng.choice(marks) + rng.choice(gaps) for _ in range(rng.randrange(1, 120)))
    for most in (1, 3, 8, 20, 64, 500):
        cut = split(text, most, len)
        assert joined(cut) == text
        leading, chunks = cut
        assert all(char in WHITE_SPACE for char in leading)
        for chunk, separator in chunks:
            assert 0 < len(chunk) <= most, (most, chunk)
            assert chunk[0] not in WHITE_SPACE and chunk[-1] not in WHITE_SPACE
            assert all(char in WHITE_SPACE for char in separator)


def test_the_history_is_the_latest_that_fit():
    pairs = [("a", "A"), ("b", "B"), ("c", "C")]
    assert recent(pairs, [5, 5, 5], 10) == [("b", "B"), ("c", "C")]
    assert recent(pairs, [5, 5, 5], 15) == pairs
    assert recent(pairs, [5, 5, 5], 4) == []
    # (from the latest back until one does not fit: not an earlier one that would)
    assert recent(pairs, [1, 9, 5], 10) == [("c", "C")]
    assert recent([], [], 10) == []


def test_the_budgets_are_shares_of_what_the_context_leaves():
    assert budgets(len, 4096) == (min((4096 - ROOM) // 6, CHUNK_TOKENS), min((4096 - ROOM) // 3, HISTORY_TOKENS)) == (640, 1280)
    assert budgets(len, 2048) == ((2048 - ROOM) // 6, (2048 - ROOM) // 3)
    assert budgets(len, 2048, context="c" * 100, glossary=[("ab", "xyz")]) == \
        ((2048 - ROOM - 100 - 7) // 6, (2048 - ROOM - 100 - 7) // 3)
    assert budgets(len, 4096, context="c" * 5000) == (LEAST // 6, LEAST // 3)
    assert budgets(len, 4096, chunk_tokens=100, history_tokens=200) == (100, 200)


class Translator:
    """the stand-in of a PLaMo 3 Translate: PLaMo's made-up tokenizer and a generate() that says what it was asked and
    writes the next of answers ((pieces, whether a stop token ended it))"""

    def __init__(self, tokenizer, answers, seq_len=4096):
        self.tokenizer, self.answers, self.seq_len, self.asked, self.stats = tokenizer, list(answers), seq_len, [], {}

    def generate(self, request, steps=256, echo=True, **settings):
        self.asked.append((list(request), steps, echo, settings))
        pieces, stopped = self.answers.pop(0)
        written = sum(len(piece) for piece in pieces)
        self.stats = {"tokens": len(request) + written, "sampled": written + (1 if stopped else 0),
                      "prompt_tokens": len(request), "seconds": 1.0}
        yield from pieces


def test_the_chunks_are_translated_in_order_with_the_latest_as_their_history(made):
    """each chunk a request of its own, with the chunks before and their translations (stripped) the latest first
    back as far as the budget goes; the whitespace around them written as it is; a translation that ran out of room
    (no stop token) is written but is not history"""
    tokenizer, _ = made
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    cost = lambda chunk, said: count(chunk) + count(said) + HISTORY_PAIR  # noqa: E731
    history_tokens = max(cost("aaaa", "AA"), cost("bbbb", "B"))
    assert history_tokens < cost("aaaa", "AA") + cost("bbbb", "B")  # (one pair fits, not two)
    text = "  aaaa\n\nbbbb\n\ncccc\n\ndddd\n"
    model = Translator(tokenizer, [(["A", "A "], True), ([" B"], True), (["C"], False), (["D"], True)])
    told = []
    out = "".join(translate_document(model, text, "ja", source="en", chunk_tokens=4, history_tokens=history_tokens,
                                     told=lambda *said: told.append(said), temperature=0.0))
    assert out == "  AA \n\n B\n\nC\n\nD\n"
    assert told == [(0, 4, "aaaa"), (1, 4, "bbbb"), (2, 4, "cccc"), (3, 4, "dddd")]
    histories = [[], [("aaaa", "AA")], [("bbbb", "B")], [("bbbb", "B")]]  # (C ran out of room)
    for (request, steps, echo, settings), chunk, history in zip(model.asked, ["aaaa", "bbbb", "cccc", "dddd"], histories):
        assert request == pfml_tokens(tokenizer, chunk, "ja", source="en", history=history)
        assert steps == len(request) + 1 + LONGER * count(chunk) + MORE
        assert (echo, settings) == (False, {"temperature": 0.0})
    assert model.stats["chunks"] == 4 and model.stats["unfinished"] == 1 and model.stats["seconds"] == 4.0


def test_a_longer_history_takes_more_pairs_the_earliest_first(made):
    tokenizer, _ = made
    model = Translator(tokenizer, [(["A"], True), (["B"], True), (["C"], True)])
    list(translate_document(model, "aaaa\n\nbbbb\n\ncccc", "en", chunk_tokens=4, history_tokens=1000,
                            context="ctx", glossary=[("a", "b")]))
    assert model.asked[2][0] == pfml_tokens(tokenizer, "cccc", "en", context="ctx", glossary=[("a", "b")],
                                            history=[("aaaa", "A"), ("bbbb", "B")])


def test_generate_takes_the_ids_of_a_request_as_it_takes_their_text():
    """T666: a prompt of ids goes to the model as the ids its text is read as"""
    texts, ids = [], []
    for prompt, log in (("ab", texts), ([ord("a"), ord("b")], ids)):
        model = stand_in_model(log, [ord("x"), 0])
        assert "".join(generation.generate(model, prompt, steps=20)) == "abx"
        assert (model.stats["tokens"], model.stats["prompt_tokens"]) == (3, 2)
    assert texts == ids
    # an id that is no token (negative: another row of the table, or past it) is refused before any is read
    for ids in ([-1], [ord("a"), 1000]):
        with pytest.raises(ValueError, match="no token of the vocabulary"):
            list(generation.generate(stand_in_model([], [0]), ids, steps=20))
    # ids that end in the opening mark of a thought open it, as a text that ends in the mark does
    X = ord("x")
    for prompt in ("a<think>", [ord("a"), 200]):
        model = thinker([], [X, X, X, 0])
        assert "".join(generation.generate(model, prompt, steps=40, echo=False, thinking=2)) == "xx</think>x"
        assert (model.stats["thought"], model.stats["thought_cut"]) == (2, True)
