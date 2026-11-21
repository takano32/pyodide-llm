"""T666: engine/document.py, a long text cut into chunks of paragraphs and translated a chunk at a time with the
chunks before as its history. The cutting is counted in characters here (a token a character), where its order of
the ways to cut shows; the translation runs on a stand-in of the model's generate() and PLaMo's made-up vocabulary
(tests/test_pfml.py's), where the requests and their histories show. generate() taking the ids of a request is
held with test_parts.py's stand-ins. T667: a translation that runs away is taken back and translated again in
smaller pieces, and cut where no piece is smaller, on the same stand-in (which says how each run ended)."""
import random

import pytest

from engine import document, generation
from engine.document import (CHUNK_TOKENS, HISTORY_PAIR, HISTORY_TOKENS, LEAST, LONG, LONGER, LOOP, MORE, ROOM, SHORT,
                             Cut, Withdrawn, applied, budgets, recent, sentences_of, split, translate_document)
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


def test_the_pieces_put_together_are_counted_again_near_the_limit():
    """a count is not the sum of its pieces' (a token may join two, or cross the empty line): near the limit the whole
    is counted, and what fits as a whole is one chunk, where the sum would have cut it"""
    words = lambda said: len(said.split(" "))  # noqa: E731 (the empty line joins the word before it to the one after)
    assert words("aa bb") + words("\n\n") + words("cc dd") == 5 > words("aa bb\n\ncc dd") == 3
    assert split("aa bb\n\ncc dd", 4, words) == ("", [("aa bb\n\ncc dd", "")])
    # and the whole counted over the limit is cut where the sum would have let it through
    over = lambda said: len(said) + (4 if said[0] != "\n" and said[-1] != "\n" and "\n" in said else 0)  # noqa: E731
    one, two = "x" * 7, "y" * 7
    assert over(one) + over("\n\n") + over(two) == 16 < over(one + "\n\n" + two) == 20
    assert split(one + "\n\n" + two, 16, over) == ("", [(one, "\n\n"), (two, "")])


def test_a_line_that_ends_in_a_japanese_quote_ends_a_sentence_without_a_terminal():
    assert split("「はい」\n「いいえ」\n「たぶん」", 11, len) == \
        ("", [("「はい」\n「いいえ」", "\n"), ("「たぶん」", "")])
    # a line that ends in the quote's closing ends a sentence (lines after sentences are cut before sentences are):
    # else the cut would be after "あ", where the line ends none
    assert split("「はい」\nあ\ne", 6, len) == ("", [("「はい」", "\n"), ("あ\ne", "")])
    # (a line that ends in neither: the lines are not cut as lines that end sentences)
    assert split("あ、い\nう、え\nお、か", 5, len) == ("", [("あ、い", "\n"), ("う、え", "\n"), ("お、か", "")])


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
    writes the next of answers ((pieces, how it ended: True a stop token, False out of room, a number a loop and the
    characters of it worth keeping), a character a token)"""

    def __init__(self, tokenizer, answers, seq_len=4096):
        self.tokenizer, self.answers, self.seq_len, self.asked, self.stats = tokenizer, list(answers), seq_len, [], {}

    def generate(self, request, steps=256, echo=True, loops=False, **settings):
        assert loops  # (T667: every run of a document tells loops)
        self.asked.append((list(request), steps, echo, settings))
        pieces, ended = self.answers.pop(0)
        written = sum(len(piece) for piece in pieces)
        self.stats = {"tokens": len(request) + written, "sampled": written + (1 if ended is True else 0),
                      "prompt_tokens": len(request), "seconds": 1.0, "loop": None if isinstance(ended, bool) else ended}
        yield from pieces


def test_the_chunks_are_translated_in_order_with_the_latest_as_their_history(made):
    """each chunk a request of its own, with the chunks before and their translations (stripped) the latest first
    back as far as the budget goes; the whitespace around them written as it is; a translation that ran out of room
    (no stop token) that no finer way cuts is written, cut (T667) and not history"""
    tokenizer, _ = made
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    cost = lambda chunk, said: count(chunk) + count(said) + HISTORY_PAIR  # noqa: E731
    history_tokens = max(cost("aaaa", "AA"), cost("bbbb", "B"))
    assert history_tokens < cost("aaaa", "AA") + cost("bbbb", "B")  # (one pair fits, not two)
    text = "  aaaa\n\nbbbb\n\ncccc\n\ndddd\n"
    model = Translator(tokenizer, [(["A", "A "], True), ([" B"], True), (["C"], False), (["D"], True)])
    told = []
    pieces = list(translate_document(model, text, "ja", source="en", chunk_tokens=4, history_tokens=history_tokens,
                                     told=lambda *said: told.append(said), temperature=0.0))
    assert applied(pieces, "[CUT]") == "  AA \n\n B\n\nC[CUT]\n\nD\n"
    assert [repr(piece) for piece in pieces if not isinstance(piece, str)] == [repr(Cut(2, LONG, "cccc"))]
    assert told == [(0, 4, "aaaa"), (1, 4, "bbbb"), (2, 4, "cccc"), (3, 4, "dddd")]
    histories = [[], [("aaaa", "AA")], [("bbbb", "B")], [("bbbb", "B")]]  # (C ran out of room)
    for (request, steps, echo, settings), chunk, history in zip(model.asked, ["aaaa", "bbbb", "cccc", "dddd"], histories):
        assert request == pfml_tokens(tokenizer, chunk, "ja", source="en", history=history)
        assert steps == len(request) + 1 + LONGER * count(chunk) + MORE
        assert (echo, settings) == (False, {"temperature": 0.0})
    prompts = sum(len(request) for request, *_ in model.asked)
    assert model.stats == {"tokens": prompts + 7, "sampled": 10, "prompt_tokens": prompts, "seconds": 4.0, "chunks": 4,
                           "retries": 1, "cut": 1}


def test_a_longer_history_takes_more_pairs_the_earliest_first(made):
    tokenizer, _ = made
    model = Translator(tokenizer, [(["A"], True), (["B"], True), (["C"], True)])
    list(translate_document(model, "aaaa\n\nbbbb\n\ncccc", "en", chunk_tokens=4, history_tokens=1000,
                            context="ctx", glossary=[("a", "b")]))
    assert model.asked[2][0] == pfml_tokens(tokenizer, "cccc", "en", context="ctx", glossary=[("a", "b")],
                                            history=[("aaaa", "A"), ("bbbb", "B")])


def test_the_pieces_are_applied_in_order():
    """T667: a Withdrawn takes characters back from the end (across pieces), a Cut is the mark"""
    assert applied(["ab", "cd", Withdrawn(3), "x", Cut(0, LOOP, "s"), " ", "yz", Withdrawn(1)], "[!]") == "ax[!] y"
    assert applied(["ab", Cut(0, LONG, "s")]) == "ab"
    assert sentences_of("  Aa. Bb.\n\nCc!") == ("  ", [("Aa.", " "), ("Bb.", "\n\n"), ("Cc!", "")])
    assert sentences_of("Aa\nbb.") == ("", [("Aa\nbb.", "")])  # (a newline in a paragraph is whitespace)


def test_a_translation_that_runs_away_is_translated_again_finer_and_cut_last(made, monkeypatch):
    """T667: a chunk that runs away taken back and translated again in pieces (split() at RETRY_TOKENS), a piece that
    runs away again a sentence at a time, and a sentence that still runs away cut after its loop's first time and
    marked; what ran away is never history, and the pieces after it go on with what did not"""
    tokenizer, _ = made
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    text = "Aa. Bb.\n\nCc. Dd."
    monkeypatch.setattr(document, "RETRY_TOKENS", count("Aa. Bb."))  # (a paragraph a piece)
    model = Translator(tokenizer, [
        (["xxx", "xxx"], 3),  # the chunk: a loop
        (["yyyy"], False),  # its first piece: out of room
        (["zz", "zz"], 1),  # that piece's first sentence: a loop, of which a character is kept
        (["B"], True),  # its second sentence
        (["CD"], True)])  # the chunk's second piece
    pieces = list(translate_document(model, text, "ja", source="en", chunk_tokens=1000))
    assert [repr(piece) for piece in pieces] == [repr(piece) for piece in [
        "xxx", "xxx", Withdrawn(6), "yyyy", Withdrawn(4), "zz", "zz", Withdrawn(3), Cut(0, LOOP, "Aa."), " ", "B",
        "\n\n", "CD"]]
    assert applied(pieces, "[CUT]") == "z[CUT] B\n\nCD"
    sources = [text, "Aa. Bb.", "Aa.", "Bb.", "Cc. Dd."]
    histories = [[], [], [], [], [("Bb.", "B")]]  # (the sentence that was cut is no history)
    for (request, steps, echo, settings), source, history in zip(model.asked, sources, histories, strict=True):
        assert request == pfml_tokens(tokenizer, source, "ja", source="en", history=history)
        assert steps == len(request) + 1 + LONGER * count(source) + MORE
    assert (model.stats["chunks"], model.stats["retries"], model.stats["cut"]) == (1, 3, 1)


def test_a_way_that_cuts_nothing_is_passed_over(made):
    """a chunk that fits in RETRY_TOKENS goes straight to its sentences (translating the same text again would write
    the same), and a sentence that ran out of room is cut as far as it ran"""
    tokenizer, _ = made
    model = Translator(tokenizer, [(["xx"], False), (["A"], True), (["bbb"], False)])
    pieces = list(translate_document(model, "Aa. Bb.", "ja"))
    assert applied(pieces, "|") == "A bbb|"
    assert repr(pieces[-1]) == repr(Cut(0, LONG, "Bb."))
    assert len(model.asked) == 3


def test_a_translation_far_too_short_runs_away(made, monkeypatch):
    """a translation of less than a quarter of a text of SHORT_SOURCE tokens or more stopped far too short: tanakh's
    200, made 8 here"""
    tokenizer, _ = made
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    monkeypatch.setattr(document, "SHORT_SOURCE", 8)
    source = "aaaa bbbb cccc dddd eeee"
    assert count(source) >= 8
    short = "x" * ((count(source) - 1) // 4)
    model = Translator(tokenizer, [([short], True), ([short + "x"], True)])
    pieces = list(translate_document(model, source, "ja"))
    assert repr(pieces[-1]) == repr(Cut(0, SHORT, source)) and model.stats["cut"] == 1
    # as long as a quarter is not too short (and a text of fewer tokens is never)
    while count(source) % 4:
        source += " ffff"
    model = Translator(tokenizer, [(["x" * (count(source) // 4)], True)])
    assert applied(translate_document(model, source, "ja")) == "x" * (count(source) // 4) and model.stats["retries"] == 0
    monkeypatch.setattr(document, "SHORT_SOURCE", count(source) + 1)
    model = Translator(tokenizer, [(["x"], True)])
    assert applied(translate_document(model, source, "ja")) == "x" and model.stats["retries"] == 0


def test_the_penalized_settings_are_one_more_try_before_the_cut(made):
    """the last try (T668's DRY) with its settings over the document's: what does not run away is the translation and
    history; what still loops is cut after its first time"""
    tokenizer, _ = made
    count = lambda said: len(tokenizer.encode(said, literal=True))  # noqa: E731
    model = Translator(tokenizer, [(["xxxx"], 2), (["A"], True), (["B"], True)])
    pieces = list(translate_document(model, "Aa.\n\nBb.", "ja", chunk_tokens=max(count("Aa."), count("Bb.")),
                                     penalized={"dry": 0.8}, temperature=0.0))
    assert [repr(piece) for piece in pieces] == [repr(piece) for piece in ["xxxx", Withdrawn(4), "A", "\n\n", "B"]]
    assert [settings for *_, settings in model.asked] == [{"temperature": 0.0}, {"temperature": 0.0, "dry": 0.8},
                                                          {"temperature": 0.0}]
    assert model.asked[2][0] == pfml_tokens(tokenizer, "Bb.", "ja", history=[("Aa.", "A")])
    assert (model.stats["retries"], model.stats["cut"]) == (1, 0)
    model = Translator(tokenizer, [(["xxxx"], 2), (["yyyy"], 1)])
    assert applied(translate_document(model, "Aa.", "ja", penalized={"dry": 0.8}), "|") == "y|"


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
