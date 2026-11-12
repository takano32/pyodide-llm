# T617: a tokenizer's replacements (its options' replaced): rinna's japanese-gpt-neox instruction models read "<NL>"
# for a newline, which their vocabulary has not. What goes in has the newline written as "<NL>" before anything else;
# what comes out has "<NL>" a newline again, also where it is written over several tokens ("<", "N", "L", ">": four
# pieces of their spiece.model), as generate() writes a token at a time.
import pytest

from conftest import pack_tokenizer, tiny_vocab
from engine import generation
from engine.tokenizer import Tokenizer, Unreplaced
from test_parts import Letters, stand_in_model

NEWLINE = [["\n", "<NL>"]]


def tokenizer(kind="unigram", **more):
    pieces = tiny_vocab(320)
    return Tokenizer(pack_tokenizer(pieces), len(pieces), kind=kind, **more)


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
def test_a_newline_goes_in_as_what_replaces_it(kind):
    replacing = tokenizer(kind, replaced=NEWLINE)
    plain = tokenizer(kind)
    assert replacing.encode("ab\ncd\n") == plain.encode("ab<NL>cd<NL>") != plain.encode("ab\ncd\n")


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
def test_a_tokenizer_with_no_replacements_reads_as_before(kind):
    text = "ab\ncd <NL>"
    assert tokenizer(kind, replaced=None).encode(text) == tokenizer(kind, replaced=[]).encode(text) == tokenizer(kind).encode(text)
    assert tokenizer(kind).replaced == ()


@pytest.mark.parametrize("replaced", [[["\n"]], [["\n", ""]], [["", "<NL>"]], [["\n", 1]], [["a", "b", "c"]]])
def test_a_replacement_that_is_no_pair_of_strings_is_refused(replaced):
    with pytest.raises(ValueError, match="Unsupported replaced"):
        tokenizer(replaced=replaced)


def test_what_comes_out_is_turned_back_a_piece_at_a_time():
    for pieces, said in [(["a<", "N", "L>b"], ["a", "", "\nb", ""]),
                         (["<NL>"], ["\n", ""]),
                         (["x<NL><", "NL>"], ["x\n", "\n", ""]),
                         (["<", "x"], ["", "<x", ""]),  # not the beginning of one after all: as it was
                         (["<N", "<NL>"], ["", "<N\n", ""]),
                         (["a<N"], ["a", "<N"]),  # held to the end, and written as it is there
                         (["<NL", ">"], ["", "\n", ""])]:
        out = Unreplaced((("\n", "<NL>"),))
        assert [out(piece) for piece in pieces] + [out("", final=True)] == said, pieces


class Spelled(Letters):
    """Letters with "<NL>" for a newline, as the options of rinna's entry make the real Tokenizer"""
    replaced = (("\n", "<NL>"),)

    def encode(self, text, specials=(), after=None):
        return super().encode(text.replace("\n", "<NL>"), specials, after)


def test_generate_writes_a_newline_where_the_model_wrote_it_over_four_tokens():
    answer = [ord(character) for character in "ok<NL>no<N"]
    model = stand_in_model([], answer + [0], tokenizer=Spelled())
    written = list(generation.generate(model, "q\n", steps=40, echo=False))
    assert "".join(written) == "ok\nno<N"
    # (the prompt is read with "<NL>": four tokens for its newline, then the answer)
    assert model.stats["prompt_tokens"] == len("q<NL>")
    # echoed, the prompt's newline comes out a newline too
    model = stand_in_model([], answer + [0], tokenizer=Spelled())
    assert "".join(generation.generate(model, "q\n", steps=40)) == "q\nok\nno<N"


def test_generate_calls_nothing_for_a_tokenizer_with_no_replacements(monkeypatch):
    monkeypatch.setattr(generation, "Unreplaced", None)  # (called, it would fail)
    model = stand_in_model([], [ord("a"), ord("<"), 0])
    assert "".join(generation.generate(model, "q", steps=40, echo=False)) == "a<"
