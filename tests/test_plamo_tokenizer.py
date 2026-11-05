"""T649.1: PLaMo's tokenizer.jsonl -> the converter (convert.vocabulary's plamo()) -> tokenizer.bin -> the engine's
"plamo" kind, against PLaMo's search written out here by its definition: of every way to cut each stretch of the text
into pieces (and characters no piece begins with), the one of the least sum of round(score * 1e4), a character
without a piece 10,000,000; of equal sums, the one whose first piece is the longest, then its second, and so on.

The vocabulary is made up, of the shape of the real one (control pieces, the 256 byte pieces, pieces of four
decimals and of 0). The real file is held to PLaMo's own tokenization_plamo.py by tests/plamo_tokenizer_check.py,
by hand.
"""
import functools
import json
import random

import pytest

from conftest import detokenize
from convert.vocabulary import plamo, vocabulary_kind
from engine.tokenizer import PLAMO_CUT, PLAMO_SPECIAL, SETTINGS, Tokenizer, plamo_breaks

CONTROLS = ["<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:eos|>", "<|plamo:msg|>", "<|plamo:begin_"]
# (the piece, its score): ties of the sums that only the order of the lengths settles (日本+語 and 日+本語, which
# unigram's search reads as the latter; ab+c and a+bc), a tie that the float32 scores of tokenizer.bin add up
# otherwise than the integers do (pq+r against p+qr, 10.0191 both: as floats the latter is the less), a piece of 0
# and runs of spaces and of one character
NORMAL = [("a", -0.3), ("b", -2.0), ("c", -0.2), ("d", -2.5), ("ab", -0.1), ("bc", 0.0), ("cd", -1.2),
          ("bcd", -1.7), ("x", -5.0), ("xx", -3.0), ("xxxx", -1.0), (" ", -1.5), ("  ", -1.0), ("    ", -0.5),
          (" a", -0.8), ("a ", -0.9), ("aaaa", -1.2), ("p", -9.1205),
          ("pq", -9.9914), ("q", -2.0), ("r", -0.0277), ("qr", -0.8986), ("日", -3.1), ("本", -3.0), ("日本", -2.2), ("本語", -2.2), ("語", -3.1),
          ("<|", -4.0), ("|>", -4.0), ("plamo", -3.0), (":", -2.0), ("\n", -2.0), ("ー", -2.5)]
DRAWN = ["a", "b", "p", "q", "r", "pq", "c", "d", "ab", "x", "xx", " ", "  ", "日", "本", "語", "é", "😀", "\n", "ー", "<|plamo:msg|>",
         "<|plamo:x|>", "<|plamo:begin_", "<|", "|>", PLAMO_CUT, "<0x41>", "A"]
BREAKS = {"break_around_consecutive_spaces_threshold": 2, "break_around_repeated_chars_threshold": 4}


def rows(normal=NORMAL):
    made = [[text, 0.0, "UNKNOWN" if "unk" in text else "CONTROL"] for text in CONTROLS]
    made += [[f"<0x{byte:02X}>", 0.0, "BYTE"] for byte in range(256)]
    return made + [[text, score, "NORMAL"] for text, score in normal]


def jsonl(made):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in made).encode("utf-8") + b"\n"


def tokenizer_of(made, said=BREAKS):
    vocabulary = plamo(jsonl(made), len(made), said, True)
    return Tokenizer(vocabulary.tokenizer, len(made), **{SETTINGS[key]: value for key, value in vocabulary.options.items()}), vocabulary


def by_the_definition(text, made, breaks=BREAKS):
    """PLaMo's ids of a text, by the definition at the top"""
    ids = {row[0]: id for id, row in enumerate(made) if row[2] != "BYTE"}
    points = {row[0]: round(row[1] * 1e4) for row in made if row[2] != "BYTE"}
    byte = {int(row[0][3:5], 16): id for id, row in enumerate(made) if row[2] == "BYTE"}
    text = PLAMO_SPECIAL.sub(PLAMO_CUT + r"\1" + PLAMO_CUT, text)
    pattern = plamo_breaks({"spaces": breaks["break_around_consecutive_spaces_threshold"],
                            "repeats": breaks["break_around_repeated_chars_threshold"]})
    text = pattern.sub(PLAMO_CUT + r"\1" + PLAMO_CUT, text)
    out = []
    for stretch in text.split(PLAMO_CUT):
        @functools.cache
        def best(i):
            """(the least sum, the lengths: minus each, to be the least where they are the longest) of every way to
            cut stretch[i:], and its parts (None for a character no piece is)"""
            if i == len(stretch):
                return 0, (), ()
            ways = [(-points[stretch[i:j]], j, stretch[i:j]) for j in range(i + 1, len(stretch) + 1) if stretch[i:j] in points]
            ways += [(10_000_000, i + 1, None)]
            return min(((cost + best(j)[0], (-(j - i),) + best(j)[1], (part,) + best(j)[2]) for cost, j, part in ways),
                       key=lambda way: way[:2])

        i = 0
        for part in best(0)[2]:
            out += [ids[part]] if part is not None else [byte[b] for b in stretch[i].encode("utf-8")]
            i += len(part) if part is not None else 1
    return out


def test_the_engine_tokenizes_as_plamos_search_by_its_definition():
    made = rows()
    engine, vocabulary = tokenizer_of(made)
    assert vocabulary.options == {"tokenizer_kind": "plamo", "breaks": {"spaces": 2, "repeats": 4}}
    rng = random.Random(649)
    texts = ["abc", "abcd", "xxxxx", "xxxxxxxx", "a  b", "a    b", "日本語", "<|plamo:msg|>abc", "<|plamo:x|>ab",
             "<|plamo:begin_ab", "<0x41>", "A", "é😀", PLAMO_CUT + "ab" + PLAMO_CUT, "ーーーー", "\n\n\n\n\n", ""]
    texts += ["".join(rng.choice(DRAWN) for _ in range(rng.randint(1, 9))) for _ in range(1500)]
    for text in texts:
        want = by_the_definition(text, made)
        assert engine.encode(text) == want, text
        assert engine.encode(text, vocabulary.specials) == want, text


def test_equal_sums_go_to_the_longest_first_piece_and_the_sums_are_integers():
    """The ties of NORMAL, spelled out: unigram's search (the longest last piece) and sums of floats read them otherwise"""
    made = rows()
    engine, _ = tokenizer_of(made)
    id = {row[0]: i for i, row in enumerate(made)}
    assert engine.encode("日本語") == [id["日本"], id["語"]]
    assert engine.encode("abc") == [id["ab"], id["c"]]
    # one sum of the integers, the longest first piece; the float32 scores added up take p+qr
    assert engine.encode("pqr") == [id["pq"], id["r"]]


def test_the_cuts_byte_pieces_and_the_control_pieces():
    made = rows()
    engine, vocabulary = tokenizer_of(made)
    id = {row[0]: i for i, row in enumerate(made)}
    # the control pieces PLaMo cuts around are the specials; <|plamo:begin_ is found by the search alone
    assert vocabulary.specials == ["<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:eos|>", "<|plamo:msg|>"]
    assert engine.encode("<|plamo:begin_") == [id["<|plamo:begin_"]]
    # a <|plamo:…|> no piece is is cut around all the same: "ab" does not join its neighbours across
    assert engine.encode("a<|plamo:z|>b")[0] == id["a"] and engine.encode("a<|plamo:z|>b")[-1] == id["b"]
    # a byte piece's text is not matched: <0x41> is spelled by the other pieces (bytes), A by its byte piece
    assert id["<0x41>"] not in engine.encode("<0x41>") and engine.encode("A") == [id["<0x41>"]]
    # U+EE00 is left out wherever it is
    assert engine.encode(PLAMO_CUT) == [] and engine.encode("a" + PLAMO_CUT + "bc") == [id["a"], id["bc"]]
    # without breaks, a run of spaces and of one character is searched as any text
    plain, _ = tokenizer_of(made, {})
    assert plain.encode(" aaaa") == [id[" a"], id["a"], id["a"], id["a"]]
    assert engine.encode(" aaaa") == [id[" "], id["aaaa"]] == by_the_definition(" aaaa", made)


def test_the_engine_decodes_the_pieces_as_they_are():
    made = rows()
    engine, _ = tokenizer_of(made)
    for text in [" a b", "é😀 ab", "日本語\n", " "]:
        assert detokenize(engine, engine.encode(text), bos=1) == text
    # no dummy prefix: the space of the first piece after the BOS stays
    assert engine.decode(1, made.index([" a", -0.8, "NORMAL"]), 1) == b" a"


def test_the_converter_refuses_what_is_no_plamo_vocabulary():
    made = rows()
    with pytest.raises(ValueError, match="for each of the 256 bytes"):
        plamo(jsonl([row for row in made if row[0] != "<0x7F>"]), len(made) - 1, {}, True)
    with pytest.raises(ValueError, match=r"not \[piece, score, kind\]"):
        plamo(jsonl(made + [["q", -1.0, "USER"]]), len(made) + 1, {}, True)
    with pytest.raises(ValueError, match="four decimals"):
        plamo(jsonl(made + [["q", -0.00015, "NORMAL"]]), len(made) + 1, {}, True)
    with pytest.raises(ValueError, match="four decimals"):  # json.loads reads Infinity
        plamo(jsonl(made).replace(b"\n", b"\n", 1) + b'["q", Infinity, "NORMAL"]\n', len(made) + 1, {}, True)
    with pytest.raises(ValueError, match="twice"):
        plamo(jsonl(made + [["ab", -1.0, "NORMAL"]]), len(made) + 1, {}, True)
    with pytest.raises(ValueError, match="not a line of JSON"):
        plamo(b"[\n", 1, {}, True)
    assert vocabulary_kind("tokenizer.jsonl") == "tokenizer.jsonl" and vocabulary_kind("Tokenizer.JSONL") == "tokenizer.jsonl"
