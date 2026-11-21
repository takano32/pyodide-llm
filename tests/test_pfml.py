"""T665: engine/pfml.py, PLaMo 3 Translate's request put together from its parts, on a made-up vocabulary of PLaMo's
shape (tests/test_plamo_tokenizer.py's, with PFML3's marks): the texts of the model card's examples, the ids of the
whole text as the tokenizer reads it, a mark typed in a text read as its characters, and what is refused. The real
files are held to the official chat_template.jinja and PLaMo's own tokenizer by tests/pfml_check.py, by hand."""
import itertools

import pytest

from conftest import detokenize
from engine.pfml import EASY_ALIAS, EASY_JAPANESE, LANGUAGES, mark, pfml_pieces, pfml_tokens
from engine.tokenizer import Tokenizer
from test_plamo_tokenizer import CONTROLS, NORMAL, jsonl, tokenizer_of

# (bos and msg are CONTROLS' already, and <|plamo:begin_, a control piece of the real vocabulary as :plamo|> is)
MARKS = [mark(name) for name in ("tag", "key", "val", "index", "item")] + [":plamo|>"]
WORDS = [(word, -1.0 - i / 10) for i, word in enumerate(
    ["dataset", "translation", "input", "output", "lang", "form", "context", "glossary", "en", "ja", "plain", "polite",
     "zh", "-", "hans", "(", "easy", ")", "e", "l", "n", "j", "t", "o", "i", "s", "g", "m", "<", "|"])]


@pytest.fixture(scope="module")
def made():
    rows = [[text, 0.0, "UNKNOWN" if "unk" in text else "CONTROL"] for text in CONTROLS + MARKS]
    rows += [[f"<0x{byte:02X}>", 0.0, "BYTE"] for byte in range(256)]
    rows += [[text, score, "NORMAL"] for text, score in NORMAL + WORDS]
    tokenizer, vocabulary = tokenizer_of(rows)
    return tokenizer, tuple(vocabulary.specials)


def text_of(pieces):
    return "".join(piece for _, piece in pieces)


def test_the_requests_of_the_model_cards_examples():
    """the request's text in the official template's order, as the card writes its examples (after the BOS)"""
    head = "<|plamo:tag|>dataset<|plamo:msg|>translation"
    assert text_of(pfml_pieces("会議は来週の月曜日に変更されました。", "en")) == \
        head + "<|plamo:tag|>input<|plamo:msg|>会議は来週の月曜日に変更されました。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    history = [("プロジェクトの管理画面を開きます。", "Open the project dashboard."),
               ("進行中のタスクがすべて表示されます。", "It lists all active tasks.")]
    assert text_of(pfml_pieces("オーナーでフィルタリングすることもできます。", "en", source="ja", history=history)) == \
        head + "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>プロジェクトの管理画面を開きます。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Open the project dashboard." \
               "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>進行中のタスクがすべて表示されます。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>It lists all active tasks." \
               "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>オーナーでフィルタリングすることもできます。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    assert text_of(pfml_pieces("プロジェクトの管理画面を開いてください。", "en",
                               glossary=[("管理画面", "dashboard"), ("プロジェクト", "project")])) == \
        head + "<|plamo:tag|>glossary<|plamo:index|>管理画面<|plamo:item|>dashboard" \
               "<|plamo:index|>プロジェクト<|plamo:item|>project" \
               "<|plamo:tag|>input<|plamo:msg|>プロジェクトの管理画面を開いてください。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    assert text_of(pfml_pieces("かみを切りました。", "en", context="工作で折り紙を作っています。")) == \
        head + "<|plamo:tag|>context<|plamo:msg|>工作で折り紙を作っています。" \
               "<|plamo:tag|>input<|plamo:msg|>かみを切りました。" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
    # the form after the language (the template's), and the context element whenever the context is not empty: one
    # of spaces too, as the template writes it (tanakh's leaves that one out)
    assert text_of(pfml_pieces("x", "ja", form="polite", context=" ")) == \
        head + "<|plamo:tag|>context<|plamo:msg|> <|plamo:tag|>input<|plamo:msg|>x" \
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"
    assert text_of(pfml_pieces("x", "ja", context="")) == text_of(pfml_pieces("x", "ja"))


def test_the_ids_are_those_of_the_whole_text(made):
    """each mark its token and each text read alone are the ids of the whole text, which the tokenizer cuts around
    every <|plamo:…|>: what the template's text is (and the list's format, with the converter's specials)"""
    tokenizer, specials = made
    texts = ["abc", "  a b\n\nxxxx  日本語", "pq r"]
    for text, (target, source), form, context, glossary, history in itertools.product(
            texts, [("ja", None), ("en", "ja"), (EASY_JAPANESE, "en")], [None, "plain"], [None, "cd", " "],
            [(), [("ab", "c d"), ("日本", "x")]], [(), [("a", "b "), (" c", "日本語\n")]]):
        if form is not None and target != "ja":
            continue
        parts = {"source": source, "form": form, "context": context, "glossary": glossary, "history": history}
        whole = text_of(pfml_pieces(text, target, **parts))
        ids = pfml_tokens(tokenizer, text, target, **parts)
        assert ids == tokenizer.encode(whole, specials) == tokenizer.encode(whole), (text, target, parts)


def test_a_mark_typed_in_a_text_is_its_characters(made):
    """what a person typed stands for itself: a <|plamo:tag|> in it is not the mark, where the tokenizer reading the
    text whole (the list's format) makes it the mark, named a special or not"""
    tokenizer, specials = made
    tag = tokenizer.index[mark("tag").encode()]
    controls = {tokenizer.index[piece.encode()] for piece in CONTROLS + MARKS}
    for typed in [mark("tag"), "a<|plamo:tag|>b", "<|plamo:msg|><|plamo:bos|>", "<|plamo:x|>", "<|plamo:begin_", "a:plamo|>"]:
        frame, at = pfml_tokens(tokenizer, "a", "en"), pfml_tokens(tokenizer, typed, "en")
        head, tail = frame[:frame.index(tokenizer.encode("a")[0])], frame[frame.index(tokenizer.encode("a")[0]) + 1:]
        middle = at[len(head):len(at) - len(tail)]
        assert at[:len(head)] == head and at[len(at) - len(tail):] == tail
        assert not controls & set(middle), typed
        assert middle == tokenizer.encode(typed, literal=True)
        assert detokenize(tokenizer, middle, bos=-1) == typed
    assert tokenizer.encode(mark("tag")) == [tag] == tokenizer.encode(mark("tag"), specials)
    # (the cuts are the same: the stretch after a typed mark is searched alone either way)
    assert tokenizer.encode("<|plamo:tag|>ab", literal=True)[-1] == tokenizer.encode("ab")[0]


def test_only_a_plamo_tokenizer_reads_a_text_literally(made):
    tokenizer, specials = made
    with pytest.raises(ValueError, match="no specials"):
        tokenizer.encode("a", specials, literal=True)
    rows = [["<unk>", 0.0, "UNKNOWN"]]
    other = Tokenizer(tokenizer_of(rows + [[f"<0x{b:02X}>", 0.0, "BYTE"] for b in range(256)] + [["a", -1.0, "NORMAL"]])[1].tokenizer,
                      258, kind="unigram")
    with pytest.raises(ValueError, match="unigram tokenizer reads no text literally"):
        other.encode("a", literal=True)


@pytest.mark.parametrize("parts, said", [
    ({"text": "x", "target": "en", "form": "plain"}, "only a translation into ja has one"),
    ({"text": "x", "target": EASY_JAPANESE, "form": "polite"}, "only a translation into ja"),
    ({"text": "x", "target": "ja", "form": "casual"}, "A form is one of plain, polite"),
    ({"text": "x", "target": "en-us"}, "target language 'en-us'"),
    ({"text": "x", "target": "JA"}, "target language 'JA'"),
    ({"text": "x", "target": "ja", "source": "xx"}, "source language 'xx'"),
    ({"text": "x", "target": None}, "needs the language"),
    ({"text": " \n", "target": "ja"}, "The text is empty"),
    ({"text": 3, "target": "ja"}, "The text is a text"),
    ({"text": "x", "target": "ja", "context": 3}, "The context is a text"),
    ({"text": "x", "target": "ja", "glossary": [("a", " ")]}, "Glossary pair 1 is empty"),
    ({"text": "x", "target": "ja", "glossary": [("a", "b", "c")]}, "two texts"),
    ({"text": "x", "target": "ja", "history": [("a", "b"), ("", "c")]}, "Translated pair 2 is empty"),
    ({"text": "x", "target": "ja", "history": ["ab"]}, "two texts"),
    # the template's order: the texts of the messages first, then the languages, the context, the form, the glossary
    ({"text": "", "target": "xx", "form": "plain"}, "The text is empty"),
    ({"text": "x", "target": "xx", "form": "plain", "glossary": [("", "b")]}, "target language"),
    ({"text": "x", "target": "en", "form": "plain", "glossary": [("", "b")]}, "only a translation into ja"),
])
def test_what_is_refused(parts, said):
    with pytest.raises(ValueError, match=said):
        pfml_pieces(**parts)


def test_easy_japanese_is_written_ja_easy():
    """the template's: ja-easy is what PFML3 writes, and ja(easy) (the first release's) an alias of it (a form is
    still ja's only)"""
    for code in (EASY_JAPANESE, EASY_ALIAS):
        written = text_of(pfml_pieces("x", code, source=code))
        assert written.count("<|plamo:val|>ja-easy<|plamo:msg|>") == 2 and "(" not in written, code
    with pytest.raises(ValueError, match="only a translation into ja"):
        pfml_pieces("x", EASY_ALIAS, form="plain")


def test_the_languages_of_the_card():
    assert len(LANGUAGES) == 53 and list(LANGUAGES)[:4] == ["ja", "en", "zh-hans", "zh-hant"]
    assert all(code.replace("-", "").isalpha() and code == code.lower() for code in LANGUAGES)
    for code in [*LANGUAGES, EASY_JAPANESE]:
        assert text_of(pfml_pieces("x", code)).endswith(f"<|plamo:val|>{code}<|plamo:msg|>")
