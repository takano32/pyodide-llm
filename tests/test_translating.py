# T648.4: engine/translating.py, a translator as the page's worker has a model: generate() writes the translation of
# each sentence a piece a step, sets the sentences apart (the source's lines, or the target's space), cuts a sentence
# past 127 pieces into stretches, and leaves a Llama's stats. The made-up model is tests/test_translation.py's; the real
# ones (make models' translations/, or MARIAN_FILES) are read where they are found.
import re

import pytest

from engine.translating import PIECES, Translation, translation
from test_marian import found
from test_translation import tokenizer, translator


def made_up(sentences=None, space=" ", seed=2):
    """The made-up model of a seed that writes words (seed 1 writes byte pieces that are no UTF-8)."""
    return Translation(translator(seed=seed), "int8, NumPy", sentences, space)


def joined(answering, prompt, **settings):
    pieces = answering.generate(prompt, **settings)
    return "".join(pieces)


@pytest.mark.parametrize("seed", [1, 2])
def test_one_sentence_is_the_translators_text(seed):
    answering = made_up(seed=seed)
    for text in ["a b", "b a c", "c"]:
        assert joined(answering, text) == answering.translator.translate(text)
        assert joined(answering, text)


def test_the_settings_of_an_answer_are_not_read():
    answering = made_up()
    assert joined(answering, "a b", temperature=1.3, topp=0.5, seed=7, steps=3, echo=True) == answering.translator.translate("a b")


def test_sentences_are_translated_apart_and_set_apart():
    # (a sentence ends with c here; as Intl.Segmenter does, every character is in a sentence)
    cut = lambda text: re.findall(r"[^c]*c?\s*", text)[:-1]
    answering = made_up(cut, space="+")
    one, two = (answering.translator.translate(text) for text in ("a b c", "c"))
    assert one and two and one != two
    assert joined(answering, "a b c c") == f"{one}+{two}"
    assert joined(answering, "a b c\n\nc\n") == f"{one}\n\n{two}"
    assert joined(answering, "\n  a b c") == one, "the lines before the first sentence are not written"
    assert joined(made_up(cut, space=""), "a b c c") == one + two
    assert joined(answering, " \n ") == ""


def test_a_long_sentence_is_cut_into_stretches():
    answering = made_up()
    text = " ".join(["a b c"] * 60)  # (more than PIECES pieces)
    pieces = tokenizer().encode(text)
    assert len(pieces) > PIECES
    stretches = answering.stretches(text)
    assert [len(ids) for ids in stretches] == [PIECES + 1, len(pieces) - PIECES + 1]
    assert all(ids[-1] == answering.translator.eos[0] for ids in stretches)


def test_the_stats_are_a_llamas():
    answering = made_up()
    written = list(answering.generate("a b"))
    stats = answering.stats
    assert set(stats) == {"tokens", "seconds", "tokens_per_second", "sampled", "prompt_tokens", "prompt_seconds",
                          "prompt_tokens_per_second", "first_token_seconds", "thought", "thought_cut"}
    ids = answering.translator.source_ids("a b")
    assert stats["prompt_tokens"] == len(ids)
    assert stats["tokens"] == len(ids) + len(answering.translator.greedy(ids))
    assert stats["sampled"] >= stats["tokens"] - stats["prompt_tokens"]
    assert stats["thought"] is None and stats["thought_cut"] is False
    assert written


def test_an_abandoned_answer_does_not_write_the_stats_of_a_newer_one():
    answering = made_up()
    first = answering.generate("a b")
    next(first, None)
    joined(answering, "c")
    newer = dict(answering.stats)
    first.close()
    assert answering.stats == newer


# ---- the real models, where make models put them

REAL = {
    "en-ja 3.0": (("model.enja.intgemm.alphas.bin", 43_849_787), ("srcvocab.enja.spm", 796_275), ("trgvocab.enja.spm", 827_144),
                  "", ["Hello. ", "How are you?"]),
    "ja-en 3.1": (("model.jaen.intgemm.alphas.bin", 43_977_787), ("vocab.jaen.spm", 1_443_222), ("vocab.jaen.spm", 1_443_222),
                  " ", ["こんにちは。", "今日はいい天気ですね。"]),
}

SHORTLISTS = {"en-ja 3.0": ("lex.50.50.enja.s2t.bin", 4_128_360), "ja-en 3.1": ("lex.50.50.jaen.s2t.bin", 9_348_172)}


def test_a_language_it_does_not_know_is_refused():
    with pytest.raises(ValueError, match="No translation into 'fr'"):
        translation(b"", b"", b"", "fr")


@pytest.mark.parametrize("which", REAL)
def test_a_real_model_answers_as_its_translator(which):
    model, source, target, space, sentences = REAL[which]
    read = lambda file: found(*file).read_bytes()
    language = "ja" if which.startswith("en") else "en"
    answering = translation(read(model), read(source), read(target), language, sentences=lambda text: sentences)
    assert answering.space == space
    assert answering.backend == "int8, NumPy"  # (pytest has no kernels)
    alone = [answering.translator.translate(sentence) for sentence in sentences]
    assert joined(answering, "".join(sentences)) == space.join(alone)


@pytest.mark.parametrize("which", REAL)
def test_a_real_model_with_its_shortlist(which):
    """T650: translation() reads the shortlist's bytes and its translator takes it."""
    model, source, target, space, sentences = REAL[which]
    read = lambda file: found(*file).read_bytes()
    lex = SHORTLISTS[which]
    language = "ja" if which.startswith("en") else "en"
    answering = translation(read(model), read(source), read(target), language, sentences=lambda text: sentences,
                            shortlist=read(lex))
    assert answering.backend == "int8, NumPy, shortlist"
    assert answering.translator.shortlist is not None
    alone = [answering.translator.translate(sentence) for sentence in sentences]
    assert joined(answering, "".join(sentences)) == space.join(alone)
