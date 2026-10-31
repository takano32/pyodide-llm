"""T.257: a sentencepiece BPE model written out as a tokenizer.json (a Gemma's, CAT-Translate 7B's): the whole path
tokenizer.json -> converter -> tokenizer.bin -> engine, against the tokenizers library reading the same file, against
sentencepiece reading the model the file was written from, and against the definition written out here.

Nothing is downloaded: sentencepiece trains a small BPE model in a tenth of a second, and written_out() below writes it
out the way transformers' SentencePieceExtractor does (every way to join two pieces into a third is a merge, in the
order of the scores), by hand and not through the converter. The real files are held to the real tokenizers by
tests/tokenizer_json_check.py, by hand.
"""
import io
import json
import random

import pytest

import spelled_bpe_cases
from conftest import CORPUS, TEXTS, detokenize, vocabulary_conversion
from convert.tokenizer import Later, tokenizer_bin, tokenizer_json_options, tokenizer_json_pieces
from engine.tokenizer import Tokenizer
from spelled_bpe_cases import by_the_definition

tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers to check against the real one")
spm = pytest.importorskip("sentencepiece", reason="pip install sentencepiece to check against the real one")

SPACE = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
PREPEND = {"type": "Prepend", "prepend": "▁"}
GEMMAS_SPLIT = {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False}
WORDS = ("the quick brown fox jumps over lazy dog and then there other think thing than that this these those theirs "
         "they ab abab aba baba bab aab abb 日本語 日本 本語 語学 学校 の文章 文章も 混ぜる 混ぜ ぜる る。 です ます ました").split()
# what a random text is drawn from: letters and pieces that the merges join in more ways than one, spaces, a line
# break (a byte piece: the model's vocabulary has no such character) and a character it lacks
DRAWN = ["a", "b", "a", "b", "th", "e", "the", " ", "n", "o", "r", "s", "日", "本", "語", "学", "の", "。", "\n", "ab", "ba", "é", "  "]
# the shapes: (a dummy prefix, a Gemma's pre_tokenizer, how many neighbouring scores are made one)
SHAPES = [(True, False, 1), (False, True, 1), (False, False, 1), (False, True, 4)]
MODELS = {}


def model(dummy):
    """A sentencepiece BPE model with byte pieces (bytes: the file), with the dummy prefix or without."""
    if dummy not in MODELS:
        rng, out = random.Random(7), io.BytesIO()
        lines = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 12))) for _ in range(400)]
        lines += [line for line in CORPUS.split("\n") if line.strip()]
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines), model_writer=out, vocab_size=700, model_type="bpe", byte_fallback=True,
            add_dummy_prefix=dummy, normalization_rule_name="identity", remove_extra_whitespaces=False,
            split_by_whitespace=False, character_coverage=0.98, hard_vocab_limit=False, allow_whitespace_only_pieces=True,
            minloglevel=2)
        MODELS[dummy] = out.getvalue()
    return MODELS[dummy]


def written_out(dummy, gemma=False, tied=1, added=()):
    """(the tokenizer.json of the model as transformers writes one out, sentencepiece's own reading of the model).
    tied: so many neighbouring scores are one score, as a Gemma's are here and there: the merges of pieces of one score
    are in the order of their left part's length, those of one piece no longer next to each other. added: added tokens
    that are not special, as a Gemma 3's are: pieces of the model (its user-defined ones, of the best score), here
    after its own, and added tokens of the file besides."""
    read = spm.SentencePieceProcessor(model_proto=model(dummy))
    vocab = {read.id_to_piece(id): id for id in range(read.get_piece_size())}
    scores = {piece: read.get_score(id) // tied for piece, id in vocab.items()}
    token = lambda id, content, special: {"id": id, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
                                          "normalized": False, "special": special}
    tokens = [token(id, read.id_to_piece(id), True) for id in range(3)]
    for content in added:
        tokens.append(token(vocab.setdefault(content, len(vocab)), content, False))
        scores[content] = 0.0
    merges = []
    for piece, score in scores.items():
        ways = [(piece[:cut], piece[cut:], score) for cut in range(1, len(piece)) if piece[:cut] in vocab and piece[cut:] in vocab]
        merges += sorted(ways, key=lambda way: (vocab[way[0]], vocab[way[1]]))
    merges = sorted(merges, key=lambda way: (way[2], len(way[0]), len(way[1])), reverse=True)
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": tokens,
            "normalizer": {"type": "Sequence", "normalizers": [PREPEND, SPACE]} if dummy else SPACE,
            "pre_tokenizer": GEMMAS_SPLIT if gemma else None, "post_processor": None,
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                                                         {"type": "ByteFallback"}, {"type": "Fuse"},
                                                         *([{"type": "Strip", "content": " ", "start": 1, "stop": 0}] if dummy else [])]},
            "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                      "vocab": vocab, "merges": [[left, right] for left, right, _ in merges]}}, read


def engine_of(spec):
    options, pieces = tokenizer_json_options(spec), list(tokenizer_json_pieces(spec))
    assert options["tokenizer_kind"] == "bpe"
    return Tokenizer(tokenizer_bin(pieces, len(pieces)), len(pieces), kind="bpe", prefixed=options["prefixed"]), options


def real_of(spec):
    return tokenizers.Tokenizer.from_str(json.dumps(spec))


def random_texts(count, seed=1):
    rng = random.Random(seed)
    return ["".join(rng.choice(DRAWN) for _ in range(rng.randint(1, 40))) for _ in range(count)]


@pytest.mark.parametrize("dummy, gemma, tied", SHAPES)
def test_matches_the_real_tokenizer(dummy, gemma, tied):
    spec, _ = written_out(dummy, gemma, tied)
    (engine, options), real = engine_of(spec), real_of(spec)
    assert options == {"tokenizer_kind": "bpe", "prefixed": "every" if dummy else "never"}
    texts = [*TEXTS, *WORDS, " ".join(WORDS), "", "▁", "a▁▁b", "é", "\x00", "🙂 and 🙂", "a" * 50, "ab" * 40, "aba" * 30, *random_texts(4000)]
    wrong = [text for text in texts if engine.encode(text) != real.encode(text, add_special_tokens=False).ids]
    assert not wrong, f"{len(wrong)} of {len(texts)} texts differ, as {wrong[0]!r}"
    # and the merges did something: a word of the corpus is a piece or two, not its letters
    assert len(engine.encode("the quick brown fox")) < 12


@pytest.mark.parametrize("dummy, gemma, tied", SHAPES[:3])
def test_matches_the_definition_and_sentencepiece_itself(dummy, gemma, tied):
    """Three more readings of the same text: the definition, written out here; sentencepiece's of the model the file was
    written from (the pieces and the scores themselves: what the file's merges stand for); and, since the file gives
    the engine other scores than the model's own file does, the engine's of the model's own file (the model's dummy
    prefix by hand, which that path does not read)."""
    spec, sentencepiece = written_out(dummy, gemma, tied)
    engine, _ = engine_of(spec)
    texts = [text for text in [*TEXTS, *WORDS, *random_texts(1500, seed=2)] if text]
    assert [engine.encode(text) for text in texts] == [by_the_definition(spec, text) for text in texts]
    assert [engine.encode(text) for text in texts] == [sentencepiece.encode(text) for text in texts]


@pytest.mark.parametrize("dummy", [False, True])
def test_the_cases_of_the_smoke_test_are_the_real_tokenizers(dummy):
    """tests/spelled_bpe_cases.py's file, made by hand, which tests/smoke.ts reads inside Pyodide: the engine's ids
    of its texts are the tokenizers library's and the definition's, and the number the smoke test holds Pyodide's to
    is theirs."""
    spec, specials = spelled_bpe_cases.spec(dummy), spelled_bpe_cases.SPECIALS
    (engine, _), real = engine_of(spec), real_of(spec)
    ids = [engine.encode(text, specials) for text in spelled_bpe_cases.TEXTS]
    assert ids == [real.encode(text, add_special_tokens=False).ids for text in spelled_bpe_cases.TEXTS]
    assert ids == [by_the_definition(spec, text, specials) for text in spelled_bpe_cases.TEXTS]
    assert max(map(len, ids)) > 8 and any(len(found) == 1 for found in ids)
    assert spelled_bpe_cases.digest(ids) == spelled_bpe_cases.IDS[dummy]


def test_the_scores_are_minus_the_rank_of_the_first_merge_that_makes_a_piece():
    spec, _ = written_out(False, True, 4)
    merges, vocab = spec["model"]["merges"], spec["model"]["vocab"]
    made = [left + right for left, right in merges]
    assert len(set(made)) < len(merges), "a piece is made in more ways than one"
    assert any(made.index(text) + made.count(text) < len(made) - made[::-1].index(text) for text in set(made)), \
        "and the ways to make one piece are not all next to each other (the tied scores)"
    pieces = list(tokenizer_json_pieces(spec))
    for text, score, matchable in pieces:
        if text in made:
            assert score == -float(made.index(text)) and matchable
        else:  # a single character, a byte piece, a control piece: no merge makes it
            assert score < -1e8 and not matchable
    assert [text for text, _, _ in pieces] == sorted(vocab, key=vocab.get)


def test_a_character_the_vocabulary_lacks_is_its_bytes():
    spec, _ = written_out(False, True)
    (engine, _), real, vocab = engine_of(spec), real_of(spec), spec["model"]["vocab"]
    assert "é" not in vocab and "🙂" not in vocab
    assert engine.encode("é") == [vocab["<0xC3>"], vocab["<0xA9>"]]
    for text in ("aéb", "🙂", "é🙂é", "the é", "\x00\x01", "日本語と韓国語 한국어"):
        assert engine.encode(text) == real.encode(text, add_special_tokens=False).ids, text
        assert detokenize(engine, engine.encode(text), bos=-1) == text


@pytest.mark.parametrize("dummy", [True, False])
def test_the_dummy_prefix_is_before_every_text_or_before_none(dummy):
    """A Llama tokenizer's file prepends U+2581 to every stretch of text between two added tokens; a Gemma's to none,
    not to the first either, and what it decodes keeps a space it begins with."""
    spec, _ = written_out(dummy, not dummy)
    (engine, options), real, vocab = engine_of(spec), real_of(spec), spec["model"]["vocab"]
    specials = ("<s>", "</s>")
    for text in ("the", " the", "the</s>the", "the</s> the", "<s>the dog</s>  and", "</s>", "</s></s>the"):
        assert engine.encode(text, specials) == real.encode(text, add_special_tokens=False).ids, text
    assert engine.encode("the") == [vocab["▁the" if dummy else "the"]]
    assert engine.encode("the</s>the", specials)[-1] == vocab["▁the" if dummy else "the"]
    # the first piece written after the BOS: the dummy prefix goes, a Gemma's own space stays
    assert engine.decode(1, vocab["▁the"], 1) == (b"the" if dummy else b" the")
    assert engine.decode(5, vocab["▁the"], 1) == b" the"
    for text in ("the dog", " the dog", "  the dog "):
        assert detokenize(engine, engine.encode(text), bos=-1) == real.decode(real.encode(text, add_special_tokens=False).ids) == text


def test_added_tokens_that_are_not_special_are_one_token_wherever_they_are():
    """A Gemma 3's: runs of line breaks and of U+2581 and thousands of <unusedN>, each one token wherever it is written
    (T.143), named to the engine by the conversion, the longest first. A run of U+2581 is a piece the engine's
    vocabulary spells with spaces."""
    added = ["\n" * count for count in range(1, 6)] + ["▁" * count for count in range(2, 6)] + [f"<unused{n}>" for n in range(40)] + ["[multimodal]"]
    spec, _ = written_out(False, True, added=added)
    real, vocab = real_of(spec), spec["model"]["vocab"]
    conversion = vocabulary_conversion(json.dumps(spec).encode(), "tokenizer.json", len(vocab))
    options = conversion.options
    specials = tuple(options["specials"])
    assert set(specials) == set(added) and list(specials) == sorted(specials, key=lambda token: (-len(token), token))
    engine = Tokenizer(conversion.tokenizer, len(vocab), kind=options["tokenizer_kind"], prefixed=options["prefixed"])
    texts = ["a\nb", "a\n\n\nb", "\n" * 7, "the\n\n dog\n", "<unused3>", "x<unused12><unused1>y", "<unused40>", "<unused 3>", "[multimodal] the",
             "a▁▁b", "▁▁▁▁▁▁▁", "a  b", "a     b", "the ▁▁ dog", "the\n▁▁\n", *TEXTS]
    wrong = [text for text in texts if engine.encode(text, specials) != real.encode(text, add_special_tokens=False).ids]
    assert not wrong, f"{len(wrong)} of {len(texts)} texts differ, as {wrong[0]!r}"
    assert engine.encode("a\n\n\nb", specials)[1] == vocab["\n\n\n"] and engine.encode("▁▁▁", specials) == [vocab["▁▁▁"]]
    assert engine.encode("x<unused12>", specials)[-1] == vocab["<unused12>"]


def test_the_specials_are_cut_as_they_were_and_the_pattern_is_made_once():
    spec, _ = written_out(True)
    engine, _ = engine_of(spec)
    specials = ("</s>", "<s>")
    first = engine.encode("the</s>dog<s>", specials)
    pattern = engine.cut(specials)[0]
    assert engine.encode("the</s>dog<s>", list(specials)) == first and engine.cut(specials)[0] is pattern
    # the first of the alternatives that matches, as before: a list that names a shorter one first is cut at that one
    vocab = spec["model"]["vocab"]
    short, long = next((short, long) for long in vocab for short in vocab if long.startswith(short) and len(long) > len(short) > 1
                       and "▁" not in long and "<" not in long and long[len(short):] in vocab)
    assert engine.encode("x" + long, (short, long)) == [*engine.encode("x"), vocab[short], *engine.encode(long[len(short):])]
    assert engine.encode("x" + long, (long, short)) == [*engine.encode("x"), vocab[long]]
    with pytest.raises(KeyError):  # a special the vocabulary has no piece for, where it is written (as before)
        engine.encode("the<none>", ("<none>",))
    assert engine.encode("the", ("<none>",)) == engine.encode("the")


def test_refuses_what_the_engine_would_not_follow():
    spec, _ = written_out(False, True)
    changed = lambda **model: {**spec, "model": {**spec["model"], **model}}
    vocab, merges = spec["model"]["vocab"], spec["model"]["merges"]
    read = lambda spec: (tokenizer_json_options(spec), list(tokenizer_json_pieces(spec)))
    read(spec)
    # a character the vocabulary lacks: only as the byte pieces, all of them
    for other in (changed(byte_fallback=False), changed(vocab={text: id for text, id in vocab.items() if text != "<0x80>"})):
        with pytest.raises(ValueError, match="spells a character it lacks in a way the engine does not know"):
            read(other)
    # merges the engine's BPE would not follow: a way to spell a piece with two pieces that is no merge (which is how a
    # BPE trained by the tokenizers library names them: one way for each piece), a pair of no pieces, a piece
    # written with a space, a piece taken whole before the merges, marks at the edges of a word, merges left out at random
    made = [left + right for left, right in merges]
    several = next(at for at, text in enumerate(made) if made.count(text) > 1)
    for other in (changed(merges=merges[:several] + merges[several + 1:]), changed(merges=[*merges, ["zz", "y"]]),
                  changed(vocab={**vocab, "a b": len(vocab)}), changed(ignore_merges=True), changed(end_of_word_suffix="</w>"),
                  changed(continuing_subword_prefix="##"), changed(dropout=0.1)):
        with pytest.raises(ValueError, match="merges its pieces in a way the engine does not know"):
            read(other)
    # and what reads a text otherwise is no sentencepiece model written out: refused as it was, by how it splits
    for other in ({**spec, "normalizer": None}, {**spec, "normalizer": {"type": "NFKC"}}, {**spec, "normalizer": PREPEND},
                  {**spec, "normalizer": {"type": "Sequence", "normalizers": [SPACE, {"type": "NFKC"}]}},
                  {**spec, "normalizer": {**SPACE, "content": "_"}}, {**spec, "normalizer": {**SPACE, "pattern": {"Regex": " "}}},
                  {**spec, "pre_tokenizer": {**GEMMAS_SPLIT, "invert": True}}, {**spec, "pre_tokenizer": {**GEMMAS_SPLIT, "pattern": {"String": "▁"}}},
                  {**spec, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}},
                  {**spec, "pre_tokenizer": {"type": "Whitespace"}}):
        with pytest.raises(ValueError, match="splits text in a way the engine does not know"):
            tokenizer_json_options(other)


def test_a_byte_level_bpe_is_read_as_it_was():
    """The pieces and the options of a byte-level tokenizer.json go through none of this (its merges name one way to
    make a piece, which merged_by_text() would refuse)."""
    from made_up_tokenizers import byte_level
    spec, _ = byte_level(("<|im_end|>",))
    assert tokenizer_json_options(spec)["tokenizer_kind"] == "bytebpe"
    assert len(list(tokenizer_json_pieces(spec))) == 320


def test_the_models_own_file_comes_first_where_others_are_left_to_try():
    """Conversion(last=False), as the conduct asks for each candidate: a tokenizer.json that writes out a sentencepiece
    model with its dummy prefix is put off, before anything is made of the weights; one without the prefix (a Gemma's:
    nothing of it is tokenizer_config.json's to say) and any other tokenizer are read either way."""
    import struct
    import llama2_convert
    from conftest import synthetic_weights
    from made_up_tokenizers import byte_level
    from test_convert import hugging_face, safetensors_file
    spec, _ = written_out(True)
    size = len(spec["model"]["vocab"])

    class NoWeights:
        opened = 0

        def open(self, *args):
            NoWeights.opened += 1

        def write(self, *args):
            pass

    def conversion(tokenizer, name, size, **more):
        settings, weights = synthetic_weights(vocab_size=size)
        tensors, published = hugging_face(settings, weights, True)
        file = safetensors_file(tensors)
        head = struct.unpack("<Q", file[:8])[0]
        return llama2_convert.Conversion(file[8:8 + head].decode(), 8 + head, json.dumps(published), tokenizer, name, dtype="float32",
                                         max_seq_len=settings["seq_len"], start=8 + head, sink=NoWeights(), **more)

    made = conversion(json.dumps(spec).encode(), "tokenizer.json", size)
    assert made.options["tokenizer_kind"] == "bpe" and made.options["prefixed"] == "every" and NoWeights.opened == 1
    with pytest.raises(Later):
        conversion(json.dumps(spec).encode(), "tokenizer.json", size, last=False)
    assert NoWeights.opened == 1 and isinstance(Later("x"), ValueError)  # (refused in words where nothing catches it)
    for gemma in (True, False):
        bare, _ = written_out(False, gemma)
        assert conversion(json.dumps(bare).encode(), "tokenizer.json", size, last=False).options["prefixed"] == "never"
    assert NoWeights.opened == 3
    other, _ = byte_level(("<|im_end|>",))
    assert conversion(json.dumps(other).encode(), "tokenizer.json", 320, last=False).options["tokenizer_kind"] == "bytebpe"
    assert conversion(model(True), "tokenizer.model", size, last=False).options["tokenizer_kind"] == "bpe"


def test_the_conduct_reads_the_file_where_no_other_tokenizer_will_do(monkeypatch):
    """The candidates in their order, each asked for once (the requests are what they were): a tokenizer.json that
    writes out a sentencepiece model is read only where no candidate after it is there and readable."""
    from test_conduct import HF, repository, requests_of
    asked = lambda told: [request[2] for request in told if request[0] == "bytes"]
    anyones = {key: value for key, value in HF.items() if key != "tokenizer"}  # (no tokenizer named: the candidates)
    requests = requests_of
    requests_of = lambda files, monkeypatch: requests(files, anyones, monkeypatch)
    written = {"tokenizer.json": b"written out, as a Gemma's"}
    # the model's own file is there: read, as before
    told, last, stand_in = requests_of(repository({**written, "tokenizer.model": b"the model's own"}), monkeypatch=monkeypatch)
    assert asked(told) == ["tokenizer.json", "tokenizer.model"] and last[0] == "done"
    made = [line for line in stand_in.handed if line.startswith("Conversion(")]
    assert len(made) == 2 and made[0].endswith(": put off") and "tokenizer.model of 15 bytes" in made[1]
    # no other tokenizer: the file itself, after the others were asked for
    told, last, stand_in = requests_of(repository(written), monkeypatch=monkeypatch)
    assert asked(told) == ["tokenizer.json", "tokenizer.model", "spiece.model"] and last[0] == "done"
    made = [line for line in stand_in.handed if line.startswith("Conversion(")]
    assert len(made) == 2 and made[0].endswith(": put off") and "tokenizer.json of 25 bytes" in made[1] and "dtype" in made[1]
    # another that cannot be read: the file itself
    told, last, stand_in = requests_of(repository({**written, "tokenizer.model": b"unreadable"}), monkeypatch=monkeypatch)
    made = [line for line in stand_in.handed if line.startswith("Conversion(")]
    assert last[0] == "done" and len(made) == 3 and made[1].endswith(": refused") and "tokenizer.json of 25 bytes" in made[2]
    # and the one candidate an item names is read at once... after being put off, with nothing asked for twice
    told, last, stand_in = requests(repository(written), HF, monkeypatch)
    assert asked(told) == ["tokenizer.json"] and last[0] == "done"


def test_an_added_token_past_the_models_last_row_is_left_out():
    """A text-only Gemma 3's tokenizer.json names <image_soft_token> one past the model's vocabulary (262,144, the id
    of the token in the models that see): no token of this model, and not named to the engine. A model with fewer rows
    than the tokenizer's own vocabulary is refused as it was, and so is one with many more."""
    spec, _ = written_out(False, True, added=["\n"])
    size = len(spec["model"]["vocab"])
    token = lambda id, content, special: {**spec["added_tokens"][0], "id": id, "content": content, "special": special}
    spec["added_tokens"] += [token(size, "<image_soft_token>", True), token(size + 1, "<seen>", False)]
    assert len(list(tokenizer_json_pieces(spec))) == size + 2
    conversion = vocabulary_conversion(json.dumps(spec).encode(), "tokenizer.json", size)
    assert conversion.options["specials"] == ["\n"]
    engine = Tokenizer(conversion.tokenizer, size, kind="bpe", prefixed="never")
    assert len(engine.vocab) == size and engine.encode("the\ndog", ("\n",))[1] == spec["model"]["vocab"]["\n"]
    for rows, said in ((size - 1, "but the model has a vocabulary of"), (size + 1, "but the model has a vocabulary of"),
                       (2 * size, "they do not belong together")):
        if rows == size + 1:  # (one row more: the first of the two is a token of the model, the second is not)
            assert vocabulary_conversion(json.dumps(spec).encode(), "tokenizer.json", rows).options["specials"] == ["\n"]
            continue
        with pytest.raises(ValueError, match=said):
            vocabulary_conversion(json.dumps(spec).encode(), "tokenizer.json", rows)
    # a byte-level tokenizer.json's too (the same pieces), and its own vocabulary's are never cut
    from made_up_tokenizers import byte_level
    other, ids = byte_level(("<|im_end|>",), ("<seen>",))
    assert vocabulary_conversion(json.dumps(other).encode(), "tokenizer.json", 320).options["specials"] == ["<seen>"]
    assert "specials" not in vocabulary_conversion(json.dumps(other).encode(), "tokenizer.json", 319).options
    assert "specials" not in vocabulary_conversion(json.dumps(other).encode(), "tokenizer.json", 318).options
    other["model"]["vocab"]["one more"] = 320
    with pytest.raises(ValueError, match="but the model has a vocabulary of"):
        vocabulary_conversion(json.dumps(other).encode(), "tokenizer.json", 320)


def test_a_gemmas_turn_begins_with_its_bos_and_ends_at_its_end_of_turn():
    """What T.261 gets of a Gemma 3's files (made up here in their shape): the BOS the tokenizer names, which its
    template writes first, is the engine's and is taken off the format; the turn's own tokens are specials; the answer
    stops at <end_of_turn>, the EOS tokenizer_config.json names, and at config.json's."""
    from made_up_tokenizers import sent
    marks = ["<start_of_turn>", "<end_of_turn>"]
    spec, _ = written_out(False, True, added=["\n", "\n\n"])
    vocab = spec["model"]["vocab"]
    for mark in marks:
        vocab[mark] = len(vocab)
        spec["added_tokens"].append({**spec["added_tokens"][0], "id": vocab[mark], "content": mark})
    template = ("{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\n'"
                " + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}")
    described = {"bos_token": "<s>", "eos_token": "<end_of_turn>", "chat_template": template, "tokenizer_class": "GemmaTokenizer"}
    conversion = vocabulary_conversion(json.dumps(spec).encode(), "tokenizer.json", len(vocab), json.dumps(described),
                                       bos_token_id=vocab["<s>"], eos_token_id=[vocab["</s>"], vocab["<end_of_turn>"]])
    options = conversion.options
    assert options["tokenizer_kind"] == "bpe" and options["prefixed"] == "never" and options["bos"] == vocab["<s>"]
    assert options["template"] == "<start_of_turn>user\n{prompt:trim}<end_of_turn>\n<start_of_turn>model\n"
    assert options["stop_tokens"] == [vocab["<s>"], vocab["</s>"], vocab["<end_of_turn>"]]
    assert options["specials"] == ["<start_of_turn>", "<end_of_turn>", "\n\n", "\n"]
    engine, real = Tokenizer(conversion.tokenizer, len(vocab), kind="bpe", prefixed="never"), real_of(spec)
    ids = sent({**options, "template": options["template"].replace("{prompt:trim}", "{prompt}")}, engine, "the dog")
    assert ids == [vocab["<s>"]] + real.encode("<start_of_turn>user\nthe dog<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False).ids
    assert ids[1] == vocab["<start_of_turn>"] and ids[-1] == vocab["\n"]
