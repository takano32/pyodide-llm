# tests/conducting.py (T374.4): what the tools that convert a model answer the conduct of a conversion with. Alone
# first: each answerer with files of a temporary folder (and stand-ins for tests/fetching.py: no network), the loop
# with a stand-in for the conduct that plays a list of requests. Then with the real conduct and the real converter, on
# real files of every kind of source, against a conversion with no conduct in it (tests/conduct_hub.py's direct()).
import json
import urllib.error
from pathlib import Path

import pytest
from conduct_hub import THREE, direct
from conftest import synthetic_weights
from test_convert import hugging_face, safetensors_file
from test_gguf import EPS, gguf_file, sentencepiece, unigram

import conducting
from conducting import Directory, Fetched, Heads, Mapped, Missing, Nothing, converted, listed

REVISION, OTHER = "0123456789abcdef0123456789abcdef01234567", "76543210fedcba9876543210fedcba9876543210"


def folder_of(tmp_path, files, name="model"):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    for file, data in files.items():
        (folder / file).write_bytes(data.encode() if isinstance(data, str) else data)
    return folder


# ---- a folder's answers, one kind at a time
def test_a_folder_answers_a_file_whole_and_none_where_it_is_not_there(tmp_path):
    answerer = Directory(folder_of(tmp_path, {"config.json": "{\"a\": \"あ\"}", "tokenizer.model": b"\x00\xff\x01"}))
    assert answerer.text("weights", "config.json") == "{\"a\": \"あ\"}"
    assert answerer.bytes("weights", "tokenizer.model") == b"\x00\xff\x01"
    assert answerer.text("weights", "chat_template.jinja") is None and answerer.bytes("weights", "spiece.model") is None
    assert answerer.range("weights", "model.safetensors", 0, 8) is None
    # (a folder of that name is no file)
    (tmp_path / "model" / "tokenizer.json").mkdir()
    assert answerer.bytes("weights", "tokenizer.json") is None


def test_a_range_is_the_bytes_asked_for_wherever_it_begins_and_fewer_at_the_files_end(tmp_path):
    data = bytes(range(200))
    answerer = Directory(folder_of(tmp_path, {"model.gguf": data}))
    assert answerer.range("weights", "model.gguf", 0, 50) == (data[:50], 200)
    # T374.3: a further piece of a head begins where the bytes in hand end, not at the file's first byte
    assert answerer.range("weights", "model.gguf", 50, 120) == (data[50:120], 200)
    assert answerer.range("weights", "model.gguf", 120, 1 << 20) == (data[120:], 200)
    assert answerer.range("weights", "model.gguf", 200, 300) == (b"", 200)
    assert answerer.range("weights", "model.gguf", 300, 400) == (b"", 200)
    assert answerer.size("weights", "model.gguf") == 200


def test_each_place_is_its_own_folder_or_all_are_one(tmp_path):
    weights, original = folder_of(tmp_path, {"a.gguf": "the weights"}, "maker"), folder_of(tmp_path, {"config.json": "the original's"}, "original")
    two = Directory({"weights": weights, "vocabulary": original})
    assert two.text("vocabulary", "config.json") == "the original's" and two.text("weights", "config.json") is None
    assert two.text("weights", "a.gguf") == "the weights" and two.text("vocabulary", "a.gguf") is None
    assert two.place("vocabulary") == str(original)
    one = Directory(weights)
    assert one.text("vocabulary", "a.gguf") == one.text("weights", "a.gguf") == "the weights"


def test_a_text_given_instead_of_a_file_is_what_is_answered_for_that_place_alone(tmp_path):
    folder = folder_of(tmp_path, {"config.json": "the file's"})
    answerer = Directory(folder, instead={("vocabulary", "config.json"): "another", ("weights", "more.json"): "made up"})
    assert answerer.text("vocabulary", "config.json") == "another" and answerer.bytes("vocabulary", "config.json") == b"another"
    assert answerer.text("weights", "config.json") == "the file's"
    assert answerer.text("weights", "more.json") == "made up"


def test_a_stream_goes_to_the_feed_in_parts_in_the_order_of_the_file(tmp_path):
    data = bytes(i % 251 for i in range(1000))
    fed, after = [], []
    answerer = Directory(folder_of(tmp_path, {"model.safetensors": data}), part=300, fed=lambda: after.append(len(fed)))
    assert answerer.stream("weights", "model.safetensors", 100, 950, 100, 1000, fed.append) is None
    assert [len(part) for part in fed] == [300, 300, 250] and b"".join(fed) == data[100:950]
    assert after == [1, 2, 3]  # (told after each part, once it is fed)
    # the default: 8 MiB a part, as the worker's first
    assert Directory(tmp_path).part == 8 << 20


def test_a_stream_with_nothing_in_it_feeds_nothing_and_opens_no_file(tmp_path):
    fed = []
    answerer = Directory(folder_of(tmp_path, {}))
    assert answerer.stream("weights", "not-there.gguf", 64, 64, 64, 64, fed.append) is None and fed == []


def test_a_file_that_ends_before_its_stream_does_is_an_error_and_not_a_short_conversion(tmp_path):
    fed = []
    answerer = Directory(folder_of(tmp_path, {"model.safetensors": bytes(500)}), part=200)
    with pytest.raises(OSError, match="ends at 500"):
        answerer.stream("weights", "model.safetensors", 0, 700, 0, 700, fed.append)
    assert [len(part) for part in fed] == [200, 200]


# ---- the loop, with a stand-in for the conduct
class Conversion:
    def __init__(self):
        self.fed = []

    def feed(self, part):
        self.fed.append(bytes(part))
        return 1


def played(monkeypatch, requests):
    """conducting.conduct replaced by a generator that asks for requests, one after another. Returns what it was made
    with, the answers it got and whether it was closed."""
    seen = {"answers": [], "closed": False}

    def playing():
        try:
            for request in requests:
                seen["answers"].append((yield request))
        finally:
            seen["closed"] = True

    def conduct(hf, **make):
        # (the generator is kept: one that nobody holds is closed when it is let go, whoever forgot to close it)
        seen["hf"], seen["make"], seen["steps"] = hf, make, playing()
        return seen["steps"]

    monkeypatch.setattr(conducting, "conduct", conduct)
    return seen


def test_every_request_is_answered_with_the_answerers_answer_until_the_conduct_is_done(tmp_path, monkeypatch):
    data = bytes(range(256)) * 4
    folder = folder_of(tmp_path, {"config.json": "{}", "tokenizer.json": b"vocabulary", "model.safetensors": data})
    conversion = Conversion()
    seen = played(monkeypatch, [("text", "weights", "config.json"), ("range", "weights", "model.safetensors", 8, 24),
                                ("size", "weights", "model.safetensors"), ("text", "weights", "tokenizer_config.json"),
                                ("bytes", "weights", "tokenizer.json"),
                                ("stream", "weights", "model.safetensors", 24, 1024, 24, 1024, conversion.feed),
                                ("done", conversion)])
    hf = {"weights": "model.safetensors"}
    assert converted(Directory(folder, part=512), hf, dtype="float32", max_seq_len=64) is conversion
    assert seen["answers"] == ["{}", (data[8:24], 1024), 1024, None, b"vocabulary", None]
    assert seen["hf"] is hf and seen["make"] == {"dtype": "float32", "max_seq_len": 64}
    assert conversion.fed == [data[24:536], data[536:]] and seen["closed"]


def test_a_conduct_that_ends_for_want_of_a_file_says_where_and_which(tmp_path, monkeypatch):
    seen = played(monkeypatch, [("text", "vocabulary", "config.json"), ("missing", "vocabulary", "config.json"), ("done", None)])
    with pytest.raises(Missing, match="original has no config.json"):
        converted(Directory({"weights": tmp_path, "vocabulary": tmp_path / "original"}), {"weights": "a.gguf"})
    assert seen["answers"] == [None] and seen["closed"]
    places = {"weights": ("owner/model", REVISION, tmp_path)}
    assert Fetched(places).place("weights") == f"owner/model at {REVISION}"


def test_what_nothing_answers_and_what_an_answer_fails_with_end_the_loop_and_close_the_conduct(tmp_path, monkeypatch):
    seen = played(monkeypatch, [("folder", "weights", "model"), ("done", None)])
    with pytest.raises(RuntimeError, match="asked for folder"):
        converted(Directory(tmp_path), {})
    assert seen["answers"] == [] and seen["closed"]

    class Broken(Directory):
        def text(self, where, name):
            raise OSError("the line")

    seen = played(monkeypatch, [("text", "weights", "config.json"), ("done", None)])
    with pytest.raises(OSError, match="the line"):
        converted(Broken(tmp_path), {})
    assert seen["answers"] == [] and seen["closed"]
    # (a private name of the answerer is no kind of request)
    seen = played(monkeypatch, [("place", "weights"), ("done", None)])
    with pytest.raises(RuntimeError, match="asked for place"):
        converted(Directory(tmp_path), {})


def test_without_its_weights_a_conversion_is_what_the_first_stream_brings(tmp_path, monkeypatch):
    conversion, streamed = Conversion(), []

    class Watched(Directory):
        def stream(self, *request):
            streamed.append(request)

    seen = played(monkeypatch, [("text", "weights", "config.json"), ("stream", "weights", "a.gguf", 4, 8, 4, 8, conversion.feed),
                                ("stream", "weights", "b.gguf", 0, 8, 8, 16, conversion.feed), ("done", "finished")])
    assert converted(Watched(folder_of(tmp_path, {"a.gguf": bytes(8)})), {"weights": "a.gguf"}, weights=False) is conversion
    # nothing is fed, the stream is never answered and the conduct is closed where it stood
    assert seen["answers"] == [None] and streamed == [] and conversion.fed == [] and seen["closed"]


# ---- the files of a repository, with a stand-in for tests/fetching.py
def refused(code):
    return urllib.error.HTTPError("https://huggingface.co/x", code, "refused", {}, None)


def downloads(monkeypatch, files):
    """fetching.download from a dictionary {url: bytes or an error}; what was asked for is written down."""
    asked = []

    def download(url, target, **more):
        if target.exists():
            return target
        asked.append(url)
        if isinstance(files.get(url), Exception):
            raise files[url]
        if url not in files:
            raise refused(404)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[url])
        return target

    monkeypatch.setattr(conducting.fetching, "download", download)
    return asked


AT = f"https://huggingface.co/owner/model/resolve/{REVISION}"
ORIGINAL = f"https://huggingface.co/first/original/resolve/{OTHER}"
HF = {"repo": "owner/model", "revision": REVISION, "weights": "model.gguf",
      "vocabulary": {"repo": "first/original", "revision": OTHER, "tokenizer": "tokenizer.json"}}


def test_a_file_of_a_repository_is_fetched_whole_when_it_is_first_asked_for_and_kept(tmp_path, monkeypatch):
    data = bytes(range(100))
    asked = downloads(monkeypatch, {f"{AT}/model.gguf": data, f"{ORIGINAL}/config.json": b"{}"})
    answerer = Fetched.of(HF, tmp_path)
    assert asked == []  # (nothing before it is asked for)
    assert answerer.range("weights", "model.gguf", 10, 30) == (data[10:30], 100)
    assert answerer.text("vocabulary", "config.json") == "{}"
    assert answerer.range("weights", "model.gguf", 30, 300) == (data[30:], 100)
    assert asked == [f"{AT}/model.gguf", f"{ORIGINAL}/config.json"]
    # each repository's files in a folder of its own, by its name and revision
    assert (tmp_path / "owner--model" / REVISION / "model.gguf").read_bytes() == data
    assert answerer.folder("vocabulary") == tmp_path / "first--original" / OTHER
    assert Fetched.of(HF, tmp_path, between="__").folder("weights") == tmp_path / "owner__model" / REVISION
    # a model that names no other repository has one place
    assert set(Fetched.of({"repo": "owner/model", "revision": REVISION}, tmp_path).places) == {"weights"}


def test_only_a_404_is_a_file_that_is_not_there(tmp_path, monkeypatch):
    asked = downloads(monkeypatch, {f"{AT}/gated.json": refused(403), f"{AT}/busy.json": refused(429)})
    answerer = Fetched.of(HF, tmp_path)
    assert answerer.text("weights", "tokenizer_config.json") is None
    assert answerer.text("weights", "tokenizer_config.json") is None and len(asked) == 2  # (asked again: nothing remembers)
    for name in ("gated.json", "busy.json"):
        with pytest.raises(urllib.error.HTTPError):
            answerer.bytes("weights", name)


def test_a_file_that_is_not_there_is_asked_for_once_where_that_is_remembered(tmp_path, monkeypatch):
    asked = downloads(monkeypatch, {f"{AT}/gated.json": refused(403)})
    answerer = Fetched.of(HF, tmp_path, remember=True)
    assert answerer.text("weights", "chat_template.jinja") is None and answerer.text("weights", "chat_template.jinja") is None
    assert asked == [f"{AT}/chat_template.jinja"]
    assert (tmp_path / "owner--model" / REVISION / "chat_template.jinja.missing").exists()
    assert Fetched.of(HF, tmp_path, remember=True).bytes("weights", "chat_template.jinja") is None and len(asked) == 1
    # (a refusal that is no 404 is not remembered as one)
    with pytest.raises(urllib.error.HTTPError):
        answerer.bytes("weights", "gated.json")
    assert not (tmp_path / "owner--model" / REVISION / "gated.json.missing").exists()


def ranges(monkeypatch, files):
    """fetching.sized and fetching.ranged from a dictionary {url: bytes}; what was asked for is written down. ranged()
    holds an answer to its length, as the real one does."""
    asked = []

    def sized(url, **more):
        asked.append(f"size {url.rsplit('/', 1)[1]}")
        return len(files[url]) if url in files else None

    def ranged(url, start, length, **more):
        asked.append(f"{url.rsplit('/', 1)[1]} {start}-{start + length}")
        data = files[url][start:start + length]
        if len(data) != length:
            raise OSError(f"{len(data)} bytes of {length}")
        return data

    monkeypatch.setattr(conducting.fetching, "sized", sized)
    monkeypatch.setattr(conducting.fetching, "ranged", ranged)
    return asked


def test_the_head_of_a_file_grows_from_where_it_ends_and_never_past_the_file(tmp_path, monkeypatch):
    data = bytes(i % 253 for i in range(1000))
    asked = ranges(monkeypatch, {f"{AT}/model.gguf": data})
    downloads(monkeypatch, {})
    answerer = Heads.of(HF, tmp_path)
    assert answerer.range("weights", "model.gguf", 0, 200) == (data[:200], 1000)
    assert answerer.range("weights", "model.gguf", 200, 800) == (data[200:800], 1000)
    assert answerer.range("weights", "model.gguf", 0, 100) == (data[:100], 1000)    # (in hand already)
    assert answerer.range("weights", "model.gguf", 800, 3200) == (data[800:], 1000)  # (the file ends first)
    assert answerer.range("weights", "model.gguf", 500, 3200) == (data[500:], 1000)
    assert asked == ["size model.gguf", "model.gguf 0-200", "model.gguf 200-800", "model.gguf 800-1000"]
    folder = tmp_path / "owner--model" / REVISION
    assert (folder / "model.gguf.head").read_bytes() == data and (folder / "model.gguf.size").read_text() == "1000"
    assert not (folder / "model.gguf").exists() and not list(folder.glob("*.part"))
    # another run asks for nothing that is kept
    assert Heads.of(HF, tmp_path).range("weights", "model.gguf", 0, 1000) == (data, 1000) and len(asked) == 4
    assert Heads.of(HF, tmp_path).size("weights", "model.gguf") == 1000 and len(asked) == 4


def test_a_piece_past_the_head_in_hand_is_fetched_from_where_the_head_ends(tmp_path, monkeypatch):
    data = bytes(i % 253 for i in range(1000))
    asked = ranges(monkeypatch, {f"{AT}/model.gguf": data})
    answerer = Heads.of(HF, tmp_path)
    assert answerer.range("weights", "model.gguf", 0, 100) == (data[:100], 1000)
    # (no conduct asks so; the head kept must still be the beginning of the file, with no hole in it)
    assert answerer.range("weights", "model.gguf", 300, 400) == (data[300:400], 1000)
    assert asked[1:] == ["model.gguf 0-100", "model.gguf 100-400"]
    assert (tmp_path / "owner--model" / REVISION / "model.gguf.head").read_bytes() == data[:400]


def test_a_head_kept_by_a_tree_of_before_is_used_once_the_size_is_known(tmp_path, monkeypatch):
    data = bytes(i % 253 for i in range(1000))
    asked = ranges(monkeypatch, {f"{AT}/model.gguf": data})
    folder_of(tmp_path / "owner--model", {"model.gguf.head": data[:600]}, REVISION)
    assert Heads.of(HF, tmp_path).range("weights", "model.gguf", 0, 500) == (data[:500], 1000)
    assert asked == ["size model.gguf"]


def test_the_weights_a_repository_does_not_have_are_not_there_and_no_head_has_a_stream(tmp_path, monkeypatch):
    asked = ranges(monkeypatch, {})
    answerer = Heads.of(HF, tmp_path, remember=True)
    assert answerer.range("weights", "model.safetensors", 0, 8) is None and answerer.range("weights", "model.safetensors", 0, 8) is None
    assert asked == ["size model.safetensors"]
    assert Heads.of(HF, tmp_path).range("weights", "model.safetensors", 0, 8) is None and len(asked) == 2  # (not remembered: asked)
    with pytest.raises(RuntimeError, match="reads no weights"):
        answerer.stream("weights", "model.gguf", 0, 8, 0, 8, None)


# ---- the real conduct and the real converter
def a_model():
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = hugging_face(config, weights, False)
    published["rms_norm_eps"] = 1e-5
    gguf, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS])
    return tensors, json.dumps(published), gguf, config["vocab_size"]


ALPACA = "{% for message in messages %}### Instruction:\n{{ message['content'] }}\n{% endfor %}### Response:\n"


def sources(tmp_path):
    """{a kind of source: what tests/hf_fetch.py would print for it}: real files of a made-up Llama."""
    tensors, config, gguf, vocabulary = a_model()
    names = list(tensors)
    shards = {"model-00001-of-00002.safetensors": names[len(names) // 2:], "model-00002-of-00002.safetensors": names[:len(names) // 2]}
    small = {"config.json": config, "tokenizer.json": unigram(vocabulary)}
    return {
        "one safetensors file": folder_of(tmp_path, {**small, "model.safetensors": safetensors_file(tensors),
                                                     "tokenizer_config.json": json.dumps({"chat_template": ALPACA})}, "one"),
        "the shards an index names": folder_of(tmp_path, {
            **small, "model.safetensors.index.json": json.dumps({"weight_map": {t: s for s, mine in shards.items() for t in mine}}),
            **{shard: safetensors_file({name: tensors[name] for name in mine}) for shard, mine in shards.items()},
            "tokenizer_config.json": "{}", "chat_template.jinja": ALPACA}, "shards"),
        "a GGUF alone": folder_of(tmp_path, {"model.Q8_0.gguf": gguf}, "alone") / "model.Q8_0.gguf",
        "a GGUF with the original's vocabulary": folder_of(tmp_path, {**small, "b.Q8_0.gguf": gguf, "a.Q8_0.gguf": gguf[:64]}, "with"),
    }


KINDS = ["one safetensors file", "the shards an index names", "a GGUF alone", "a GGUF with the original's vocabulary"]


def without_a_conduct(hf, answerer, **making):
    whole = lambda where, name: answerer.bytes(where, name)
    return direct(whole, hf, THREE, **making)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["int8", "float32"])
def test_a_tools_conversion_is_the_conversion_of_the_same_files_with_no_conduct(kind, dtype, tmp_path):
    source = sources(tmp_path)[kind]
    if kind.startswith("a GGUF with"):
        (source / "a.Q8_0.gguf").unlink()  # (the first .gguf by its name is the model's)
    hf, answerer = listed(source, part=4096)
    sink = Mapped(tmp_path / "out.bin")
    got = converted(answerer, hf, dtype=dtype, sink=sink)
    sink.data.flush()
    expected = without_a_conduct(hf, answerer, dtype=dtype)
    assert (tmp_path / "out.bin").read_bytes() == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer) and got.options == expected.options
    assert sink.size == len(expected.checkpoint) and sink.dtype == dtype and len(sink.header) == 7
    # the template is the model's own, wherever the page would find it
    if "safetensors" in kind or "shards" in kind:
        assert got.options["template"] == "### Instruction:\n{prompt}\n### Response:"


@pytest.mark.parametrize("kind", KINDS)
def test_without_its_weights_a_conversion_says_what_the_finished_one_says(kind, tmp_path):
    source = sources(tmp_path)[kind]
    if kind.startswith("a GGUF with"):
        (source / "a.Q8_0.gguf").unlink()
    hf, answerer = listed(source)
    finished = converted(answerer, hf, dtype="int8", sink=Nothing())

    class Unstreamed(Directory):
        def stream(self, *request):
            raise AssertionError("the weights were asked for")

    made = converted(Unstreamed(answerer.places), hf, weights=False, dtype="int8", sink=Nothing())
    assert made.options == finished.options and bytes(made.tokenizer) == bytes(finished.tokenizer)
    assert list(made.stream.header) == list(finished.stream.header)


def test_what_a_path_stands_for(tmp_path):
    found = sources(tmp_path)
    hf, answerer = listed(found["a GGUF alone"])
    assert hf == {"weights": "model.Q8_0.gguf"} and Path(answerer.place("weights")) == tmp_path / "alone"
    # a folder with a GGUF: the first by its name, with a vocabulary that is the same folder's; no tokenizer is named
    hf, answerer = listed(found["a GGUF with the original's vocabulary"])
    assert hf["weights"] == "a.Q8_0.gguf" and hf["vocabulary"] and "tokenizer" not in hf["vocabulary"] and "tokenizer" not in hf
    assert answerer.place("vocabulary") == answerer.place("weights") == str(tmp_path / "with")
    for kind in ("one safetensors file", "the shards an index names"):
        assert listed(found[kind])[0] == {"weights": "model.safetensors"}
    assert listed(str(found["one safetensors file"]), part=5)[1].part == 5


def test_the_tokenizer_is_the_conducts_choice_of_what_the_folder_has(tmp_path):
    tensors, config, _, vocabulary = a_model()
    model, other = sentencepiece(vocabulary), sentencepiece(vocabulary).replace("▁w7".encode(), "▁x7".encode())
    files = {"config.json": config, "model.safetensors": safetensors_file(tensors)}

    def made(name, more):
        hf, answerer = listed(folder_of(tmp_path, {**files, **more}, name))
        return converted(answerer, hf, weights=False, sink=Nothing())

    alone = made("alone", {"tokenizer.model": model})
    # tokenizer.model before spiece.model (the conduct's order, which the tools' own was not), and past a
    # tokenizer.json the converter cannot read
    assert bytes(made("both", {"tokenizer.model": model, "spiece.model": other}).tokenizer) == bytes(alone.tokenizer)
    assert bytes(made("other", {"spiece.model": other}).tokenizer) != bytes(alone.tokenizer)
    assert bytes(made("past", {"tokenizer.json": "unreadable", "tokenizer.model": model}).tokenizer) == bytes(alone.tokenizer)
    with pytest.raises(ValueError, match="tokenizer.json"):
        made("none", {"tokenizer.json": "unreadable"})
    with pytest.raises(Missing, match="has no tokenizer.json"):
        made("empty", {})


def made_directly(config, vocabulary, gguf):
    import llama2_convert
    header, base = llama2_convert.gguf_weights(gguf, config)
    return llama2_convert.Conversion(header, base, config, unigram(vocabulary), "tokenizer.json", start=base, sink=Nothing())


def test_the_heads_of_a_repository_are_all_a_conversion_without_weights_needs(tmp_path, monkeypatch):
    """The real conduct over Heads (as tests/format_check.py has it): a GGUF with the vocabulary of another
    repository, from made-up fetches. What is fetched of the weights is their head; the small files come whole."""
    tensors, config, gguf, vocabulary = a_model()
    asked = ranges(monkeypatch, {f"{AT}/model.gguf": gguf})
    whole = downloads(monkeypatch, {f"{ORIGINAL}/config.json": config.encode(), f"{ORIGINAL}/tokenizer.json": unigram(vocabulary),
                                    f"{ORIGINAL}/tokenizer_config.json": json.dumps({"chat_template": ALPACA}).encode()})
    made = converted(Heads.of(HF, tmp_path, between="__", remember=True), HF, weights=False, dtype="int8", sink=Nothing())
    assert made.options["template"] == "### Instruction:\n{prompt}\n### Response:"
    assert bytes(made.tokenizer) == bytes(made_directly(config, vocabulary, gguf).tokenizer)
    assert asked == ["size model.gguf", f"model.gguf 0-{len(gguf)}"]  # (smaller than the first piece of a head: all of it)
    assert whole == [f"{ORIGINAL}/config.json", f"{ORIGINAL}/tokenizer_config.json", f"{ORIGINAL}/tokenizer.json"]
    # and again, from what was kept: nothing is asked for
    converted(Heads.of(HF, tmp_path, between="__", remember=True), HF, weights=False, dtype="int8", sink=Nothing())
    assert len(asked) == 2 and len(whole) == 3
