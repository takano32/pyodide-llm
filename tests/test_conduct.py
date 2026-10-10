# The conduct of a conversion (T374.1, public/convert/conduct.py): a generator that asks for a model's files in the
# order the worker asks for them today, driven here by a dictionary (tests/conduct_hub.py's Hub): no browser, no
# worker, no Pyodide.
#
# Held to three things. The requests of the worker itself, for the 19 repositories of tests/worker-fetches-check.mjs
# (tests/fixtures/conversion-cases.json is that check's record of them, conversion-fetches.json what the worker asked):
# line by line, with the converter's stand-in of that check. The worker's ladder written out in Python (today(), itself
# held to the same fixture), wherever there is no fixture: the real converter on real files of the same 19 kinds (the
# checkpoint, the options and the tokenizer.bin as well), and made-up repositories by the hundred. And what the
# requests are, one at a time, with a stand-in that converts nothing.
import json
import random
import struct
from pathlib import Path

import pytest
from conduct_hub import File, Hub, MiB, StandIn, answer, answered, cut, gguf, made, safetensors, today
from conftest import synthetic_weights
from test_convert import hugging_face, safetensors_file
from test_gguf import EPS, gguf_file, sentencepiece, unigram

import llama2_convert
from convert import conduct as conducting
from convert.conduct import SOURCES, TOKENIZERS, conduct, says_a_template, shards_of
from convert.gguf import Incomplete

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "conversion-cases.json").read_text())
FETCHES = json.loads((FIXTURES / "conversion-fetches.json").read_text())
NAMES = [case["name"] for case in CASES]
REVISION = "0123456789abcdef0123456789abcdef01234567"


def choice(header, form, sizes):  # (the worker's choice of a dtype, once the header is known: any function here)
    return "int8"


def listed(case):
    """(the model as it is listed, what its conversion is made with) of a case."""
    hf = dict(case["hf"])
    return hf, {"dtype": hf.pop("dtype", choice)}


def ending(hub, last=None, error=None):
    """How a conversion ended, in the words of the fixture's "ended"."""
    if error is not None:
        return f"failed: {error}"
    return "converted" if last[0] == "done" else f"failed: {hub.refusal(*last[1:])}"


def conducted(hub, hf, making):
    """A conduct answered from the hub to its end: (how it ended, the conversion or None)."""
    try:
        last = answered(hub, conduct(hf, **making))
    except Exception as error:
        return ending(hub, error=error), None
    return ending(hub, last), last[1] if last[0] == "done" else None


def as_today(hub, hf, converter, making, **more):
    try:
        return "converted", today(hub, hf, converter, **more, **making)
    except Exception as error:
        return ending(hub, error=error), None


# ---- the worker's requests, line by line
def test_the_cases_are_the_ones_the_worker_was_asked_for():
    assert NAMES == list(FETCHES) and len(NAMES) == 19
    assert sum(len(fetches["requests"]) for fetches in FETCHES.values()) == 161


def expected_of(name):
    fetches = FETCHES[name]
    # (what the worker keeps afterwards is not the conduct's)
    return fetches["requests"], [line for line in fetches["converter"] if not line.startswith("kept as ")], fetches["ended"].split(";")[0]


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_a_conduct_asks_what_the_worker_asks(case, monkeypatch):
    """The 19 repositories of worker-fetches-check.mjs: the same requests in the same order (the file, the range), the
    same things handed to the converter, the same end. The hub is given what only the answerer decides: how the line
    cuts a stream into parts, and that a server does not say the size of a file."""
    stand_in = StandIn().into(monkeypatch)
    hf, making = listed(case)
    hub = Hub(made(case["files"]), hf, unsaid=case["line"].get("unsaid", False), **cut(case["line"]))
    ended, _ = conducted(hub, hf, making)
    requests, handed, end = expected_of(case["name"])
    assert hub.asked == requests
    assert stand_in.handed == handed
    assert ended == end


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_the_ladder_written_out_in_python_is_the_workers(case):
    """today() is what a conduct is compared with below: it must be the worker's ladder, by the same fixture."""
    stand_in = StandIn()
    hf, making = listed(case)
    hub = Hub(made(case["files"]), hf, unsaid=case["line"].get("unsaid", False), **cut(case["line"]))
    ended, _ = as_today(hub, hf, stand_in, making)
    requests, handed, end = expected_of(case["name"])
    assert (hub.asked, stand_in.handed, ended) == (requests, handed, end)


# ---- the real converter, on real files of the same 19 kinds
CHATML = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}" \
         "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
ALPACA = "{% for message in messages %}### Instruction:\n{{ message['content'] }}\n{% endfor %}### Response:\n"


def real_model():
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = hugging_face(config, weights, False)
    published["rms_norm_eps"] = 1e-5
    file, same = gguf_file(tensors, published, config["vocab_size"], more=[EPS])
    return tensors, published, file, same, config["vocab_size"]


def real_files(case):
    """The repository of a case with real files in the place of the made-up ones, file for file: a Llama's tensors
    (in one safetensors file, in the shards the index names, or in a GGUF), its config.json, a Unigram tokenizer.json,
    a sentencepiece model, templates the converter reads. What is not there stays not there, and the tokenizer.json
    that cannot be read stays one."""
    tensors, published, gguf_bytes, same, vocab_size = real_model()
    whole = safetensors_file(tensors)
    names = list(tensors)
    shards = sorted(name for name in case["files"] if "-of-" in name)
    files = {}
    for name, value in case["files"].items():
        file = name.split("/")[-1]
        if file == "model.safetensors.index.json":
            # (the first shard has the later tensors, the second the earlier ones: the order of the names is the order fed)
            cuts = [names[i * len(names) // len(shards):(i + 1) * len(names) // len(shards)] for i in range(len(shards))]
            files[name] = json.dumps({"weight_map": {tensor: shard.split("/")[-1] for shard, mine in zip(shards, reversed(cuts)) for tensor in mine}})
            for shard, mine in zip(shards, reversed(cuts)):
                files[shard] = safetensors_file({tensor: tensors[tensor] for tensor in mine})
        elif "-of-" in file:
            continue
        elif file.endswith(".safetensors"):
            files[name] = whole
        elif file.endswith(".gguf"):
            files[name] = gguf_bytes
        elif file == "tokenizer_config.json":
            files[name] = json.dumps({"chat_template": CHATML} if "chat_template" in value else json.loads(value))
        elif file == "chat_template.jinja":
            files[name] = ALPACA
        elif file == "tokenizer.json":
            files[name] = value if value == "unreadable" else unigram(vocab_size)
        elif file in ("tokenizer.model", "spiece.model"):
            files[name] = sentencepiece(vocab_size)
        elif "maker's" in value:
            files[name] = value
        else:
            assert json.loads(value) == {"model_type": "llama"}
            files[name] = json.dumps(published)
    return files


def in_turn(requests):
    """The files asked for, each once for as long as it is asked for again and again (its head's pieces, its parts)."""
    files = [" ".join(line.split()[1:3]) for line in requests]
    return [file for i, file in enumerate(files) if i == 0 or file != files[i - 1]]


def gguf_base(file):
    return llama2_convert.gguf_read(file)[2]


@pytest.mark.parametrize("case", CASES, ids=NAMES)
def test_a_conduct_converts_what_the_worker_converts(case, monkeypatch):
    """The real converter and real files: the requests are today's ladder's, one for one, and so are the checkpoint,
    the options and the tokenizer.bin. Scaled to files of a few hundred kilobytes: the parts of the line are cut by
    2 KiB where the worker's are cut by a MiB, and the first piece of a head is as short as the case needs (a header
    past it; a GGUF's head read at the second try, and at the third)."""
    name, files = case["name"], real_files(case)
    hf, making = listed(case)
    making["dtype"] = making["dtype"] if isinstance(making["dtype"], str) else "int8"
    weights = next((file for path, file in files.items() if path.endswith(".gguf")), None)
    head = 128 if "past the first 512 KiB" in name else 512 * 1024 if not weights or "small head" in name \
        else gguf_base(weights) // 8 if "alone" in name else gguf_base(weights) // 40
    monkeypatch.setattr(conducting, "HEAD", head)
    monkeypatch.setattr(conducting, "GGUF_HEAD", 4 * head)
    line = dict(unsaid=case["line"].get("unsaid", False), **cut(case["line"], unit=2048))
    theirs, mine = Hub(files, hf, **line), Hub(files, hf, **line)
    ended, expected = as_today(theirs, hf, llama2_convert, making, head=head)
    end, got = conducted(mine, hf, making)
    assert mine.asked == theirs.asked
    assert end == ended and (ended == "converted") == (expected_of(name)[2] == "converted")
    # and they are the fixture's but for the sizes: the same files of the same repositories, one after another
    assert in_turn(mine.asked) == in_turn(FETCHES[name]["requests"])
    if expected is None:
        assert got is None
        return
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert got.options == expected.options
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options["dtype"] == making["dtype"] and len(got.checkpoint) > 20000
    # the template is the one of the file the worker takes it from
    if any(path.endswith("tokenizer_config.json") and "im_start" in text for path, text in files.items()):
        assert "<|im_start|>" in got.options["template"]
    elif any(path.endswith("chat_template.jinja") for path in files):
        assert "### Instruction" in got.options["template"]
    else:
        assert "template" not in got.options
    # (the cases are of the size they are meant to be: several parts, a second piece of the head, three of a GGUF's)
    asked = "\n".join(mine.asked)
    if "MiB parts" in name:
        assert len([line for line in mine.asked if "model.safetensors bytes=" in line]) > 5
    if "past the first" in name:
        assert asked.count("model.safetensors bytes=0-") == 2
    if weights:
        assert asked.count(".gguf bytes=0-") == (1 if "small head" in name else 2 if "alone" in name else 3)


# ---- made-up repositories by the hundred: the conduct against the ladder
def made_up(rng):
    """A repository nobody would publish, the model as it might be listed for it, and the line."""
    head = rng.choice([40, 64, 300])
    repository, files = "owner/model", {}
    hf = {"repo": repository, "revision": REVISION, "weights": "model.safetensors"}

    def some_safetensors():
        kind = rng.random()
        if kind < 0.12:
            return None
        if kind < 0.2:
            return File(rng.choice([b"", b"short", struct.pack("<Q", 1) + b"{}", struct.pack("<Q", 10 ** 9) + b"{}", b"\xff" * 30]))
        return safetensors(rng.choice([0, 1, 700, 5000]), header=rng.choice([20, 30, 50, 100, 400]))

    kind = rng.choice(["gguf", "gguf", "vocabulary", "safetensors", "safetensors", "safetensors", "safetensors"])
    if kind != "safetensors":
        hf["weights"] = "model.Q8_0.gguf"
        base = rng.choice([12, 30, 100, 170, 700, 1500, 5000])
        files[f"{repository}/model.Q8_0.gguf"] = rng.choice([gguf(base, rng.choice([0, 3, 900, 4000])), gguf(base, 10)] * 4
                                                             + [File(b"GGUF" + struct.pack("<Q", base), max(12, base - 5)), None])
    else:
        files[f"{repository}/model.safetensors"] = some_safetensors()
        shards = [f"model-{i + 1:05}-of-{n:05}.safetensors" for n in [rng.choice([1, 2, 3])] for i in range(n)]
        for shard in shards:
            files[f"{repository}/{shard}"] = some_safetensors()
        rng.shuffle(shards)
        files[f"{repository}/model.safetensors.index.json"] = rng.choice(
            [None, None, "not JSON", "{}", "[]", "null", "{\"weight_map\":null}", "{\"weight_map\":{}}"]
            + [json.dumps({"weight_map": {f"tensor.{i}": shard for i, shard in enumerate(shards + shards[:1])}})] * 6)
    place = repository
    if kind == "vocabulary":
        place = "original/model"
        hf["vocabulary"] = {"repo": place, "revision": REVISION[::-1]}
    if rng.random() < 0.2:
        hf["config"] = "configs/text.json"
    files[f"{place}/{hf.get('config', 'config.json')}"] = rng.choice(["{\"model_type\":\"llama\"}"] * 6 + [None])
    files[f"{place}/tokenizer_config.json"] = rng.choice([
        None, "", "{}", "null", "[1]", "7", "not JSON", "{\"bos_token\":\"<s>\"}", "{\"chat_template\":\"{{ messages }}\"}",
        "{\"chat_template\":[]}", "{\"chat_template\":{}}", "{\"chat_template\":\"\"}", "{\"chat_template\":0}", "{\"chat_template\":0.0}",
        "{\"chat_template\":false}", "{\"chat_template\":true}", "{\"chat_template\":null}", "{\"chat_template\":NaN}",
        "{\"chat_template\":\"x\",\"more\":Infinity}", "{\"chat_template\":[{\"name\":\"default\",\"template\":\"x\"}]}"])
    files[f"{place}/chat_template.jinja"] = rng.choice([None, "a template", ""])
    candidates = rng.sample(list(TOKENIZERS), rng.choice([1, 1, 2, 3]))
    for candidate in TOKENIZERS:
        files[f"{place}/{candidate}"] = rng.choice([None, None, "unreadable", "a tokenizer", "a tokenizer"])
    (hf["vocabulary"] if kind == "vocabulary" and rng.random() < 0.8 else hf)["tokenizer"] = candidates[0] if len(candidates) == 1 and rng.random() < 0.5 else candidates
    if kind == "vocabulary":
        hf.setdefault("tokenizer", "tokenizer.json")  # (which the vocabulary's own comes before)
        hf["vocabulary"].setdefault("tokenizer", None)
    files = {name: value for name, value in files.items() if value is not None}
    unit = rng.choice([16, 100, 512])
    line = dict(unsaid=rng.random() < 0.25, first=unit, ahead=rng.choice([1, 6, 12]), rest=rng.choice([1, 2]) * unit)
    return head, hf, files, line


def test_made_up_repositories_are_asked_as_the_worker_asks_them(monkeypatch):
    """Files that are not there, heads that are no heads, indexes that say nothing, tokenizers the converter refuses,
    templates of every kind of JSON, a GGUF that ends before its head does: for 1500 of them, the conduct asks what
    the ladder asks, hands the converter what the ladder hands it, and ends as it ends. (Only a file that is not there
    fails here: another failure is the answerer's, who stops asking.)"""
    rng = random.Random(374)
    ends = {}
    for _ in range(1500):
        head, hf, files, line = made_up(rng)
        monkeypatch.setattr(conducting, "HEAD", head)
        monkeypatch.setattr(conducting, "GGUF_HEAD", 4 * head)
        theirs, mine = Hub(files, hf, **line), Hub(files, hf, **line)
        ladder, stand_in = StandIn(), StandIn().into(monkeypatch)
        expected = as_today(theirs, hf, ladder, {"dtype": choice}, head=head)[0]
        got = conducted(mine, hf, {"dtype": choice})[0]
        told = (hf, sorted(files), line, head)
        assert mine.asked == theirs.asked, told
        assert stand_in.handed == ladder.handed, told
        assert got == expected, told
        kind = "converted" if got == "converted" else "a file is not there" if "has no" in got else "refused" if "cannot be converted" in got \
            else "no safetensors file" if "not a safetensors" in got else "the head of a GGUF never ends" if got == "failed: " else got
        ends[kind] = ends.get(kind, 0) + 1
    # every way to end was among them, many times
    assert set(ends) == {"converted", "a file is not there", "refused", "no safetensors file", "the head of a GGUF never ends"}, ends
    assert min(ends.values()) >= 20, ends


# ---- the requests, one at a time, with a stand-in that converts nothing
HF = {"repo": "owner/model", "revision": REVISION, "weights": "model.safetensors", "tokenizer": "tokenizer.json"}
CONFIG = "{\"model_type\":\"llama\"}"


def repository(more=(), without=(), name="owner/model", weights=None):
    files = {"config.json": CONFIG, "tokenizer_config.json": "{\"chat_template\":\"{{ messages }}\"}", "tokenizer.json": "a tokenizer",
             "model.safetensors": weights or safetensors(1000), **dict(more)}
    return {f"{name}/{file}": value for file, value in files.items() if file not in without}


def requests_of(files, hf=HF, monkeypatch=None, **making):
    stand_in = StandIn().into(monkeypatch)
    hub, told = Hub(files, hf, first=400, rest=400), []
    last = answered(hub, conduct(hf, **making), told)
    return told, last, stand_in


def test_the_requests_of_one_safetensors_file(monkeypatch):
    told, last, stand_in = requests_of(repository(), monkeypatch=monkeypatch)
    assert told == [("text", "weights", "config.json"),
                    ("range", "weights", "model.safetensors", 0, 512 * 1024),
                    ("text", "weights", "tokenizer_config.json"),
                    ("bytes", "weights", "tokenizer.json"),
                    # before and total: the header counts as arrived, of the whole file
                    ("stream", "weights", "model.safetensors", 308, 1308, 308, 1308),
                    ("more", 0.4), ("more", 0.8), ("more", 1.0)]
    assert last[0] == "done" and last[1].feeds == 3
    assert stand_in.handed[-1] == "fed 1000 bytes in 3 pieces; finish()"


def test_a_conduct_ends_after_done_and_after_missing(monkeypatch):
    for files in (repository(), repository(without=["config.json"])):
        StandIn().into(monkeypatch)
        steps = conduct(HF)
        assert answered(Hub(files, HF), steps)[0] in ("done", "missing")
        with pytest.raises(StopIteration):
            steps.send(None)


def test_the_parts_go_to_the_converter_as_they_were_answered(monkeypatch):
    """The very objects, in order, and nothing of them is kept: the large bytes pass through as they did."""
    fed = []

    class Fed:
        base = 0

        def __init__(self, *given, **more):
            pass

        def feed(self, part):
            fed.append(part)
            return len(fed) / 4

        def finish(self):
            fed.append("finish")

    monkeypatch.setattr(conducting, "Conversion", Fed)
    steps = conduct(HF)
    hub, request = Hub(repository(), HF), next(steps)
    while request[0] != "stream":
        request = steps.send(getattr(hub, request[0])(*request[1:]))
    parts = [bytearray(b"a"), memoryview(b"bc"), b"", b"def"]
    for count, part in enumerate(parts, 1):
        request = steps.send(part)
        assert request == ("more", count / 4) and fed[-1] is part
    assert fed == parts  # finish() only once the answerer says there is no more
    request = steps.send(None)
    assert fed[-1] == "finish" and request[0] == "done" and isinstance(request[1], Fed)


def test_a_stream_with_nothing_in_it_is_still_asked_for(monkeypatch):
    """Whether there is anything to fetch is the answerer's to see (as inOrder() saw it)."""
    told, last, stand_in = requests_of(repository(weights=safetensors(0)), monkeypatch=monkeypatch)
    assert told[-1] == ("stream", "weights", "model.safetensors", 308, 308, 308, 308) and last[0] == "done"
    assert stand_in.handed[-1] == "fed 0 bytes in 0 pieces; finish()"


def test_what_conversion_is_made_with_is_handed_on(monkeypatch):
    sink, rows, readers = object(), object(), object()
    for hf, files in ((HF, repository()),
                      ({**HF, "weights": "model.gguf"}, repository({"model.gguf": gguf(100, 50)}))):
        _, last, _ = requests_of(files, hf, monkeypatch, dtype="int6", max_seq_len=512, sink=sink, quantize_rows=rows, readers=readers)
        assert last[1].given == dict(dtype="int6", max_seq_len=512, sink=sink, quantize_rows=rows, readers=readers)


def test_an_optional_file_that_is_not_there_is_an_answer(monkeypatch):
    """tokenizer_config.json and chat_template.jinja: without them the conversion goes on, with none."""
    told, last, stand_in = requests_of(repository(without=["tokenizer_config.json"]), monkeypatch=monkeypatch)
    assert [request[2] for request in told[2:5]] == ["tokenizer_config.json", "chat_template.jinja", "tokenizer.json"]
    assert last[0] == "done"
    assert "tokenizer_config \"\", chat_template null" in stand_in.handed[0]


def test_the_template_is_the_jinja_files_only_where_the_config_has_none(monkeypatch):
    plain = repository({"tokenizer_config.json": "{\"bos_token\":\"<s>\"}", "chat_template.jinja": "the file's"})
    told, _, stand_in = requests_of(plain, monkeypatch=monkeypatch)
    assert ("text", "weights", "chat_template.jinja") in told
    assert "tokenizer_config \"{\\\"bos_token\\\":\\\"<s>\\\"}\", chat_template \"the file's\"" in stand_in.handed[0]
    both = repository({"chat_template.jinja": "the file's"})
    told, _, stand_in = requests_of(both, monkeypatch=monkeypatch)
    assert ("text", "weights", "chat_template.jinja") not in told
    assert "chat_template null" in stand_in.handed[0] and "{{ messages }}" in stand_in.handed[0]
    # an empty file is no template
    told, _, stand_in = requests_of(repository({"tokenizer_config.json": "{}", "chat_template.jinja": ""}), monkeypatch=monkeypatch)
    assert "chat_template null" in stand_in.handed[0]


@pytest.mark.parametrize("text, says", [
    ("{\"chat_template\":\"x\"}", True), ("{\"chat_template\":[]}", True), ("{\"chat_template\":{}}", True), ("{\"chat_template\":true}", True),
    ("{\"chat_template\":1}", True), ("{\"chat_template\":[{\"name\":\"default\"}]}", True),
    ("{\"chat_template\":\"\"}", False), ("{\"chat_template\":null}", False), ("{\"chat_template\":false}", False), ("{\"chat_template\":0}", False),
    ("{\"chat_template\":0.0}", False), ("{\"chat_template\":-0.0}", False), ("{}", False), ("", False), ("null", False), ("[]", False),
    ("\"chat_template\"", False), ("7", False), ("not JSON", False), ("{\"chat_template\":\"x\",\"more\":NaN}", False),
    ("{\"chat_template\":Infinity}", False)])
def test_a_template_is_said_where_javascript_would_call_it_true(text, says):
    """As the worker told (Boolean(JSON.parse(text).chat_template), false where that throws): a list of named
    templates is one, even an empty one, and so chat_template.jinja is not asked for."""
    assert says_a_template(text) is says


def test_a_refused_tokenizer_is_followed_by_the_next_candidate(monkeypatch):
    hf = {**HF, "tokenizer": ["tokenizer.json", "tokenizer.model", "spiece.model"]}
    told, last, stand_in = requests_of(repository({"tokenizer.json": "unreadable", "spiece.model": "a sentencepiece model"}), hf, monkeypatch)
    assert [request for request in told if request[0] == "bytes"] == [("bytes", "weights", name) for name in hf["tokenizer"]]
    assert last[0] == "done"
    assert "tokenizer.json of 10 bytes" in stand_in.handed[0] and stand_in.handed[0].endswith(": refused")
    assert "spiece.model of 21 bytes" in stand_in.handed[1]
    # the first that will do is the last asked for
    told, _, _ = requests_of(repository({"tokenizer.model": "a sentencepiece model", "spiece.model": "another"}), hf, monkeypatch)
    assert [request[2] for request in told if request[0] == "bytes"] == ["tokenizer.json"]


def test_where_no_tokenizer_will_do_the_refusal_says_why_and_else_the_first_missing_one(monkeypatch):
    hf = {**HF, "tokenizer": ["tokenizer.model", "tokenizer.json", "spiece.model"]}
    # a refusal of one that is there comes before a file that is not (T144), whichever came first
    StandIn().into(monkeypatch)
    with pytest.raises(ValueError, match="tokenizer.json is of a kind the engine does not read"):
        answered(Hub(repository({"tokenizer.json": "unreadable", "spiece.model": "unreadable too"}), hf), conduct(hf))
    # none there: the first candidate is the one said
    _, last, _ = requests_of(repository(without=["tokenizer.json"]), hf, monkeypatch)
    assert last == ("missing", "weights", "tokenizer.model")


def test_the_candidates_are_the_models_or_the_three_of_a_repository_nobody_looked_at(monkeypatch):
    def asked(hf):
        told, _, _ = requests_of(repository(without=["tokenizer.json"]), hf, monkeypatch)
        return [request[2] for request in told if request[0] == "bytes"]
    assert TOKENIZERS == ("tokenizer.json", "tokenizer.model", "spiece.model")
    assert asked({**HF, "tokenizer": "spiece.model"}) == ["spiece.model"]
    assert asked({**HF, "tokenizer": ["spiece.model", "tokenizer.json"]}) == ["spiece.model", "tokenizer.json"]
    for nothing in ({}, {"tokenizer": None}, {"tokenizer": []}):
        assert asked({**{key: value for key, value in HF.items() if key != "tokenizer"}, **nothing}) == list(TOKENIZERS)


def test_a_file_that_is_needed_and_not_there_ends_with_missing(monkeypatch):
    told, last, _ = requests_of(repository(without=["config.json"]), monkeypatch=monkeypatch)
    assert told == [("text", "weights", "config.json")] and last == ("missing", "weights", "config.json")
    # config.json under the name the model gives it
    hf = {**HF, "config": "configs/text.json"}
    told, last, _ = requests_of(repository(), hf, monkeypatch)
    assert last == ("missing", "weights", "configs/text.json")
    assert requests_of(repository({"configs/text.json": CONFIG}, without=["config.json"]), hf, monkeypatch)[1][0] == "done"


def test_no_weights_at_all_ends_with_the_one_file_that_was_asked_for_first(monkeypatch):
    for index in (None, "{}", "not JSON", "{\"weight_map\":{}}", "[\"model-00001-of-00001.safetensors\"]"):
        files = repository({} if index is None else {"model.safetensors.index.json": index}, without=["model.safetensors"])
        told, last, _ = requests_of(files, monkeypatch=monkeypatch)
        assert told == [("text", "weights", "config.json"), ("range", "weights", "model.safetensors", 0, 512 * 1024),
                        ("text", "weights", "model.safetensors.index.json")]
        assert last == ("missing", "weights", "model.safetensors")


def test_a_file_that_is_no_safetensors_file_is_looked_for_in_the_index_and_refused_as_itself(monkeypatch):
    StandIn().into(monkeypatch)
    for head in (b"", b"1234567", struct.pack("<Q", 1) + b"{}", struct.pack("<Q", 100_000_001) + b"{}"):
        hub = Hub(repository(weights=File(head)), HF)
        with pytest.raises(ValueError, match="This is not a safetensors file."):
            answered(hub, conduct(HF))
        assert hub.asked[-1].endswith("model.safetensors.index.json")
    # a header of two bytes is one (which the stand-in then finds nothing in), and so is one of 100 MB
    for length in (2, 100_000_000):
        hub = Hub(repository(weights=File(struct.pack("<Q", length) + b"{}", 8 + length)), HF)
        with pytest.raises((KeyError, ValueError)) as refused:
            answered(hub, conduct(HF))
        assert "not a safetensors" not in str(refused.value) and hub.asked[-1].endswith("tokenizer.json")


def test_the_header_of_a_safetensors_file_past_its_first_piece_is_asked_for_from_the_start(monkeypatch):
    told, last, stand_in = requests_of(repository(weights=safetensors(10, header=600_000)), monkeypatch=monkeypatch)
    assert told[1:3] == [("range", "weights", "model.safetensors", 0, 524288), ("range", "weights", "model.safetensors", 0, 600_008)]
    assert "a header of 600000 characters, base 600008, start 600008" in stand_in.handed[0]
    # exactly the first piece: not asked for twice
    told, _, _ = requests_of(repository(weights=safetensors(10, header=524288 - 8)), monkeypatch=monkeypatch)
    assert [request[0] for request in told[1:3]] == ["range", "text"]
    told, _, _ = requests_of(repository(weights=safetensors(10, header=524288 - 7)), monkeypatch=monkeypatch)
    assert told[2] == ("range", "weights", "model.safetensors", 0, 524289)


def test_the_size_is_asked_for_where_a_range_did_not_say_it_and_only_there(monkeypatch):
    StandIn().into(monkeypatch)
    hub, told = Hub(repository(weights=safetensors(10, header=600_000)), HF, unsaid=True), []
    answered(hub, conduct(HF), told)
    # after the first piece of the head, not after the second: the size is known by then
    assert [request[0] for request in told[:5]] == ["text", "range", "size", "range", "text"]
    assert told[2] == ("size", "weights", "model.safetensors")
    # whatever the answerer has where it has no size: nothing, 0, or JavaScript's NaN
    for nothing in (None, 0, float("nan")):
        steps = conduct(HF)
        assert next(steps)[0] == "text"
        assert steps.send(CONFIG)[0] == "range"
        assert steps.send((safetensors(10).head, nothing)) == ("size", "weights", "model.safetensors")
        assert steps.send(318.0)[0] == "text"  # (a JavaScript number comes as a float)
        steps.send(None), steps.send(None)
        stream = steps.send(b"a tokenizer")
        assert stream == ("stream", "weights", "model.safetensors", 308, 318, 308, 318)
        assert all(type(value) is int for value in stream[3:])  # (and goes back as an integer: a place in a file)
    # a GGUF's head: after every piece, as the worker asked (T381 is T374.3's)
    hf = {**HF, "weights": "model.gguf"}
    hub, told = Hub(repository({"model.gguf": gguf(3 * MiB, 100)}), hf, unsaid=True), []
    answered(hub, conduct(hf), told)
    assert [request[0] for request in told] == ["range", "size", "range", "size", "stream", "more"]


def test_shards_are_fed_one_after_another_and_the_progress_counts_across_them(monkeypatch):
    names = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    index = json.dumps({"weight_map": {"a": names[2], "b": names[0], "c": names[1], "d": names[0]}})
    shards = dict(zip(names, (safetensors(500), safetensors(30, header=100), safetensors(700, header=40))))
    told, last, stand_in = requests_of(repository({"model.safetensors.index.json": index, **shards}, without=["model.safetensors"]), monkeypatch=monkeypatch)
    assert [request[2] for request in told[2:6]] == ["model.safetensors.index.json", *names]
    assert stand_in.handed[0] == "joined_shards(3 headers)"
    assert "base 0, start 0" in stand_in.handed[1]
    # each from its own base, for the bytes of its tensors; before: the bytes of the shards before it, of them all
    assert [request for request in told if request[0] == "stream"] == [
        ("stream", "weights", names[0], 308, 808, 0, 1230), ("stream", "weights", names[1], 108, 138, 500, 1230),
        ("stream", "weights", names[2], 48, 748, 530, 1230)]
    assert stand_in.handed[-1] == "fed 1230 bytes in 5 pieces; finish()"
    # a shard that is not there ends it, with that shard
    del shards[names[1]]
    _, last, _ = requests_of(repository({"model.safetensors.index.json": index, **shards}, without=["model.safetensors"]), monkeypatch=monkeypatch)
    assert last == ("missing", "weights", names[1])


def test_one_shard_named_by_the_index_is_that_file(monkeypatch):
    index = json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"}})
    files = repository({"model.safetensors.index.json": index, "model-00001-of-00001.safetensors": safetensors(900)}, without=["model.safetensors"])
    told, last, stand_in = requests_of(files, monkeypatch=monkeypatch)
    assert told[-4] == ("stream", "weights", "model-00001-of-00001.safetensors", 308, 1208, 308, 1208)
    assert not any("joined_shards" in line for line in stand_in.handed) and "base 308, start 308" in stand_in.handed[0]
    del files["owner/model/model-00001-of-00001.safetensors"]
    assert requests_of(files, monkeypatch=monkeypatch)[1] == ("missing", "weights", "model-00001-of-00001.safetensors")


def test_the_shards_of_an_index_are_in_the_order_of_their_names():
    assert shards_of("{\"weight_map\":{\"a\":\"b-2\",\"b\":\"b-10\",\"c\":\"a\",\"d\":\"b-2\",\"e\":\"B\"}}") == ["B", "a", "b-10", "b-2"]
    # (JavaScript's order, by UTF-16 code units: a character past the BMP sorts before U+FF5E)
    assert shards_of(json.dumps({"weight_map": {"a": "～", "b": "\U0001f600"}})) == ["\U0001f600", "～"]
    for nothing in ("", "not JSON", "{}", "null", "[]", "7", "{\"weight_map\":null}", "{\"weight_map\":[]}", "{\"weight_map\":\"x\"}"):
        assert shards_of(nothing) == []


GGUF_HF = {"repo": "maker/model-GGUF", "revision": REVISION, "weights": "model.Q8_0.gguf"}


def test_a_ggufs_head_is_asked_for_in_growing_pieces_from_its_start(monkeypatch):
    for base, pieces in ((1000, [2]), (2 * MiB, [2]), (2 * MiB + 1, [2, 8]), (8 * MiB + 1, [2, 8, 32]), (32 * MiB + 1, [2, 8, 32, 128])):
        told, last, stand_in = requests_of({"maker/model-GGUF/model.Q8_0.gguf": gguf(base, 700)}, GGUF_HF, monkeypatch)
        assert [request for request in told if request[0] == "range"] == [("range", "weights", "model.Q8_0.gguf", 0, piece * MiB) for piece in pieces]
        assert [request for request in told if request[0] == "stream"] == [("stream", "weights", "model.Q8_0.gguf", base, base + 700, base, base + 700)]
        assert last[0] == "done" and last[1].base == base
        # nothing but the GGUF is asked for
        assert {request[2] for request in told if len(request) > 2} == {"model.Q8_0.gguf"}


def test_a_gguf_whose_head_never_ends_is_the_converters_to_refuse(monkeypatch):
    """The file ends before its head does: Incomplete for good, once a piece as long as the file was asked for."""
    StandIn().into(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": File(b"GGUF" + struct.pack("<Q", 20 * MiB), 3 * MiB)}, GGUF_HF)
    with pytest.raises(Incomplete):
        answered(hub, conduct(GGUF_HF))
    assert [line.split("bytes=")[1] for line in hub.asked] == ["0-2097151", "0-8388607"]
    # and a GGUF that is not there
    assert answered(Hub({}, GGUF_HF), conduct(GGUF_HF)) == ("missing", "weights", "model.Q8_0.gguf")


def test_a_ggufs_head_is_not_asked_for_again_where_the_file_is_as_long_as_the_piece(monkeypatch):
    """The worker's end of the growing is "the piece asked for is as long as the file" (>=): a file of exactly 2 MiB
    that is still incomplete is refused after one range, not asked for a second time."""
    StandIn().into(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": File(b"GGUF" + struct.pack("<Q", 20 * MiB), 2 * MiB)}, GGUF_HF)
    with pytest.raises(Incomplete):
        answered(hub, conduct(GGUF_HF))
    assert [line.split("bytes=")[1] for line in hub.asked] == ["0-2097151"]


def test_what_the_converter_refuses_of_a_head_is_not_asked_for_again(monkeypatch):
    def refuse(*given, **more):
        raise ValueError("This GGUF names no BOS token")
    stand_in = StandIn().into(monkeypatch)
    monkeypatch.setattr(stand_in.Conversion, "from_gguf", refuse)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": gguf(3 * MiB, 10)}, GGUF_HF)
    with pytest.raises(ValueError, match="names no BOS"):
        answered(hub, conduct(GGUF_HF))
    assert len(hub.asked) == 1


def test_a_gguf_with_a_vocabulary_asks_the_original_for_everything_but_the_weights(monkeypatch):
    hf = {**GGUF_HF, "tokenizer": "never asked for", "config": "config.json",
          "vocabulary": {"repo": "owner/model", "revision": REVISION[::-1], "tokenizer": ["tokenizer.json", "tokenizer.model"]}}
    files = {**repository({"tokenizer.model": "a sentencepiece model", "chat_template.jinja": "a template",
                           "tokenizer_config.json": "{}"}, without=["tokenizer.json", "model.safetensors"]),
             **repository({"tokenizer.model": "the maker's"}, name="maker/model-GGUF", weights=gguf(9 * MiB, 500))}
    files["maker/model-GGUF/model.Q8_0.gguf"] = files.pop("maker/model-GGUF/model.safetensors")
    told, last, stand_in = requests_of(files, hf, monkeypatch)
    assert told[:7] == [("text", "vocabulary", "config.json"),
                        *[("range", "weights", "model.Q8_0.gguf", 0, piece * MiB) for piece in (2, 8, 32)],
                        ("text", "vocabulary", "tokenizer_config.json"), ("text", "vocabulary", "chat_template.jinja"),
                        ("bytes", "vocabulary", "tokenizer.json")]
    assert told[7:9] == [("bytes", "vocabulary", "tokenizer.model"),
                         ("stream", "weights", "model.Q8_0.gguf", 9 * MiB, 9 * MiB + 500, 9 * MiB, 9 * MiB + 500)]
    assert "tokenizer.model of 21 bytes" in stand_in.handed[-2] and "a header of 11 characters" in stand_in.handed[-2]
    assert last[0] == "done"
    # the model's own tokenizer where the vocabulary names none; the original's config.json is one there must be
    del hf["vocabulary"]["tokenizer"]
    assert requests_of(files, hf, monkeypatch)[1] == ("missing", "vocabulary", "never asked for")
    del files["owner/model/config.json"]
    assert requests_of(files, hf, monkeypatch)[1] == ("missing", "vocabulary", "config.json")


def test_a_vocabulary_makes_the_weights_a_gguf_whatever_their_name(monkeypatch):
    """As the worker read it: a model that names a vocabulary is read as a GGUF's weights."""
    hf = {**HF, "vocabulary": {"repo": "owner/original", "revision": REVISION}}
    files = {**repository(weights=gguf(100, 10)), **repository(name="owner/original")}
    told, last, stand_in = requests_of(files, hf, monkeypatch)
    assert told[1] == ("range", "weights", "model.safetensors", 0, 2 * MiB) and stand_in.handed[0].startswith("gguf_weights(")


def test_the_rows_of_the_table_take_a_model_by_how_it_is_listed():
    def taken(hf):
        return [row.title for row in SOURCES if row.takes(hf)]
    vocabulary = {"vocabulary": {"repo": "owner/original"}}
    assert [row.title for row in SOURCES] == ["a GGUF alone", "a GGUF with the vocabulary of another repository",
                                             "one safetensors file", "the safetensors files its index names"]
    assert taken({"weights": "model.Q8_0.gguf"}) == ["a GGUF alone"]
    assert taken({"weights": "model.Q8_0.gguf", "vocabulary": None}) == ["a GGUF alone"]
    assert taken({"weights": "model.Q8_0.gguf", **vocabulary}) == ["a GGUF with the vocabulary of another repository"]
    assert taken({"weights": "model.safetensors", **vocabulary}) == ["a GGUF with the vocabulary of another repository"]
    assert taken({"weights": "model.safetensors"}) == ["one safetensors file", "the safetensors files its index names"]
    assert taken({"weights": "weights/model.gguf.safetensors"}) == ["one safetensors file", "the safetensors files its index names"]
    assert [row.alone for row in SOURCES] == [True, False, False, False]


def test_a_conduct_that_is_closed_leaves_nothing_running(monkeypatch):
    StandIn().into(monkeypatch)
    for stop in range(8):
        steps, hub, streams = conduct(HF), Hub(repository(), HF, first=400, rest=400), []
        request = next(steps)
        for _ in range(stop):
            request = steps.send(answer(hub, request, streams))
        steps.close()
        with pytest.raises(StopIteration):
            next(steps)


def test_what_the_answerer_fails_with_is_not_the_conducts_to_take(monkeypatch):
    """A failure that is no "not there" is thrown at whoever answers, or the generator is closed: the conduct catches
    neither and asks for nothing more."""
    StandIn().into(monkeypatch)
    for stop in range(5):
        steps, hub, streams = conduct(HF), Hub(repository(), HF), []
        request = next(steps)
        for _ in range(stop):
            request = steps.send(answer(hub, request, streams))
        with pytest.raises(ConnectionError):
            steps.throw(ConnectionError("the line"))
