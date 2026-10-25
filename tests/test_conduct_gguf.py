# The conduct of a conversion (src/python/convert/conduct.py) on a GGUF's head (T.374.3): asked for in pieces that end
# ever further, each from where the last ended, until the converter reads all of it; what of the tensors came with it
# fed to the conversion and not asked for again. Moved out of tests/test_conduct.py (T.420: 1,042 lines, past the 800 of AGENTS.md's measure), whose
# stand-ins and repository of one model it shares; the tests are the same.
import struct

import pytest
from conduct_hub import ASKED, File, Hub, MiB, StandIn, answer, answered, answering, gguf, sound
from test_conduct import HF, REVISION, repository, requests_of

from convert import conduct as conducting
from convert.conduct import conduct
from convert.gguf import Incomplete

GGUF_HF = {"repo": "maker/model-GGUF", "revision": REVISION, "weights": "model.Q8_0.gguf"}


def fed_to(monkeypatch):
    """The stand-in, whose conversion also writes down every piece it is fed: (the stand-in, [the pieces, as bytes])."""
    stand_in, pieces = StandIn().into(monkeypatch), []

    class Fed(stand_in.Conversion):
        def feed(self, data):
            pieces.append(bytes(data))
            return super().feed(data)

    monkeypatch.setattr(conducting, "Conversion", Fed)
    return stand_in, pieces


def same(pieces, data):
    """Whether the pieces, one after another, are data. (A function, so that a failure does not have pytest write out
    how hundreds of kilobytes differ: that took gigabytes.)"""
    return b"".join(pieces) == data


def marked(base, data):
    """A GGUF of the stand-in's whose tensors say where each of their bytes stands in the file."""
    tensors = b"".join(struct.pack("<I", at) for at in range(base, base + data, 4))
    return File(b"GGUF" + struct.pack("<Q", base) + bytes(base - 12) + tensors)


def test_a_ggufs_head_is_asked_for_in_pieces_that_end_ever_further_each_from_where_the_last_ended(monkeypatch):
    """T.374.3: 2 MiB, then what is missing of 8, of 32, of 128: no byte twice. The stream begins where the last piece
    ends (or the file, where that came first), and so does the count of what has arrived."""
    for base, ends in ((1000, [2]), (2 * MiB, [2]), (2 * MiB + 1, [2, 8]), (8 * MiB + 1, [2, 8, 32]), (32 * MiB + 1, [2, 8, 32, 128])):
        for data in (700, 200 * MiB):
            told, last, stand_in = requests_of({"maker/model-GGUF/model.Q8_0.gguf": gguf(base, data)}, GGUF_HF, monkeypatch)
            begins = [0, *ends[:-1]]
            assert [request for request in told if request[0] == "range"] == \
                [("range", "weights", "model.Q8_0.gguf", begin * MiB, end * MiB) for begin, end in zip(begins, ends)]
            size = base + data
            hand = min(ends[-1] * MiB, size)
            assert [request for request in told if request[0] == "stream"] == [("stream", "weights", "model.Q8_0.gguf", hand, size, hand, size)]
            assert last[0] == "done" and last[1].base == base and last[1].fed == data
            # the converter was handed the beginning joined, as long as what was asked for of it
            assert f"the first {hand} bytes)" in stand_in.handed[-2]
            # nothing but the GGUF is asked for
            assert {request[2] for request in told if len(request) > 2} == {"model.Q8_0.gguf"}


@pytest.mark.parametrize("base, data, first", [
    (1000, 5000, 4096),     # the head and some tensors with the first piece, the rest streamed
    (1000, 5000, 1000),     # the first piece ends where the head does: nothing in hand, all of it streamed
    (1000, 5000, 1004),     # four bytes of the tensors in hand
    (5000, 5000, 4096),     # a second piece (to 16384): the file ends within it
    (5000, 60000, 4096),    # a second piece, and a stream after it
    (17000, 60000, 4096),   # a third (to 65536)
    (17000, 600000, 4096),  # and a stream after the third
    (1000, 5000, 6000),     # the file ends exactly where the first piece does
    (1000, 5000, 60000),    # a file shorter than the first piece
    (1000, 0, 4096),        # a file that ends where its head does: no tensors
    (4096, 0, 4096),        # and exactly at the first piece's end
    (5000, 0, 4096),        # and after a second piece
])
def test_the_tensors_that_came_with_a_ggufs_head_are_fed_and_not_asked_for_again(base, data, first, monkeypatch):
    """T.374.3: every byte of the tensors reaches the conversion once and in the order of the file: first what came
    with the head, from the conduct, then the stream's parts, from whoever answers. No byte of the file is asked for
    twice, and none past what the pieces of the head and one stream cover."""
    monkeypatch.setattr(conducting, "GGUF_HEAD", first)
    monkeypatch.setattr(conducting, "AHEAD", 1000)
    file = marked(base, data)
    for hf, files in ((GGUF_HF, {}), ({**GGUF_HF, "vocabulary": {"repo": "owner/model", "revision": REVISION, "tokenizer": "tokenizer.json"}},
                                      repository(without=["model.safetensors"]))):
        stand_in, pieces = fed_to(monkeypatch)
        hub, told = Hub({"maker/model-GGUF/model.Q8_0.gguf": file, **files}, hf, first=400, rest=400), []
        last = answered(hub, conduct(hf), told)
        assert last[0] == "done" and same(pieces, file[base:])
        ranges = [request[3:5] for request in told if request[0] == "range"]
        hand = min(ranges[-1][1], file.size)
        assert ranges[0][0] == 0 and all(before[1] == after[0] for before, after in zip(ranges, ranges[1:]))
        # the stream is asked for even where nothing is left of the file (whether there is anything to fetch is the
        # answerer's to see), from where the bytes in hand end, which count as arrived
        assert told[-1] == ("stream", "weights", "model.Q8_0.gguf", hand, file.size, hand, file.size)
        # what the conduct fed: pieces of AHEAD at most, nothing where the head came alone; then the parts of the line
        mine = -(-(hand - base) // 1000)
        assert [len(piece) for piece in pieces[:mine]] == [min(1000, hand - base - at) for at in range(0, hand - base, 1000)]
        assert all(len(piece) <= 400 for piece in pieces[mine:]) and len(pieces) == mine + -(-(file.size - hand) // 400)
        # of the file's bytes, the hub was asked for each once
        asked = sorted(tuple(int(edge) for edge in line.split("bytes=")[1].split("-")) for line in hub.asked if ".gguf bytes=" in line)
        assert asked[0][0] == 0 and all(before[1] + 1 == after[0] for before, after in zip(asked, asked[1:]))
        assert min(asked[-1][1] + 1, file.size) == file.size
        sound(told, last[1], "converted", hf, hub)


def test_a_conversion_made_only_after_the_tokenizer_is_fed_what_came_with_the_head_before_the_stream(monkeypatch):
    """A GGUF with another repository's vocabulary: its beginning is in hand while the tokenizer is asked for, and what
    it holds of the tensors goes to the conversion as soon as there is one: before the stream is asked for."""
    hf = {**GGUF_HF, "vocabulary": {"repo": "owner/model", "revision": REVISION, "tokenizer": ["tokenizer.json", "spiece.model"]}}
    stand_in, pieces = fed_to(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": marked(1000, 3 * MiB), **repository({"tokenizer.json": "unreadable", "spiece.model": "a model"})}, hf)
    steps = conduct(hf)
    request = next(steps)
    for _ in range(ASKED):
        if request[0] == "stream":
            break
        assert pieces == []
        request = steps.send(answer(hub, request))
    assert request[0] == "stream", f"no stream asked for in {ASKED} requests"
    assert [len(piece) for piece in pieces] == [2 * MiB - 1000] and request[3:7] == (2 * MiB, 3 * MiB + 1000, 2 * MiB, 3 * MiB + 1000)
    assert stand_in.handed[1].endswith(": refused") and "spiece.model" in stand_in.handed[2]


def test_what_came_with_a_head_is_not_kept_once_it_is_fed(monkeypatch):
    """The beginning of a GGUF (as much as 32 MiB) is let go of before the stream is asked for: nothing of the conduct
    holds it while the file is streamed."""
    import gc
    import weakref

    class Kept(bytearray):  # (bytes that a weak reference can be made to)
        pass

    held = []

    def joined(data, more):
        held.append(Kept(bytes(data) + bytes(more)))
        return held[-1]

    monkeypatch.setattr(conducting, "joined", joined)
    StandIn().into(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": gguf(3 * MiB, 20 * MiB)}, GGUF_HF)
    steps = conduct(GGUF_HF)
    request = next(steps)
    request = answering(hub, steps, request, ("stream",))
    alive = [weakref.ref(data) for data in held]
    assert len(held) == 2 and len(held[-1]) == 8 * MiB
    del held[:]
    gc.disable()
    try:
        assert [data() for data in alive] == [None, None]
    finally:
        gc.enable()
    steps.close()


def test_a_server_that_answers_a_piece_short_is_asked_from_where_its_answer_ended(monkeypatch):
    """A piece of a head with fewer bytes than were asked for, of a file that goes on: the next piece begins where
    the bytes in hand end, and the stream after them. Nothing is skipped and nothing asked for twice."""
    file = marked(40000, 30000)

    class Short(Hub):
        def range(self, where, name, begin, end):
            data, size = super().range(where, name, begin, end)
            return data[:len(data) * 2 // 3], size

    monkeypatch.setattr(conducting, "GGUF_HEAD", 4096)
    stand_in, pieces = fed_to(monkeypatch)
    hub, told = Short({"maker/model-GGUF/model.Q8_0.gguf": file}, GGUF_HF, first=5000, rest=5000), []
    last = answered(hub, conduct(GGUF_HF), told)
    assert last[0] == "done" and same(pieces, file[40000:])
    # (two thirds of 4096; then of what is missing of 16384; of 65536: 47634 in hand, past the head's 40000)
    assert [request[3:5] for request in told if request[0] == "range"] == [(0, 4096), (2730, 16384), (11832, 65536)]
    assert told[-1][3:7] == (47634, 70000, 47634, 70000)


def test_a_gguf_whose_head_never_ends_is_the_converters_to_refuse(monkeypatch):
    """The file ends before its head does: Incomplete for good, once a piece that ends at the file's end or past it
    was asked for."""
    StandIn().into(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": File(b"GGUF" + struct.pack("<Q", 20 * MiB), 3 * MiB)}, GGUF_HF)
    with pytest.raises(Incomplete):
        answered(hub, conduct(GGUF_HF))
    assert [line.split("bytes=")[1] for line in hub.asked] == ["0-2097151", "2097152-8388607"]
    # and a GGUF that is not there
    assert answered(Hub({}, GGUF_HF), conduct(GGUF_HF)) == ("missing", "weights", "model.Q8_0.gguf")
    # or that is there no more when the second piece of its head is asked for
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": gguf(3 * MiB, 10)}, GGUF_HF)
    steps = conduct(GGUF_HF)
    assert steps.send(answer(hub, next(steps)))[:5] == ("range", "weights", "model.Q8_0.gguf", 2 * MiB, 8 * MiB)
    assert steps.send(None) == ("missing", "weights", "model.Q8_0.gguf")


def test_a_ggufs_head_is_not_asked_for_again_where_the_file_is_as_long_as_the_piece(monkeypatch):
    """The end of the growing is "the piece asked for ends at the file's end or past it" (>=): a file of exactly
    2 MiB that is still incomplete is refused after one range, not asked for a second time."""
    StandIn().into(monkeypatch)
    hub = Hub({"maker/model-GGUF/model.Q8_0.gguf": File(b"GGUF" + struct.pack("<Q", 20 * MiB), 2 * MiB)}, GGUF_HF)
    with pytest.raises(Incomplete):
        answered(hub, conduct(GGUF_HF))
    assert [line.split("bytes=")[1] for line in hub.asked] == ["0-2097151"]


def test_a_head_that_never_ends_of_a_file_said_to_be_endless_is_given_up(monkeypatch):
    """An answerer that says a size its file has not (here: larger than any file) and never has more bytes: the
    pieces are bounded, and the end is the converter's Incomplete all the same."""
    class Endless(Hub):
        def range(self, where, name, begin, end):
            data, _ = super().range(where, name, begin, end)
            return data, 1 << 200

        def parts(self, where, name, begin, end):  # (it would never end)
            raise AssertionError("a file whose head was never read is streamed")

    StandIn().into(monkeypatch)
    hub = Endless({"maker/model-GGUF/model.Q8_0.gguf": File(b"GGUF" + struct.pack("<Q", 1 << 62), 3 * MiB)}, GGUF_HF)
    with pytest.raises(Incomplete):
        answered(hub, conduct(GGUF_HF))
    assert len(hub.asked) == 1 + conducting.GGUF_PIECES == 17
    assert hub.asked[1].endswith("bytes=2097152-8388607") and hub.asked[2].endswith("bytes=3145728-33554431")


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
                        *[("range", "weights", "model.Q8_0.gguf", begin * MiB, end * MiB) for begin, end in ((0, 2), (2, 8), (8, 32))],
                        ("text", "vocabulary", "tokenizer_config.json"), ("text", "vocabulary", "chat_template.jinja"),
                        ("bytes", "vocabulary", "tokenizer.json")]
    assert told[7:9] == [("bytes", "vocabulary", "tokenizer.model"),
                         # (the file ends within the third piece of its head: nothing is left to stream)
                         ("stream", "weights", "model.Q8_0.gguf", 9 * MiB + 500, 9 * MiB + 500, 9 * MiB + 500, 9 * MiB + 500)]
    assert stand_in.handed[-1] == "fed 500 bytes in 1 pieces; finish()"
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
