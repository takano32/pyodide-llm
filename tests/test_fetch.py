# tests/fetching.py (the files of a model of the list for the measurements in CI: the tools reach it through
# tests/conducting.py's answerers since T374.4, which a model of the list is fetched by here): a
# download that stops short is asked for again. (T357: the loop is download(), which the reference
# tools use too; and ranged(), a range of a file, held to its length the same way. T374.4: sized().) http.client's read(amount) returns what came when the connection
# closes early and says nothing, so a 4.5 GB GGUF that stopped short was a file that "ended before all of its tensors
# were read" a minute into its conversion (the review of T247, a runner's CI run), and the failure named the converter.
import io

import urllib.error

import pytest
import fetching
from conducting import Fetched


class Response:
    def __init__(self, body, length=None):
        self.stream = io.BytesIO(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def read(self, amount):
        return self.stream.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False


ENTRY = {"hf": {"repo": "owner/repository", "revision": "0123abc"}}


def fetch(entry, name, directory):
    """A file of the entry's repository as tests/fixed_outputs.py and tests/hf_fetch.py get it: the answerer's path()."""
    return Fetched.of(entry["hf"], directory).path("weights", name)


def serving(monkeypatch, answers):
    asked = []

    def urlopen(url, timeout):
        asked.append(url)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(fetching.urllib.request, "urlopen", urlopen)
    return asked


def test_a_download_that_stopped_short_is_asked_for_again(tmp_path, monkeypatch):
    asked = serving(monkeypatch, [Response(b"x" * 40, 100), Response(b"y" * 100, 100)])
    path = fetch(ENTRY, "model.gguf", tmp_path)
    assert path.read_bytes() == b"y" * 100 and len(asked) == 2
    assert not path.with_suffix(".gguf.part").exists()


def test_three_downloads_that_stopped_short_are_an_error_and_leave_no_file(tmp_path, monkeypatch):
    asked = serving(monkeypatch, [Response(b"x" * 40, 100) for _ in range(3)])
    with pytest.raises(OSError, match="40 of 100 bytes"):
        fetch(ENTRY, "model.gguf", tmp_path)
    assert len(asked) == 3
    assert not (tmp_path / "owner--repository" / "0123abc" / "model.gguf").exists()


def test_a_whole_download_and_one_that_names_no_length_are_taken_as_they_come(tmp_path, monkeypatch):
    serving(monkeypatch, [Response(b"z" * 100, 100), Response(b"w" * 7)])
    assert fetch(ENTRY, "a.bin", tmp_path).read_bytes() == b"z" * 100
    assert fetch(ENTRY, "b.bin", tmp_path).read_bytes() == b"w" * 7
    # T357: a file that is there is not asked for again; a file the repository does not have is refused at once (a split
    # model's index is found by that 404), or is nothing where it is optional; a server's own failure is asked again
    refused = lambda code: urllib.error.HTTPError("https://huggingface.co/x", code, "refused", {}, None)
    asked = serving(monkeypatch, [refused(404), refused(404), refused(503), Response(b"v" * 5, 5)])
    assert fetch(ENTRY, "a.bin", tmp_path).read_bytes() == b"z" * 100 and asked == []
    # (the answerer: a 404 is "not there", any other refusal is raised)
    assert fetch(ENTRY, "c.bin", tmp_path) is None and len(asked) == 1
    serving(monkeypatch, [refused(403)])
    with pytest.raises(urllib.error.HTTPError):
        fetch(ENTRY, "c.bin", tmp_path)
    asked = serving(monkeypatch, [refused(404), refused(404), refused(503), Response(b"v" * 5, 5)])
    with pytest.raises(urllib.error.HTTPError):
        fetching.download("https://huggingface.co/x", tmp_path / "c.bin")
    assert len(asked) == 1
    assert fetching.download("https://huggingface.co/x", tmp_path / "d.bin", optional=True) is None and len(asked) == 2
    assert fetching.download("https://huggingface.co/x", tmp_path / "e.bin").read_bytes() == b"v" * 5 and len(asked) == 4
    # a range that came short is asked for again, and the last try's error is the caller's
    class Whole(Response):
        def read(self, amount=None):
            return self.stream.read()

    told = []
    serving(monkeypatch, [Whole(b"r" * 3), Whole(b"s" * 8)])
    assert fetching.ranged("https://huggingface.co/x", 16, 8, said=told.append) == b"s" * 8 and len(told) == 1
    serving(monkeypatch, [Whole(b"r" * 3), Whole(b"r" * 3)])
    with pytest.raises(OSError, match="3 bytes of 8"):
        fetching.ranged("https://huggingface.co/x", 16, 8, tries=2)


def test_a_body_that_closed_before_its_length_is_asked_for_again(monkeypatch):
    # T357's review: read() with no amount raises http.client.IncompleteRead there, which is no OSError (a real server
    # closing early, tried by hand: the first version of ranged() let it through unasked-again)
    import http.client

    class Closed(Response):
        def read(self, amount=None):
            raise http.client.IncompleteRead(b"abc", 5)

    class Whole(Response):
        def read(self, amount=None):
            return self.stream.read()
    asked = serving(monkeypatch, [Closed(b""), Whole(b"s" * 8)])
    assert fetching.ranged("https://huggingface.co/x", 0, 8) == b"s" * 8 and len(asked) == 2


def test_the_size_of_a_file_is_what_the_answer_to_a_range_of_one_byte_says(monkeypatch):
    # T374.4: for whoever reads a file by its ranges and must not ask past its end
    class Ranged(Response):
        def __init__(self, said, status=206):
            super().__init__(b"x")
            self.headers, self.status = said, status

    refused = lambda code: urllib.error.HTTPError("https://huggingface.co/x", code, "refused", {}, None)
    asked = serving(monkeypatch, [Ranged({"Content-Range": "bytes 0-0/1234"})])
    assert fetching.sized("https://huggingface.co/x") == 1234
    assert asked[0].get_header("Range") == "bytes=0-0"
    # a server that sends the whole file says its length; a file that is not there is None, at once
    serving(monkeypatch, [Ranged({"Content-Length": "77"}, status=200)])
    assert fetching.sized("https://huggingface.co/x") == 77
    asked = serving(monkeypatch, [refused(404)])
    assert fetching.sized("https://huggingface.co/x") is None and len(asked) == 1
    # a server's own failure and a broken connection are asked again; another refusal, and an answer that says no size, are raised
    asked = serving(monkeypatch, [refused(503), OSError("reset"), Ranged({"Content-Range": "bytes 0-0/5"})])
    assert fetching.sized("https://huggingface.co/x") == 5 and len(asked) == 3
    serving(monkeypatch, [refused(403)])
    with pytest.raises(urllib.error.HTTPError):
        fetching.sized("https://huggingface.co/x")
    serving(monkeypatch, [Ranged({"Content-Range": "bytes 0-0/*"}) for _ in range(2)])
    with pytest.raises(OSError, match="does not say the size"):
        fetching.sized("https://huggingface.co/x", tries=2)
