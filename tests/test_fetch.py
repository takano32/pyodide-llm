# tests/fixed_outputs.py's fetch() (the files of a model of the list for the measurements in CI, tests/hf_fetch.py): a
# download that stops short is asked for again. http.client's read(amount) returns what came when the connection
# closes early and says nothing, so a 4.5 GB GGUF that stopped short was a file that "ended before all of its tensors
# were read" a minute into its conversion (the review of T247, a runner's CI run), and the failure named the converter.
import io

import pytest
import fixed_outputs


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


def serving(monkeypatch, answers):
    asked = []

    def urlopen(url, timeout):
        asked.append(url)
        return answers.pop(0)

    monkeypatch.setattr(fixed_outputs.urllib.request, "urlopen", urlopen)
    return asked


def test_a_download_that_stopped_short_is_asked_for_again(tmp_path, monkeypatch):
    asked = serving(monkeypatch, [Response(b"x" * 40, 100), Response(b"y" * 100, 100)])
    path = fixed_outputs.fetch(ENTRY, "model.gguf", tmp_path)
    assert path.read_bytes() == b"y" * 100 and len(asked) == 2
    assert not path.with_suffix(".gguf.part").exists()


def test_three_downloads_that_stopped_short_are_an_error_and_leave_no_file(tmp_path, monkeypatch):
    asked = serving(monkeypatch, [Response(b"x" * 40, 100) for _ in range(3)])
    with pytest.raises(OSError, match="40 of 100 bytes"):
        fixed_outputs.fetch(ENTRY, "model.gguf", tmp_path)
    assert len(asked) == 3
    assert not (tmp_path / "owner--repository" / "0123abc" / "model.gguf").exists()


def test_a_whole_download_and_one_that_names_no_length_are_taken_as_they_come(tmp_path, monkeypatch):
    serving(monkeypatch, [Response(b"z" * 100, 100), Response(b"w" * 7)])
    assert fixed_outputs.fetch(ENTRY, "a.bin", tmp_path).read_bytes() == b"z" * 100
    assert fixed_outputs.fetch(ENTRY, "b.bin", tmp_path).read_bytes() == b"w" * 7
