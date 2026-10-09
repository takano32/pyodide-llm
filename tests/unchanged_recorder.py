# unchanged_recorder.py (T346)
# A pytest plugin that writes down what the unit tests make the converter and the engine do, so that a refactoring can
# be held to "nothing changed" on every made-up model the tests already build: every architecture, every dtype, GGUF
# and safetensors, every tokenizer.
#
#   UNCHANGED_RECORD=<a .json file> PYTHONPATH=<this directory> python -m pytest <a tree>/tests -q -p unchanged_recorder
#
# For every test, a count and a hash of: the checkpoints its conversions finish (Stream.finish: the bytes), the options
# and the tokenizer.bin of each Conversion, the logits of every Llama.forward, the ids of every Tokenizer.encode.
# tests/unchanged.mjs runs it on the working tree and on another commit's tree and compares the two files. The names
# it wraps are the ones a refactoring keeps where they are (the windows llama2_convert and llama2_numpy).
import hashlib
import json
import os

import pytest

RECORDS, NOW = {}, [None]


def note(kind, data):
    if NOW[0] is None:
        return
    count_and_hash = RECORDS.setdefault(NOW[0], {}).setdefault(kind, [0, hashlib.sha256()])
    count_and_hash[0] += 1
    count_and_hash[1].update(data)


def after(owner, name, tell):
    """owner.name, telling tell(self, result, arguments) what it returned"""
    plain = getattr(owner, name)

    def wrapped(self, *args, **kwargs):
        result = plain(self, *args, **kwargs)
        try:
            tell(self, result, args)
        except Exception as error:  # the recorder must not fail a test: what it could not hash is said in the record
            note("the recorder failed", f"{name}: {type(error).__name__}".encode())
        return result

    setattr(owner, name, wrapped)


def as_bytes(value):
    return value.tobytes() if hasattr(value, "tobytes") else bytes(value)


def pytest_collection_finish(session):
    import numpy as np
    import llama2_convert
    import llama2_numpy

    def finished(stream, result, args):
        out = getattr(stream, "out", None)
        note("checkpoints", as_bytes(out) if out is not None else b"(to a sink)")

    def started(conversion, result, args):
        note("options", json.dumps(conversion.options, sort_keys=True, default=repr).encode())
        tokenizer = getattr(conversion, "tokenizer", None)
        note("tokenizers", as_bytes(tokenizer) if tokenizer is not None else b"(none)")

    def went(llama, result, args):
        if isinstance(result, np.ndarray):
            note("logits", np.ascontiguousarray(result).tobytes())

    def encoded(tokenizer, result, args):
        note("ids", repr(list(result)).encode())

    after(llama2_convert.Stream, "finish", finished)
    after(llama2_convert.Conversion, "start", started)
    after(llama2_numpy.Llama, "forward", went)
    after(llama2_numpy.Tokenizer, "encode", encoded)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    NOW[0] = item.nodeid
    yield
    NOW[0] = None


def pytest_sessionfinish(session, exitstatus):
    where = os.environ.get("UNCHANGED_RECORD")
    if where:
        with open(where, "w", encoding="utf-8") as file:
            json.dump({test: {kind: f"{count} {digest.hexdigest()[:16]}" for kind, (count, digest) in sorted(kinds.items())}
                       for test, kinds in sorted(RECORDS.items())}, file, indent=0)
