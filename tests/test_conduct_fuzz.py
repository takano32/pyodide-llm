# T374.3 review: a GGUF's head and tensors through an answerer that answers short (mid-file), long, or a file with bytes
# after its tensors, over random first pieces and ahead sizes: always direct()'s checkpoint, options and tokenizer.bin.
import random

import pytest
from conduct_hub import THREE, Hub, MiB, cut, direct
from test_conduct import CASES, conducted, conducting, listed, real_files

class Fuzzy(Hub):
    def __init__(s, *a, rng, mode, **k):
        super().__init__(*a, **k); s.rng, s.mode = rng, mode
    def range(s, where, name, begin, end):
        r = super().range(where, name, begin, end)
        if r is None: return r
        data, size = r
        if s.mode == "short" and len(data) > 1 and s.rng.random() < .6:
            data = data[:s.rng.randrange(1, len(data))]
        if s.mode == "long":
            full = s.whole(where, name)
            data = full[begin:end + s.rng.randrange(0, 5000)]
        return data, size

@pytest.mark.parametrize("name", ["a GGUF alone (a head of 3 MiB)", "a GGUF with the vocabulary of another repository"])
@pytest.mark.parametrize("mode", ["short", "long", "trail"])
def test_fuzz(name, mode, monkeypatch):
    case = next(c for c in CASES if c["name"] == name)
    files = real_files(case); hf, making = listed(case); making["dtype"] = "int8"
    for k, v in list(files.items()):
        if k.endswith(".gguf") and mode == "trail":
            files[k] = type(v)(bytes(v[:]) + b"\x07" * 3000) if not hasattr(v, "size") else v
    exp = direct(Hub(files, hf).whole, hf, THREE, **making)
    rng = random.Random(5)
    for i in range(60):
        monkeypatch.setattr(conducting, "GGUF_HEAD", rng.choice([64, 5000, 40000, 100000, 2*MiB]))
        monkeypatch.setattr(conducting, "AHEAD", rng.choice([1000, 7777, 16*MiB]))
        hub = Fuzzy(files, hf, rng=rng, mode=mode, unsaid=rng.random()<.5, **cut({}, unit=2048))
        end, got = conducted(hub, hf, making, [])
        assert end == "converted", (i, end)
        assert bytes(got.checkpoint) == bytes(exp.checkpoint) and got.options == exp.options and bytes(got.tokenizer) == bytes(exp.tokenizer), i
