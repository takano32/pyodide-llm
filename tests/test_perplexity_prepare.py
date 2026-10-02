# tests/perplexity_prepare.py --entry (the review of T247): the options a measurement of a model of the list runs with. Without
# it they are the converter's alone, what ?hf= opens: a Qwen3.5 of the list begins with <|im_start|> where the converter's
# BOS is <|endoftext|>, which costs a Qwen3.5 18% to 45% of perplexity on plain text (T236, T247's review) and makes its
# perplexity move by 3% under a rounding of its weights, so a measurement that leaves the entry's options out measures
# another model's way. Needs node (it reads src/models.js), which the suite has.
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import synthetic_weights
from test_convert import hugging_face, safetensors_file
from test_gguf import unigram

ROOT = Path(__file__).resolve().parent.parent


def tiny_directory(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shape, weights = synthetic_weights(vocab_size=320, seq_len=64)
    tensors, published = hugging_face(shape, weights, True)
    (tmp_path / "config.json").write_text(json.dumps(published))
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(tensors))
    (tmp_path / "tokenizer.json").write_bytes(unigram(320))
    return tmp_path


def prepare(directory, out, *more):
    run = subprocess.run([sys.executable, str(ROOT / "tests" / "perplexity_prepare.py"), str(directory), str(out), "int8", *more],
                         capture_output=True, text=True)
    return run, (json.loads(Path(f"{out}.json").read_text()) if Path(f"{out}.json").exists() else None)


def test_the_options_of_an_entry_go_over_the_converters(tmp_path):
    directory = tiny_directory(tmp_path / "model")
    plain_run, plain = prepare(directory, tmp_path / "plain")
    assert plain_run.returncode == 0, plain_run.stderr
    entry_run, entry = prepare(directory, tmp_path / "entry", "--entry", "hf-qwen3.5-0.8b")
    assert entry_run.returncode == 0, entry_run.stderr
    # the entry's BOS (the format's own first token) and stops, its specials in place of the converter's; the converter's
    # own options (the tokenizer's kind, the dtype ...) stay
    assert entry["bos"] == 248045 and entry["stop_tokens"] == [248044, 248045, 248046]
    assert len(entry["specials"]) == 14 and "<|im_start|>" in entry["specials"]
    assert plain["bos"] != 248045 and plain.get("stop_tokens") != [248044, 248045, 248046]
    assert len(plain.get("specials", [])) != 14
    assert {key: value for key, value in entry.items() if key not in ("bos", "stop_tokens", "specials")} == \
        {key: value for key, value in plain.items() if key not in ("bos", "stop_tokens", "specials")}


def test_an_id_that_is_no_entry_is_refused(tmp_path):
    directory = tiny_directory(tmp_path / "model")
    run, _ = prepare(directory, tmp_path / "none", "--entry", "no-such-model")
    assert run.returncode != 0 and "no entry" in (run.stderr + run.stdout)
