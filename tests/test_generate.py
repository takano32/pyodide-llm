# generate(): the prompt, the stop tokens, the seed, and the reference output of a real checkpoint.
import numpy as np
import pytest

from conftest import (checkpoint_vocab_size, detokenize, model_file, pack_checkpoint,
                      pack_tokenizer, synthetic_weights, tiny_vocab)
from llama2_numpy import BOS, Llama


def build(**options):
    config, weights = synthetic_weights()
    return Llama(pack_checkpoint(config, weights), pack_tokenizer(tiny_vocab(config["vocab_size"])), **options)


def forces(llama, tokens):
    """Replace forward() with one that always predicts the given tokens, one per position."""
    def forward(token, pos, need_logits=True):
        if not need_logits:
            return None
        logits = np.full(llama.vocab_size, -10.0, dtype=np.float32)
        logits[tokens[min(pos, len(tokens) - 1)]] = 10.0
        return logits
    llama.forward = forward


def test_a_prompt_longer_than_the_run_is_refused():
    llama = build()
    with pytest.raises(ValueError):
        list(llama.generate("hello world hello world hello world", steps=3))


def test_steps_are_clamped_to_the_context_length():
    llama = build()
    forces(llama, [100])  # never a stop token
    list(llama.generate(steps=10 ** 6))
    assert llama.stats["tokens"] == llama.seq_len


def test_generation_stops_at_a_stop_token():
    llama = build(stop_tokens=(2, 5))
    forces(llama, [100, 101, 5, 102])
    text = "".join(llama.generate(steps=20))
    assert text == detokenize(llama.tokenizer, [100, 101])
    assert llama.stats["tokens"] == 2


def test_bos_stops_generation_by_default():
    llama = build()
    forces(llama, [100, BOS, 101])
    assert llama.stats == {}
    list(llama.generate(steps=20))
    assert llama.stats["tokens"] == 1


def test_the_prompt_comes_back_before_the_generated_text():
    llama = build()
    forces(llama, [100, 101])
    assert "".join(llama.generate("hello world", steps=20)).startswith("hello world")


def test_the_same_seed_repeats_the_text():
    llama = build()
    runs = ["".join(llama.generate("hello", steps=16, temperature=0.9, topp=0.9, seed=7)) for _ in range(2)]
    assert runs[0] == runs[1] and len(runs[0]) > len("hello")


def test_another_seed_writes_something_else():
    llama = build()
    texts = {"".join(llama.generate("hello", steps=16, temperature=0.9, topp=0.9, seed=seed)) for seed in range(4)}
    assert len(texts) > 1


def test_a_repetition_penalty_changes_nothing_when_it_is_one():
    llama = build()
    plain = "".join(llama.generate("hello", steps=16, temperature=0.9, seed=5))
    same = "".join(llama.generate("hello", steps=16, temperature=0.9, repetition_penalty=1.0, seed=5))
    assert plain == same


def test_stats_are_filled_in():
    llama = build()
    forces(llama, [100])
    list(llama.generate(steps=8))
    assert llama.stats["tokens"] == 8
    assert llama.stats["seconds"] > 0.0 and llama.stats["tokens_per_second"] > 0.0


def test_stats_separate_the_prompt_from_the_generated_text():
    llama = build()
    forces(llama, [100])
    prompt_tokens = len(llama.tokenizer.encode("hello world"))
    list(llama.generate("hello world", steps=20))
    stats = llama.stats
    assert stats["prompt_tokens"] == prompt_tokens
    assert stats["sampled"] == stats["tokens"] - prompt_tokens == 20 - prompt_tokens
    assert 0.0 < stats["prompt_seconds"] < stats["seconds"]
    assert stats["prompt_tokens_per_second"] > 0.0


def test_the_prompt_is_counted_when_a_stop_token_ends_the_run():
    llama = build()
    prompt_tokens = len(llama.tokenizer.encode("hello world"))
    forces(llama, [100] * prompt_tokens + [101, BOS])
    list(llama.generate("hello world", steps=20))
    # the stop token is sampled but never shown, so it counts as a sampling step and not as a token
    assert llama.stats["prompt_tokens"] == prompt_tokens
    assert llama.stats["tokens"] == prompt_tokens + 1 and llama.stats["sampled"] == 2


def test_time_to_first_token_covers_the_prompt():
    llama = build()
    forces(llama, [100])
    list(llama.generate("hello world", steps=20))
    stats = llama.stats
    assert stats["prompt_seconds"] < stats["first_token_seconds"] < stats["seconds"]


def test_stats_of_a_run_without_a_prompt_have_no_prompt_time():
    llama = build()
    forces(llama, [100])
    list(llama.generate(steps=4))
    assert llama.stats["prompt_tokens"] == 0 and llama.stats["prompt_seconds"] == 0.0
    assert llama.stats["prompt_tokens_per_second"] == 0.0
    assert llama.stats["sampled"] == 4


def test_an_abandoned_generator_does_not_overwrite_newer_stats():
    llama = build()
    forces(llama, [100])
    abandoned = llama.generate(steps=8)
    next(abandoned)
    list(llama.generate(steps=4))
    abandoned.close()
    assert llama.stats["tokens"] == 4


# --------------------------------------------------------------------------- a real checkpoint

def test_stories260K_writes_the_reference_story():
    checkpoint = model_file("stories260K.bin").read_bytes()
    tokenizer = model_file("tok512.bin").read_bytes()
    llama = Llama(checkpoint, tokenizer)
    text = "".join(llama.generate("Once upon a time", steps=48, temperature=0.0))
    assert text.startswith("Once upon a time, there was a little girl named Lily.")
    assert "She loved to play outside in the park." in text


def test_stories260K_vocabulary_matches_its_tokenizer():
    llama = Llama(model_file("stories260K.bin").read_bytes(), model_file("tok512.bin").read_bytes())
    assert llama.vocab_size == checkpoint_vocab_size("stories260K.bin") == len(llama.tokenizer.vocab)
