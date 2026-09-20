"""The switches of T52: every optimization that has a fallback can be left out, to measure what it is worth."""
import pytest
from conftest import naive_logits, pack_checkpoint, pack_tokenizer, synthetic_weights, tiny_vocab

import numpy as np
from llama2_numpy import SWITCHES, Llama


def model(disable=()):
    settings, weights = synthetic_weights()
    llama = Llama(pack_checkpoint(settings, weights), pack_tokenizer(tiny_vocab(settings["vocab_size"])),
                  disable=disable)
    return settings, weights, llama


@pytest.mark.parametrize("disable", [(), ("kernels",), ("relaxed",), ("sampler",), ("int8", "relaxed"),
                                     tuple(SWITCHES)])
def test_every_switch_leaves_a_working_engine(disable):
    settings, weights, llama = model(disable)
    tokens = [1, 5, 7]
    want = naive_logits(settings, weights, tokens)
    for pos, token in enumerate(tokens):
        assert np.allclose(llama.forward(token, pos), want[pos], rtol=1e-4, atol=1e-4)
    assert llama.disabled == disable


def test_the_backend_line_says_what_was_left_out():
    assert "without" not in model()[2].backend
    assert model(("sampler",))[2].backend.endswith("(without sampler)")
    # in the order of SWITCHES, whatever order the caller gave
    assert model(("sampler", "relaxed"))[2].backend.endswith("(without relaxed, sampler)")


def test_a_switch_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="no optimization called 'turbo'"):
        model(("turbo",))
    # the message names what there is, so that a typo is easy to fix
    with pytest.raises(ValueError, match="kernels, int8, relaxed, sampler"):
        model(("Relaxed",))
