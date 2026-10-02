"""The rule of tests/format_check.py for the IDs the page sends against the real ones (T250's review)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import format_check  # noqa: E402

IM_START, USER, NEWLINE, ENDOFTEXT = 151644, 872, 198, 151643


def test_the_same_ids_are_the_same():
    assert format_check.same_ids([IM_START, USER, NEWLINE], [IM_START, USER, NEWLINE])


def test_one_more_token_in_front_is_the_bos_the_page_always_starts_with():
    # the T131 policy: a Qwen3's real tokenizer puts nothing in front, the page puts <|endoftext|>
    assert format_check.same_ids([ENDOFTEXT, IM_START, USER, NEWLINE], [IM_START, USER, NEWLINE])


def test_the_first_token_of_the_real_ids_written_twice_is_not_the_same():
    # a BOS set to the format's own first token with the format still beginning with it (the page then sends
    # <|im_start|> twice), and Llama 3's <|begin_of_text|> written by the template and by the page
    assert not format_check.same_ids([IM_START, IM_START, USER, NEWLINE], [IM_START, USER, NEWLINE])
    assert not format_check.same_ids([128000, 128000, 9906], [128000, 9906])


def test_another_format_is_not_the_same():
    assert not format_check.same_ids([ENDOFTEXT, IM_START, USER], [IM_START, USER, NEWLINE])
    assert not format_check.same_ids([IM_START, USER], [IM_START, USER, NEWLINE])
    assert not format_check.same_ids([], [IM_START])


def test_the_families_whose_bos_is_the_formats_first_token_take_nothing_in_front():
    # T236's review: <|endoftext|> in front of <|im_start|> is what a Qwen3.5 was sent before its entries began at the
    # format's own first token (it costs the model much on plain text and moves its own answers). same_ids lets that by
    # (the extra token is not the real first one), the families of STRICT do not
    real, old_design = [IM_START, USER, NEWLINE], [ENDOFTEXT, IM_START, USER, NEWLINE]
    # (T254's review: and MiniCPM5, whose template writes its BOS <s> itself first, as the page does)
    for model_id in ("hf-qwen3.5-0.8b", "hf-qwen3.5-9b-thinking", "hf-granite-4.2-3b", "hf-minicpm5-1b-thinking", "hf-ternary-bonsai-2-27b"):
        assert format_check.ids_match(model_id, real, real)
        assert not format_check.ids_match(model_id, old_design, real)
    # elsewhere the page's BOS in front is let by (T131), and the doubled one is not
    assert format_check.ids_match("hf-qwen3-4b", old_design, real)
    assert not format_check.ids_match("hf-qwen3-4b", [IM_START, IM_START, USER, NEWLINE], real)
