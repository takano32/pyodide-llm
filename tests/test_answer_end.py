"""T.524: an answer stops at the token the model's own template writes right after an answer. A Gemma 4 opened with
?hf= wrote two to five <turn|> after each answer: neither its config.json nor its tokenizer names that token (its
generation_config.json does, which the converter does not fetch)."""
import pytest

from convert.template import answer_end, model_answer_end
from made_up_tokenizers import byte_level, converted, sentencepiece

# a Gemma 4's turns, in its own tokens (<|turn> and <turn|>), with the BOS first
GEMMA4 = ("{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
          "{{ m.content | trim }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}")
CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{{ '<|im_start|>assistant\\n' }}"


def test_what_a_template_writes_after_an_answer():
    assert answer_end(GEMMA4, {"bos_token": "<bos>"}) == "<turn|>\n"
    assert answer_end(CHATML, {}) == "<|im_end|>\n<|im_start|>assistant\n"
    # a template that ends with the answer, one that leaves the answer out, one that writes it twice
    assert answer_end("{% for m in messages %}{{ m.content }}{% endfor %}", {}) == ""
    assert answer_end("{{ messages[0].content }}", {}) is None
    assert answer_end("{% for m in messages %}{{ m.content }}{{ m.content }}{% endfor %}", {}) is None
    # one that refuses a conversation that ends with an answer, and one that is no template
    assert answer_end("{% if messages[-1].role == 'assistant' %}{{ raise_exception('no') }}{% endif %}", {}) is None
    assert answer_end("{% for %}", {}) is None
    # from a tokenizer_config.json, and from chat_template.jinja before it
    assert model_answer_end({"chat_template": CHATML}) == "<|im_end|>\n<|im_start|>assistant\n"
    assert model_answer_end({"chat_template": "x"}, GEMMA4) == "<turn|>\n"
    assert model_answer_end({}) is None and model_answer_end("no config") is None


def test_the_answer_stops_at_the_token_that_ends_a_turn():
    tokenizer, ids = byte_level(["<bos>", "<eos>", "<|turn>", "<turn|>"])
    config = {"bos_token_id": ids["<bos>"], "eos_token_id": ids["<eos>"]}
    options, _ = converted(tokenizer, config=config, chat_template=GEMMA4, bos_token="<bos>", eos_token="<eos>")
    assert options["stop_tokens"] == [ids["<bos>"], ids["<eos>"], ids["<turn|>"]]
    assert options["bos"] == ids["<bos>"] and options["template"] == "<|turn>user\n{prompt:trim}<turn|>\n<|turn>model\n"
    # without a template, or with one this cannot read: as it was
    for described in ({}, {"chat_template": "{{ messages[0].content }}"}, {"chat_template": "{% for %}"}):
        assert converted(tokenizer, config=config, bos_token="<bos>", eos_token="<eos>", **described)[0]["stop_tokens"] \
            == [ids["<bos>"], ids["<eos>"]], described
    # a token said already is said once
    options, _ = converted(tokenizer, config={**config, "eos_token_id": [ids["<eos>"], ids["<turn|>"]]}, chat_template=GEMMA4,
                           bos_token="<bos>", eos_token="<eos>")
    assert options["stop_tokens"] == [ids["<bos>"], ids["<eos>"], ids["<turn|>"]]


@pytest.mark.parametrize("written", ["<turn|>x", "\n<turn|>", "turn|>"])
def test_only_a_token_right_after_the_answer(written):
    """Text, or a token after text, is no token that ends an answer."""
    tokenizer, ids = byte_level(["<bos>", "<eos>", "<|turn>", "<turn|>"])
    template = "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}" + written.replace("<turn|>x", "x<turn|>") + "{% endfor %}"
    options, _ = converted(tokenizer, config={"bos_token_id": ids["<bos>"], "eos_token_id": ids["<eos>"]}, chat_template=template,
                           bos_token="<bos>", eos_token="<eos>")
    assert ids["<turn|>"] not in options["stop_tokens"]


def test_a_sentencepiece_model_and_its_control_pieces():
    """sarashina2.2's: </s> after an answer, which is the EOS already; <|assistant|> after it would be one more."""
    data, pieces = sentencepiece()
    config = {"bos_token_id": pieces["<s>"], "eos_token_id": pieces["</s>"]}
    template = "{% for m in messages %}{{ '<|user|>' if m.role == 'user' else '<|assistant|>' }}{{ m.content }}</s>{% endfor %}"
    options, _ = converted(data, "tokenizer.model", config, chat_template=template, bos_token="<s>", tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["stop_tokens"] == [pieces["<s>"], pieces["</s>"]]
    options, _ = converted(data, "tokenizer.model", config, chat_template=template.replace("</s>", "<|assistant|>"), bos_token="<s>",
                           tokenizer_class="LlamaTokenizer", legacy=False)
    assert options["stop_tokens"] == [pieces["<s>"], pieces["</s>"], pieces["<|assistant|>"]]
