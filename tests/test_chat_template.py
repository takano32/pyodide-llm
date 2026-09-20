"""The chat_template reader (T73): the part of Jinja that a tokenizer_config.json actually uses.

The templates here are the real ones of the models this site lists, and the expected text is what transformers
produces for a single user turn (checked against jinja2 on the development machine, where transformers is
installed; CI has neither, so the expectations are written out).
"""
import json

import pytest

from llama2_convert import Unsupported, one_turn, one_turn_template, render

SMOLLM2 = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
           "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face"
           "<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
           "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

LLM_JP = ("{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}"
          "{{ '\n\n### 指示:\n' + message['content'] }}{% elif message['role'] == 'system' %}"
          "{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}"
          "{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}"
          "{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}{% endfor %}")

# Qwen2.5: the shape that made the reader worth writing (a tools branch, nested ifs, whitespace control)
QWEN = ("{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}"
        "\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}"
        "\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
        "\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") %}"
        "\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- endif %}"
        "\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n")


@pytest.mark.parametrize("template, expected", [
    (SMOLLM2, "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
              "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
    (LLM_JP, "<s>\n\n### 指示:\n{prompt}\n\n### 応答:\n"),
    (QWEN, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
])
def test_one_turn_is_what_transformers_writes(template, expected):
    assert one_turn(template, {"bos_token": "<s>", "eos_token": "</s>"}) == expected


def test_a_template_it_cannot_read_is_refused_quietly():
    # selectattr and the other filters of the fancier templates: the caller keeps the format it has
    assert one_turn("{{ messages | selectattr('role', 'equalto', 'user') | list | last }}", {}) is None
    assert one_turn("{% for message in messages %}{{ message['content'] }}", {}) is None  # never closed
    assert one_turn("{{ 'nothing about the prompt' }}", {}) is None  # no {prompt} in the result


def test_the_pieces_of_jinja_it_does_read():
    scope = {"messages": [{"role": "user", "content": "X"}], "add_generation_prompt": True,
             "bos_token": "<s>", "eos_token": "</s>"}
    assert render("{{ 'a' + 'b' }}", scope) == "ab"
    assert render("{%- if messages %}yes{%- else %}no{% endif %}", scope) == "yes"
    assert render("{% if tools is defined %}yes{% else %}no{% endif %}", scope) == "no"
    assert render("{% for m in messages %}{{ loop.first }}{{ loop.last }}{% endfor %}", scope) == "TrueTrue"
    assert render("{{ messages[0]['role'].capitalize() }}", scope) == "User"
    assert render("{# a comment #}text", scope) == "text"
    assert render("{{ '<|x|>' }}", scope) == "<|x|>"      # a | inside quotes is not a filter
    assert render("{{ ' a ' | trim }}", scope) == "a"
    assert render("{% if ('x' == 'y') or (messages[0].role == 'user') %}yes{% endif %}", scope) == "yes"
    with pytest.raises(Unsupported):
        render("{{ messages | length % 2 }}", scope)


def test_it_reads_a_tokenizer_config():
    config = {"chat_template": SMOLLM2, "bos_token": {"content": "<s>"}, "eos_token": "</s>"}
    assert one_turn_template(json.dumps(config)).endswith("<|im_start|>assistant\n")
    assert one_turn_template(json.dumps({"chat_template": [{"name": "default", "template": LLM_JP}],
                                         "bos_token": "<s>"})) == "<s>\n\n### 指示:\n{prompt}\n\n### 応答:\n"
    assert one_turn_template("{}") is None
    assert one_turn_template("not json") is None
    assert one_turn_template(json.dumps({"chat_template": ""})) is None


def test_the_conversion_hands_the_template_to_the_page():
    """The page uses options["template"] when src/models.js has none for that model. The engine must not see
    it: Llama() would refuse a keyword it does not know."""
    import json
    import struct

    import numpy as np
    from conftest import pack_tokenizer, synthetic_weights, tiny_vocab
    from test_convert import hugging_face, reader, safetensors_file
    import llama2_convert

    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(settings["vocab_size"])]}}).encode()
    tokenizer_config = json.dumps({"chat_template": SMOLLM2, "bos_token": "<s>", "eos_token": "</s>"})
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                                           "tokenizer.json", dtype="float32", max_seq_len=settings["seq_len"],
                                           tokenizer_config=tokenizer_config)
    assert conversion.options["template"].endswith("<|im_start|>assistant\n")
    assert "{prompt}" in conversion.options["template"]
    # and without a template the key is not there at all
    plain = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                                      "tokenizer.json", dtype="float32", max_seq_len=settings["seq_len"])
    assert "template" not in plain.options
