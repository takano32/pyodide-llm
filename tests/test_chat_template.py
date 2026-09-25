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
    # the filters, calls and statements it does not know: the caller keeps the format it has
    assert one_turn("{{ messages | tojson }}{{ messages[0].content }}", {}) is None
    assert one_turn("{% macro m() %}x{% endmacro %}{{ m() }}{{ messages[0].content }}", {}) is None  # a macro called
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
        render("{{ messages | tojson }}", scope)


def test_the_pieces_of_jinja_t127_adds():
    """What Qwen3's, Mistral v0.3's and sarashina2.2's templates use for one turn (each checked against jinja2 with
    transformers' trim_blocks and lstrip_blocks on the development machine)."""
    scope = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "X"}]}
    assert render("{{ messages | length % 2 }}{{ (messages|length - 1) * 3 }}{{ 7 - 2 - 1 }}", scope) == "034"
    assert render("{{ messages | selectattr('role', 'equalto', 'user') | list | length }}", scope) == "1"
    assert render("{% set others = messages | rejectattr('role', 'equalto', 'user') | list %}{{ others[0].content }}", scope) == "S"
    assert render("{% set ns = namespace(n=0, last=messages|length - 1) %}{% for m in messages %}"
                  "{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}{{ ns.last }}", scope) == "21"
    assert render("{% for m in messages[::-1] %}{{ m.role }}{% endfor %}{{ messages[1:][0].content }}", scope) == "usersystemX"
    assert render("{% set c = 'a</t>b' %}{{ messages[-1].content }}{{ c.split('</t>')[-1] }}{{ c.split('</t>')[0].strip('a') }}", scope) == "Xb"
    assert render("{% if messages[1].content is string and not(messages[1].content.startswith('<')) %}yes{% endif %}", scope) == "yes"
    assert render("{% if x is defined and x is false %}no{% elif 3 > 2 and 'a' not in 'bc' %}yes{% endif %}", scope) == "yes"
    assert render("{{ 'the user\\'s' }}", scope) == "the user's"   # an escaped quote
    assert render("{# the user's #}ok", scope) == "ok"               # a comment is prose: its quotes are not strings
    assert render("{% macro tools(x) %}{{ x }}{% endmacro %}ok", scope) == "ok"  # defined, never called
    # trim_blocks and lstrip_blocks, as transformers renders: the newline after a block goes, and the indent before it
    assert render("{% for m in messages %}\n  {% if m.role == 'user' %}\n{{ m.content }}\n  {% endif %}\n{% endfor %}", scope) == "X\n"


FIXTURES = json.load(open(__import__("pathlib").Path(__file__).parent / "fixtures" / "chat-templates.json"))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture["repo"] for fixture in FIXTURES])
def test_real_templates_read_as_jinja_writes_them(fixture):
    """T127: real templates of the families this site takes (tokenizers/fixtures: the template of the pinned
    revision, and one turn of it rendered by jinja2 with transformers' settings, the BOS the template writes first
    left out). Two of them are in chat_template.jinja, which the converter is handed on its own."""
    config = {"bos_token": fixture["bos_token"], "eos_token": fixture["eos_token"]}
    if fixture["file"] == "chat_template.jinja":
        assert one_turn_template(json.dumps(config), fixture["template"]) == fixture["one_turn"]
    else:
        assert one_turn_template(json.dumps({**config, "chat_template": fixture["template"]})) == fixture["one_turn"]


def test_it_reads_a_tokenizer_config():
    config = {"chat_template": SMOLLM2, "bos_token": {"content": "<s>"}, "eos_token": "</s>"}
    assert one_turn_template(json.dumps(config)).endswith("<|im_start|>assistant\n")
    assert one_turn_template(json.dumps({"chat_template": [{"name": "default", "template": LLM_JP}],
                                         "bos_token": "<s>"})) == "\n\n### 指示:\n{prompt}\n\n### 応答:\n"
    # the BOS the template writes first is left out: generate() starts with one already (T106)
    assert one_turn_template(json.dumps({"chat_template": "{{ bos_token }}{{ messages[0].content }}",
                                         "bos_token": "<|begin_of_text|>"})) == "{prompt}"
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


def test_a_sentencepiece_model_names_its_control_pieces_for_the_template():
    """T127: with a tokenizer.model, the template read from the model writes <|user|> and </s>, which are control
    pieces: the conversion names them as specials (without, each was spelled out as text through ?hf=)."""
    import struct
    from conftest import synthetic_weights
    from make_hf_fixture import field
    from test_convert import hugging_face, safetensors_file
    import llama2_convert

    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    NORMAL, CONTROL, UNKNOWN = 1, 3, 2
    pieces = [("<unk>", UNKNOWN), ("<s>", CONTROL), ("</s>", CONTROL), ("<|user|>", CONTROL), ("<|assistant|>", CONTROL)]
    pieces += [(f"▁w{i}", NORMAL) for i in range(settings["vocab_size"] - len(pieces))]
    model = b"".join(field(1, field(1, text.encode()) + field(2, -float(i)) + field(3, kind)) for i, (text, kind) in enumerate(pieces))
    model += field(2, field(3, 1)) + field(3, field(1, b"identity"))
    assert llama2_convert.sentencepiece_specials(model) == ["<s>", "</s>", "<|user|>", "<|assistant|>"]
    template = "{% for m in messages %}<|user|>{{ m.content }}</s>{% endfor %}<|assistant|>"
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), model,
                                           "tokenizer.model", dtype="float32", max_seq_len=settings["seq_len"],
                                           tokenizer_config=json.dumps({"chat_template": template}))
    assert conversion.options["template"] == "<|user|>{prompt}</s><|assistant|>"
    assert conversion.options["specials"] == ["<|assistant|>", "<|user|>", "</s>"]
