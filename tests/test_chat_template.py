"""The chat templates (T73): one user turn of a model's own template, rendered by jinja2 as transformers renders it.

The templates here are the real ones of the models this site lists, and the expected text is what transformers
produces for a single user turn (checked against transformers on the development machine; CI has it not, so the
expectations are written out). T640.14: jinja2 alone renders (the converter's own reader of a part of Jinja is gone),
and a converter without it raises where it reads a template.
"""
import json

import pytest
from made_up import unigram

import sys

from llama2_convert import one_turn, one_turn_template


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
    assert one_turn("{{ messages | tojson }}{{ messages[0].content }}", {}) is None  # (jinja2: the prompt twice)
    assert one_turn("{% macro m() %}x{% endmacro %}{{ m() }}{{ messages[0].content }}", {}) == "x{prompt}"  # a macro called
    assert one_turn("{{ undefined_function() }}{{ messages[0].content }}", {}) is None
    assert one_turn("{{ raise_exception('only system and user') }}{{ messages[0].content }}", {}) is None
    assert one_turn("{% for message in messages %}{{ message['content'] }}", {}) is None  # never closed
    assert one_turn("{{ 'nothing about the prompt' }}", {}) is None  # no {prompt} in the result
    # T138: trimmed, on its own and between words of the template's (RakutenAI's), or cut on one side only
    assert one_turn("[{{ messages[0].content | trim }}]", {}) == "[{prompt:trim}]"
    assert one_turn("USER: {{ messages[0].content | trim }} ASSISTANT:", {}) == "USER: {prompt:trim} ASSISTANT:"
    assert one_turn("USER: {{ messages[0].content }} ASSISTANT:", {}) == "USER: {prompt} ASSISTANT:"
    assert one_turn("[{{ messages[0].content.lstrip() }}]", {}) is None


def test_what_jinja2_reads():
    """T397: a macro called (Qwen3.5's, LFM2's), tojson, a for with a test, loop controls, {% generation %}, which
    transformers adds."""
    only_jinja = [
        ("{% macro text(m) %}{{ m.content }}{% endmacro %}<u>{{ text(messages[0]) }}</u>", "<u>{prompt}</u>"),
        ("{{ {'a': 1} | tojson }}{{ messages[0].content }}", '{"a": 1}{prompt}'),
        ("{{ {'a': 'あ'} | tojson }}{{ messages[0].content }}", '{"a": "あ"}{prompt}'),  # (transformers' tojson keeps non-ASCII)
        ("{% for m in messages if m.role == 'user' %}[{{ m.content }}]{% endfor %}", "[{prompt}]"),
        ("{% for m in messages %}{{ m.content }}{% break %}{% endfor %}", "{prompt}"),
        ("{{ messages[0].content }}{% generation %}<a>{% endgeneration %}", "{prompt}<a>"),
        # the environment is transformers': a block's line leaves no newline and no indentation behind
        ("{% if true %}\n  {% if true %}\n{{ messages[0].content }}\n  {% endif %}\n{% endif %}\n", "{prompt}\n"),
        # a token the tokenizer does not name is not defined, as transformers passes only those it has
        ("{% if bos_token is defined %}B{% endif %}{% if eos_token is defined %}E{% endif %}{{ messages[0].content }}", "E{prompt}"),
    ]
    for template, written in only_jinja[:5]:
        assert one_turn(template, {}) == written, template
    for template, written in only_jinja[5:]:
        assert one_turn(template, {"bos_token": "", "eos_token": "</s>"}) == written, template


def test_a_converter_without_jinja2_raises_where_it_reads_a_template(monkeypatch):
    """T640.14 (the owner: as the other packages): where jinja2 cannot be imported, a conversion that reads a template
    fails, as one without NumPy does; it does not go on as if the model had no format. One without a template does
    not ask for jinja2."""
    from convert.template import answer_end, jinja_environment
    monkeypatch.setitem(sys.modules, "jinja2", None)  # (import jinja2 raises ImportError)
    jinja_environment.cache_clear()
    try:
        with pytest.raises(ImportError):
            one_turn(SMOLLM2, {})
        with pytest.raises(ImportError):
            answer_end(SMOLLM2, {})
        with pytest.raises(ImportError):
            one_turn_template(json.dumps({"chat_template": SMOLLM2}))
        assert one_turn_template(json.dumps({"bos_token": "<s>"})) is None
    finally:
        jinja_environment.cache_clear()


def test_jinja2_is_imported_once_and_not_before_a_template_is_read():
    """T397: importing the converter does not import jinja2 (a third of a second in Pyodide), and the environment
    is transformers': sandboxed and immutable, so a template cannot reach into Python or change what it is given."""
    import subprocess
    import sys
    from pathlib import Path
    from tree import python_folder
    sources = python_folder(Path(__file__).resolve().parent.parent)
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import llama2_convert as c; print('jinja2' in sys.modules); "
              "print(c.one_turn('{{ messages[0].content }}', {})); print('jinja2' in sys.modules)")
    said = subprocess.check_output([sys.executable, "-c", script, sources], text=True).split()
    assert said == ["False", "{prompt}", "True"]
    from convert.template import jinja_environment
    assert jinja_environment() is jinja_environment()
    assert one_turn("{{ messages.append(1) }}{{ messages[0].content }}", {}) is None  # immutable
    assert one_turn("{{ ''.__class__.__mro__ }}{{ messages[0].content }}", {}) is None  # sandboxed


def test_the_tokens_transformers_gives_by_name():
    # the tokens transformers gives by name are all in the scope (a template may write the pad token), the ones not set are not
    assert one_turn("{{ pad_token }}{{ messages[0].content }}{{ unk_token }}", {"pad_token": "<pad>"}) == "<pad>{prompt}"
    assert one_turn("{{ messages[0].content }}", {"pad_token": ""}) == "{prompt}"
    assert one_turn_template(json.dumps({"pad_token": {"content": "<p>"}, "chat_template": "{{ pad_token }}{{ messages[0].content }}"})) == "<p>{prompt}"
    assert one_turn("{% set think = think if think is defined else True %}{% set tools = [] %}"
                    "{{ messages[0].content }}{{ '<think>' if think else '' }}", {}) == "{prompt}<think>"


def test_the_date_of_a_template_is_the_day_the_prompt_is_sent():
    # strftime_now() is the day the prompt is sent: filled() in src/models.ts makes {date:format} of it
    assert one_turn("Today Date: {{ strftime_now('%d %b %Y') }}\n{{ messages[0].content }}", {}) == "Today Date: {date:%d %b %Y}\n{prompt}"
    for unknown in ("strftime_now('%j')", "strftime_now(fmt)", "strftime_now('%d}')"):
        assert one_turn("{{ " + unknown + " }}{{ messages[0].content }}", {}) is None, unknown
    assert one_turn("{{ strftime_now('%d ' + '%b') }}{{ messages[0].content }}", {}) == "{date:%d %b}{prompt}"
    # a template that reckons with the date (an Unsloth copy of Mistral Small works out yesterday's) cannot be
    # filled later: checked on two real days, it gives up
    assert one_turn("{% set d = strftime_now('%d') %}{% if d == '01' %}first {% endif %}{{ d }} {{ messages[0].content }}", {}) is None


FIXTURES = json.load(open(__import__("pathlib").Path(__file__).parent / "fixtures" / "chat-templates.json"))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture["repo"] for fixture in FIXTURES])
def test_real_templates_read_as_jinja_writes_them(fixture):
    """T127: real templates of the families this site takes (tokenizers/fixtures: the template of the pinned
    revision, and one turn of it rendered by jinja2 with transformers' settings, the BOS the template writes first
    left out). Two of them are in chat_template.jinja, which the converter is handed on its own."""
    config = {"bos_token": fixture["bos_token"], "eos_token": fixture["eos_token"]}
    if fixture["file"] == "chat_template.jinja":
        got = one_turn_template(json.dumps(config), fixture["template"])
    else:
        got = one_turn_template(json.dumps({**config, "chat_template": fixture["template"]}))
    # T138: trims: whether jinja2 drops the spaces around what was typed, which the converter says as {prompt:trim}
    assert got == fixture["one_turn"].replace("{prompt}", "{prompt:trim}" if fixture["trims"] else "{prompt}")


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
    """The page uses options["template"] when src/models.ts has none for that model. The engine must not see
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
    vocabulary = unigram(settings["vocab_size"])
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
