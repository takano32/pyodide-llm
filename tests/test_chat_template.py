"""The chat_template reader (T73): the part of Jinja that a tokenizer_config.json actually uses.

The templates here are the real ones of the models this site lists, and the expected text is what transformers
produces for a single user turn (checked against jinja2 on the development machine, where transformers is
installed; CI has neither, so the expectations are written out).
"""
import json

import pytest

from llama2_convert import Unsupported, one_turn, one_turn_template, render


@pytest.fixture(params=["jinja2", "the reader"])
def reads(request, monkeypatch):
    """T397: what renders a template for one_turn(): jinja2 where it can be imported, else this project's own reader
    (a browser whose fetch of the package failed). What both must do is tested with both. render() is the reader
    itself, whichever of the two one_turn() takes."""
    if request.param == "jinja2":
        pytest.importorskip("jinja2")
    else:
        monkeypatch.setattr("convert.template.jinja_environment", lambda: None)
    return request.param

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
def test_one_turn_is_what_transformers_writes(template, expected, reads):
    assert one_turn(template, {"bos_token": "<s>", "eos_token": "</s>"}) == expected


def test_a_template_it_cannot_read_is_refused_quietly(reads):
    # the filters, calls and statements it does not know: the caller keeps the format it has
    assert one_turn("{{ messages | tojson }}{{ messages[0].content }}", {}) is None  # (jinja2: the prompt twice)
    macro = one_turn("{% macro m() %}x{% endmacro %}{{ m() }}{{ messages[0].content }}", {})  # a macro called
    assert macro == ("x{prompt}" if reads == "jinja2" else None)
    assert one_turn("{{ undefined_function() }}{{ messages[0].content }}", {}) is None
    assert one_turn("{{ raise_exception('only system and user') }}{{ messages[0].content }}", {}) is None
    assert one_turn("{% for message in messages %}{{ message['content'] }}", {}) is None  # never closed
    assert one_turn("{{ 'nothing about the prompt' }}", {}) is None  # no {prompt} in the result
    # T138: trimmed, on its own and between words of the template's (RakutenAI's), or cut on one side only
    assert one_turn("[{{ messages[0].content | trim }}]", {}) == "[{prompt:trim}]"
    assert one_turn("USER: {{ messages[0].content | trim }} ASSISTANT:", {}) == "USER: {prompt:trim} ASSISTANT:"
    assert one_turn("USER: {{ messages[0].content }} ASSISTANT:", {}) == "USER: {prompt} ASSISTANT:"
    assert one_turn("[{{ messages[0].content.lstrip() }}]", {}) is None


def test_jinja2_renders_where_it_is_there_and_the_reader_where_it_is_not(monkeypatch):
    """T397: what only jinja2 reads has a format with it and none without (the page then keeps what the list has, or
    sends what was typed as it is): a macro called (Qwen3.5's, LFM2's), tojson, a for with a test, loop controls,
    {% generation %}, which transformers adds."""
    pytest.importorskip("jinja2")
    only_jinja = [
        ("{% macro text(m) %}{{ m.content }}{% endmacro %}<u>{{ text(messages[0]) }}</u>", "<u>{prompt}</u>"),
        ("{{ {'a': 1} | tojson }}{{ messages[0].content }}", '{"a": 1}{prompt}'),
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
    monkeypatch.setattr("convert.template.jinja_environment", lambda: None)
    for template, _ in only_jinja[:5]:
        assert one_turn(template, {}) is None, template


def test_jinja2_is_imported_once_and_not_before_a_template_is_read():
    """T397: importing the converter does not import jinja2 (a third of a second in Pyodide), and the environment
    is transformers': sandboxed and immutable, so a template cannot reach into Python or change what it is given."""
    import subprocess
    import sys
    from pathlib import Path
    pytest.importorskip("jinja2")
    public = Path(__file__).resolve().parent.parent / "public"
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import llama2_convert as c; print('jinja2' in sys.modules); "
              "print(c.one_turn('{{ messages[0].content }}', {})); print('jinja2' in sys.modules)")
    said = subprocess.check_output([sys.executable, "-c", script, str(public)], text=True).split()
    assert said == ["False", "{prompt}", "True"]
    from convert.template import jinja_environment
    assert jinja_environment() is jinja_environment()
    assert one_turn("{{ messages.append(1) }}{{ messages[0].content }}", {}) is None  # immutable
    assert one_turn("{{ ''.__class__.__mro__ }}{{ messages[0].content }}", {}) is None  # sandboxed


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


def test_the_pieces_of_jinja_t269_adds():
    """What Granite 4.2's template uses for one turn: a if c else b, a list written out, the filter string (each the
    same as jinja2 writes with transformers' settings, on the development machine)."""
    scope = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "X"}]}
    for template, written in (
            ("{% set t = t if t is defined else True %}{{ t }}", "True"),
            ("{% set t = false %}{% set t = t if t is defined else True %}{{ t }}", "False"),
            ("{{ 'a' if messages | length > 1 else 'b' }}|{{ 'a' if messages | length > 2 else 'b' }}", "a|b"),
            # the else of one is another one; without an else the other side is undefined
            ("{{ 'a' if false else 'b' if false else 'c' }}|{{ 'a' if false else 'b' if true else 'c' }}", "c|b"),
            ("[{{ 'a' if false }}]{{ 'yes' if true }}", "[]yes"),
            # it binds loosest: the + and the or belong to a side
            ("{{ ('x' if false else 'y') + 'z' }}|{{ 'x' if false else 'y' + 'z' }}|{{ 'x' if true else 'y' + 'z' }}", "yz|yz|x"),
            ("{{ 'a' or 'b' if false else 'c' }}", "c"),
            ("{{ 'a' if messages[0].role == 'system' and messages[1].role == 'user' else 'b' }}", "a"),
            # the words inside quotes are no words of it
            ("{{ 'if' if ' if ' in ' if x else ' else ' else ' }}", "if"),
            # only the side picked is read
            ("{{ messages[0].content if messages[0].role == 'system' else '' }}{{ nothing.here if false else 'safe' }}", "Ssafe"),
            ("{% set l = [] %}{{ l | length }}{% if l %}full{% else %}empty{% endif %}", "0empty"),
            ("{% set l = ['a', 'b,c', messages[1].content] %}{{ l | length }}{{ l[1] }}{{ l[-1] }}"
             "{% for i in l %}<{{ i }}>{% endfor %}", "3b,cX<a><b,c><X>"),
            ("{% set l = [[1, 2], []] %}{{ l[0][1] }}{{ l[1] | length }}{{ 'b' in ['a', 'b'] }}{{ 'c' in ['a', 'b'] }}", "20TrueFalse"),
            ("{% for m in [] %}never{% endfor %}ok", "ok"),
            ("{{ messages[1].content | string }}|{{ 3 | string }}|{{ true | string }}|{{ none | string }}|[{{ nothing | string }}]",
             "X|3|True|None|[]"),
            ("{{ (messages | length | string) + '!' }}", "2!")):
        assert render(template, dict(scope)) == written, template
    # a for with a test keeps the items that pass, which the conditional would read as "all of them or nothing"
    # (the second would read as "messages if true" and loop over them all, which is right by chance: not read either)
    for filtered in ("{% for m in messages if m.role == 'user' %}{{ m.content }}{% endfor %}",
                     "{% for m in messages if true %}{{ m.content }}{% endfor %}"):
        with pytest.raises(Unsupported):
            render(filtered, dict(scope))
    for broken in ("{{ 'a' if }}", "{{ if true else 'b' }}", "{{ 'a' if true else }}", "{{ ['a', 'b' }}"):
        with pytest.raises(Unsupported):
            render(broken, dict(scope))
    # the tokens transformers gives by name are all in the scope (a template may write the pad token), the ones not set are not
    assert one_turn("{{ pad_token }}{{ messages[0].content }}{{ unk_token }}", {"pad_token": "<pad>"}) == "<pad>{prompt}"
    assert one_turn("{{ messages[0].content }}", {"pad_token": ""}) == "{prompt}"
    assert one_turn_template(json.dumps({"pad_token": {"content": "<p>"}, "chat_template": "{{ pad_token }}{{ messages[0].content }}"})) == "<p>{prompt}"
    assert one_turn("{% set think = think if think is defined else True %}{% set tools = [] %}"
                    "{{ messages[0].content }}{{ '<think>' if think else '' }}", {}) == "{prompt}<think>"


def test_what_the_review_of_t269_found():
    """Each against jinja2 with transformers' settings (the same snippets, the same outputs), on the development machine"""
    scope = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "X"}]}
    for template, written in (
            # lists are joined as lists (they were joined as their texts: "[][1]"), whichever side is the empty one
            ("{% set x = [] %}{% set x = x + [1, 2] %}{{ x | length }}{{ [] + [] }}{{ [3] + x }}", "2[][3, 1, 2]"),
            # {% for %} ... {% else %}: the else when there was nothing to loop over, and only then
            ("{% for i in [] %}{{ i }}{% else %}none{% endfor %}|{% for i in [1] %}{{ i }}{% else %}none{% endfor %}", "none|1"),
            ("{% for m in messages[2:] %}x{% else %}{% for i in [1, 2] %}{{ i }}{% endfor %}{% endfor %}", "12"),
            ("{% for i in [1] %}{% if i == 2 %}a{% else %}b{% endif %}{% else %}none{% endfor %}", "b"),
            # what Jinja writes of an undefined and of a namespace inside a list or on its own
            ("{{ [nothing, 1] }}", "[Undefined, 1]"),
            # without an else the other side is undefined, not an empty text
            ("{% set a = 1 if false %}{{ 'D' if a is defined else 'U' }}{% set b = 1 if true %}{{ b }}", "U1"),
            ("{% set ns = namespace(a=1) %}{{ ns | string }}", "<Namespace {'a': 1}>"),
            # None is written (Jinja: "None"), an undefined is not; a bool is a number
            ("[{{ none }}][{{ nothing }}]", "[None][]"),
            ("{{ 'a' if true is number else 'b' }}{{ 'a' if 'x' is number else 'b' }}", "ab")):
        assert render(template, dict(scope)) == written, template
    # Jinja reads a if b if c else d as (a if b) if c else d, and the reader does not take it for a if (b if c) else d;
    # a float is no name with an attribute; a list and a text, or a list and a number, are no sum
    for not_read in ("{{ 'a' if false if true else 'b' }}", "{{ 1.5 }}", "{{ [1.5] }}", "{{ [1] + 'x' }}", "{{ 'x' + [1] }}",
                     "{{ [1] * 2 }}", "{{ [1] - [1] }}",
                     "{{ [1, 2][0] }}", "{{ ['a', 'b'][1:] }}", "{{ (1, 2) }}"):
        with pytest.raises(Unsupported):
            render(not_read, dict(scope))


def test_what_the_review_of_t127_found():
    """Each against transformers' Jinja on the development machine (the same snippets, the same outputs)"""
    scope = {"messages": [{"role": "user", "content": "X"}]}
    # or / and are the operand that decides, as in Jinja: not True or False
    assert render("{% set s = none %}{{ (s or 'You are a helpful assistant.') }}|{{ 'a' and 'b' }}|{{ '' and 'b' }}|{{ none or '' }}",
                  dict(scope)) == "You are a helpful assistant.|b||"
    # a comment stands between the text before it and a {%- after it
    assert render("<s>\n{# The system prompt #}\n{%- if true %}[{{ messages[0].content }}]{% endif %}", dict(scope)) == "<s>\n[X]"
    # escapes, left to right as Python's unicode-escape (Jinja's lexer)
    assert render("{{ 'a\\\\nb' }}|{{ '\\x41' }}|{{ '\\u2581' }}|{{ 'it\\'s' }}|{{ '\\\\' }}", dict(scope)) == "a\\nb|A|\u2581|it's|\\"


def test_the_date_of_a_template_is_the_day_the_prompt_is_sent(reads):
    # strftime_now() is the day the prompt is sent: filled() in src/models.js makes {date:format} of it
    assert one_turn("Today Date: {{ strftime_now('%d %b %Y') }}\n{{ messages[0].content }}", {}) == "Today Date: {date:%d %b %Y}\n{prompt}"
    for unknown in ("strftime_now('%j')", "strftime_now(fmt)", "strftime_now('%d}')"):
        assert one_turn("{{ " + unknown + " }}{{ messages[0].content }}", {}) is None, unknown
    # (a format put together is one jinja2 reads and the reader does not)
    assert one_turn("{{ strftime_now('%d ' + '%b') }}{{ messages[0].content }}", {}) == ("{date:%d %b}{prompt}" if reads == "jinja2" else None)
    # a template that reckons with the date (an Unsloth copy of Mistral Small works out yesterday's) cannot be
    # filled later: checked on two real days, it gives up
    assert one_turn("{% set d = strftime_now('%d') %}{% if d == '01' %}first {% endif %}{{ d }} {{ messages[0].content }}", {}) is None


FIXTURES = json.load(open(__import__("pathlib").Path(__file__).parent / "fixtures" / "chat-templates.json"))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture["repo"] for fixture in FIXTURES])
def test_real_templates_read_as_jinja_writes_them(fixture, reads):
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


def test_it_reads_a_tokenizer_config(reads):
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
