# The chat templates of models: one user turn of a model's own template, rendered by jinja2 as transformers'
# apply_chat_template renders it, given back as the format the page fills (T73, T127, T269, T397).
#
# T640.14: jinja2 is what renders, as NumPy is what computes: the build and the tests have it installed, and the
# page's worker loads Pyodide's package before it converts and fails as it fails without NumPy where it does not
# come. Where jinja2 cannot be imported, a conversion that reads a template raises (jinja_environment()): it does not
# go on without the format.
import functools
import json
import re
import time


class Unsupported(Exception):
    """A date format the page's filled() cannot write (date_format())."""


DAY = "\x00day"  # in the scope: the day strftime_now() writes, instead of {date:format} (see one_turn())
# two days that differ in every field strftime_now() may write, the first and the last of a year
CHECK_DAYS = [time.strptime(day, "%Y-%m-%d %H:%M:%S") for day in ("2025-01-01 00:00:00", "2026-12-31 23:59:59")]


def date_format(form):
    """A format of strftime_now() that the page's filled() can write on the day the prompt is sent, or Unsupported."""
    directives = form.replace("%%", "").split("%")[1:]  # what follows each %: the ones filled() knows
    if "}" in form or not all(directive[:1] and directive[0] in "dmYybBaAHMS" for directive in directives):
        raise Unsupported(f"the date format {form!r}")
    return form


@functools.cache
def jinja_environment():
    """jinja2's environment as transformers makes it for apply_chat_template (utils/chat_template_utils.py: sandboxed
    and immutable, trim_blocks and lstrip_blocks, the loop controls, {% generation %}, its tojson and
    raise_exception). Raises ImportError where jinja2 is not there to import (T640.14). Asked once: the import is a
    third of a second in Pyodide, and only a conversion with a template pays it (functools.cache: cache_clear() asks
    again)."""
    import jinja2
    from jinja2.ext import Extension, loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    class Generation(Extension):
        """{% generation %} ... {% endgeneration %}, which transformers adds to mark what the assistant wrote: the
        body as it is"""
        tags = {"generation"}

        def parse(self, parser):
            line = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(line)

        def _body(self, caller):
            return caller()

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation])
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    return environment


def rendered(template, scope):
    """The template with the names of scope, by jinja2. scope is one_turn()'s: the tokens the tokenizer does not name
    are left out, as transformers passes only those it has (special_tokens_map), and strftime_now() writes
    {date:format} or the day under DAY."""
    environment = jinja_environment()
    day = scope.get(DAY)
    names = {name: value for name, value in scope.items()
             if name != DAY and not (name.endswith("_token") and value == "")}
    now = lambda form: "{date:" + date_format(form) + "}" if day is None else time.strftime(date_format(form), day)
    return environment.from_string(template).render(**names, strftime_now=now)


def config_token(config, name):
    """The text of a token a tokenizer_config.json names (bos_token ...): a string, or {"content": ...}; "" for none."""
    token = config.get(name) if isinstance(config, dict) else None
    token = token.get("content") if isinstance(token, dict) else token
    return token if isinstance(token, str) else ""


def one_turn_template(tokenizer_config, chat_template=None):
    """The template of one user turn from a tokenizer_config.json, or None when there is none this can read.
    chat_template: the text of chat_template.jinja, where the repository has one (T127: transformers now saves the
    template there rather than in tokenizer_config.json, and reads it first); the tokens it names are still the
    tokenizer_config's."""
    try:
        config = json.loads(tokenizer_config) if isinstance(tokenizer_config, (str, bytes)) else tokenizer_config
    except ValueError:
        config = None
    if not isinstance(config, dict):
        if not chat_template:
            return None
        config = {}
    turn, bos = model_turn(config, chat_template)
    # generate() starts every run with the BOS token already: one written by the template would be a second one
    return turn[len(bos):] if turn and bos and turn.startswith(bos) else turn


def model_turn(config, chat_template=None):
    """(one user turn of a model's own template, whole: with the BOS where the template writes it; the text of the BOS
    its tokenizer names, "" for none). The turn is None where there is no template this can read. config: a
    tokenizer_config.json, read (a dict; anything else is one that says nothing)."""
    config = config if isinstance(config, dict) else {}
    bos = config_token(config, "bos_token")
    template = template_of(config, chat_template)
    return (one_turn(template, named_tokens(config)) if template else None), bos


def model_answer_end(config, chat_template=None):
    """T524: what a model's own template writes right after an answer (answer_end()), or None where there is no
    template this can read. config, chat_template: as model_turn() takes them."""
    config = config if isinstance(config, dict) else {}
    template = template_of(config, chat_template)
    return answer_end(template, named_tokens(config)) if template else None


def template_of(config, chat_template=None):
    """The text of a model's chat template: chat_template.jinja's, else tokenizer_config.json's; None for none."""
    template = chat_template or config.get("chat_template")
    if isinstance(template, list):  # some models publish several; the first is the chat one
        template = template[0].get("template") if template and isinstance(template[0], dict) else None
    return template if isinstance(template, str) and template.strip() else None


def named_tokens(config):
    """The tokens transformers gives a template by their names (special_tokens_map): a template may write the pad
    token."""
    names = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")
    return {name: config_token(config, name) for name in names}


def scope_of(specials, messages, generation):
    """The names a template is rendered with, as transformers' apply_chat_template passes them."""
    return {"messages": messages, "add_generation_prompt": generation,
            "bos_token": specials.get("bos_token", ""), "eos_token": specials.get("eos_token", ""),
            **{name: token for name, token in specials.items() if token and name not in ("bos_token", "eos_token")},
            "tools": None, "tools_json": None, "documents": None}


def answer_end(template, specials, mark="\x00prompt\x00", answer="\x00answer\x00"):
    """T524: the text a template writes right after an assistant's message that ends a conversation, rendered as
    transformers renders one to train on (no generation prompt): where a turn of the model's ends, the token the
    model writes when its answer is done (a Gemma 4's <turn|>, a ChatML's <|im_end|>, a harmony template's <|return|>,
    which is not the <|end|> after its analysis). "" where the answer ends the text; None where the template raises
    for it, or does not write the answer once as it was given. Raises ImportError where jinja2 is not there."""
    jinja_environment()  # (outside the try: a converter without jinja2 does not go on as if there were no template)
    try:
        text = rendered(template, scope_of(specials, [{"role": "user", "content": mark},
                                                      {"role": "assistant", "content": answer}], False))
    except Exception:  # (an unwritable date, and whatever a template raises of a conversation it does not take)
        return None
    return text.split(answer, 1)[1] if text.count(answer) == 1 else None


def one_turn(template, specials, mark="\x00prompt\x00"):
    """The template of one user turn, as src/models.ts writes it: the text around a {prompt}.

    specials: the tokenizer's special tokens, for bos_token and eos_token. Returns None when the template raises for
    one turn or writes what filled() cannot fill, and then the caller keeps the format it has. Raises ImportError
    where jinja2 is not there (T640.14).
    """
    jinja_environment()  # (outside the try: a converter without jinja2 does not go on as if there were no template)
    scope = scope_of(specials, [{"role": "user", "content": mark}], True)
    try:
        text = rendered(template, dict(scope))
        # strftime_now() is written {date:format}, which the page fills with the day it sends the prompt. That holds
        # where the template only writes the date; one that reckons with it (yesterday's date from today's day of
        # the month) would come out otherwise, so the two are compared on two days, and it gives up where they differ
        days = CHECK_DAYS if "{date:" in text else []
        fill = lambda day: re.sub(r"\{date:([^}]*)\}", lambda found: time.strftime(found.group(1), day), text)
        if any(fill(day) != rendered(template, {**scope, DAY: day}) for day in days):
            return None
    except Exception:  # (an unwritable date, and whatever a template raises of one turn)
        return None
    if text.count(mark) != 1:
        return None
    # T138: a template that trims what was typed ({{ message['content'] | trim }}) says so as {prompt:trim}, which the
    # page's filled() trims: rendered once more with spaces around the mark, it either keeps them or drops both.
    # (Whether the spaces are there is no test: RakutenAI's writes "USER: " and " ASSISTANT:" around a trimmed one)
    try:
        spaced = rendered(template, {**scope, "messages": [{"role": "user", "content": f" {mark} "}]})
    except Exception:
        return None
    if spaced == text.replace(mark, f" {mark} "):
        return text.replace(mark, "{prompt}")
    if spaced == text:
        return text.replace(mark, "{prompt:trim}")
    return None  # it drops one of the two, or changes more than that: nothing filled() would do
