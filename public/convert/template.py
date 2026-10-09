# The reader of chat templates: a part of Jinja, enough to render one user turn of a model's own template and give it
# back as the format the page fills (T73, T127, T269).
import json
import re
import time

# A chat_template is Jinja. This reads the part of Jinja those templates actually use: a loop over the
# messages, if / elif / else with the usual comparisons, set, string concatenation, the trim filter, and the
# whitespace control of {%- -%}; since T127 also the filters length, list and selectattr, namespace() and the
# setting of its attributes, integer arithmetic, the tests of "is", slices and string methods with arguments,
# which Qwen3's, Mistral v0.3's and sarashina2.2's templates use; since T269 also a if c else b and a list written
# out ([a, b]), which Granite 4.2's uses. A macro is skipped where it is defined (the
# templates define them for tools, which one turn has none of); calling one is Unsupported. Anything else raises
# Unsupported, and then the caller keeps whatever format src/models.js has for that model. Chosen over a real
# Jinja (jinja2 through micropip) to add no dependency.

class Unsupported(Exception):
    """This template uses something this reader does not know."""


def tokenize_template(text):
    """The template as a list of ("text", str) | ("say", expression) | ("do", statement)."""
    if text.endswith("\n"):
        text = text[:-1]   # Jinja drops one trailing newline of the source (keep_trailing_newline=False)
    out, i = [], 0
    while i < len(text):
        start = min([p for p in (text.find("{{", i), text.find("{%", i), text.find("{#", i)) if p != -1], default=-1)
        if start == -1:
            out.append(("text", text[i:]))
            break
        if start > i:
            out.append(("text", text[i:start]))
        opening, closing = {"{{": ("{{", "}}"), "{%": ("{%", "%}"), "{#": ("{#", "#}")}[text[start:start + 2]]
        # a comment is prose, and may hold an apostrophe (Llama 3.1's "user's"): no quotes inside it
        end = text.find(closing, start + 2) if opening == "{#" else find_outside_quotes(text, closing, start + 2)
        if end == -1:
            raise Unsupported(f"a {opening} that never closes")
        inner = text[start + len(opening):end]
        # {%- and -%} strip the whitespace next to them
        if inner.startswith("-"):
            inner = inner[1:]
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1].rstrip())
        elif opening != "{{" and not inner.startswith("+"):
            # lstrip_blocks (transformers renders with it): the spaces and tabs from the start of its line to a
            # block or a comment go, when there is nothing else on the line before it
            indent = text[text.rfind("\n", 0, start) + 1:start]
            if indent and not indent.strip(" \t") and out and out[-1][0] == "text" and out[-1][1].endswith(indent):
                out[-1] = ("text", out[-1][1][:-len(indent)])
        strip_after = inner.endswith("-")
        if strip_after:
            inner = inner[:-1]
        # a comment says nothing, but it stands between: a {%- after it strips up to it, not the text before it
        out.append(("say" if opening == "{{" else "do", inner.strip()) if opening != "{#" else ("text", ""))
        i = end + len(closing)
        if strip_after:
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
        elif opening != "{{" and text.startswith("\n", i):
            i += 1  # trim_blocks (transformers renders with it): the newline right after a block or a comment goes
    return out


# expressions, from the loosest to the tightest: or, and, not, the tests of "is", in, comparisons, + and -, * / %
# (integers), filters (|), and single values: 'text', "text", numbers, name, name['key'], name[a:b], name.attribute,
# name.method(...), namespace(...)
def evaluate(expression, scope):
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")") and balanced(expression[1:-1]):
        expression = expression[1:-1].strip()
    # T269: a if condition else b, which binds loosest of all and reads only the side the condition picks (Granite 4.2's
    # "enable_thinking if enable_thinking is defined else True"). The first " if " ends a, the first " else " after it
    # the condition, and b may be another of the kind; without an else the other side is undefined, as in Jinja
    parts = split_outside_quotes(expression, " if ")
    if len(parts) > 1:
        chosen, rest = parts[0], expression[len(parts[0]) + len(" if "):]
        condition = split_outside_quotes(rest, " else ")[0]
        otherwise = rest[len(condition) + len(" else "):]
        if not chosen.strip() or not condition.strip() or (len(condition) < len(rest) and not otherwise.strip()):
            raise Unsupported(f"the expression {expression!r}")
        if len(split_outside_quotes(condition, " if ")) > 1:
            # (the review of T269) a if b if c else d is (a if b) if c else d in Jinja, not a if (b if c) else d
            raise Unsupported(f"the expression {expression!r}")
        if truthy(evaluate(condition, scope)):
            return evaluate(chosen, scope)
        return evaluate(otherwise, scope) if otherwise else MISSING
    for joiner, decided in ((" or ", truthy), (" and ", lambda value: not truthy(value))):
        # as in Jinja, the operand that decides, not True or False ('x' or 'default' is 'x'), and what follows it
        # unread (the review of T127: (system_message or 'You are ...') wrote "True")
        parts = split_outside_quotes(expression, joiner)
        if len(parts) > 1:
            for part in parts:
                value = evaluate(part, scope)
                if decided(value):
                    break
            return value
    if expression.startswith("not ") or (expression.startswith("not(") and balanced(expression[3:])):
        return not truthy(evaluate(expression[3:], scope))
    parts = split_outside_quotes(expression, " is ")
    if len(parts) == 2:
        test = parts[1].strip()
        negated = test.startswith("not ")
        return is_test(evaluate(parts[0], scope), test[4:].strip() if negated else test) != negated
    parts = split_outside_quotes(expression, " not in ")
    if len(parts) == 2:
        return not evaluate(f"({parts[0]}) in ({parts[1]})", scope)
    parts = split_outside_quotes(expression, " in ")
    if len(parts) == 2:
        needle, haystack = (evaluate(part, scope) for part in parts)
        return needle in haystack if isinstance(haystack, (str, list, tuple, dict)) else False
    for operator in ("==", "!=", ">=", "<=", ">", "<"):
        parts = split_outside_quotes(expression, operator)
        if len(parts) == 2:
            left, right = (evaluate(part, scope) for part in parts)
            return compare(left, right, operator)
    for operators in (("+", " - "), ("*", "/", "%")):
        terms = split_operators(expression, operators)
        if len(terms) > 1:
            value = evaluate(terms[0][1], scope)
            for operator, term in terms[1:]:
                value = arithmetic(value, operator.strip(), evaluate(term, scope))
            return value
    parts = split_outside_quotes(expression, "|")   # a | inside quotes, as in '<|im_start|>', is not a filter
    if len(parts) > 1:
        value, *filters = parts
        result = evaluate(value, scope)
        for spec in filters:
            result = apply_filter(result, spec.strip(), scope)
        return result
    return value_of(expression, scope)


def is_test(value, test):
    """value is test: defined, none, string, number, mapping, iterable, sequence, true, false. None counts as not
    defined: one_turn() gives tools and documents as None where transformers leaves them out or passes None, and
    the templates test them with "is defined" (T73 matched transformers on 23 templates so)."""
    tests = {"defined": lambda v: v is not MISSING and v is not None, "undefined": lambda v: v is MISSING or v is None,
             "none": lambda v: v is None or v is MISSING,
             "string": lambda v: isinstance(v, str), "number": lambda v: isinstance(v, (int, float)),  # (a bool is a number, as in Jinja)
             "mapping": lambda v: isinstance(v, (dict, Namespace)), "iterable": lambda v: isinstance(v, (str, list, tuple, dict)),
             "sequence": lambda v: isinstance(v, (str, list, tuple)), "true": lambda v: v is True, "false": lambda v: v is False}
    if test not in tests:
        raise Unsupported(f"the test is {test}")
    return tests[test](value)


def compare(left, right, operator):
    if operator in ("==", "!="):
        same = left == right and (left is MISSING) == (right is MISSING)
        return same if operator == "==" else not same
    if not (isinstance(left, (int, float)) and isinstance(right, (int, float))) and not (isinstance(left, str) and isinstance(right, str)):
        raise Unsupported(f"{left!r} {operator} {right!r}")
    return {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right}[operator]


def arithmetic(left, operator, right):
    """+ joins text (Jinja's templates add strings far more than numbers) and adds numbers; the others are integers'."""
    numbers = all(isinstance(v, int) and not isinstance(v, bool) for v in (left, right))
    if any(isinstance(v, (list, tuple)) for v in (left, right)):
        # (the review of T269) lists, now that [] can be written: [] + x is the lists joined, not their texts; a list and a
        # text is a TypeError in Jinja, and the other operators on a list are not read
        if operator == "+" and isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return [*left, *right]
        raise Unsupported(f"{left!r} {operator} {right!r}")
    if operator == "+":
        return left + right if numbers else as_text(left) + as_text(right)
    if not numbers:
        raise Unsupported(f"{left!r} {operator} {right!r}")
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        raise Unsupported("a division by zero")
    return left % right if operator == "%" else left / right


def apply_filter(value, spec, scope):
    """The filters the templates use on one turn: trim, length, list, first, last, and selectattr / rejectattr
    (attribute, test[, value]) with the tests equalto (==), defined and none."""
    name, _, rest = spec.partition("(")
    name = name.strip()
    arguments = [evaluate(argument, scope) for argument in split_outside_quotes(rest[:-1], ",") if argument.strip()] if rest else []
    if name == "trim" and not arguments:
        return as_text(value).strip()
    if name == "string" and not arguments:  # (T269) Jinja's: an undefined is "", anything else Python's str()
        return "" if value is MISSING else str(value)
    if name in ("length", "count") and not arguments and isinstance(value, (str, list, tuple, dict)):
        return len(value)
    if name == "list" and not arguments and isinstance(value, (str, list, tuple)):
        return list(value)
    if name in ("first", "last") and not arguments and isinstance(value, (list, tuple)):
        return (value[0] if name == "first" else value[-1]) if value else MISSING
    if name in ("selectattr", "rejectattr") and isinstance(value, (list, tuple)) and 1 <= len(arguments) <= 3:
        attribute, test, *expected = arguments + ([] if len(arguments) > 1 else ["defined"])
        def passes(item):
            got = item.get(attribute, MISSING) if isinstance(item, dict) else getattr(item, str(attribute), MISSING)
            if test in ("equalto", "eq", "==", "sameas"):
                return expected and got == expected[0]
            if test in ("defined", "none") and not expected:
                return is_test(got, test)
            raise Unsupported(f"selectattr with the test {test}")
        return [item for item in value if passes(item) == (name == "selectattr")]
    raise Unsupported(f"the filter {spec}")


STRFTIME = object()   # so that "strftime_now is defined" is true, as it is in transformers
DAY = "\x00day"  # in the scope: the day strftime_now() writes, instead of {date:format} (see one_turn())
# two days that differ in every field strftime_now() may write, the first and the last of a year
CHECK_DAYS = [time.strptime(day, "%Y-%m-%d %H:%M:%S") for day in ("2025-01-01 00:00:00", "2026-12-31 23:59:59")]
class Undefined:
    """What a name nothing set is. Printed inside a list, Jinja writes Undefined (the review of T269)."""

    def __repr__(self):
        return "Undefined"


MISSING = Undefined()  # a name the template asks for and nothing set: Jinja calls it undefined, and it is false


def truthy(value):
    return bool(value) and value is not MISSING


def as_text(value):
    if value is MISSING or value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def balanced(text):
    """Whether the brackets of text are balanced, so that its outer pair can be dropped."""
    depth, quote, escaped = 0, "", False
    for char in text:
        if quote:
            quote, escaped = ("" if char == quote and not escaped else quote), char == "\\" and not escaped
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def find_outside_quotes(text, needle, start):
    """Where needle is, skipping what is inside quotes: a template may write }} or %} inside a string."""
    quote, i = "", start
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 1  # an escaped character, as in 'the user\'s'
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif text.startswith(needle, i):
            return i
        i += 1
    return -1


def split_outside_quotes(text, separator):
    """text.split(separator), but not inside quotes or brackets."""
    parts, depth, quote, start, i = [], 0, "", 0, 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 1  # an escaped character
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [part for part in parts]


class Namespace:
    """What namespace(a=1, b=2) makes: attributes a {% set ns.a = ... %} may change inside a loop."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def __repr__(self):  # as Jinja prints one, for {{ ns }} and ns | string (the review of T269)
        return f"<Namespace {self.__dict__!r}>"


def split_operators(expression, operators):
    """[(operator, term)] of a chain of + and - (or * / %), outside quotes and brackets; the first operator is ""."""
    marks = []
    for operator in operators:
        at = 0
        for part in split_outside_quotes(expression, operator)[:-1]:
            at += len(part)
            marks.append((at, operator))
            at += len(operator)
    marks.sort()
    terms, start, previous = [], 0, ""
    for at, operator in marks:
        term = expression[start:at]
        if not term.strip():  # a sign, as in -1: not an operator
            return [("", expression)]
        terms.append((previous, term))
        start, previous = at + len(operator), operator
    terms.append((previous, expression[start:]))
    return terms if all(term.strip() for _, term in terms) else [("", expression)]


def call_arguments(text, scope):
    """The values of "a, 'b', c=1" (the part between the brackets of a call): ([positional], {keyword})."""
    positional, keyword = [], {}
    for argument in split_outside_quotes(text, ","):
        if not argument.strip():
            continue
        name, equals, value = argument.partition("=")
        if equals and name.strip().isidentifier() and not value.startswith("="):
            keyword[name.strip()] = evaluate(value, scope)
        else:
            positional.append(evaluate(argument, scope))
    return positional, keyword


def string_end(text):
    """Where the string that text begins with ends (its closing quote), a backslash escaping the next character."""
    i = 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == text[0]:
            return i
        i += 1
    raise Unsupported(f"a string that never closes: {text!r}")


def closing_bracket(text, start):
    """Where the bracket opened at text[start] closes, outside quotes."""
    depth, quote, escaped = 0, "", False
    for i in range(start, len(text)):
        char = text[i]
        if quote:
            quote, escaped = ("" if char == quote and not escaped else quote), char == "\\" and not escaped
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return i
    raise Unsupported(f"a bracket that never closes in {text!r}")


# the string methods a template may call on one turn, and how many arguments each takes at most
METHODS = {"capitalize": 0, "lower": 0, "upper": 0, "title": 0, "strip": 1, "lstrip": 1, "rstrip": 1,
           "startswith": 1, "endswith": 1, "split": 2, "replace": 2}


def value_of(expression, scope):
    """One value: a literal, a name, or a name with ['key'], [a:b], .attribute and .method(...) after it."""
    expression = expression.strip()
    if not expression:
        raise Unsupported("an empty expression")
    if expression[0] in "'\"" and string_end(expression) == len(expression) - 1:
        return unescape(expression[1:-1])
    if expression.lstrip("-").isdigit():
        return int(expression)
    if expression[0] == "[" and closing_bracket(expression, 0) == len(expression) - 1:
        # T269: a list written out, as in {% set tools = [] %}
        return [evaluate(item, scope) for item in split_outside_quotes(expression[1:-1], ",") if item.strip()]
    if expression.startswith("namespace(") and closing_bracket(expression, len("namespace")) == len(expression) - 1:
        positional, keyword = call_arguments(expression[len("namespace("):-1], scope)
        if positional:
            raise Unsupported("namespace() with values without names")
        return Namespace(**keyword)
    if expression in ("true", "True"):
        return True
    if expression in ("false", "False"):
        return False
    if expression in ("none", "None"):
        return None
    if expression.startswith("strftime_now(") and expression.endswith(")"):
        # the one function these templates call: the date of today, for a system prompt. Written as {date:format},
        # which filled() in src/models.js makes the visitor's day when the prompt is sent: the format is kept with
        # the conversion, and a date written now went stale from the next day on (the review of T127)
        argument = expression[len("strftime_now("):-1].strip()
        if not argument or argument[0] not in "'\"" or string_end(argument) != len(argument) - 1:
            raise Unsupported(f"strftime_now of {argument!r}")
        form = unescape(argument[1:-1])
        directives = form.replace("%%", "").split("%")[1:]  # what follows each %: the ones filled() knows
        if "}" in form or not all(directive[:1] and directive[0] in "dmYybBaAHMS" for directive in directives):
            raise Unsupported(f"the date format {form!r}")
        day = scope.get(DAY)  # a day to write, when one_turn() checks the template against real dates
        return "{date:" + form + "}" if day is None else time.strftime(form, day)
    name, rest = expression, ""
    for cut in ("[", "."):
        at = expression.find(cut)
        if at != -1 and at < len(name):
            name, rest = expression[:at], expression[at:]
    if not name.replace("_", "").isalnum() or name.isdigit():  # (a digit then a dot is a float, 1.5: not read)
        raise Unsupported(f"the expression {expression!r}")
    value = scope.get(name, MISSING)
    while rest:
        if rest.startswith("["):
            end = closing_bracket(rest, 0)
            inside, rest = rest[1:end], rest[end + 1:]
            if ":" in split_outside_quotes(inside, ":")[0] or len(split_outside_quotes(inside, ":")) > 1:
                # a slice, as in messages[1:] and messages[::-1]
                bounds = [evaluate(bound, scope) if bound.strip() else None for bound in split_outside_quotes(inside, ":")]
                if len(bounds) > 3 or not all(bound is None or isinstance(bound, int) for bound in bounds):
                    raise Unsupported(f"the slice [{inside}]")
                value = value[slice(*bounds)] if isinstance(value, (str, list, tuple)) else MISSING
                continue
            key = evaluate(inside, scope)
        elif rest.startswith("."):
            piece = rest[1:]
            length = len(piece) - len(piece.lstrip("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            key, rest = piece[:length], piece[length:]
            if rest.startswith("("):  # a method, of the few the templates call
                end = closing_bracket(rest, 0)
                arguments, keyword = call_arguments(rest[1:end], scope)
                rest = rest[end + 1:]
                if key not in METHODS or keyword or len(arguments) > METHODS[key]:
                    raise Unsupported(f"the method {key}()")
                if value is not MISSING:
                    value = getattr(as_text(value), key)(*arguments)
                continue
        else:
            raise Unsupported(f"the expression {expression!r}")
        if value is MISSING:
            continue
        if isinstance(value, dict):
            value = value.get(key, MISSING)
        elif isinstance(value, (list, tuple)) and isinstance(key, int):
            value = value[key] if -len(value) <= key < len(value) else MISSING
        else:
            value = getattr(value, str(key), MISSING)
    return value


def unescape(text):
    """A string literal's escapes, as Jinja's lexer reads them: Python's unicode-escape, left to right (the review of
    T127: "\\\\n" was a backslash and a newline, and \\x41 and \\u2581 stayed as they were)."""
    return text.replace("\r\n", "\n").encode("ascii", "backslashreplace").decode("unicode-escape")


class Loop:
    """What {% for %} puts in scope as loop."""

    def __init__(self, index, total):
        self.index0, self.index = index, index + 1
        self.first, self.last = index == 0, index == total - 1
        self.length = total


def render(template, scope):
    """The template, with the names of scope. Raises Unsupported for anything this reader does not know."""
    pieces = tokenize_template(template)
    out = []
    run(pieces, 0, len(pieces), scope, out)
    return "".join(out)


def run(pieces, start, stop, scope, out):
    i = start
    while i < stop:
        kind, body = pieces[i]
        if kind == "text":
            out.append(body)
            i += 1
        elif kind == "say":
            value = evaluate(body, scope)
            out.append("None" if value is None else as_text(value))  # Jinja writes None, and nothing for an undefined
            i += 1
        elif body.startswith("for "):
            end = matching(pieces, i, stop, "for ", "endfor")
            name, _, source = body[4:].partition(" in ")
            if "," in name:
                raise Unsupported("a for over pairs")
            if len(split_outside_quotes(source, " if ")) > 1:
                # (T269) {% for x in xs if test %} keeps the xs that pass, which is not xs if test: not read
                raise Unsupported("a for with a test")
            values = evaluate(source, scope)
            if not isinstance(values, (list, tuple)):
                raise Unsupported(f"a for over {source.strip()!r}")
            # {% for %} ... {% else %} ... {% endfor %}: the else is written when there was nothing to loop over
            # (the review of T269: [] can be written now, and messages[1:] of one message was always empty)
            at = next_branch(pieces, i, end)
            body_end, otherwise = (at, at + 1) if pieces[at][1] == "else" else (end, end)
            for index, value in enumerate(values):
                run(pieces, i + 1, body_end, {**scope, name.strip(): value, "loop": Loop(index, len(values))}, out)
            if not values:
                run(pieces, otherwise, end, scope, out)
            i = end + 1
        elif body.startswith("if "):
            end = matching(pieces, i, stop, "if ", "endif")
            branches, at = [], i
            while at < end:  # if / elif / elif / else, as (condition or None, where the body begins)
                statement = pieces[at][1]
                if statement.startswith(("if ", "elif ")):
                    branches.append((statement.split(" ", 1)[1], at + 1))
                elif statement == "else":
                    branches.append((None, at + 1))
                at = next_branch(pieces, at, end)
            ends = [begin - 1 for _, begin in branches[1:]] + [end]
            for (condition, begin), finish in zip(branches, ends):
                if condition is None or truthy(evaluate(condition, scope)):
                    run(pieces, begin, finish, scope, out)
                    break
            i = end + 1
        elif body.startswith("set "):
            name, _, expression = body[4:].partition("=")
            target, _, attribute = name.strip().partition(".")
            if attribute:  # {% set ns.index = ... %}: an attribute of a namespace()
                if not isinstance(scope.get(target), Namespace) or not attribute.isidentifier():
                    raise Unsupported(f"{{% {body} %}}")
                setattr(scope[target], attribute, evaluate(expression, scope))
            else:
                scope[target] = evaluate(expression, scope)
            i += 1
        elif body.startswith("macro "):
            # defined for tools, which one turn has none of: skipped (a call of it is an expression it cannot read)
            i = matching(pieces, i, stop, "macro ", "endmacro") + 1
        elif body in ("endfor", "endif", "else", "endmacro") or body.startswith("elif "):
            raise Unsupported(f"{body!r} without its opening")
        else:
            raise Unsupported(f"{{% {body} %}}")


def matching(pieces, start, stop, opening, closing):
    """Where the {% endfor %} or {% endif %} of the statement at start is."""
    depth = 0
    for i in range(start, stop):
        kind, body = pieces[i]
        if kind != "do":
            continue
        if body.startswith(opening):
            depth += 1
        elif body == closing:
            depth -= 1
            if depth == 0:
                return i
    raise Unsupported(f"a {opening.strip()} without its {closing}: {pieces[start][1][:40]!r} at {start}, looking to {stop}")


def next_branch(pieces, start, end):
    """The elif / else / endif that follows the branch beginning at start, at the same level."""
    depth = 0
    for i in range(start + 1, end + 1):
        kind, body = pieces[i]
        if kind != "do":
            continue
        if body.startswith(("if ", "for ")):
            depth += 1
        elif body in ("endif", "endfor"):
            if depth == 0:
                return i
            depth -= 1
        elif depth == 0 and (body == "else" or body.startswith("elif ")):
            return i
    return end


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
    template = chat_template or config.get("chat_template")
    if isinstance(template, list):  # some models publish several; the first is the chat one
        template = template[0].get("template") if template and isinstance(template[0], dict) else None
    if not isinstance(template, str) or not template.strip():
        return None
    bos = config_token(config, "bos_token")
    # the tokens transformers gives a template by their names (special_tokens_map): a template may write the pad token
    names = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")
    turn = one_turn(template, {name: config_token(config, name) for name in names})
    # generate() starts every run with the BOS token already: one written by the template would be a second one
    return turn[len(bos):] if turn and bos and turn.startswith(bos) else turn


def one_turn(template, specials, mark="\x00prompt\x00"):
    """The template of one user turn, as src/models.js writes it: the text around a {prompt}.

    specials: the tokenizer's special tokens, for bos_token and eos_token. Returns None when the template uses
    something this reader does not know, and then the caller keeps the format it has.
    """
    scope = {"messages": [{"role": "user", "content": mark}], "add_generation_prompt": True,
             "bos_token": specials.get("bos_token", ""), "eos_token": specials.get("eos_token", ""),
             **{name: token for name, token in specials.items() if token and name not in ("bos_token", "eos_token")},
             "tools": None, "tools_json": None, "documents": None, "strftime_now": STRFTIME}
    try:
        text = render(template, dict(scope))
        # strftime_now() is written {date:format}, which the page fills with the day it sends the prompt. That holds
        # where the template only writes the date; one that reckons with it (yesterday's date from today's day of
        # the month) would come out otherwise, so the two are compared on two days, and it gives up where they differ
        days = CHECK_DAYS if "{date:" in text else []
        fill = lambda day: re.sub(r"\{date:([^}]*)\}", lambda found: time.strftime(found.group(1), day), text)
        if any(fill(day) != render(template, {**scope, DAY: day}) for day in days):
            return None
    except Unsupported:
        return None
    except Exception:
        return None
    if text.count(mark) != 1:
        return None
    # T138: a template that trims what was typed ({{ message['content'] | trim }}) says so as {prompt:trim}, which the
    # page's filled() trims: rendered once more with spaces around the mark, it either keeps them or drops both.
    # (Whether the spaces are there is no test: RakutenAI's writes "USER: " and " ASSISTANT:" around a trimmed one)
    try:
        spaced = render(template, {**scope, "messages": [{"role": "user", "content": f" {mark} "}]})
    except Exception:
        return None
    if spaced == text.replace(mark, f" {mark} "):
        return text.replace(mark, "{prompt}")
    if spaced == text:
        return text.replace(mark, "{prompt:trim}")
    return None  # it drops one of the two, or changes more than that: nothing filled() would do
