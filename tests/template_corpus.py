# template_corpus.py
# The converter's reader of chat templates (llama2_convert.one_turn_template, T73 / T127 / T269) against Jinja itself
# on real templates: those of the most downloaded text-generation models of the Hub that are not gated, and of any
# repository named. Only tokenizer_config.json and chat_template.jinja are fetched (kept in the directory, so a second
# run asks the Hub for nothing).
#
#   python3 tests/template_corpus.py <directory> [--top 500] [--before <another llama2_convert.py>] [owner/repo[@revision] ...]
#
# T397: the converter renders with jinja2 itself where it can import it, so here "the converter" is its wrapping of
# jinja2 (the date, the prompt written once, what it trims) against the plain rendering below, and "its own reader" is
# what a browser without the package gets: both are judged, and neither may write another text than Jinja.
#
# For each distinct template, one user turn is rendered by jinja2 with transformers' settings (what
# apply_chat_template does: trim_blocks, lstrip_blocks, the tokenizer's special tokens by name, tools and documents
# as None, strftime_now, raise_exception, tojson, the generation tag) and by the reader. The reader may refuse (the page then has no format for ?hf=); what it
# must not do is read a template and write another text than Jinja: any such is DIFFERENT and the exit status is 1.
# --before: the same with another copy of the converter (public/llama2_convert.py of another tree: main's is under
# .tmp/unchanged/<commit> once `node tests/unchanged.mjs sizes` or any of its checks has run, tests/other-tree.mjs),
# and what changed between the two is listed: a change of the reader is to move templates from refused to the same,
# and nothing else.
#
# Needs jinja2 (tests/requirements-reference.txt has it through transformers); no weights, no torch.
import hashlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "\x00prompt\x00"
DAY = time.strptime("2026-12-31 23:59:59", "%Y-%m-%d %H:%M:%S")
TOKENS = ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token")


def converter(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))  # (its own llama2_numpy, where it imports one)
    # T347: the converter is a window over the package convert/ beside it. Two converters have two packages of that
    # name, so each is imported with none loaded and taken out again: a second would else get the first one's parts
    # and be compared with itself. (The file given is the window: public/llama2_convert.py of a tree.)
    parts = lambda: [key for key in sys.modules if key == "convert" or key.startswith("convert.")]
    aside = {key: sys.modules.pop(key) for key in parts()}
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        for key in parts():
            del sys.modules[key]
        sys.modules.update(aside)
    return module


def get(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pyodide-llm tests"}), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                return None
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(5 * (attempt + 1))


def most_downloaded(count):
    """[(repo, revision)] of the text-generation models of the Hub by downloads, the gated ones left out."""
    found, url = [], ("https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1"
                      f"&limit={min(count * 2, 1000)}&expand[]=sha&expand[]=gated")
    for model in json.loads(get(url) or b"[]"):
        if not model.get("gated") and not model.get("private") and model.get("sha"):
            found.append((model["id"], model["sha"]))
    return found[:count]


def files(directory, repo, revision):
    """(tokenizer_config.json's text or None, chat_template.jinja's text or None), fetched once"""
    folder = directory / repo.replace("/", "--") / revision
    folder.mkdir(parents=True, exist_ok=True)
    texts = []
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        kept, missing = folder / name, folder / (name + ".missing")
        if not kept.exists() and not missing.exists():
            data = get(f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}")
            (missing if data is None else kept).write_bytes(data or b"")
        texts.append(kept.read_text(encoding="utf-8", errors="replace") if kept.exists() else None)
    return texts


def token_text(config, name):
    token = config.get(name)
    token = token.get("content") if isinstance(token, dict) else token
    return token if isinstance(token, str) else ""


def by_jinja(template, config, content, day):
    """One user turn as transformers' apply_chat_template writes it (its environment and what it passes)"""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    class Generation(jinja2.ext.Extension):
        """{% generation %} ... {% endgeneration %}, which transformers adds to mark what the assistant wrote: the body as it is"""
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
    environment.globals["strftime_now"] = lambda form: time.strftime(form, day)
    names = {name: token_text(config, name) for name in TOKENS if token_text(config, name)}
    return environment.from_string(template).render(messages=[{"role": "user", "content": content}],
                                                    add_generation_prompt=True, tools=None, documents=None, **names)


def real_turn(template, config):
    """The one turn the page would send, by Jinja: (text with {prompt} or {prompt:trim}, with the dates of DAY), or
    (None, why) where the template does not write one user turn the page can fill."""
    try:
        text = by_jinja(template, config, MARK, DAY)
        spaced = by_jinja(template, config, f" {MARK} ", DAY)
    except Exception as error:  # the template raises for a lone user turn, or uses what transformers does not give it
        return None, f"{type(error).__name__}: {str(error)[:80]}"
    if text.count(MARK) != 1:
        return None, f"the prompt {text.count(MARK)} times"
    bos = token_text(config, "bos_token")
    if bos and text.startswith(bos):
        text, spaced = text[len(bos):], spaced[len(bos):]
    if spaced == text.replace(MARK, f" {MARK} "):
        return text.replace(MARK, "{prompt}"), None
    if spaced == text:
        return text.replace(MARK, "{prompt:trim}"), None
    return None, "trims one side of the prompt"


def by_reader(module, template, config, in_file, own=False):
    """The converter's one turn with its {date:format} filled with DAY, or None where it refuses. own: by its own
    reader, as where jinja2 is not there (a converter of before T397 has no other)"""
    import re
    tokens = {name: config[name] for name in TOKENS if name in config}
    parts = module.one_turn.__globals__  # convert/template.py's names
    had = parts.get("jinja_environment")
    if own and had:
        parts["jinja_environment"] = lambda: None
    try:
        if in_file:
            turn = module.one_turn_template(json.dumps(tokens), template)
        else:
            turn = module.one_turn_template(json.dumps({**tokens, "chat_template": template}))
    finally:
        if own and had:
            parts["jinja_environment"] = had
    if turn is None:
        return None
    return re.sub(r"\{date:([^}]*)\}", lambda found: time.strftime(found.group(1), DAY), turn)


def judge(module, template, config, in_file, real, own=False):
    turn = by_reader(module, template, config, in_file, own)
    if turn is None:
        return "refused"
    if real is None:
        return "DIFFERENT"  # it read what Jinja does not write as one turn
    return "the same" if turn == real else "DIFFERENT"


def main():
    arguments = sys.argv[1:]

    def take(flag, default):
        if flag not in arguments:
            return default
        at = arguments.index(flag)
        value = arguments[at + 1]
        del arguments[at:at + 2]
        return value

    top, before = int(take("--top", "0")), take("--before", None)
    directory, repos = Path(arguments[0]), [tuple((argument + "@main").split("@")[:2]) for argument in arguments[1:]]
    from tree import python_folder
    now = converter(Path(python_folder(HERE.parent, "llama2_convert.py")), "converter_now")
    then = converter(before, "converter_before") if before else None
    if top:
        repos += most_downloaded(top)
    seen, counts, changes, different = {}, {}, [], []
    alone, alone_changes = {}, []  # the same of the converter's own reader
    without = 0
    for repo, revision in repos:
        try:
            config_text, template_file = files(directory, repo, revision)
        except Exception as error:
            print(f"templates: {repo}: not fetched ({type(error).__name__})")
            continue
        try:
            config = json.loads(config_text) if config_text else {}
        except ValueError:
            config = {}
        config = config if isinstance(config, dict) else {}
        template = template_file or config.get("chat_template")
        if isinstance(template, list):
            template = template[0].get("template") if template and isinstance(template[0], dict) else None
        if not isinstance(template, str) or not template.strip():
            without += 1
            continue
        key = hashlib.sha256(json.dumps([template, [token_text(config, name) for name in TOKENS]]).encode()).hexdigest()
        if key in seen:
            continue
        seen[key] = repo
        real, why = real_turn(template, config)
        verdict = judge(now, template, config, bool(template_file), real)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "DIFFERENT":
            different.append(repo)
            print(f"templates: {repo}@{revision[:8]}: DIFFERENT: Jinja {json.dumps(real) if real is not None else why}, "
                  f"the reader {json.dumps(by_reader(now, template, config, bool(template_file)))}")
        by_itself = judge(now, template, config, bool(template_file), real, own=True)
        alone[by_itself] = alone.get(by_itself, 0) + 1
        if by_itself == "DIFFERENT":
            different.append(repo)
            print(f"templates: {repo}@{revision[:8]}: DIFFERENT by the converter's own reader: Jinja "
                  f"{json.dumps(real) if real is not None else why}, the reader "
                  f"{json.dumps(by_reader(now, template, config, bool(template_file), own=True))}")
        if then:
            was = judge(then, template, config, bool(template_file), real, own=True)
            if was != verdict:
                changes.append((repo, was, verdict))
                print(f"templates: {repo}@{revision[:8]}: {was} before, {verdict} now"
                      f"{'' if real is not None else f' (Jinja: {why})'}")
            if was != by_itself:
                alone_changes.append((repo, was, by_itself))
                print(f"templates: {repo}@{revision[:8]}: {was} before, {by_itself} now by the converter's own reader")
    print(f"templates: {len(repos)} repositories, {without} without a template, {len(seen)} distinct templates: "
          + ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items())))
    print("templates: by the converter's own reader (where jinja2 is not there): "
          + ", ".join(f"{count} {verdict}" for verdict, count in sorted(alone.items())))
    if then:
        for what, changed in (("the converter", changes), ("its own reader", alone_changes)):
            kinds = {}
            for _, was, verdict in changed:
                kinds[f"{was} -> {verdict}"] = kinds.get(f"{was} -> {verdict}", 0) + 1
            print(f"templates: against the converter before, {what}: {len(changed)} changed"
                  + (": " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items())) if kinds else ""))
    bad = [change for change in [*changes, *alone_changes] if change[1:] != ("refused", "the same")]
    if different or bad:
        print(f"templates: FAILED ({len(different)} DIFFERENT, {len(bad)} changes other than refused -> the same)")
        sys.exit(1)
    print("templates: ok")


if __name__ == "__main__":
    main()
