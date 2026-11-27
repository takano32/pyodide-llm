"""T665: PLaMo 3 Translate's request as engine/pfml.py puts it together, against the official chat_template.jinja: the
template rendered by transformers' apply_chat_template (format_check.py's way, with the repository's own tokenizer,
tokenization_plamo.py's Plamo3Tokenizer) and its ids, for many requests of languages, forms, contexts, glossaries and
pairs translated before. And the request the list's entries had until T669 (formats.ts's plamoTranslate, as the page
filled it and the engine read it with the converter's specials; T669's entries send pfml_tokens()'s own) against
pfml_tokens() of the same direction, and a mark typed in a text read as its characters. Run by hand (the files are PFN's; pytest has a made-up vocabulary, tests/test_pfml.py):

    python tests/pfml_check.py <a folder with chat_template.jinja, tokenizer.jsonl, tokenizer_config.json,
        special_tokens_map.json and tokenization_plamo.py of a pfnet/plamo-3-translate-* revision of 2026-11-07 or
        later: the template of the first release writes easy Japanese ja(easy), not ja-easy (T665's review)>

tokenization_plamo.py imports numba: PYTHONPATH names a folder of .tmp that has it (tests/plamo_tokenizer_check.py
says how). Any request whose ids differ fails (exit 1) and is printed.
"""
import itertools
import sys
from pathlib import Path

from tree import python_folder

sys.path.insert(0, python_folder())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import detokenize  # noqa: E402
from engine.pfml import pfml_tokens  # noqa: E402
from format_check import PROMPTS, filled, reference_of  # noqa: E402
from plamo_tokenizer_check import read  # noqa: E402

TEXTS = PROMPTS + ["The weather is nice today. Let's go for a walk.", "今日はいい天気ですね。散歩に行きましょう。",
                   "  two leading spaces and a line\nand    four", "表\tと\n\n段落"]
LISTED = ("<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>{source}"
          "<|plamo:msg|>{{prompt}}<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>")
GLOSSARY = [("管理画面", "dashboard"), ("プロジェクト", "project")]
HISTORY = [("プロジェクトの管理画面を開きます。", "Open the project dashboard."),
           ("Open the project dashboard.", "  It lists  all active tasks.\n")]


def requests():
    """(text, target, parts) of every mix: each part with and without, over a few languages"""
    for text, (target, source), form, context, glossary, history in itertools.product(
            TEXTS[:4], [("ja", None), ("en", "ja"), ("ja", "en"), ("zh-hans", None), ("ja(easy)", "en"), ("ceb", "ja(easy)"), ("ja-easy", "en")],
            [None, "plain", "polite"], [None, "", "工作で折り紙を作っています。", " "], [[], GLOSSARY], [[], HISTORY]):
        if form is not None and target != "ja":
            continue
        yield text, target, {"source": source, "form": form, "context": context, "glossary": glossary, "history": history}


def official(reference, text, target, parts):
    """the ids the official template's text is, after its BOS"""
    messages = [message for said, translated in parts["history"]
                for message in ({"role": "user", "content": said}, {"role": "assistant", "content": translated})]
    messages.append({"role": "user", "content": text})
    said = {"target_lang": target, **{name: parts[name] for name in ("source", "form", "context") if parts[name] is not None}}
    said = {("source_lang" if name == "source" else name): value for name, value in said.items()}
    if parts["glossary"]:
        said["glossary"] = [{"source": term, "target": translation} for term, translation in parts["glossary"]]
    ids = reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **said)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    assert ids[0] == 1, ids[:4]
    return ids[1:]


def main(arguments):
    if len(arguments) != 1:
        raise SystemExit(__doc__)
    folder = Path(arguments[0])
    engine, specials, real, controls, _ = read(folder)
    reference = reference_of(folder)
    wrong, count = [], 0
    for text, target, parts in requests():
        count += 1
        got, want = pfml_tokens(engine, text, target, **parts), official(reference, text, target, parts)
        if got != want:
            wrong.append(("template", text, target, parts, got, want))
    print(f"the official template: {count} requests, {len(wrong)} with other ids")
    listed = 0
    for text in TEXTS:
        for source, target in (("en", "ja"), ("ja", "en")):
            listed += 1
            page = engine.encode(filled(LISTED.format(source=source, target=target), text), specials, 1)
            got = pfml_tokens(engine, text, target, source=source)
            if got != page:
                wrong.append(("listed", text, target, source, got, page))
    print(f"the list's format: {listed} requests, {sum(kind == 'listed' for kind, *_ in wrong)} with other ids")
    control_ids = {engine.index[piece.encode()] for piece in controls if piece.encode() in engine.index}
    typed = ["<|plamo:tag|>", "a<|plamo:tag|>output<|plamo:msg|>b", "<|plamo:bos|> <|plamo:x|> <|plamo:index|>",
             "<|plamo:begin_x:plamo|><|plamo:end_"]
    frame, x = pfml_tokens(engine, "x", "ja"), engine.encode("x")
    at = next(i for i in range(len(frame)) if frame[i:i + len(x)] == x)
    head, tail = frame[:at], frame[at + len(x):]
    for text in typed:
        ids = pfml_tokens(engine, text, "ja")
        middle = ids[len(head):len(ids) - len(tail)]
        back = detokenize(engine, middle, bos=-1)
        if ids[:len(head)] != head or ids[len(ids) - len(tail):] != tail or control_ids & set(middle) or back != text:
            wrong.append(("typed", text, None, None, ids, back))
    print(f"marks typed in a text: {len(typed)}, {sum(kind == 'typed' for kind, *_ in wrong)} read as marks or written back otherwise")
    for kind, text, target, parts, got, want in wrong[:20]:
        print(f"  WRONG ({kind}) {text[:60]!r} into {target} {parts}\n    the engine {got[:40]}\n    the real   {want[:40] if isinstance(want, list) else want!r}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
