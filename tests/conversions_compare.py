# conversions_compare.py
# What a change of the converter changes for the list (T143's rule, T369): for every Hugging Face entry, the options
# and the tokenizer.bin the converter makes, and the ids the page sends for format_check.py's prompts, as the list has
# the entry (its own options and format over the converter's) and as ?hf= has the repository (the converter's alone),
# in two trees. Only config.json, the tokenizer files and the heads of the weights are fetched (format_check.py's
# converted()); no weights: the converter writes into a sink that does nothing.
#
#   python3 tests/conversions_compare.py <directory for the downloads> [--before <commit>] [model id ...]
#   python3 tests/conversions_compare.py --dump <tree> <directory for the downloads> <out.json> [model id ...]
#
# --before: the commit to compare with (origin/main; its tree is tests/other-tree.mjs's). Each tree is read by its own
# tests/format_check.py and src/models.js, in a process of its own (--dump): what an entry says may change too.
# The table: one line an entry whose options, tokenizer.bin or ids differ, and the count of those that do not. The exit
# status is 0 either way: what may differ is the change's own to say. Needs no reference tools (numpy only).
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dump(tree, directory, out, only):
    """The conversions of one tree, by that tree's own format_check.py (which puts its tree's public/ first)."""
    spec = importlib.util.spec_from_file_location("format_check", Path(tree) / "tests" / "format_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    numpy_engine = sys.modules["llama2_numpy"]
    accepted = set(inspect.signature(numpy_engine.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    results = {}
    for entry in check.entries():
        if only and entry["id"] not in only:
            continue
        try:
            if hasattr(check, "converted"):
                _, made = check.converted(entry, Path(directory))
            else:  # (a tree of before T374.4: gone once no such tree is compared with)
                made = check.conversion(entry, *check.fetch(entry, Path(directory)))
        except Exception as error:  # said, and compared as it is: a refusal that comes or goes is a difference too
            results[entry["id"]] = {"refused": f"{type(error).__name__}: {error}"}
            print(f"refused {entry['id']}: {error}", file=sys.stderr, flush=True)
            continue
        sent = {}
        for name, own in (("list", True), ("hf", False)):
            options = {**made.options, **(entry.get("options", {}) if own else {})}
            template = (entry.get("template") if own else None) or made.options.get("template")
            tokenizer = numpy_engine.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                               **{key: value for key, value in options.items() if key in accepted})
            ids = [[options["bos"]] + tokenizer.encode(check.filled(template, prompt) if template else prompt,
                                                      tuple(options.get("specials", ())))
                   for prompt in check.PROMPTS]
            sent[name] = {"ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "first": ids[0][:6],
                          "stop_tokens": sorted(set(options["stop_tokens"])), "template": template}
        results[entry["id"]] = {"options": made.options, "sent": sent,
                                "tokenizer": hashlib.sha256(bytes(made.tokenizer)).hexdigest()[:16]}
        print(f"made {entry['id']}", file=sys.stderr, flush=True)
    Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True))


def short(value):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 70 else text[:67] + "..."


def compare(before, after):
    same = 0
    for id in sorted(set(before) | set(after), key=lambda id: list(after).index(id) if id in after else -1):
        old, new = before.get(id), after.get(id)
        if old == new:
            same += 1
            continue
        if old is None or new is None or "refused" in old or "refused" in new:
            print(f"{id}: {short(old)} -> {short(new)}")
            continue
        keys = sorted(key for key in set(old["options"]) | set(new["options"]) if old["options"].get(key) != new["options"].get(key))
        said = [f"tokenizer.bin {'CHANGED' if old['tokenizer'] != new['tokenizer'] else 'same'}"]
        for name in ("list", "hf"):
            for what in ("ids", "stop_tokens", "template"):
                if old["sent"][name][what] != new["sent"][name][what]:
                    said.append(f"{name}'s {what} CHANGED" + (f" (begins {old['sent'][name]['first']} -> {new['sent'][name]['first']})" if what == "ids" else ""))
        print(f"{id}: options {', '.join(keys) or 'same'}; {'; '.join(said)}")
        for key in keys:
            print(f"    {key}: {short(old['options'].get(key))} -> {short(new['options'].get(key))}")
    print(f"{same} of {len(set(before) | set(after))} entries the same in everything")


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--dump"]:
        return dump(arguments[1], arguments[2], arguments[3], arguments[4:])
    commit = "origin/main"
    if "--before" in arguments:
        at = arguments.index("--before")
        commit = arguments[at + 1]
        del arguments[at:at + 2]
    directory, only = arguments[0], arguments[1:]
    script = f"import('./tests/other-tree.mjs').then(({{ otherTree }}) => console.log(otherTree({json.dumps(commit)}).folder))"
    other = subprocess.check_output(["node", "-e", script], cwd=HERE.parent, text=True).strip()
    work = HERE.parent / ".tmp" / "conversions"
    work.mkdir(parents=True, exist_ok=True)
    made = []
    # (the other tree's is made once for a commit and the entries asked for: its tree does not change)
    asked = hashlib.sha256(" ".join(only).encode()).hexdigest()[:8]
    for out, tree in ((work / f"before-{Path(other).name[:12]}-{asked}.json", other), (work / "after.json", HERE.parent)):
        if tree is not other or not out.exists():
            subprocess.check_call([sys.executable, __file__, "--dump", str(tree), directory, str(out), *only])
        made.append(json.loads(out.read_text()))
    compare(*made)


if __name__ == "__main__":
    main()
