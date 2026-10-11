"""tests/tree.py (T367.1): where in a tree what is, for the Python tools. The rule is tests/tree.json, which
tests/tree.mjs reads too (the kinds and what is assumed are said there); tests/tree-check.mjs holds the two to the
same answers.

    from tree import python_folder
    sys.path.insert(0, python_folder())          this tree's Python sources (public/ today, src/python/ after T367.2)
    python_folder(root)                          another tree's
    runtime_folder(root), built_folder(root), served_folder(root)

    python tests/tree.py [--root <a tree>] runtime|python|built|served [<a file>]

A tool says `from tree import …`: tests/ is on sys.path where a tool is run as `python tests/<tool>.py` and under
pytest (tests/conftest.py is there)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree.json"), encoding="utf-8") as file:
    RULE = json.load(file)


def _found(root, kind):
    rule = RULE[kind]
    holding = [folder for folder in rule["folders"] if os.path.exists(os.path.join(root, folder, rule["mark"]))]
    if len(holding) != 1:
        raise ValueError(f"{root}: {' and '.join(holding)} both hold {rule['mark']}: a tree half moved" if holding else
                         f"{root}: no {rule['mark']} in {' or '.join(rule['folders'])}: is this a tree of the project?")
    return holding[0]


def folders(root=None):
    """The folders of the four kinds, as the tree names them."""
    root = os.path.abspath(str(root or HERE))
    runtime = _found(root, "runtime")
    return {"runtime": runtime, "python": _found(root, "python"), "built": RULE["built"]["beside"][runtime],
            "served": RULE["served"]["folder"]}


def _folder(kind):
    def folder(root=None, name=""):
        root = os.path.abspath(str(root or HERE))
        return os.path.join(root, folders(root)[kind], *([name] if name else []))
    return folder


runtime_folder, python_folder, built_folder, served_folder = (_folder(kind) for kind in ("runtime", "python", "built", "served"))

if __name__ == "__main__":
    args = sys.argv[1:]
    root = None
    if args[:1] == ["--root"]:
        root, args = args[1], args[2:]
    if not args or args[0] not in ("runtime", "python", "built", "served") or len(args) > 2:
        sys.exit("python tests/tree.py [--root <a tree>] runtime|python|built|served [<a file>]")
    print(_folder(args[0])(root, *args[1:]))
