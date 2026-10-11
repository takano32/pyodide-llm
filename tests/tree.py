"""tests/tree.py (T367.1): where in a tree what is, for the Python tools: tests/tree.mjs's way in Python (the kinds,
the search and what stops it from taking the wrong file are said there). No place is written down: a kind's folder is
where its anchor (tests/tree.json, which tests/tree.mjs reads too) is found in the tree that is asked about;
tests/tree-check.mjs holds the two to the same answers.

    from tree import python_folder
    sys.path.insert(0, python_folder())          this tree's Python sources, wherever they are
    python_folder(root)                          another tree's, searched under its own root
    runtime_folder(root, "kept.js"), built_folder(root), served_folder(root, "models/tokenizer.bin")

    python tests/tree.py [--root <a tree>] runtime|python|built|served [<a file>]

A tool says `from tree import …`: tests/ is on sys.path where a tool is run as `python tests/<tool>.py` and under
pytest (tests/conftest.py is there)."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree.json"), encoding="utf-8") as file:
    RULE = json.load(file)
_walked = {}  # a root -> its files: a tree is walked once a process


def _files(root):
    if root not in _walked:
        found = []
        for folder, folders, files in os.walk(root, followlinks=False):
            folders[:] = [name for name in folders if name not in RULE["skip"] and not os.path.islink(os.path.join(folder, name))]
            under = os.path.relpath(folder, root).replace(os.sep, "/")
            found += [name if under == "." else f"{under}/{name}" for name in files if os.path.isfile(os.path.join(folder, name)) and not os.path.islink(os.path.join(folder, name))]
        _walked[root] = found
    return _walked[root]


def _ending(root, name):
    return sorted(file for file in _files(root) if file == name or file.endswith("/" + name))


def _looked():
    return f"(looked in every folder but {', '.join(RULE['skip'])})"


def _folder_of(root, kind):
    """A kind's folder, as the tree names it: where its anchor is (the built files' of a tree where nothing is
    built: where its Makefile builds them)."""
    anchor = RULE["anchors"][kind]
    found = _ending(root, anchor)
    if kind == "built" and not found:
        makefile = os.path.join(root, "Makefile")
        made = os.path.exists(makefile) and re.search(rf"^(\S*?)/?{re.escape(anchor)}:", open(makefile, encoding="utf-8").read(), re.M)
        if not made:
            raise ValueError(f"{root}: no {anchor} anywhere in the tree, and no rule of its Makefile makes one: where are the kernels built?")
        return made.group(1)
    what = f"{anchor} (what says where the {kind} files are)"
    if len(found) > 1:
        raise ValueError(f"{root}: {what} is in {len(found)} places: {' and '.join(found)}")
    if not found:
        raise ValueError(f"{root}: no {what} anywhere in the tree {_looked()}")
    return os.path.dirname(found[0])


def folders(root=None):
    """The folders of the kinds, as the tree names them."""
    root = os.path.abspath(str(root or HERE))
    return {kind: _folder_of(root, kind) for kind in RULE["anchors"]}


def _folder(kind):
    def folder(root=None, name="", maybe=False):
        root = os.path.abspath(str(root or HERE))
        inside = _folder_of(root, kind)
        under = os.path.normpath(os.path.join(root, inside, name))
        if not name or maybe or os.path.exists(under):
            return under
        found = _ending(root, name.rstrip("/"))
        if len(found) > 1:
            raise ValueError(f"{root}: {name} is not in {inside or 'the root'}/ (the {kind} files), and is in {len(found)} places: {' and '.join(found)}")
        if not found:
            raise ValueError(f"{root}: no {name} in {inside or 'the root'}/ (the {kind} files) nor anywhere in the tree {_looked()}")
        return os.path.join(root, found[0])
    return folder


runtime_folder, python_folder, built_folder, served_folder = (_folder(kind) for kind in ("runtime", "python", "built", "served"))

if __name__ == "__main__":
    args = sys.argv[1:]
    root = None
    if args[:1] == ["--root"]:
        root, args = args[1], args[2:]
    if not args or args[0] not in RULE["anchors"] or len(args) > 2:
        sys.exit("python tests/tree.py [--root <a tree>] runtime|python|built|served [<a file>]")
    try:
        print(_folder(args[0])(root, *args[1:]))
    except ValueError as error:
        sys.exit(str(error))
