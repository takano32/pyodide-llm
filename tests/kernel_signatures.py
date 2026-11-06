"""tests/kernel_signatures.py (T639.11): the ctypes signatures Python gives the SIMD kernels against the kernels'
own declarations, so that a kernel's arguments, their number and their order are no longer something nothing says.

    python tests/kernel_signatures.py

Python's side is read off the product, not copied: engine/kernels.py's load_kernels() and kernel_wideners() (through
convert/readers.py's kernel_readers(), which hands it every reader that has a kernel) run with ctypes.CDLL replaced
by a library that only writes down the argtypes and restype it is given. The kernels' side is AssemblyScript's: the
module Python loads (simdkernel.so) is built from the source kernels/build.py names for it (kernel.ts), whose
`export { … } from "./kernel/<part>"` lines say where each export's `export function name(a: usize, n: i32, …): r`
is. The types are one to one: usize is c_void_p, i32 c_int32, f32 c_float, f64 c_double, and void is None.

A difference fails, by the kernel's name: an argument of another type, one argument more or fewer, a kernel Python
binds that the module does not export, a return Python reads where the kernel returns nothing. What the kernels
export and Python does not bind (forward.ts's) is said in a line, and fails nothing.

Where the module is built (src/runtime/built/simdkernel.so), its own export types are read too (the wasm's type
section: usize and i32 are both i32 there), so that the reading of the source is checked against what was compiled.

KNOWN holds the differences there are today, each with why it is left: a known one that is gone fails too (take it
out of KNOWN), so the list does not outlive what it says."""
import ctypes
import os
import re
import sys

from tree import built_folder, python_folder

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = {"usize": ctypes.c_void_p, "i32": ctypes.c_int32, "f32": ctypes.c_float, "f64": ctypes.c_double}
NAMES = {ctypes.c_void_p: "usize", ctypes.c_int32: "i32", ctypes.c_float: "f32", ctypes.c_double: "f64", None: "void"}
# the value types of the wasm (its type section): an address is an i32 in a 32-bit memory, as Pyodide's is
WASM = {0x7F: "i32", 0x7E: "i64", 0x7D: "f32", 0x7C: "f64", 0x7B: "v128"}
AS_WASM = {"usize": "i32", "i32": "i32", "f32": "f32", "f64": "f64"}
# (kernel, what differs) -> why it is left as it is. T639.11 changes no product code: these are for the owner to decide
KNOWN = {}


class Recorded:
    """A kernel as ctypes would give it: what is set on it is kept."""

    def __init__(self, name):
        self.name, self.argtypes, self.restype = name, None, ctypes.c_int  # (ctypes's own default restype)

    def __call__(self, *args):
        raise AssertionError(f"{self.name} was called while its signature was read")


class Library:
    def __init__(self, path):
        self.path, self.made = path, {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.made.setdefault(name, Recorded(name))


def pythons():
    """{kernel: (its argtypes, its restype)} as engine/kernels.py sets them."""
    sys.path.insert(0, python_folder())
    from engine.kernels import load_kernels
    from convert.readers import kernel_readers

    libraries, real = [], ctypes.CDLL

    def made(path, *args, **kwargs):
        libraries.append(Library(path))
        return libraries[-1]

    ctypes.CDLL = made
    try:
        loaded = load_kernels("simdkernel.so")
        assert loaded is not None, "load_kernels() gave None with a library that has every kernel: it fails on its own"
        readers = kernel_readers("simdkernel.so")
        assert readers, "kernel_readers() gave no reader with a library that has every kernel"
    finally:
        ctypes.CDLL = real
    found = {}
    for library in libraries:
        for name, kernel in library.made.items():
            if kernel.argtypes is None:
                continue  # looked for (hasattr) and not bound
            said = (tuple(kernel.argtypes), kernel.restype)
            if name in found and found[name] != said:
                raise AssertionError(f"{name} is bound twice in engine/kernels.py, with two signatures: {shown(*found[name])} and {shown(*said)}")
            found[name] = said
    return found


def shown(argtypes, restype):
    return f"({', '.join(NAMES.get(t, repr(t)) for t in argtypes)}): {NAMES.get(restype, repr(restype))}"


def python_module_source():
    """The AssemblyScript source of the module Python loads: the Makefile's rule for simdkernel.so says which script
    builds it, and that script's side_module() which source (no folder of the tree is written here)."""
    rule = re.search(r"^\S*simdkernel\.so:.*\n\t\S*python3? (\S+\.py)", open(os.path.join(HERE, "Makefile"), encoding="utf-8").read(), re.M)
    assert rule, "the Makefile has no rule that builds simdkernel.so with a Python script: where is Python's module built?"
    script = os.path.join(HERE, rule.group(1))
    made = re.findall(r'side_module\(here / "([^"]+)", out / "simdkernel\.so"', open(script, encoding="utf-8").read())
    assert len(made) == 1, f"{rule.group(1)} builds simdkernel.so {len(made)} times: which source is Python's?"
    return os.path.join(os.path.dirname(script), made[0])


called = {}  # a kernel's arguments' names, as its declaration says them (for the failures' lines)
DECLARED = re.compile(r"^export function (\w+)\s*\(([^)]*)\)\s*:\s*(\w+)", re.M)


def declarations(source):
    """{export: ([argument types], return type)} of a source and the parts it exports from."""
    text = open(source, encoding="utf-8").read()
    found = {name: (params, returned) for name, params, returned in DECLARED.findall(text)}
    out = {}
    for name, (params, returned) in found.items():
        out[name] = (parsed(source, name, params), returned)
    for names, part in re.findall(r'^export\s*\{([^}]*)\}\s*from\s*"([^"]+)"', text, re.M):
        path = os.path.normpath(os.path.join(os.path.dirname(source), part if part.endswith(".ts") else part + ".ts"))
        there = declarations(path)
        for name in (n.strip() for n in names.split(",") if n.strip()):
            if " as " in name:
                raise AssertionError(f"{source}: `{name}` renames an export: this check does not read it")
            if name not in there:
                raise AssertionError(f"{source} exports {name} from {part}, which declares no `export function {name}(…): …` on one line")
            if name in out:
                raise AssertionError(f"{source} exports {name} twice")
            out[name] = there[name]
    return out


def parsed(source, name, params):
    types = []
    for param in (p.strip() for p in params.split(",") if p.strip()):
        typed = re.fullmatch(r"\w+\s*:\s*(\w+)", param)
        if not typed or typed.group(1) not in TYPES:
            raise AssertionError(f"{source}: {name}'s argument `{param}` is of a type this check does not know ({', '.join(TYPES)})")
        types.append(typed.group(1))
    called[name] = [param.split(":")[0].strip() for param in params.split(",") if param.strip()]
    return types


def compiled(path):
    """{export: ([wasm argument types], [wasm results])} of the functions a built module exports."""
    wasm = open(path, "rb").read()
    at = 8

    def number():
        nonlocal at
        value = shift = 0
        while True:
            byte = wasm[at]
            at += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    def limits():
        nonlocal at
        flags = wasm[at]
        at += 1
        number()
        if flags & 1:
            number()

    def types():
        nonlocal at
        count = number()
        found = [WASM[byte] for byte in wasm[at:at + count]]
        at += count
        return found

    kinds, functions, exports, imported = [], [], {}, 0
    while at < len(wasm):
        section = wasm[at]
        at += 1
        size = number()
        end = at + size
        if section == 1:  # types
            for _ in range(number()):
                assert wasm[at] == 0x60, f"{path}: a type that is not a function's"
                at += 1
                kinds.append((types(), types()))
        elif section == 2:  # imports: a function's counts before the module's own
            for _ in range(number()):
                for _ in range(2):  # the module's name and the field's
                    length = number()  # (not `at += number()`: that adds to the at before the number was read)
                    at += length
                kind = wasm[at]
                at += 1
                if kind == 0:
                    number()
                    imported += 1
                elif kind == 1:  # a table
                    at += 1
                    limits()
                elif kind == 2:  # a memory
                    limits()
                elif kind == 3:  # a global
                    at += 2
                else:
                    raise AssertionError(f"{path}: an import of the kind {kind}")
        elif section == 3:  # the module's own functions' types
            functions = [number() for _ in range(number())]
        elif section == 7:  # exports
            for _ in range(number()):
                length = number()
                name = wasm[at:at + length].decode()
                at += length
                kind = wasm[at]
                at += 1
                index = number()
                if kind == 0:
                    exports[name] = index
        at = end
    return {name: kinds[functions[index - imported]] for name, index in exports.items() if index >= imported}


def main():
    source = python_module_source()
    declared = declarations(source)
    bound = pythons()
    wrong, known = [], set()

    def differs(name, what, said):
        if (name, what) in KNOWN:
            known.add((name, what))
        else:
            wrong.append(f"{name}: {said}")

    for name, (argtypes, restype) in sorted(bound.items()):
        if name not in declared:
            differs(name, "missing", f"engine/kernels.py binds it, and {os.path.relpath(source, HERE)} exports no such kernel")
            continue
        params, returned = declared[name]
        python = [NAMES.get(t, repr(t)) for t in argtypes]
        if len(python) != len(params):
            differs(name, "count", f"{len(python)} arguments in engine/kernels.py {shown(argtypes, restype)}, {len(params)} in the kernel ({', '.join(params)}): {returned}")
        else:
            for at, (mine, theirs) in enumerate(zip(python, params), 1):
                if mine != theirs:
                    differs(name, f"argument {at}", f"argument {at} ({called[name][at - 1]}) is {mine} in engine/kernels.py and {theirs} in the kernel")
        if TYPES.get(returned) is not restype and not (returned == "void" and restype is None):
            if returned != "void" and restype is None:
                differs(name, "returns", f"the kernel returns {returned}, and engine/kernels.py reads nothing (restype None)")
            else:
                differs(name, "returns", f"the kernel returns {returned}, and engine/kernels.py reads {NAMES.get(restype, repr(restype))}")
    for gone in sorted(set(KNOWN) - known):
        wrong.append(f"{gone[0]}: KNOWN says {gone[1]} differs ({KNOWN[gone]}), and it does not any more: take it out of KNOWN")

    built = built_folder(name="simdkernel.so", maybe=True)
    if os.path.exists(built):
        module = compiled(built)
        for name, (params, returned) in sorted(declared.items()):
            if name not in module:
                wrong.append(f"{name}: {os.path.relpath(source, HERE)} exports it and the built {os.path.relpath(built, HERE)} does not (make kernels?)")
                continue
            want = ([AS_WASM[p] for p in params], [] if returned == "void" else [AS_WASM[returned]])
            if list(module[name][0]) != want[0] or list(module[name][1]) != want[1]:
                wrong.append(f"{name}: the built module's ({', '.join(module[name][0])}) -> {module[name][1]} is not the source's ({', '.join(want[0])}) -> {want[1]} (make kernels?)")
        checked = f"; the built {os.path.relpath(built, HERE)} has the same {len(declared)} exports' types"
    else:
        checked = f"; no built {os.path.relpath(built, HERE)} to read (make kernels)"

    unbound = sorted(set(declared) - set(bound))
    print(f"{len(bound)} kernels bound in engine/kernels.py against {os.path.relpath(source, HERE)}{checked}")
    print(f"exported and not bound by Python ({len(unbound)}, forward.ts's): {', '.join(unbound)}")
    for name, what in sorted(known):
        print(f"known: {name} ({what}): {KNOWN[(name, what)]}")
    if wrong:
        print("FAILED:\n  " + "\n  ".join(wrong))
        sys.exit(1)
    print("the same")


if __name__ == "__main__":
    main()
