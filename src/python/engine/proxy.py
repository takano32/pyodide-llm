# What JavaScript hands Python, made Python's own (T639.5). In Pyodide a JavaScript object reaches Python as a
# JsProxy: a dict's or an array's comes as one, and a Uint8Array's too, which to_py() makes a memoryview. Outside
# Pyodide (the tools, pytest) the same value is Python's already. This file reads nothing of the engine; the
# converter reads it too (convert/ imports engine/, never the other way).
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository).


def plain(value):
    """value as Python has it: a JsProxy's to_py(), anything else as it is (None stays None)."""
    return value.to_py() if hasattr(value, "to_py") else value
