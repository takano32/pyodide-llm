# engine_plans.py (T.357)
# The plan Llama(external=) hands src/runtime/forward.js, as Python builds it today (the dict of engine/external.py's
# ExternalForward), for made-up headers: nothing of a model is there but its header and a final norm's weight, and
# start(plan) keeps the plan and ends before the engine is built (and before the tokenizer is read).
#   engine_plan(L, header, dtype, form, more, size)   L: llama2_numpy of the tree asked; more: what Llama takes besides
#                                                     the form (rotary, unturned, ...), and "outliers": True for a final
#                                                     norm with three channels far above the rest
#   python tests/engine_plans.py <the root of a tree> < [{header, dtype, form, more}]     -> the plans, as JSON
# The derived tables (bytes) are given as a hash each. tests/unchanged_layouts.py and tests/plan-keys-check.mjs read it.
import hashlib
import json
import struct
import sys

import numpy as np


class Handed(Exception):
    """The plan is in hand: nothing of the engine is built."""


class External:
    """forward.js as far as the plan, for Llama(external=): size and read(offset, length) of the checkpoint, and
    start(plan), which keeps the plan and ends (Handed) before any engine is built. The one stand-in of it (T.357.5):
    tests/plan_facts.py hands it a real checkpoint's bytes to read, the others nowhere()'s."""

    def __init__(self, size, read):
        self.size, self.read, self.plan = size, read, None

    def start(self, plan):
        self.plan = plan
        raise Handed()


def nowhere(header, outliers):
    """read(offset, length) of a file that is nowhere: its header, and wherever else Llama reads (the final norm's
    weight, for the outlier channels) a weight a little above 1, three channels far above the rest where outliers."""
    def read(offset, length):
        if offset == 0:
            return struct.pack("<7i", *header)
        weight = 1.0 + np.arange(length // 4, dtype=np.float32) / 1024
        if outliers:
            weight[[5, 77, 130]] = (50.0, 60.0, 70.0)
        return weight.tobytes()
    return read


def engine_plan(L, header, dtype, form, more, size):
    external = External(size, nowhere(header, more.get("outliers", False)))
    options = {key: value for key, value in more.items() if key != "outliers"}
    try:
        L.Llama(None, b"", dtype=dtype, external=external, **form, **options)
    except Handed:
        pass
    plan = dict(external.plan)
    plan["derived"] = {name: hashlib.sha256(bytes(value)).hexdigest()[:16] for name, value in plan["derived"].items()}
    return plan


if __name__ == "__main__":
    from tree import python_folder
    sys.path.insert(0, python_folder(sys.argv[1]))
    import llama2_convert as C
    import llama2_numpy as L

    plans = []
    for case in json.loads(sys.stdin.read()):
        header, dtype, form, more = case["header"], case["dtype"], case.get("form", {}), case.get("more", {})
        if form.get("rotated") is True:
            # (a rotated basis for the widths the model's matrices read: every seventh sign minus)
            q_dim = header[3] * (form.get("head_dim") or header[0] // header[3])
            widths = L.rotated_widths(header[0], q_dim, header[1], L.linear_form(form.get("linear")))
            form = {**form, "rotated": {"block": 8, "signs": {str(width): L.sign_bits(np.where(np.arange(width) % 7 == 3, -1.0, 1.0)) for width in widths}}}
        plans.append(engine_plan(L, header, dtype, form, more, C.checkpoint_size(header, dtype, {**L.FORM, **form})))
    print(json.dumps(plans))
