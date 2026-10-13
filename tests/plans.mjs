// plans.mjs
// What createForward() is handed for a made-up header, without Pyodide and without built kernels: the plan Python
// makes (by the native Python: where every tensor is, llama2_numpy's external_tensors(), and the plan itself,
// engine/plan.py's forward_plan()) and kernels that do nothing. For the checks that run forward.js on its own:
// tests/memory-check.mjs (T130), tests/unchanged-calls.mjs (T346) and tests/gpu-alone-check.mjs (T450).
//   const { plansOf, planOf, FORM, empty, nothing } = plans(<the repository's root, a URL or a path>)
// T375.2: planOf() writes no plan by hand any more. The engine reads the facts of a model's layers from the plan, and
// a second writer of them here would be the copy T375 takes away: the plan is Python's own, with the few things a
// check sets for itself laid over it (planOf() says which).
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { treeOf } from "./tree.mjs";

// the plan Python hands createForward (llama2_numpy's external_tensors()), for a header, a form and a dtype:
// where every tensor is, and the bytes of the RoPE tables Python computes (the checkpoint's size is where the places end)
const python = `
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import llama2_convert, llama2_numpy as L

def keeps_int8(header, form, dtype):
    # the engine's own condition for keeping int8: the int8 kernels work on groups of 32 only (T229: and the rows of a
    # linear-attention layer's output matrix, its value heads together)
    dim, hidden, _, heads, kv_heads = header[:5]
    head = form["head_dim"] or dim // heads
    linear = L.linear_form(form["linear"])
    return dtype in ("int8", *L.PACKED) and all(n % 32 == 0 for n in (dim, heads * head, kv_heads * head, hidden)) \
        and (linear is None or L.linear_widths(linear)[2] % 32 == 0)

def plan_of(header, form, dtype):
    # where every tensor is: external_tensors(), the rows of engine/layout.py placed (T359)
    tensors = L.external_tensors(header, dtype, form)
    head_size = form["head_dim"] or header[0] // header[3]
    table = header[6] * (head_size // 2) * 4
    derived = {"freq_cis_real": table, "freq_cis_imag": table} if form["arch"] == "gpt2" or dtype != "float32" else {}
    # (the tensors by name: the order of a dict is not what the plan says, and tests/unchanged-calls.mjs hashes this whole)
    return {"header": header, "form": form, "dtype": dtype, "size": llama2_convert.checkpoint_size(header, dtype, form),
            "tensors": dict(sorted(tensors.items())), "derived": derived, "keep_int8": bool(keeps_int8(header, form, dtype)), "head_size": head_size}

def engines_plan(header, form, dtype):
    # T375.2: the plan the engine is handed, as Python makes it for a model (engine/plan.py's forward_plan(), from the
    # header, the form and the dtype alone): a made-up model's options are a Qwen3.5's RoPE over a quarter of a head and a
    # GPT-NeoX's parallel residual. Its rotated basis and its RoPE tables are planOf()'s (the bytes of neither matter
    # to a kernel that does nothing), and its final norm has no outlier channels (planOf() says how many there are)
    dims = L.Dims(header, form)
    flat = lambda offset, length: np.ones(length // 4, dtype=np.float32).tobytes()
    return L.forward_plan(dims, dtype, flat, int8=bool(keeps_int8(header, form, dtype)), kv_start=header[6],
                          rotary=dims.head_size // 4 if form["linear"] else 0, parallel_residual=form["arch"] == "neox")

def both(header, form, dtype):
    return {**plan_of(header, form, dtype), "plan": engines_plan(header, form, dtype)}

print(json.dumps([both(**s) for s in json.loads(sys.stdin.read())]))
`;
// (the tree's Python sources are where tests/tree.mjs finds them: the script is given the folder)
const plansIn = (root) => (shapes) => {
  const folder = typeof root === "string" ? root : fileURLToPath(root);
  return JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", python, treeOf(folder).python()], { cwd: folder, input: JSON.stringify(shapes), maxBuffer: 1 << 28 }).toString());
};
export const FORM = { bias: false, arch: "llama", qk_norm: false, head_dim: 0, linear: null, rotated: null, convolution: null };

// what createForward is handed: Python's plan of the model (p.plan), and over it what a check sets for itself: whether the
// kernels are the relaxed ones, kv_start where the cache starts, the outlier channels of the final norm there are
// (footprint() counts the most there can be for every int8 model: the second part asks for them), the block of a rotated
// basis (the form's) and the bytes Python computes, as many zeros: the RoPE tables (p.derived) and, T237's review, of a
// rotated basis a float32 for every value of every width a matrix reads (the plan's widths say which)
export const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
export function planOf(p, { relaxed = true, kvStart = p.header[6], outliers = 0 } = {}) {
  const rotated = p.form.rotated ? p.form.rotated.block : 0;
  return {
    ...p.plan, relaxed, kv_start: kvStart, rotated,
    derived: Object.fromEntries([...Object.entries(p.derived).map(([name, bytes]) => [name, new Uint8Array(bytes)]),
      ...(rotated ? p.plan.widths.rotated : []).map((width) => [`signs.${width}`, new Uint8Array(width * 4)])]),
    outliers: p.keep_int8 ? Array.from({ length: Math.min(outliers, p.header[0]) }, (_, c) => c) : [],
  };
}
// (the kernels do nothing unless a part says what one does: wrap is how the profiler swaps them, tests/profile.mjs)
export const nothing = () => new Proxy({}, { get: () => () => 0 });

export const plans = (root) => ({ plansOf: plansIn(root), planOf, FORM, empty, nothing });
