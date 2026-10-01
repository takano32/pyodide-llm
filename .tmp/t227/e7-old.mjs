import { warnings, checkVerdict } from "./old-bench.js";
import * as F from "./fixtures.mjs";

const dawn = 'Invalid ComputePipeline "tile 32x32".\n - While validating compute stage ([ShaderModule "main"]).\n - While calling [Device].CreateComputePipeline().';
const check = { ...F.aCheck, "llama.cpp tiles 32×32, f16": { worstRelative: NaN, ok: false, error: dawn }, "TF.js tiles 32×32, vec4": { worstRelative: NaN, ok: false, error: "refused | x" } };
const verdicts = Object.entries(check), said = verdicts.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
const markdown = `**Shaders against JavaScript**: ${verdicts.map(checkVerdict).join(", ")}`;
const out = warnings([{ title: "GPU", status: "wrong", said, markdown }]);
console.log(out.length, out.map((w) => w.length));
console.log("the whole check line:", markdown.length, "chars; verdicts:", verdicts.length);
// the same with the failed verdict last of all 27 (the bogus piece is the line up to the first break)
const last = { ...F.aCheck };
delete last["tokens on the GPU"];
last["tokens on the GPU"] = { worstRelative: NaN, ok: false, error: dawn };
const v2 = Object.entries(last), s2 = v2.filter(([, v]) => v.error || !v.ok).map(checkVerdict);
const out2 = warnings([{ title: "GPU", status: "wrong", said: s2, markdown: `**Shaders against JavaScript**: ${v2.map(checkVerdict).join(", ")}` }]);
console.log("failed verdict last:", out2.map((w) => w.length));
