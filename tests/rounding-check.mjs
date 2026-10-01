// T225's review: /benchmark/'s GPU worker (public/benchmark/gpu.js) checking the shaders on devices that round a float32 to a
// float16 the other ways WGSL lets a device (tests/rounding.mjs: toward zero, as Direct3D does, which the owner's NVIDIA
// PC on Windows did; away from zero; every conversion toward zero), in Node on Dawn + lavapipe at subgroups of 4 and 16
// (tests/bench-dawn.mjs, the step "check"). CI's own implementations round to the nearest, so nothing else in CI says
// whether a check of the keys and values the GPU writes rests on that (T225's did). Every check must be right on each,
// and the cache must show the rounding asked: about half of the keys and values the other way from the nearest, none
// the other way again, none farther.
//   node tests/rounding-check.mjs <the webgpu package's directory> [toward-zero] [away] [everything] [nearest]
// (about 30 s a rounding and width; gpu-prompt.yml's Dawn job runs it; tests.yml's extra= can: bash tests/rounding-check.sh)
import { spawnSync } from "node:child_process";
import fs from "node:fs";

const [webgpu, ...asked] = process.argv.slice(2);
if (!webgpu) {
  console.error("node tests/rounding-check.mjs <the webgpu package's directory> [toward-zero] [away] [everything] [nearest]");
  process.exit(2);
}
const hows = asked.length ? asked : ["toward-zero"];
const icd = process.env.VK_ICD_FILENAMES ?? `/usr/share/vulkan/icd.d/${fs.readdirSync("/usr/share/vulkan/icd.d").find((name) => /^lvp_icd.*\.json$/.test(name))}`;
// how /benchmark/'s verdicts say what the cache holds (public/benchmark/gpu.js's halvesSaid), added up over a row's words
const WORDS = /K and V (\d+) to the nearest float16(?:, (\d+) toward zero, (\d+) away from it, (\d+) farther)?/g;
const counted = (text) => {
  const sum = { same: 0, inward: 0, outward: 0, far: 0 };
  for (const [, same, inward = 0, outward = 0, far = 0] of String(text).matchAll(WORDS)) {
    sum.same += +same;
    sum.inward += +inward;
    sum.outward += +outward;
    sum.far += +far;
  }
  return sum;
};

let failures = 0;
for (const how of hows) {
  for (const width of [128, 512]) {
    const label = `${how}, subgroups of ${width / 32}`;
    const run = spawnSync(process.execPath, ["tests/bench-dawn.mjs", webgpu, "public", "check", "16"], {
      env: { ...process.env, VK_ICD_FILENAMES: icd, LP_NATIVE_VECTOR_WIDTH: String(width), GPU_ROUNDING: how === "nearest" ? "" : how },
      encoding: "utf8", timeout: 900000, maxBuffer: 1 << 28 });
    const line = (run.stdout ?? "").split("\n").find((one) => one.startsWith("RESULT "));
    const why = [];
    let said = "";
    if (!line) why.push(`no result (${(run.stderr ?? "").trim().split("\n").slice(-2).join(" / ").slice(0, 300)})`);
    else {
      const [, seconds, json] = line.match(/^RESULT (\S+) (.*)$/s);
      const result = JSON.parse(json);
      if (result.error) why.push(`error: ${result.error}`);
      else {
        const layers = Object.entries(result).filter(([name]) => name.startsWith("a layer, ")), tokens = result["tokens on the GPU"];
        if (!layers.length) why.push("no layer check in the result");
        if (!tokens) why.push("no check of the tokens in the result");
        for (const [name, verdict] of Object.entries(result)) if (!verdict?.ok) why.push(`${name}: ${verdict?.error ?? "not ok"}${verdict?.stages ? ` (${verdict.stages})` : ""}`.slice(0, 500));
        const holds = counted(`${layers.map(([, verdict]) => verdict.stages).join(" ")} ${tokens?.steps ?? ""}`);
        const all = holds.same + holds.inward + holds.outward + holds.far, share = (n) => n / all;
        said = `; K and V of ${layers.length} layer rows and the tokens: ${holds.same} to the nearest, ${holds.inward} toward zero, ${holds.outward} away from it, ${holds.far} farther`;
        if (!all) why.push("the verdicts do not say how the keys and values were rounded");
        else if (holds.far) why.push(`${holds.far} keys and values farther than a neighbour (a float16 off)`);
        else if (how === "nearest") {
          if (holds.inward || holds.outward) why.push("rounded the other way on a device of the nearest");
        } else {
          const [wanted, other] = how === "away" ? [holds.outward, holds.inward] : [holds.inward, holds.outward];
          // about half of the values of a random row are not floats16 already and lie nearer the far neighbour than the near
          if (!(share(wanted) > 0.35 && share(wanted) < 0.65)) why.push(`${(100 * share(wanted)).toFixed(0)}% rounded ${how}, where about half were to (the rounding did not take?)`);
          if (other) why.push(`${other} keys and values rounded the other way as well`);
        }
        said = `${seconds} s; layers ${layers.filter(([, verdict]) => verdict.ok).length}/${layers.length} ok, tokens ${tokens?.ok ? "ok" : "not ok"}${said}`;
      }
    }
    console.log(`rounding ${label}: ${why.length ? "FAILED" : "ok"} ${said}`);
    for (const reason of why) console.log(`  ${reason}`);
    failures += why.length > 0;
  }
}
process.exit(failures ? 1 : 0);
