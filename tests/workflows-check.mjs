// .github/workflows: the rule of T262, held by a test (the review of T262: it was a sentence in AGENTS.md). A job that waits on a
// stuck download holds a runner until GitHub's limit of 360 minutes (run 36902097346 of gpu-prompt.yml stayed in the apt of
// `npx playwright-core install --with-deps` for 120 minutes and never began the tests), so
//   - every job has a timeout-minutes of its own;
//   - every step that installs a browser or runs apt has a timeout-minutes of its own, and where it runs apt (apt-get, or
//     playwright's --with-deps, which does) the time-outs of tests/apt-timeouts.sh are set first: in the step before the first
//     apt, or in an earlier step of the job.
// Node alone, no YAML library: the workflows are read by their indentation (jobs two spaces in, their keys four, a step's
// "- " six, its keys eight), which is how GitHub's own files are written. The checks are run on made-up files first, so that a
// check that sees nothing cannot pass.
//
//   node tests/workflows-check.mjs
import assert from "node:assert/strict";
import fs from "node:fs";

const APT = /\bapt(-get)?\s+(-\S+\s+)*(install|update|upgrade)\b|--with-deps|install-deps/;
const BROWSER = /playwright(-core)? install\b/;
const SETTINGS = "apt-timeouts.sh";

/** the problems of one workflow file's text, as lines of words */
export function problems(name, text) {
  const out = [];
  const lines = text.split("\n");
  const start = lines.findIndex((line) => /^jobs:\s*$/.test(line));
  if (start < 0) return out;  // a file with no jobs (there is none)
  // jobs: the keys at two spaces; a job runs to the next one
  const heads = [];
  for (let i = start + 1; i < lines.length; i++) if (/^  [A-Za-z0-9_-]+:\s*$/.test(lines[i])) heads.push(i);
  heads.forEach((head, k) => {
    const job = lines[head].trim().slice(0, -1);
    const body = lines.slice(head + 1, k + 1 < heads.length ? heads[k + 1] : lines.length);
    const keys = body.filter((line) => /^    [A-Za-z-]+:/.test(line));
    if (!keys.some((line) => /^    timeout-minutes:\s*\S/.test(line))) out.push(`${name}: job ${job} has no timeout-minutes (GitHub's 360)`);
    // steps: "      - " at six spaces; a step runs to the next one
    const at = body.map((line, i) => (/^      - /.test(line) ? i : -1)).filter((i) => i >= 0);
    let configured = false;
    at.forEach((first, s) => {
      const step = body.slice(first, s + 1 < at.length ? at[s + 1] : body.length);
      const code = step.filter((line) => !/^\s*#/.test(line)).join("\n");  // (a comment may say apt)
      const label = (step[0].match(/name:\s*(.*)$/) ?? [])[1] ?? step[0].trim();
      const scrubbed = code.replaceAll(SETTINGS, "SETTINGS");  // (its own name has "apt" in it)
      const apt = APT.test(scrubbed), browser = BROWSER.test(scrubbed);
      const before = configured;
      if (code.includes(SETTINGS)) configured = true;
      if (!apt && !browser) return;
      if (!step.some((line) => /^        timeout-minutes:\s*\S/.test(line) || /^      - timeout-minutes:\s*\S/.test(line))) {
        out.push(`${name}: job ${job}, step "${label}" installs without a timeout-minutes of its own`);
      }
      if (apt) {
        const set = code.includes(SETTINGS) && code.indexOf(SETTINGS) < code.search(APT) || before;
        if (!set) out.push(`${name}: job ${job}, step "${label}" runs apt before ${SETTINGS} has set its time-outs`);
      }
    });
  });
  return out;
}

// ---- the check on made-up files: what it must say, and what it must not
const GOOD = `name: x
jobs:
  one:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v7
      # a comment that says apt-get install and --with-deps changes nothing
      - name: Install
        timeout-minutes: 20
        run: |
          bash tests/apt-timeouts.sh
          npm ci
          npx playwright-core install --with-deps chromium
      - name: Again, set up by the step before
        timeout-minutes: 20
        run: npx playwright-core install --with-deps firefox
  two:
    timeout-minutes: \${{ fromJSON(inputs.minutes || '90') }}
    steps:
      - name: Tests
        run: npm test
`;
assert.deepEqual(problems("good.yml", GOOD), []);
const without = (pattern) => GOOD.replace(pattern, "");
const has = (text, part) => problems("bad.yml", text).some((line) => line.includes(part));
assert.ok(has(without("    timeout-minutes: 30\n"), "job one has no timeout-minutes"), "a job without a limit");
assert.ok(has(GOOD.replace("    timeout-minutes: ${{ fromJSON(inputs.minutes || '90') }}\n", ""), "job two has no timeout-minutes"), "the other job without one");
assert.ok(has(GOOD.replace("        timeout-minutes: 20\n        run: |", "        run: |"), 'step "Install" installs without a timeout-minutes'), "an install without a step limit");
assert.ok(has(without("          bash tests/apt-timeouts.sh\n"), 'step "Install" runs apt before'), "apt without the time-outs set");
assert.ok(has(GOOD.replace("          bash tests/apt-timeouts.sh\n          npm ci\n          npx playwright-core install --with-deps chromium",
  "          npx playwright-core install --with-deps chromium\n          bash tests/apt-timeouts.sh"), 'step "Install" runs apt before'), "the time-outs set after the apt");
assert.ok(has(GOOD.replace("Again, set up by the step before\n        timeout-minutes: 20", "Again, set up by the step before"), 'step "Again, set up by the step before" installs without'),
  "the second install without a step limit");
// a browser installed without apt (Windows, macOS: no --with-deps) still needs the step's own limit
const BROWSER_ONLY = GOOD.replace("npx playwright-core install --with-deps firefox", "npx playwright-core install firefox").replace("Again, set up by the step before\n        timeout-minutes: 20", "Again, set up by the step before");
assert.ok(has(BROWSER_ONLY, 'step "Again, set up by the step before" installs without'), "a browser alone needs the limit too");

// ---- the real files
const directory = new URL("../.github/workflows/", import.meta.url);
const found = fs.readdirSync(directory).filter((file) => file.endsWith(".yml")).flatMap((file) => problems(file, fs.readFileSync(new URL(file, directory), "utf8")));
assert.deepEqual(found, [], `the workflows break T262's rule:\n${found.join("\n")}`);
console.log(`ok (${fs.readdirSync(directory).filter((file) => file.endsWith(".yml")).length} workflows)`);
