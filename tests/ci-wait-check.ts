// tests/ci-wait-check.ts (T690's review): the wait of tests/ci.ts itself, not a model of it (tests/ci-check.ts has the
// line of notFound() in tests/ci-runs.ts): ci.ts as a process, with a `gh` of this folder that answers what GitHub did on
// 2026-11-26 (the page of a run dispatched a moment ago is 404, and is there later). What it checks is the wiring: the ID
// of a dispatch has its time (`since`), a 404 is looked at again every NOT_FOUND_EVERY (5 s: the early sleep, not the
// 15 s of the ordinary wait), an ID given to `wait` is no run at its first 404. Nothing leaves this machine (about 6 s).
//
//   node tests/ci-wait-check.ts
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { leave, scratch } from "./leave.ts";
import { NOT_FOUND_EVERY } from "./ci-runs.ts";

const folder = scratch("ci-wait-");
const ID = 999000001; // no run of the repository (and a lock of its own in .tmp/ci/)
// gh: the dispatch gives the ID; the run's page is 404 for the first $FAKE_404 looks (counted in a file), then completed
fs.writeFileSync(path.join(folder, "gh"), `#!/bin/bash
echo "$*" >> "${folder}/calls"
case "$*" in
  *dispatches*) echo '{"workflow_run_id":${ID},"html_url":"u"}';;
  *actions/runs/${ID}/jobs*) echo '{"jobs":[]}';;
  *actions/runs/${ID}*)
    n=$(cat "${folder}/looks" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "${folder}/looks"
    if [ $n -le "$FAKE_404" ]; then echo "gh: Not Found (HTTP 404)" >&2; exit 1; fi
    echo '{"status":"completed","conclusion":"success","display_title":"t","html_url":"u","path":".github/workflows/x.yml","run_started_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:01:00Z"}';;
  *) echo "unexpected: $*" >&2; exit 1;;
esac
`, { mode: 0o755 });

function ci(fake404, ...words) {
  fs.rmSync(path.join(folder, "looks"), { force: true });
  const began = Date.now();
  const done = spawnSync(process.execPath, [new URL("./ci.ts", import.meta.url).pathname, ...words, "--minutes", "3"], {
    encoding: "utf8", env: { ...process.env, PATH: `${folder}:${process.env.PATH}`, FAKE_404: String(fake404) }, timeout: 60000 });
  const looks = Number(fs.readFileSync(path.join(folder, "looks"), "utf8").trim());
  return { code: done.status, text: done.stdout + done.stderr, looks, seconds: (Date.now() - began) / 1000 };
}

// a dispatched run that is 404 once and then there: looked at again after NOT_FOUND_EVERY, and the wait ends 0
{
  const made = ci(1, "run", "x.yml");
  assert.equal(made.code, 0, made.text);
  assert.equal(made.looks, 2, made.text);
  assert.match(made.text, /404 1 time, looking again/);
  assert.ok(made.seconds < NOT_FOUND_EVERY / 1000 + 8, `the wait was ${made.seconds} s: the 15 s of the ordinary wait where ${NOT_FOUND_EVERY / 1000} s were meant\n${made.text}`);
  console.log(`ok: a dispatched run 404 once is looked at again after ${NOT_FOUND_EVERY / 1000} s and the wait ends 0 (${made.seconds.toFixed(1)} s)`);
}
// ci.ts wait <ID>: an ID that was not learnt from GitHub by this call is no run at its first 404 (exit 3, one look)
{
  const given = ci(1, "wait", String(ID));
  assert.equal(given.code, 3, given.text);
  assert.equal(given.looks, 1, given.text);
  assert.match(given.text, /no run 999000001/);
  console.log("ok: an ID given to wait is no run at its first 404");
}
await leave(0);
