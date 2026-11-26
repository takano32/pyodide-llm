// tests/ci-check.ts (T679): which run of deploy.yml `node tests/ci.ts deploy` waits for (tests/ci-runs.ts), on made-up
// lists of runs shaped as the API gives them (newest first); T690: what a 404 of a run's page means, on made-up answers. Nothing leaves this machine (under a second).
//
//   node tests/ci-check.ts
import assert from "node:assert/strict";
import { againMayAppear, againOf, againWindow, deployOfCommit, newerOf, NOT_FOUND_EVERY, NOT_FOUND_GRACE, notFound } from "./ci-runs.ts";

let passed = 0;
const ok = (name) => { passed++; console.log(`ok: ${name}`); };

const A = "a".repeat(40);  // the commit whose push's deploy failed
const B = "b".repeat(40);  // a push that came in after it
const at = (minute) => new Date(Date.UTC(2026, 10, 20, 10, minute, 0)).toISOString();
const run = (id, event, sha, minute) => ({ id, event, head_sha: sha, created_at: at(minute) });

// the push's run 100 failed at minute 0; its job "again" ran from minute 10 to minute 11
const again = { started_at: at(10), completed_at: at(11) };

// ---- the deploy of a commit
{
  // after "again" ran on the same commit, the commit has two runs, the dispatch the newer: the push's is the one
  const runs = [run(130, "workflow_dispatch", A, 11), run(100, "push", A, 0)];
  assert.equal(deployOfCommit(runs).id, 100);
  ok("the push's run of a commit, not the newer dispatch of it");
  assert.equal(deployOfCommit([run(130, "workflow_dispatch", A, 11)]).id, 130);
  assert.equal(deployOfCommit([]), undefined);
  ok("a commit with a dispatch alone has that one, and with none, none");
}

// ---- the run "again" started
{
  // main did not move: the dispatch of the same commit
  assert.equal(againOf([run(130, "workflow_dispatch", A, 11)], again).id, 130);
  ok("the dispatch of the same commit");
  // a push came in between: the dispatch runs main's head, another commit, and the push of B is not it
  const moved = [run(130, "workflow_dispatch", B, 11), run(120, "push", B, 5)];
  assert.equal(againOf(moved, again).id, 130);
  assert.equal(againOf(moved, again).head_sha, B);
  ok("the dispatch of main's head when a push came in between, not that push");
  // ci.ts deploy --sha long after: later dispatches of others are past the minutes of the job "again"
  const later = [run(900, "workflow_dispatch", B, 59), run(130, "workflow_dispatch", A, 11), run(90, "workflow_dispatch", A, 8)];
  assert.equal(againOf(later, again).id, 130);
  ok("not a later dispatch of someone else");
  // a dispatch a minute before the job's clock says it started (two clocks) still counts; one of ten minutes before not
  assert.equal(againOf([run(110, "workflow_dispatch", A, 9)], again).id, 110);
  assert.equal(againOf([run(110, "workflow_dispatch", A, 0)], again), undefined);
  ok("a minute's slack either side, no more");
  // two dispatches in those minutes: the newest
  assert.equal(againOf([run(131, "workflow_dispatch", A, 11), run(130, "workflow_dispatch", A, 10)], again).id, 131);
  ok("the newest of two dispatches in those minutes");
  // the list may hold runs of the other events when the API's filter is not asked: not taken
  assert.equal(againOf([run(130, "push", B, 11)], again), undefined);
  ok("a push in those minutes is not the dispatch");
}

// ---- whether the dispatch may still appear
{
  const minute = 60000;
  assert.equal(againMayAppear(again, Date.parse(at(11)) + minute / 2), true);
  assert.equal(againMayAppear(again, Date.parse(at(11)) + 2 * minute), false);
  assert.equal(againMayAppear({ started_at: at(10) }, Date.parse(at(59))), true);
  ok("the list is final a minute after the job \"again\" ended");
  assert.equal(againWindow(again), "2026-11-20T10:09:00Z..2026-11-20T10:12:00Z");
  assert.equal(againWindow({ started_at: at(10) }), ">=2026-11-20T10:09:00Z");
  ok("the minutes asked of the API");
}

// ---- a newer deploy
{
  const runs = [run(130, "workflow_dispatch", B, 11), run(120, "push", B, 5), run(100, "push", A, 0), run(90, "push", A, 0)];
  assert.equal(newerOf(runs, 100).id, 130);
  assert.equal(newerOf([run(90, "push", A, 0)], 100), undefined);
  ok("the newest deploy after a run, and none when there is none");
}

// ---- T690: a 404 of a run's page
{
  // the wait of tests/ci.ts on made-up answers of gh: each look is 404 (null) or the run; a 404 looked at again after
  // NOT_FOUND_EVERY, as ci.ts does. What the wait ends with, and the looks it made
  const waited = (target, answers) => {
    let now = target.since ?? 0;
    for (let look = 1; look <= answers.length; look++) {
      if (answers[look - 1]) return { end: "found", look };
      if (notFound(target, now) === "missing") return { end: "missing", look };
      now += NOT_FOUND_EVERY;
    }
    return { end: "still 404", look: answers.length };
  };
  // the dispatch gave the ID, and its page was 404 twice before it was there (45801328142 on 2026-11-26)
  assert.deepEqual(waited({ id: 1, since: 0 }, [null, null, { status: "queued" }]), { end: "found", look: 3 });
  ok("a dispatched run that is 404 twice and then there is waited for");
  // ci.ts wait <ID> of an ID that is not a run: no run at the first 404, as before
  assert.deepEqual(waited({ id: 2 }, [null, null, { status: "queued" }]), { end: "missing", look: 1 });
  ok("an ID this call did not learn from GitHub is no run at its first 404");
  // a dispatched run whose page stays 404: no run once the grace is over, not before
  const looks = NOT_FOUND_GRACE / NOT_FOUND_EVERY;
  assert.deepEqual(waited({ id: 3, since: 0 }, Array(looks + 5).fill(null)), { end: "missing", look: looks + 1 });
  assert.equal(notFound({ since: 0 }, NOT_FOUND_GRACE - 1), "again");
  assert.equal(notFound({ since: 0 }, NOT_FOUND_GRACE), "missing");
  ok(`a dispatched run 404 for ${NOT_FOUND_GRACE / 1000} s is no run`);
  // a run seen once and then 404 is not looked at as one that may yet appear
  assert.equal(notFound({ since: 0, state: "queued" }, 1000), "missing");
  ok("a run seen once is not given the grace");
}

console.log(`ci-check: ${passed} passed`);
