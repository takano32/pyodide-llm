// tests/ci-runs.ts (T679): which run of deploy.yml `node tests/ci.ts deploy` waits for, chosen from lists of runs as
// the API gives them (newest first). Pure: no gh, no clock, no network, so tests/ci-check.ts runs them on made-up lists.
// A run here is the API's: { id, event, head_sha, created_at }.

// the minute either side of the job "again" in which its dispatch is made (the API's created_at and the job's started_at
// and completed_at come from two clocks)
export const AGAIN_SLACK = 60000;

// the deploy of a commit: the push's run before a dispatch's. A run that the job "again" dispatched has the same commit
// when main did not move, and is newer: taking the newest would wait for it alone and never see the push's failure
export function deployOfCommit(runs) {
  return runs.find((run) => run.event === "push") ?? runs[0];
}

// the run the job "again" started: a dispatch of main made while that job ran (it runs main's head,
// which is another commit when a push came in between), the newest of them. Bounded on both sides, so a call made long
// after (ci.ts deploy --sha of an old commit) does not take a later dispatch of someone else's
export function againOf(runs, again) {
  const from = Date.parse(again.started_at) - AGAIN_SLACK;
  const to = again.completed_at ? Date.parse(again.completed_at) + AGAIN_SLACK : Infinity;
  return runs
    .filter((run) => run.event === "workflow_dispatch")
    .filter((run) => Date.parse(run.created_at) >= from && Date.parse(run.created_at) <= to)
    .sort((a, b) => b.id - a.id)[0];
}

// whether the list of dispatches can still change: until a minute after the job "again" ended its dispatch may not be
// in the API's list yet
export const againMayAppear = (again, now) => !again.completed_at || now <= Date.parse(again.completed_at) + AGAIN_SLACK;

// a newer deploy of main than run `after` (it has the commit too): the newest
export function newerOf(runs, after) {
  return runs.filter((run) => run.id > after).sort((a, b) => b.id - a.id)[0];
}

// the dates the API takes in `created=` for the minutes of the job "again"
export function againWindow(again) {
  const day = (ms) => new Date(ms).toISOString().replace(/\.\d+Z$/, "Z");
  const from = day(Date.parse(again.started_at) - AGAIN_SLACK);
  return again.completed_at ? `${from}..${day(Date.parse(again.completed_at) + AGAIN_SLACK)}` : `>=${from}`;
}

// T690: a run's own page (actions/runs/<ID>) can answer 404 for a moment after GitHub gave the run's ID: the two runs
// of preview.yml dispatched on 2026-11-26 (45801328142, 45801328531) had the ID in the dispatch's answer, were 404 in
// the same second, and succeeded. So a 404 means "no such run" at once only for an ID this call did not learn from
// GitHub (ci.ts wait <ID>: a mistyped ID); a run already seen once gets no grace (ci.ts says it could not look and
// waits on, as before); for a run this call dispatched or found in
// a list, after NOT_FOUND_GRACE since it learnt the ID. Until then the run is looked at again every NOT_FOUND_EVERY.
export const NOT_FOUND_GRACE = 120000;
export const NOT_FOUND_EVERY = 5000;

// what a 404 of run `target` means at `now`: "missing" (its wait ends, exit 3) or "again" (look again). A target is
// { since?: when (ms) this call learnt its ID from GitHub, state?: the state last seen ("" or none before the first) }
export function notFound(target, now) {
  if (target.state || target.since === undefined) return "missing";
  return now - target.since < NOT_FOUND_GRACE ? "again" : "missing";
}
