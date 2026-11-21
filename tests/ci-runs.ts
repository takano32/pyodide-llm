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
