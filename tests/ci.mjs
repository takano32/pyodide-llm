// ci.mjs (T142): start a workflow and wait for its run, or wait for a run, always by the run's ID and with a
// deadline, printing a line whenever the run or one of its jobs changes state. Never by searching the latest runs
// (a new run pushes the one waited for out of the list) and never by pgrep -f or pkill -f (they match the loop that
// runs them: a wait of the 2026-09-26 review round went on 52 minutes after what it waited for had ended).
//
//   node tests/ci.mjs run <workflow file> [input=value ...] [--ref main] [--minutes 60] [--grep <regex>]
//   node tests/ci.mjs wait <run id> [--minutes 60] [--grep <regex>]
//   node tests/ci.mjs deploy [--sha <commit>] [--minutes 20] [--grep <regex>]   the deploy of a commit (HEAD); if a newer
//     push cancelled it while it waited for its turn, the newer deploy (which has the commit too)
//
//   node tests/ci.mjs run models.yml models="hf-qwen3-0.6b" browser=webkit --grep "ready in|FAILED|timed out"
//
// Meant to run in the background (the agent's Bash with run_in_background): it always ends, by the deadline at the
// latest. --grep prints, once the run has ended, the lines of its jobs' logs that match (timestamps cut).
// Exit: 0 the run succeeded, 1 it ended otherwise (failure, cancelled...), 2 the deadline passed and the run goes
// on, 3 there was no run to wait for (the dispatch failed, no deploy of that commit appeared).
import { execFileSync } from "node:child_process";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clock = () => new Date().toISOString().slice(11, 19);
const say = (line) => console.log(`${clock()} ${line}`);

// gh, tried again on a failure of the network or of the API (a single failed call must not end a long wait)
async function gh(args, { input } = {}) {
  for (let attempt = 1; ; attempt++) {
    try {
      return execFileSync("gh", args, { encoding: "utf8", input, stdio: ["pipe", "pipe", "pipe"], maxBuffer: 256 * 2 ** 20 });
    } catch (error) {
      const why = String(error.stderr || error.message).trim().split("\n").at(-1);
      if (attempt >= 5 || /HTTP 4(0[0-4]|22)/.test(why)) throw new Error(`gh ${args.slice(0, 3).join(" ")}: ${why}`);
      say(`(gh failed, trying again: ${why})`);
      await sleep(5000 * attempt);
    }
  }
}
const api = async (path, ...rest) => JSON.parse(await gh(["api", path, ...rest]));

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  if (at < 0) return fallback;
  const [, value] = args.splice(at, 2);
  return value;
};
const minutes = Number(option("minutes", args[0] === "deploy" ? 20 : 60));
const grep = option("grep", null);
const ref = option("ref", "main");
const sha = option("sha", null);
const deadline = Date.now() + minutes * 60000;
const [command, ...rest] = args;

async function start(workflow, pairs) {
  const fields = ["-f", `ref=${ref}`];
  for (const pair of pairs) {
    const at = pair.indexOf("=");
    if (at < 1) throw new Error(`an input is key=value: ${pair}`);
    fields.push("-f", `inputs[${pair.slice(0, at)}]=${pair.slice(at + 1)}`);
  }
  // the dispatch answers with the ID of the run it made, in the API's version of 2026-03-10 (gh 2.46 asks for an
  // older one by default, which answers 204 and nothing else)
  const made = await api(`repos/{owner}/{repo}/actions/workflows/${workflow}/dispatches`, "-X", "POST",
    "-H", "X-GitHub-Api-Version: 2026-03-10", ...fields);
  if (!made.workflow_run_id) throw new Error(`the dispatch gave no run: ${JSON.stringify(made)}`);
  say(`started ${workflow}: run ${made.workflow_run_id} ${made.html_url}`);
  return made.workflow_run_id;
}

async function deployOf(commit) {
  const full = execFileSync("git", ["rev-parse", commit ?? "HEAD"], { encoding: "utf8" }).trim();
  for (;;) {
    const { workflow_runs: runs } = await api(`repos/{owner}/{repo}/actions/workflows/deploy.yml/runs?head_sha=${full}&per_page=5`);
    if (runs.length) {
      say(`the deploy of ${full.slice(0, 7)}: run ${runs[0].id} ${runs[0].html_url}`);
      return runs[0].id;
    }
    if (Date.now() > deadline) return null;
    await sleep(10000);
  }
}

const minutesSince = (from, to) => `${((Date.parse(to) - Date.parse(from)) / 60000).toFixed(1)} min`;

async function wait(id) {
  let last = "";
  const jobs = new Map();
  for (;;) {
    const run = await api(`repos/{owner}/{repo}/actions/runs/${id}`);
    const state = run.status === "completed" ? `completed ${run.conclusion}` : run.status;
    if (state !== last) say(`run ${id} (${run.name}): ${state}`), last = state;
    const { jobs: now } = await api(`repos/{owner}/{repo}/actions/runs/${id}/jobs?per_page=100`);
    for (const job of now) {
      const jobState = job.status === "completed" ? `completed ${job.conclusion} (${minutesSince(job.started_at, job.completed_at)})` : job.status;
      if (jobs.get(job.id)?.state !== jobState) say(`  job ${job.name}: ${jobState}`);
      jobs.set(job.id, { name: job.name, state: jobState });
    }
    if (run.status === "completed") return run;
    if (Date.now() > deadline) {
      say(`the deadline of ${minutes} min passed; run ${id} is still ${run.status}: ${run.html_url}`);
      return null;
    }
    await sleep(15000);
  }
}

async function logs(id, pattern) {
  const { jobs } = await api(`repos/{owner}/{repo}/actions/runs/${id}/jobs?per_page=100`);
  const matching = new RegExp(pattern);
  for (const job of jobs) {
    let text;
    try {
      text = await gh(["api", `repos/{owner}/{repo}/actions/jobs/${job.id}/logs`]);
    } catch (error) {
      console.log(`== ${job.name}: no log (${error.message})`);
      continue;
    }
    const lines = text.split("\n").map((line) => line.replace(/^\d{4}-\d\d-\d\dT[\d:.]+Z /, "")).filter((line) => matching.test(line));
    console.log(`== ${job.name}`);
    for (const line of lines) console.log(line);
  }
}

let id;
try {
  if (command === "run" && rest[0]) id = await start(rest[0], rest.slice(1));
  else if (command === "wait" && /^\d+$/.test(rest[0] ?? "")) id = Number(rest[0]);
  else if (command === "deploy") id = await deployOf(sha);
  else {
    console.error("usage: node tests/ci.mjs run <workflow> [input=value ...] | wait <run id> | deploy [--sha <commit>]  [--minutes N] [--grep <regex>]");
    process.exit(3);
  }
} catch (error) {
  say(`no run to wait for: ${error.message}`);
  process.exit(3);
}
if (!id) {
  say(`no deploy of ${sha ?? "HEAD"} appeared in ${minutes} min`);
  process.exit(3);
}
let run = await wait(id);
// a deploy waiting for its turn is cancelled by a newer push (deploy.yml's concurrency): the newer deploy has this
// commit too, so it is the one to wait for
while (command === "deploy" && run?.conclusion === "cancelled") {
  const { workflow_runs: runs } = await api(`repos/{owner}/{repo}/actions/workflows/deploy.yml/runs?branch=main&per_page=10`);
  const newer = runs.filter((r) => r.id > id).sort((a, b) => a.id - b.id).at(-1);
  if (!newer) break;
  say(`superseded by the deploy of ${newer.head_sha.slice(0, 7)}: run ${newer.id}`);
  id = newer.id;
  run = await wait(id);
}
if (run && grep) await logs(id, grep);
process.exit(!run ? 2 : run.conclusion === "success" ? 0 : 1);
