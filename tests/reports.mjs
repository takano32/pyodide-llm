// The visitors' benchmark reports (T91) as one table, for T83's 30-measurements.md: every issue labelled benchmark,
// read the way src/bench.js reads one. Needs the gh CLI.
//
//   node tests/reports.mjs
import { execFileSync } from "node:child_process";
import { REPOSITORY, reportsTable } from "../src/bench.js";

const issues = JSON.parse(execFileSync("gh", ["issue", "list", "--repo", REPOSITORY, "--label", "benchmark", "--state", "all",
  "--limit", "500", "--json", "number,url,body"], { encoding: "utf8" }));
console.log(reportsTable(issues.sort((a, b) => a.number - b.number)));
