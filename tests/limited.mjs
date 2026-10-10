// limited.mjs (T384)
// Runs a Node script with a limit, so that no test of the suite waits forever: one run of the full suite stood still for
// 88 minutes after a test's last line, with nothing in the log to say where (run 38024999470).
//   node tests/limited.mjs <seconds> <script> [arguments ...]
// At the limit the script's Node is asked for its report (SIGUSR2: the JavaScript stack and the handles that keep it
// alive; a script in a loop that never returns to the event loop writes none, which is said), that is printed, the script is killed, and this exits 124.
// Otherwise it exits as the script does. (Not coreutils' timeout: macOS has none, and the one of the development
// machine's Ubuntu, uutils', kills where it is told to send another signal.)
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [seconds, ...script] = process.argv.slice(2);
const reports = fs.mkdtempSync(path.join(fs.existsSync(".tmp") ? ".tmp" : os.tmpdir(), "limited-"));
const child = spawn(process.execPath, ["--report-on-signal", "--report-signal=SIGUSR2", `--report-directory=${reports}`,
  ...process.execArgv, ...script], { stdio: "inherit" });
let stopped = false;
const limit = setTimeout(() => {
  stopped = true;
  console.error(`limited: STOPPED after ${seconds} s: node ${script.join(" ")}`);
  if (process.platform !== "win32") child.kill("SIGUSR2");
  setTimeout(() => {
    for (const name of fs.readdirSync(reports)) {
      try {
        const report = JSON.parse(fs.readFileSync(path.join(reports, name), "utf8"));
        const stack = [report.javascriptStack?.message, ...(report.javascriptStack?.stack ?? [])].filter(Boolean);
        console.error(stack.length ? `limited: where it stood:\n${stack.map((line) => `limited:   ${line}`).join("\n")}` : "limited: it was waiting (nothing on the JavaScript stack)");
        const alive = (report.libuv ?? []).filter((handle) => handle.is_active && handle.is_referenced).map((handle) => handle.type);
        console.error(`limited: what kept it alive: ${alive.join(", ") || "nothing"}`);
      } catch (error) {
        console.error(`limited: the report ${name} could not be read (${error.message})`);
      }
    }
    if (!fs.readdirSync(reports).length) console.error("limited: Node wrote no report: it was in JavaScript that did not return to the event loop (or in native code)");
    child.kill("SIGKILL");
  }, 5000);
}, Number(seconds) * 1000);
child.on("exit", (code, signal) => {
  clearTimeout(limit);
  fs.rmSync(reports, { recursive: true, force: true });
  process.exit(stopped ? 124 : code ?? (signal ? 1 : 0));
});
