// other-tree.mjs (T346, T354)
// Another commit's whole tree beside the working one, for the tools that compare the two: `git archive` of the commit
// under .tmp/unchanged/<its hash>, made once. A file of another commit taken alone (`git show <commit>:<file>`) stops
// working when that file becomes a window over others, so a comparison takes the tree.
//   const { commit, folder } = otherTree("origin/main")
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const git = (...command) => execFileSync("git", command, { cwd: root, maxBuffer: 1 << 28 });

export function otherTree(before = "origin/main") {
  let commit;
  try { commit = git("rev-parse", "--verify", `${before}^{commit}`).toString().trim(); } catch {
    // (a shallow clone, as CI's: the commit is fetched alone)
    git("fetch", "--depth", "1", "origin", before.replace(/^origin\//, ""));
    commit = git("rev-parse", "--verify", "FETCH_HEAD^{commit}").toString().trim();
  }
  const folder = path.join(root, ".tmp", "unchanged", commit);
  if (!fs.existsSync(path.join(folder, "done"))) {
    fs.rmSync(folder, { recursive: true, force: true });
    fs.mkdirSync(folder, { recursive: true });
    execFileSync("tar", ["-x", "-C", folder], { input: git("archive", commit) });
    fs.writeFileSync(path.join(folder, "done"), "");
  }
  // what `make models` built here is in no commit (the models' files at the root and in public/models): the other tree
  // reads this one's. Without them its tests that need a model of the site are skipped, and compared with nothing.
  // (Not the built kernels: no check here runs them, and they are the working tree's.)
  const built = git("ls-files", "--others", "--ignored", "--exclude-standard", "--directory").toString().trim().split("\n")
    .map((name) => name.replace(/\/$/, "")).filter((name) => name === "public/models" || (!name.includes("/") && !name.startsWith(".") &&
      !["node_modules", "dist", "__pycache__"].includes(name)));
  for (const name of built) if (!fs.existsSync(path.join(folder, name))) fs.symlinkSync(path.join(root, name), path.join(folder, name));
  return { commit, folder };
}
