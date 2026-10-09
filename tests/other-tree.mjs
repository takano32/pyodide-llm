// other-tree.mjs (T346, T354)
// Another commit's whole tree beside the working one, for the tools that compare the two: `git archive` of the commit
// under .tmp/unchanged/<its hash>, made once. A file of another commit taken alone (`git show <commit>:<file>`) stops
// working when that file becomes a window over others, so a comparison takes the tree.
//   const { commit, folder } = otherTree("origin/main")
// kernelSources() (T356): the kernels' sources of this tree or of a commit, whole, for the tools that compile two
// forms of the kernels side by side: kernel.ts is a window over kernel/*.ts since T356 and was one file before.
//   kernelSources("tree", dir)   kernelSources("origin/main", dir)
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const git = (...command) => execFileSync("git", command, { cwd: root, maxBuffer: 1 << 28 });

export function otherTree(before = "origin/main") {
  let commit;
  try { commit = git("rev-parse", "--verify", `${before}^{commit}`).toString().trim(); } catch {
    // (the commit is fetched alone, into FETCH_HEAD: no ref is written. --depth only in a shallow clone, as CI's: in a
    // whole one it would make the fetched commit a shallow boundary and cut the history behind it)
    const shallow = git("rev-parse", "--is-shallow-repository").toString().trim() === "true";
    git("fetch", ...(shallow ? ["--depth", "1"] : []), "origin", before.replace(/^origin\//, ""));
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

/** The kernels' sources (kernels/ and its folders, the .ts files) of this tree (form "tree") or of a commit, copied
 * into dir, which is emptied first: what was compiled there before is of another form. Returns the files, as their
 * paths under kernels/. A commit of before T356 gives kernel.ts as the one file it was, one after it the window and
 * kernel/: `asc <dir>/kernel.ts` compiles either. */
export function kernelSources(form, dir) {
  const from = path.join(form === "tree" ? root : otherTree(form).folder, "kernels");
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const files = fs.readdirSync(from, { recursive: true }).filter((file) => file.endsWith(".ts")).sort();
  for (const file of files) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.copyFileSync(path.join(from, file), path.join(dir, file));
  }
  if (!files.includes("kernel.ts")) throw new Error(`${form} has no kernels/kernel.ts`);
  return files;
}
