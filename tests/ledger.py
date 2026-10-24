"""The ledger: TASKS.md holds the tasks still moving, docs/tasks/ the ones done.

TASKS.md is read by people and agents alike, and at 4.5 MB nobody could read it
any more (2026-10-15, the owner: split it, keep a table of contents). So a task
that is done leaves TASKS.md for a file of its own, docs/tasks/T<number>.md (a
subtask stays in its parent's file: T329.2 goes to T329.md), and two tables of
contents are made from the headings, never written by hand:

  - the head of TASKS.md: every task still in it, in the order of the file,
    with its state;
  - docs/tasks/README.md: every task in docs/tasks/, by number.

    python tests/ledger.py split      move the tasks that are done (once more
                                      whenever tasks are done) and remake both
    python tests/ledger.py index      remake both tables of contents
    python tests/ledger.py --check    fail if either table is not what index
                                      would write (the light suite runs this)

A task's heading is a line "### T<number> …" with "状態: …" in it. It is done
when its state begins with 完了, or for the few older ones that said it
otherwise, when its number is in OLDER_DONE. Nothing is rewritten: a block of
lines moves as it is, and split checks that every line of TASKS.md before is in
exactly one place after.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "TASKS.md"
TASKS = ROOT / "docs" / "tasks"
INDEX = TASKS / "README.md"

HEADING = re.compile(r"^### (T\d+(?:\.\d+)*)\b")
STATE = re.compile(r"状態: (.*)")
DATE = re.compile(r"20\d\d-\d\d-\d\d")

# Done tasks whose state was written before "完了" was the one word for it
# (read one by one on 2026-10-15). T397, T136 and T137 are only partly done
# and stay.
OLDER_DONE = {
    "T95", "T134", "T173", "T174", "T176", "T179", "T180", "T181", "T182",
    "T188", "T192", "T210", "T217", "T272",
}

TOC_BEGIN = "<!-- 目次: tests/ledger.py index が見出しから作る。手で書かない -->"
TOC_END = "<!-- 目次ここまで -->"


def blocks(lines):
    """The file cut at every heading of level 2 or 3: (heading or None, lines)."""
    out, start = [], 0
    for i, line in enumerate(lines):
        if i and (line.startswith("## ") or line.startswith("### ")):
            out.append(lines[start:i])
            start = i
    out.append(lines[start:])
    return out


def state_of(heading):
    m = STATE.search(heading)
    return m.group(1) if m else ""


def done(heading):
    m = HEADING.match(heading)
    if not m:
        return False
    s = state_of(heading).lstrip("*")
    return s.startswith("完了") or m.group(1) in OLDER_DONE


def parent(number):
    return number.split(".")[0]


def title(heading):
    rest = HEADING.sub("", heading).strip()
    rest = re.split(r"\s*[—–-]+\s*状態[:：]", rest)[0]
    return rest.strip()


def short(state):
    s = state.replace("**", "")
    s = re.split(r"[（(、。:]", s)[0]
    return s[:24].strip()


def without_toc(lines):
    if TOC_BEGIN not in lines:
        return lines
    a = lines.index(TOC_BEGIN)
    b = lines.index(TOC_END)
    # the blank line the toc is set off with
    end = b + 1
    if end < len(lines) and lines[end] == "":
        end += 1
    return lines[:a] + lines[end:]


def todo_toc(lines):
    rows = []
    for line in lines:
        m = HEADING.match(line)
        if m:
            rows.append(f"- {m.group(1)} {title(line)} — {short(state_of(line))}")
    return [TOC_BEGIN, "## 目次（動いているタスク）", "",
            "済んだタスクは [docs/tasks/](docs/tasks/README.md) に 1 つずつのファイルで。", "",
            *rows, TOC_END, ""]


def with_toc(lines):
    lines = without_toc(lines)
    # after the opening paragraphs, before the first "## " section
    at = next(i for i, line in enumerate(lines) if line.startswith("## "))
    return lines[:at] + todo_toc(lines) + lines[at:]


def index_text():
    rows = []
    def key(p):
        return int(p.stem[1:])
    for p in sorted(TASKS.glob("T*.md"), key=key):
        heads = [l for l in p.read_text().split("\n") if HEADING.match(l)]
        first = heads[0] if heads else ""
        dates = DATE.findall(state_of(first))
        when = dates[0] if dates else ""
        rows.append(f"- [{p.stem}]({p.name}) {title(first)}" + (f" — {when}" if when else ""))
    return "\n".join([
        "# 済んだタスク",
        "",
        "TASKS.md から移した、済んだタスクの本文（1 つに 1 ファイル、子タスクは親のファイルに）。"
        "この目次は `python tests/ledger.py index` が見出しから作る（手で書かない）。"
        "記録やコメントが「TASKS.md の T○○」と指していたら、済んだものはここの `T○○.md` を読む。",
        "",
        "もっと古い済んだタスク（T1〜、1 行ずつの一覧）は [done-list.md](done-list.md)。",
        "",
        *rows,
        "",
    ])


def split():
    before = LEDGER.read_text().split("\n")
    lines = without_toc(before)
    kept, moved = [], {}
    section = ""
    for block in blocks(lines):
        head = block[0]
        if head.startswith("## "):
            section = head
            if head.startswith("## 完了したタスク"):
                # the old one-line list of done tasks, as it is
                moved.setdefault("done-list", []).extend(block)
                continue
        if done(head):
            moved.setdefault(parent(HEADING.match(head).group(1)), []).extend(block)
        else:
            kept.extend(block)
    TASKS.mkdir(parents=True, exist_ok=True)
    for name, body in moved.items():
        path = TASKS / f"{name}.md"
        old = path.read_text().rstrip("\n").split("\n") + [""] if path.exists() else []
        path.write_text("\n".join(old + body).rstrip("\n") + "\n")
    LEDGER.write_text("\n".join(with_toc(kept)))
    INDEX.write_text(index_text())
    # every line of TASKS.md before (without its toc) is in exactly one place now
    after = without_toc(LEDGER.read_text().split("\n"))
    spread = list(after)
    for name in moved:
        spread += (TASKS / f"{name}.md").read_text().split("\n")
    from collections import Counter
    want, have = Counter(l for l in lines if l.strip()), Counter(l for l in spread if l.strip())
    missing = want - have
    if missing:
        sys.exit(f"ledger: {sum(missing.values())} lines went missing, e.g. {next(iter(missing))[:80]!r}")
    print(f"ledger: moved {len(moved)} files' worth of done tasks; TASKS.md is "
          f"{len(LEDGER.read_text().encode())} bytes (was {len(chr(10).join(before).encode())})")


def index():
    LEDGER.write_text("\n".join(with_toc(LEDGER.read_text().split("\n"))))
    INDEX.write_text(index_text())


def check():
    lines = LEDGER.read_text().split("\n")
    wrong = []
    if lines != with_toc(lines):
        wrong.append("TASKS.md's table of contents")
    if INDEX.exists() and INDEX.read_text() != index_text():
        wrong.append("docs/tasks/README.md")
    still = [HEADING.match(l).group(1) for l in lines if done(l)]
    if still:
        wrong.append(f"done tasks still in TASKS.md ({', '.join(still[:5])})")
    if wrong:
        sys.exit("ledger: not as `python tests/ledger.py split` would leave it: " + "; ".join(wrong))
    print("ledger: ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"split": split, "index": index, "--check": check}[arg]()
