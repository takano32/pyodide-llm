# tests/wllama_compare.py (T519): this page against wllama (llama.cpp compiled to WebAssembly, MIT, read from its npm
# package as it is) on the same GGUF files, the CPU alone, in CI's Chromium. The owner (2026-10-15): compare, but show
# it on none of the product's pages: the numbers go to docs/notes/t519-wllama-*.md and the ledger.
#
#   python tests/wllama_compare.py files                      # the sha256 of each file, the list's and wllama's own
#   python tests/wllama_compare.py run [model id …] [--rounds 5] [--tokens 128] [--threads 0] [--query without=relaxed] [--results file.jsonl]
#   python tests/wllama_compare.py table <results.jsonl …>    # the tables, one per system and CPU
#
# run needs a build (npm run build) and Chromium for playwright-core; it fetches the npm package of the version pinned
# below (held to its sha1, the registry's shasum) into .tmp/, and runs tests/wllama-in-browser.ts once per model.
# --threads 0: half the logical cores, wllama's own default (src/wllama.ts: floor(hardwareConcurrency / 2)), and the
# page gets the same with ?threads=. The prompt and the answer are each side's own count and speed: the page's line
# under the answer, and llama-server's timings (tests/wllama-in-browser.ts says how each is counted).
#
# The GPU is not compared here: a runner has no GPU, and SwiftShader's speed means nothing (AGENTS.md). wllama puts every
# layer on WebGPU by default; on the owner's devices, by hand, if ever.
import argparse
import hashlib
import http.client
import json
import os
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

from fetching import download

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
WORK = REPOSITORY / ".tmp" / "wllama"

# the npm package: the latest when T519 measured (2026-11-20), its llama.cpp b12211-7d7f028 (esm/index.js's LIBLLAMA_VERSION)
VERSION = "3.13.0"
TARBALL = f"https://registry.npmjs.org/@wllama/wllama/-/wllama-{VERSION}.tgz"
SHASUM = "2b8d7ce0b8b91b0d33a7c5bef1e847825bd65737"

STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."
# (a paragraph of about 150 tokens, so that the translation is about as long as the answers of the other two)
JAPANESE = ("富士山は日本でいちばん高い山で、標高は三千七百七十六メートルあります。静岡県と山梨県にまたがり、"
            "昔から多くの人に親しまれてきました。夏になると、山頂を目指して世界中から登山者が集まります。"
            "夜のうちに登り始めて、山頂でご来光を見るのが人気です。冬の富士山は雪に覆われ、"
            "遠くの町からもその美しい姿を見ることができます。二〇一三年には世界文化遺産に登録されました。")

# One row per model: the id of the list, the file both read (the list's own revision), its sha256, and how wllama is
# asked: chat (the GGUF's chat template, rendered by llama.cpp's jinja) or raw (the text the page's format writes: the
# page sends that and its BOS).
MODELS = {
    "hf-smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "revision": "593b5a2e04c8f3e4ee880263f93e0bd2901ad47f",
        "file": "smollm2-360m-instruct-q8_0.gguf", "sha256": "48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201",
        # wllama's demo (examples/main/src/config.ts at the package's commit 6647c1ea) names ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF,
        # which huggingface.co now sends to this repository's main: the same file
        "theirs": "https://huggingface.co/ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
        "prompt": STORY, "chat": True},
    "hf-qwen3-0.6b": {
        "repo": "unsloth/Qwen3-0.6B-GGUF", "revision": "50968a4468ef4233ed78cd7c3de230dd1d61a56b",
        "file": "Qwen3-0.6B-Q8_0.gguf", "sha256": "e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c",
        "theirs": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
        # (src/models/formats.ts's QWEN3_AT_ONCE after the list's BOS, <|endoftext|>: QWEN3_OWN_BOS)
        "prompt": STORY, "raw": "<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"},
    "hf-plamo-3-translate-2b": {
        "repo": "pfnet/plamo-3-translate-2b-GGUF", "revision": "232fbd7916519edecddfdae0beae7e76fe70453f",
        "file": "plamo-3-translate-2b-Q4_K_M.gguf", "sha256": "f36d0813dd95573a852ab5d76e374c543632d9e292929e922bcdbf7e53b494e0",
        # tanakh/plamo-translate-desktop's extension (extension/engine.js at 3d69a116) pins a later upload of PFN's, of
        # another sha256 (3,648 bytes more): this compares the list's file, read by both
        "theirs": "https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF/resolve/228a27cc0bb4f111219963848d50a3127c16ca73/plamo-3-translate-2b-Q4_K_M.gguf",
        # (the request of Japanese into English, as T649.3's entry wrote it, the source said: the page's own of a
        # Japanese text says none since T669, engine/pfml.py's; the BOS (1) is llama.cpp's to add, as the GGUF says)
        "prompt": JAPANESE,
        "raw": "<|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>{prompt}"
               "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"},
}


def url_of(row):
    return f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['file']}"


def linked(url):
    """The size and sha256 huggingface.co says of a file (its first answer: x-linked-size and x-linked-etag), following
    the redirects of a renamed repository within huggingface.co."""
    for _ in range(5):
        address = url.removeprefix("https://huggingface.co")
        connection = http.client.HTTPSConnection("huggingface.co", timeout=60)
        connection.request("HEAD", address)
        answer = connection.getresponse()
        size, etag, where = answer.getheader("x-linked-size"), answer.getheader("x-linked-etag"), answer.getheader("location")
        connection.close()
        if etag:
            return int(size), etag.strip('"')
        if not where or not where.startswith("/"):
            raise OSError(f"{url}: no sha256 said ({answer.status})")
        url = f"https://huggingface.co{where}"
    raise OSError(f"{url}: too many redirects")


def files():
    """Each file's sha256 held to the table's, and the one wllama's own example uses beside it."""
    same = True
    for model, row in MODELS.items():
        size, sha = linked(url_of(row))
        theirs = linked(row["theirs"])
        ok = sha == row["sha256"]
        same &= ok
        print(f"{model}: {row['repo']}@{row['revision'][:8]} {size:,} B sha256 {sha} ({'as the table says' if ok else 'NOT the table'})"
              f"; wllama's example's: {theirs[0]:,} B {theirs[1]} ({'the same file' if theirs[1] == sha else 'another file'})")
    return same


def package():
    """The npm package of VERSION, unpacked under .tmp/wllama/ (its folder), held to the registry's sha1."""
    tarball = download(TARBALL, WORK / f"wllama-{VERSION}.tgz")
    sha1 = hashlib.sha1(tarball.read_bytes()).hexdigest()
    if sha1 != SHASUM:
        tarball.unlink()
        raise SystemExit(f"{TARBALL}: sha1 {sha1}, not {SHASUM}")
    folder = WORK / VERSION
    if not (folder / "package" / "esm" / "index.js").exists():
        with tarfile.open(tarball) as archive:
            archive.extractall(folder, filter="data")
    return folder / "package"


def run(models, rounds, tokens, threads, query, results, timeout):
    folder = package()
    failed = []
    for model in models:
        row = MODELS[model]
        spec = {"model": model, "url": url_of(row), "wllama": str(folder), "version": VERSION, "prompt": row["prompt"],
                "chat": row.get("chat", False), "raw": row.get("raw"), "kwargs": row.get("kwargs"), "threads": threads, "query": query,
                "tokens": tokens, "rounds": rounds, "results": str(results), "context": 4096, "timeout": timeout}
        path = WORK / f"spec-{model}.json"
        path.write_text(json.dumps(spec, ensure_ascii=False))
        print(f"wllama_compare: {model}", flush=True)
        code = subprocess.call(["node", str(HERE / "wllama-in-browser.ts"), str(path)], cwd=REPOSITORY)
        if code:
            failed.append(f"{model} ({code})")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    return not failed


def read(paths):
    lines = []
    for path in paths:
        lines += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return lines


def same_start(a, b):
    """How many characters the two answers begin with alike, as words: the page's is the text of its bubble, which drew
    the Markdown (no "**", one line where the text had a blank one), wllama's is the text as written, so the marks of
    Markdown are taken out of both and every run of spaces is one space."""
    a, b = (" ".join(text.replace("*", "").replace("#", "").split()) for text in (a, b))
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return f"{n} of {min(len(a), len(b))}"


def fixed(value, digits=1):
    return "" if value is None else f"{value:.{digits}f}"


def table(lines):
    """Markdown: one table per system and CPU (and threads), a row per model and side."""
    out = []
    groups = {}
    for line in lines:
        groups.setdefault((line["os"], line["cpu"], line["threads"], line.get("query", ""), line.get("browserVersion", "")), []).append(line)
    for (system, cpu, threads, query, browser), group in groups.items():
        out.append(f"{system}, {cpu}, {group[0]['cores']} logical cores, {threads} threads each{f', the page with ?{query}' if query else ''}; Chromium {browser}, wllama {group[0]['wllama']}\n")
        out.append("| model | side | ready with the fetch (s) | ready again (s) | prompt tokens | prompt tok/s | answer tokens | answer tok/s | page ÷ wllama, prompt | page ÷ wllama, answer | answers begin alike (characters, of the shorter) |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model in dict.fromkeys(line["model"] for line in group):
            of = [line for line in group if line["model"] == model]
            sides = {}
            for side in ("page", "wllama"):
                loads = {line["fetched"]: line["seconds"] for line in of if line["side"] == side and line["what"] == "load"}
                answers = [line for line in of if line["side"] == side and line["what"] == "answer"]
                sides[side] = {
                    "fetched": loads.get(True), "again": loads.get(False),
                    "prompt_n": statistics.median([a["promptTokens"] for a in answers]) if answers else None,
                    "prompt": statistics.median([a["promptPerSecond"] for a in answers]) if answers else None,
                    "answer_n": statistics.median([a["answerTokens"] for a in answers]) if answers else None,
                    "answer": statistics.median([a["answerPerSecond"] for a in answers]) if answers else None,
                    "range": (min(a["answerPerSecond"] for a in answers), max(a["answerPerSecond"] for a in answers)) if answers else None,
                    "text": answers[0]["text"] if answers else ""}
            page, theirs = sides["page"], sides["wllama"]
            ratio = lambda a, b: fixed(a / b, 2) if a and b else ""
            alike = same_start(page["text"], theirs["text"])
            for side, s in sides.items():
                mine = side == "page"
                out.append(f"| {model} | {side} | {fixed(s['fetched'])} | {fixed(s['again'])} | {fixed(s['prompt_n'], 0)} | {fixed(s['prompt'])} | "
                           f"{fixed(s['answer_n'], 0)} | {fixed(s['answer'])} ({fixed(s['range'][0]) if s['range'] else ''}–{fixed(s['range'][1]) if s['range'] else ''}) | "
                           f"{ratio(page['prompt'], theirs['prompt']) if mine else ''} | {ratio(page['answer'], theirs['answer']) if mine else ''} | {alike if mine else ''} |")
        out.append("")
        out.append("The speeds are the medians of the rounds (the range of the answer's beside it). The page's prompt count leaves out the prompt's last token, llama.cpp's counts it.")
        out.append("")
    fails = [line for line in lines if line["side"] == "tool"]
    for line in fails:
        out.append(f"- {line['model']}: {'; '.join(line['failures'])}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["files", "run", "table"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--query", default="", help="more of the page's address, such as without=relaxed")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--results", default=str(WORK / "results.jsonl"))
    asked = parser.parse_args()
    if asked.what == "files":
        sys.exit(0 if files() else 1)
    if asked.what == "table":
        sys.stdout.write(table(read(asked.names)))
        return
    unknown = [name for name in asked.names if name not in MODELS]
    if unknown:
        raise SystemExit(f"no row for {', '.join(unknown)}: {', '.join(MODELS)}")
    ok = run(asked.names or list(MODELS), asked.rounds, asked.tokens, asked.threads, asked.query, Path(asked.results), asked.timeout)
    if Path(asked.results).exists():
        sys.stdout.write(table(read([asked.results])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
