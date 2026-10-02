#!/usr/bin/env bash
# tests/page_27b.sh (T233): Ternary Bonsai 2 27B on the page's forward pass, held to its references, on a CI runner
# (never on the development machine: it fetches 5.95 GB and makes a checkpoint of 7.66 GB). On demand, not nightly:
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=180 extra="bash tests/page_27b.sh" \
#     --grep "^(fork|f32|reference|runner|page|pyodide):" --minutes 190 --ref <branch>
#   ... extra="STAGES='fork convert speed memory' bash tests/page_27b.sh" ...
# On x86-64 or arm64 (runner=ubuntu-24.04-arm: twice as fast for the fork and the page). Liftoff, the first compiler of the
# V8 of Node 24 (13.6) on arm64, reads v128.load32_splat above 4 GiB at the address's low 32 bits (the review of T230 and
# T231; fixed in V8 14.3, Chrome 143): where tests/page-27b.mjs's canary says so it sets --no-liftoff, and what it then
# computes is the model's (the numbers equal x86-64's to four digits: the review of T233); where the canary says so even
# then, it refuses, and ANYWAY=1 runs page and speed all the same: the comparison then says whether that engine computed
# the model. LONG_TOKENS=600 (with LONG_ROWS below 600) makes a short pass through the whole of a stage, a check of the
# tool and not of the model.
#
# Run it when public/forward.js, the ternary kernels (kernels/ternary.ts), rotate and unrotate, the converter's reading
# of a GGUF or the list's entry change in a way that could move what this model computes.
#
# STAGES (the default: "reference convert page breaks", an hour and a half on 4 logical cores):
#   dry        a made-up model of the 27B's kind through compare, speed and memory (a minute, no download): the tool
#              itself, before an hour is spent on the real model
#   reference  tests/reference_27b.sh: the fork, the fork with float32 activations and the engine's NumPy forward pass
#              over the GGUF, which leaves the ids, the float32 fork's logits and (SAVE_LOGITS) the logits, keys and
#              values of the run that rounds its activations as the ternary kernels do. REFERENCE_STAGES and TEXTS are
#              its STAGES and TEXTS
#   fork       the fork alone: its tokens a second on this runner
#   convert    the native conversion of the list's GGUF with the original's vocabulary and config.json, as the page
#              converts it (tests/page_27b.py): its seconds, its memory, the options it yields
#   pq2        the PQ2_0 file of the same weights (7.21 GB) converted the same way: the same checkpoint, byte for byte
#   pyodide    the page's conversion itself, in Pyodide (NumPy with 32-bit integers, the kernels' quantizer), into a
#              64-bit shared memory: the checkpoint's sha256 against the native file's (and of the PQ2_0 file after pq2)
#   page       tests/page-27b.mjs compare: forward.js against the references, and the engine broken on purpose in the
#              ways that need no other checkpoint (signs of the rotated basis, the embedding's rows not turned back)
#   breaks     the conversion broken on purpose (the output matrices' columns in llama.cpp's order of value heads; a
#              PTQ1_0 block read in the order of its bytes), each through the comparison of the first text: it must fail
#   speed      tokens a second by the count of threads, the logits the same to the bit, a prompt in blocks, the GPU refused
#   memory     the whole context: what is placed after the checkpoint against footprint()
#   write      the page's generate() for QUESTIONS (lines of a file, or the three here), with THINKING=1 the entry that
#              thinks, at most TOKENS positions each, PICK="0,1" only these of them (from 0)
#   long       (the review of T233: nothing in the default; a context past 4096, never computed on the real model) a text
#              of 5,987 tokens (tests/fixtures/long-27b.txt) through the fork (reference_27b.sh's long stage: 1.5 to 2.7
#              hours at its speed) and through the page's forward pass with a context of LONG_CONTEXT (8192: the header's
#              4 bytes), the logits of some 40 positions held to the fork's and the growth over the floor of the first ones
#              looked at (page-27b.mjs's long mode, which has the lines); the memory after the checkpoint against
#              footprint() at that context, and the tokens a second as the prompt grows. long-fork and long-page are its
#              two halves (LONG_REFERENCES=<a run's id>: the fork's rows an earlier run kept, tests.yml keeps them a week).
#              The page's half on arm64 too: page-27b.mjs turns Liftoff off where V8 reads load32_splat wrongly
# The stages after convert use its checkpoint; page and breaks use the reference's files: of the reference stage of
# the same run, or of an earlier run that ran it (REFERENCES=<its id>: tests.yml keeps them a week).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
stages=${STAGES:-reference convert page breaks}
has() { [[ " $stages " == *" $1 "* ]]; }
id=hf-ternary-bonsai-2-27b
pq2_file=Ternary-Bonsai-2-27B-PQ2_0.gguf
pq2_bytes=7206168928
pq2_sha256=3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1

# (Node says "unknown" for an arm64 runner's CPU; lscpu names it)
echo "page: stages \"$stages\" on $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory"
python -m pip install -q numpy
# the kernels and Pyodide, for every stage that runs forward.js (only_extra builds neither)
if has dry || has pyodide || has page || has breaks || has speed || has memory || has write || has long || has long-page; then
  began=$SECONDS
  [ -d node_modules/pyodide ] || npm ci > /dev/null
  make kernels > /dev/null
  echo "page: npm ci and make kernels in $((SECONDS - began)) s"
fi

if has dry; then
  dry=.tmp/page-27b-dry
  mkdir -p "$dry"
  python -m pip install -q pytest  # (tests/make_ternary.py takes its made-up tensors from the unit tests)
  python tests/page_27b.py made-up "$dry"
  # (the widths of the made-up model's signs are 128, 256 and 384)
  node tests/page-27b.mjs "$dry/page" compare "$dry" --entry none --wide --threads 2 --lines none --broken embedding,sign-128-5,sign-384-all
  node tests/page-27b.mjs "$dry/page" compare "$dry" --entry none --wide --threads 2 --lines 100,100,100 --broken embedding --texts 1 || echo "page: dry: lines of 100 let the broken engine through, as they must (exit 1)"
  # (the long mode on the made-up model's own context: the same files as the fork's, made by the engine's NumPy forward pass, and
  # the engine broken on purpose, which must be caught by lines the right one passes)
  node tests/page-27b.mjs "$dry/page" long "$dry" --entry none --wide --threads 2 --lines none --broken embedding
  # (with a context the header does not say: the 4 bytes changed in memory, as the 27B's run changes them to 8192)
  node tests/page-27b.mjs "$dry/page" long "$dry" --entry none --wide --threads 2 --context 2048 --lines "${DRY_LONG_LINES:-3,1,2}" --broken embedding,sign-128-all
  node tests/page-27b.mjs "$dry/page" speed --entry none --wide
  node tests/page-27b.mjs "$dry/page" memory --entry none --wide
  rm -rf "$dry"
  [ "$stages" = dry ] && exit 0
fi

# the GGUF and the texts, where tests/reference_27b.sh works (it says where)
mkdir -p .tmp
STAGES=none bash tests/reference_27b.sh | tee .tmp/page-27b.log
work=$(sed -n 's/^runner: working in \([^ ]*\) .*/\1/p' .tmp/page-27b.log)
rm -f .tmp/page-27b.log
[ -d "$work" ] || { echo "page: no work directory"; exit 1; }
file=$(ls "$work"/*PTQ1_0.gguf)
# the list's entry is what the reference pins
node --input-type=module -e '
  const { MODELS } = await import("./src/models.js");
  const entry = MODELS.find((model) => model.id === process.argv[1]);
  if (!entry) throw new Error(`${process.argv[1]} is not in the list`);
  const script = (await import("node:fs")).readFileSync("tests/reference_27b.sh", "utf8");
  for (const [what, value] of [["model_revision", entry.hf.revision], ["file", entry.hf.weights], ["bytes", String(entry.download)]]) {
    if (!script.includes(`${what}=${value}\n`)) throw new Error(`the entry has ${what} ${value}, tests/reference_27b.sh another`);
  }
  console.log(`page: ${entry.id} is ${entry.hf.repo}@${entry.hf.revision}/${entry.hf.weights} with the vocabulary of ${entry.hf.vocabulary.repo}@${entry.hf.vocabulary.revision}, context ${entry.conversion?.max_seq_len ?? 4096}`);
' "$id"
context=$(node --input-type=module -e 'const { MODELS } = await import("./src/models.js"); console.log(MODELS.find((model) => model.id === process.argv[1]).conversion?.max_seq_len ?? 4096)' "$id")

if has reference; then
  STAGES="${REFERENCE_STAGES:-fork f32 numpy}" SAVE_LOGITS="$work/saved" bash tests/reference_27b.sh
  # what the comparison reads, kept with the run (tests.yml keeps .tmp/keep): 0.3 GB, 50 minutes of a runner
  keep=.tmp/keep/reference-27b
  mkdir -p "$keep/f32" "$keep/saved"
  cp "$work"/fork-*.ids "$work"/prompt-*.txt "$keep/"
  cp "$work"/f32/fork-*.single "$keep/f32/"
  cp "$work"/saved/engine-*-as-8-bits-round.* "$keep/saved/"
  echo "page: the references are kept with this run: $(du -sh "$keep" | cut -f1) (REFERENCES=<this run's id> takes them)"
fi
if [ -n "${REFERENCES:-}" ] || [ -n "${LONG_REFERENCES:-}" ]; then
  # the references an earlier run kept, in place of the reference stage
  gh run download "${REFERENCES:-$LONG_REFERENCES}" --repo "${GITHUB_REPOSITORY:-takano32/pyodide-llm}" --name kept --dir .tmp/kept
  cp -r .tmp/kept/reference-27b/. "$work/"
  [ -d .tmp/kept/reference-27b-long ] && cp -r .tmp/kept/reference-27b-long/. "$work/"
  rm -rf .tmp/kept
  echo "page: the references of run ${REFERENCES:-$LONG_REFERENCES}: $(ls "$work"/fork-*.ids | wc -l) ids files"
fi
if has fork; then
  STAGES=fork bash tests/reference_27b.sh
fi
if has long || has long-fork; then
  # the fork on the long text, which the page's half reads; kept with the run like the other references (60 MB)
  STAGES=long bash tests/reference_27b.sh
  keep=.tmp/keep/reference-27b-long
  mkdir -p "$keep"
  cp "$work"/fork-long.ids "$work"/fork-long.rows "$work"/fork-long.logits "$work"/long.txt "$keep/"
  echo "page: the long references are kept with this run: $(du -sh "$keep" | cut -f1) (LONG_REFERENCES=<this run's id> takes them)"
fi

# the original's config.json and tokenizer, and the GGUF beside them (tests/hf_fetch.py finds the file where it would put it)
folder_of() {  # <the GGUF's file> -> a folder with it and the original's small files
  local gguf=$1 name
  name=$(basename "$gguf")
  local repo
  repo=$(node --input-type=module -e 'const { MODELS } = await import("./src/models.js"); const { hf } = MODELS.find((model) => model.id === process.argv[1]); console.log(`${hf.repo.replace("/", "--")}/${hf.revision}`)' "$id")
  mkdir -p "$work/hf/$repo"
  ln -sf "$file" "$work/hf/$repo/$(basename "$file")"
  local original
  original=$(python tests/hf_fetch.py "$id" "$work/hf" | tail -1)
  if [ "$name" = "$(basename "$file")" ]; then echo "$original"; return; fi
  mkdir -p "$work/hf-$name"
  for small in "$original"/*; do
    case "$small" in *.gguf) ;; *) ln -sf "$(realpath "$small")" "$work/hf-$name/" ;; esac
  done
  ln -sf "$gguf" "$work/hf-$name/$name"
  echo "$work/hf-$name"
}

if has convert || has pq2 || has pyodide || has breaks; then
  folder=$(folder_of "$file")
  ls -laL "$folder" | sed 's/^/page: /'
fi
if has convert; then
  python tests/page_27b.py convert "$folder" "$work/page" --context "$context"
  began=$SECONDS
  sha256sum "$work/page.bin" | cut -d' ' -f1 > "$work/page.sha256"
  echo "page: sha256 of the native conversion's checkpoint $(cat "$work/page.sha256") ($((SECONDS - began)) s); of its tokenizer.bin $(sha256sum "$work/page.tokenizer.bin" | cut -d' ' -f1)"
fi
if has pq2; then
  began=$SECONDS
  revision=$(sed -n 's/^model_revision=//p' tests/reference_27b.sh)
  curl -sS -L -f --retry 5 --retry-delay 10 -C - -o "$work/$pq2_file" \
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/$revision/$pq2_file"
  [ "$(stat -c %s "$work/$pq2_file")" = "$pq2_bytes" ] || { echo "page: $pq2_file has $(stat -c %s "$work/$pq2_file") bytes, not $pq2_bytes"; exit 1; }
  [ "$(sha256sum "$work/$pq2_file" | cut -d' ' -f1)" = "$pq2_sha256" ] || { echo "page: $pq2_file has not the revision's sha256"; exit 1; }
  echo "page: fetched $pq2_file in $((SECONDS - began)) s, the revision's sha256"
  pq2_folder=$(folder_of "$work/$pq2_file")
  python tests/page_27b.py convert "$pq2_folder" "$work/from-pq2" --context "$context"
  cmp "$work/from-pq2.bin" "$work/page.bin"
  cmp "$work/from-pq2.tokenizer.bin" "$work/page.tokenizer.bin"
  cmp "$work/from-pq2.json" "$work/page.json"
  echo "page: the PQ2_0 file converts to the same ternary checkpoint as the PTQ1_0 file, byte for byte ($(stat -c %s "$work/page.bin") bytes), the same tokenizer.bin and options"
  rm "$work/from-pq2.bin"
fi
if has pyodide; then
  for source in "$folder" ${pq2_folder:-}; do
    node tests/page-27b.mjs "$work/pyodide" convert "$source" --context "$context" | tee "$work/pyodide.log"
    got=$(sed -n 's/^pyodide: sha256 of the checkpoint //p' "$work/pyodide.log")
    if [ "$got" = "$(cat "$work/page.sha256")" ]; then
      echo "pyodide: the page's conversion of $(basename "$(ls "$source"/*.gguf)") in Pyodide is the native conversion's checkpoint, byte for byte"
    else
      echo "pyodide: the page's conversion in Pyodide has the sha256 $got, the native one $(cat "$work/page.sha256") — FAILED"; exit 1
    fi
    [ "$(sed -n 's/^pyodide: sha256 of tokenizer.bin //p' "$work/pyodide.log")" = "$(sha256sum "$work/page.tokenizer.bin" | cut -d' ' -f1)" ] \
      || { echo "pyodide: tokenizer.bin differs — FAILED"; exit 1; }
    python - "$work/pyodide.pyodide.json" "$work/page.json" <<'PYTHON'
import json, sys
pyodide, native = (json.load(open(path)) for path in sys.argv[1:3])
same = pyodide == native
print(f"pyodide: the options are {'the native conversion' + chr(39) + 's' if same else 'NOT the native ones — FAILED: ' + str({key: (pyodide.get(key), native.get(key)) for key in {*pyodide, *native} if pyodide.get(key) != native.get(key) and key != 'rotated'})}")
sys.exit(0 if same else 1)
PYTHON
  done
fi
[ -f "$work/$pq2_file" ] && rm "$work/$pq2_file"

status=0
if has page; then
  # (the widths the 27B's matrices read: 5120, 6144 and 17408. One sign of the 17408 or of the 6144 moves a logit by 0.11
  # to 0.17, T238's review: less than the rounding of the activations moves the page's forward pass from the reference
  # it is compared with, 0.13 to 0.24, so this comparison does not see them (run 36952311034); the engine's NumPy forward
  # pass against the float32 fork does, tests/reference_27b.py, and the signs it reads are the ones the plan hands on)
  node tests/page-27b.mjs "$work/page" compare "$work" ${ANYWAY:+--anyway} \
    --broken "${BROKEN:-embedding,sign-17408-17407,sign-5120-2560,sign-6144-all}" --weak "${WEAK:-sign-17408-0,sign-6144-0}" || status=1
fi
if has breaks; then
  for broken in tiled order; do
    python tests/page_27b.py convert "$folder" "$work/broken" --context "$context" --broken "$broken"
    node tests/page-27b.mjs "$work/broken" compare "$work" --texts 0 --expect-failure || status=1
    rm "$work/broken.bin"
  done
fi
if has speed; then
  node tests/page-27b.mjs "$work/page" speed --threads "${THREADS:-1,2,4}" ${ANYWAY:+--anyway} || status=1
fi
if has memory; then
  node tests/page-27b.mjs "$work/page" memory ${ANYWAY:+--anyway} || status=1
fi
if has long || has long-page; then
  # (the context is the header's 4 bytes: the checkpoint made for 4096 serves; no second conversion of 7.66 GB)
  node tests/page-27b.mjs "$work/page" long "$work" --context "${LONG_CONTEXT:-8192}" ${LONG_LINES:+--lines "$LONG_LINES"} ${LONG_TOKENS:+--prefix} ${ANYWAY:+--anyway} || status=1
fi
if has write; then
  questions=${QUESTIONS:-}
  if [ -z "$questions" ]; then
    questions="$work/questions.txt"
    printf '%s\n' "これからの流行りを3つ挙げてください。" "What is 17 times 24?" "日本でいちばん高い山と、その高さを教えてください。" > "$questions"
  fi
  # PICK="0,1,2": only these (from 0) of the questions, so that the dozen of tests/fixtures/questions-12.txt (the review of T233: the
  # six Japanese and six English of tests/answers.mjs) may be split over runs of a runner each; lines that begin with # are no questions
  index=-1
  while IFS= read -r question; do
    [ -n "$question" ] || continue
    case "$question" in \#*) continue ;; esac
    index=$((index + 1))
    if [ -n "${PICK:-}" ] && [[ ",$PICK," != *",$index,"* ]]; then continue; fi
    echo "page: question $index"
    node tests/page-27b.mjs "$work/page" write "$question" --tokens "${TOKENS:-1500}" ${THINKING:+--entry "$id-thinking"} ${ANYWAY:+--anyway} < /dev/null || status=1
  done < "$questions"
fi
exit $status
