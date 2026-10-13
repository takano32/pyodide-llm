#!/usr/bin/env bash
# tests/page_saluki.sh (T364.1, T364.2): Underdog Saluki 27B on the page's forward pass, its matrices kept in the blocks
# llama.cpp stores them in, held to its references, on a CI runner (never on the development machine: it fetches 7.9 GB
# and makes a checkpoint of 7.96 GB). On demand, not nightly (but for `dry`):
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=170 extra="bash tests/page_saluki.sh" \
#     --grep "^(llama|f32|reference|saluki|runner|page|pyodide):" --minutes 180 --ref <branch>
#   ... extra="STAGES='convert write' WRITE='at-once sampled thinking' bash tests/page_saluki.sh" ...
#   ... extra="REFERENCES=<a run's id> STAGES='convert page breaks' bash tests/page_saluki.sh" ...
# On x86-64 or arm64 (runner=ubuntu-24.04-arm: the page's forward pass is faster there, and its prompts take the tile of
# the signed dot product). The tool is tests/page_27b.sh's, Ternary Bonsai 2 27B's, made for another file: the stages
# have its names, tests/page-27b.mjs runs both models (the list has no entry of this one yet, T364.3: what one would
# say is tests/page_saluki.py's, handed over with --as), and what that file's head says of Liftoff above 4 GiB on the
# V8 of Node 24 on arm64 holds here.
#
# Run it when src/runtime/forward (the layers, jobs.ts's tables), the kernels of the blocks (kernels/blocks*), the
# conversion that keeps the blocks (convert/stream.py's Kept) or the readers of the blocks change in a way that could
# move what this model computes.
#
# STAGES (the default: "reference convert pyodide page breaks speed", an hour and a half on 4 logical cores):
#   dry        a made-up model of the kind (ten types by the layer, three value heads to a key head) through the
#              reference's NumPy pass over its GGUF, both conversions and every mode of tests/page-27b.mjs
#              (a minute, no download): the tool itself, before an hour is spent on the real model
#   reference  tests/reference_saluki.sh: llama.cpp as it is, llama.cpp with float32 activations and the engine's NumPy
#              forward pass over the GGUF, kept with the run (REFERENCE_STAGES: its STAGES; TEXTS, F32_TEXTS,
#              SECONDS_FOR_F32, TIGHT_LINES are its own)
#   convert    the native conversion of the GGUF with the original's vocabulary and config.json, its blocks kept
#              (tests/page_saluki.py): its seconds, its memory, the options it yields, the checkpoint's size
#   pyodide    the page's conversion itself, in Pyodide, into a 64-bit shared memory: the checkpoint's sha256 against
#              the native file's, its seconds and Pyodide's heap
#   page       tests/page-27b.mjs compare: forward.ts against the references, and the plan broken on purpose (BROKEN)
#   breaks     the conversion broken on purpose (the value heads left in the file's order), through the comparison of
#              the first text: it must fail
#   speed      tokens a second by the count of threads, the logits the same to the bit, a prompt in blocks, prompts of
#              PROMPTS (40,256) tokens, the two tiles of a prompt against each other, the GPU refused
#   memory     the whole context in blocks: what is placed after the checkpoint against footprint() (an hour on x86-64)
#   starts     a bare text (STARTS_TEXT: en or ja, Wikipedia's Mount Fuji) after each first token of HEADS
#              (248045,248044,none: <|im_start|>, <|endoftext|>, nothing), STARTS_POSITIONS (128) tokens of it
#   write      the page's generate() for QUESTIONS (lines of a file, or the four here) in each form of WRITE
#              ("at-once sampled thinking": tests/page_saluki.py's ENTRIES), at most TOKENS (300; THINKING_TOKENS 600
#              for the form that thinks) positions each, PICK="0,1" only these of the questions (from 0)
# The stages after convert use its checkpoint; page and breaks use the reference's files: of the reference stage of the
# same run, or of an earlier run that ran it (REFERENCES=<its id>: tests.yml keeps them a week).
# LINES: the lines of the comparison (logits,kl,kv), as tests/page-27b.mjs takes them; the default below is what the
# runs of T364 measured (TODO.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
stages=${STAGES:-reference convert pyodide page breaks speed}
has() { [[ " $stages " == *" $1 "* ]]; }
lines=${LINES:-none}
context=4096

echo "page: stages \"$stages\" on $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory"
python -c "import numpy" 2>/dev/null || python -m pip install -q numpy
python -c "import jinja2" 2>/dev/null || python -m pip install -q jinja2
began=$SECONDS
[ -d node_modules/pyodide ] || npm ci > /dev/null
make kernels > /dev/null
echo "page: npm ci and make kernels in $((SECONDS - began)) s"

if has dry; then
  dry=.tmp/page-saluki-dry
  rm -rf "$dry"
  mkdir -p "$dry"
  python -c "import pytest" 2>/dev/null || python -m pip install -q pytest  # (the made-up GGUF is the unit tests': tests/test_kept.py)
  python tests/reference_saluki_numpy.py made-up "$dry"
  python tests/page_saluki.py entries "$dry/entries.json" --made-up
  python tests/page_saluki.py convert "$dry/gguf" "$dry/page" --context 128
  node tests/page-27b.mjs "$dry/pyodide" convert "$dry/gguf" --blocks --context 128 | tee "$dry/pyodide.log"
  [ "$(sed -n 's/^pyodide: sha256 of the checkpoint //p' "$dry/pyodide.log")" = "$(sha256sum "$dry/page.bin" | cut -d' ' -f1)" ] \
    || { echo "page: dry: the conversion in Pyodide is not the native one's checkpoint — FAILED"; exit 1; }
  as="--as $dry/entries.json:sampled --wide"
  # (lines the right engine passes on this model and every broken plan fails: against the NumPy run rounded alike the
  # made-up model is 0.2 of a logit off, against the float32 one 5: its random blocks make the rounding count for more
  # than the real model's. The layer of w1 is a matrix every token reads)
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$dry/page" compare "$dry" $as --threads 2 --lines 10,2,0.05 --broken untiled,codebooks,embedding-kind,layer-w1-3
  python tests/page_saluki.py convert "$dry/gguf" "$dry/broken" --context 128 --broken tiled
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$dry/broken" compare "$dry" $as --threads 2 --lines 10,2,0.05 --texts 0 --expect-failure
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$dry/page" speed $as --prompts 40,100
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$dry/page" memory $as
  printf 'w3 w9 w27 w5 w1 w30 w2 w4 w8 w10 w11 w12' > "$dry/text.txt"
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$dry/page" starts "$dry/text.txt" $as --heads 7,3,none --positions 9
  # (the writing, with each form's sampling: on a small Qwen3.5 of the unit tests with a whole tokenizer, tests/make_qwen35.py's;
  # the made-up GGUF's vocabulary is 40 tokens of a tokenizer that has not all of them)
  python tests/make_qwen35.py "$dry/q35" int8 > /dev/null
  for form in at-once sampled thinking; do
    node tests/page-27b.mjs "$dry/q35" write "w3 w9" --as "$dry/entries.json:$form" --tokens 90 < /dev/null | tail -1 | cut -c1-300
  done
  echo "page: dry: the made-up model of blocks went through the reference, both conversions and every mode"
  rm -rf "$dry"
  [ "$stages" = dry ] && exit 0
fi

# the GGUF, where tests/reference_saluki.sh works (it says where), to its length and sha256
mkdir -p .tmp
REFERENCES= STAGES=fetch bash tests/reference_saluki.sh | tee .tmp/page-saluki.log
work=$(sed -n 's/^runner: working in \([^ ]*\) .*/\1/p' .tmp/page-saluki.log)
rm -f .tmp/page-saluki.log
[ -d "$work" ] || { echo "page: no work directory"; exit 1; }
file=$(ls "$work"/*.gguf)
python tests/page_saluki.py entries "$work/entries.json"

status=0
if has reference; then
  STAGES="${REFERENCE_STAGES:-llama f32 numpy agree keep}" bash tests/reference_saluki.sh || status=1
fi
if [ -n "${REFERENCES:-}" ]; then
  gh run download "$REFERENCES" --repo "${GITHUB_REPOSITORY:-takano32/pyodide-llm}" --name kept --dir ".tmp/kept/$REFERENCES"
  cp -r ".tmp/kept/$REFERENCES/reference-saluki/." "$work/"
  rm -rf .tmp/kept
  echo "page: the references of run $REFERENCES: $(ls "$work"/fork-*.ids 2>/dev/null | wc -l) ids files, $(ls "$work"/saved 2>/dev/null | wc -l) files of the engine's NumPy runs, $(ls "$work"/f32 2>/dev/null | wc -l) of llama.cpp with float32 activations"
fi

if has convert || has pyodide || has breaks; then
  folder=$(python tests/page_saluki.py folder "$work" "$work/hf" | tail -1)
  ls -laL "$folder" | sed 's/^/page: /'
fi
if has convert; then
  python tests/page_saluki.py convert "$folder" "$work/page" --context "$context"
  began=$SECONDS
  sha256sum "$work/page.bin" | cut -d' ' -f1 > "$work/page.sha256"
  echo "page: sha256 of the native conversion's checkpoint $(cat "$work/page.sha256") ($((SECONDS - began)) s); of its tokenizer.bin $(sha256sum "$work/page.tokenizer.bin" | cut -d' ' -f1)"
  [ "$(stat -c %s "$work/page.bin")" = 7956712492 ] && echo "page: the checkpoint is the 7,956,712,492 bytes its header and its form say (T363.3)" \
    || { echo "page: the checkpoint has $(stat -c %s "$work/page.bin") bytes, not 7,956,712,492 — FAILED"; status=1; }
fi
if has pyodide; then
  node tests/page-27b.mjs "$work/pyodide" convert "$folder" --blocks --context "$context" | tee "$work/pyodide.log" || status=1
  got=$(sed -n 's/^pyodide: sha256 of the checkpoint //p' "$work/pyodide.log")
  if [ -n "$got" ] && [ "$got" = "$(cat "$work/page.sha256")" ]; then
    echo "pyodide: the page's conversion in Pyodide is the native conversion's checkpoint, byte for byte"
  else
    echo "pyodide: the page's conversion in Pyodide has the sha256 $got, the native one $(cat "$work/page.sha256") — FAILED"; status=1
  fi
  [ "$(sed -n 's/^pyodide: sha256 of tokenizer.bin //p' "$work/pyodide.log")" = "$(sha256sum "$work/page.tokenizer.bin" | cut -d' ' -f1)" ] \
    || { echo "pyodide: tokenizer.bin differs — FAILED"; status=1; }
fi

as="--as $work/entries.json:at-once"
if has page; then
  # (the plan said wrong in ways only the real file's sizes show, each on the first text: the output matrices' input not
  # put in the file's order of the value heads, the codebooks a byte off, the embedding and a layer of a matrix taken
  # for another type's blocks: the feed-forward matrix of layer 20, the linear layers' qkv of their layer 10)
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$work/page" compare "$work" $as ${ANYWAY:+--anyway} --lines "$lines" ${TEXTS_OF_PAGE:+--texts "$TEXTS_OF_PAGE"} \
    --broken "${BROKEN:-untiled,codebooks,embedding-kind,layer-w1-20,layer-wqkv-10}" || status=1
fi
if has breaks; then
  python tests/page_saluki.py convert "$folder" "$work/broken" --context "$context" --broken tiled
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$work/broken" compare "$work" $as --lines "$lines" --texts 0 --expect-failure || status=1
  rm "$work/broken.bin"
fi
if has speed; then
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$work/page" speed $as --threads "${THREADS:-1,2,4}" --prompts "${PROMPTS:-40,256}" ${ANYWAY:+--anyway} || status=1
fi
if has starts; then
  language=${STARTS_TEXT:-en}
  node tests/wikipedia.mjs "$language" "$work/starts.txt"
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$work/page" starts "$work/starts.txt" $as --heads "${HEADS:-248045,248044,none}" --positions "${STARTS_POSITIONS:-128}" ${ANYWAY:+--anyway} || status=1
fi
if has memory; then
  # shellcheck disable=SC2086
  node tests/page-27b.mjs "$work/page" memory $as ${ANYWAY:+--anyway} || status=1
fi
if has write; then
  questions=${QUESTIONS:-}
  if [ -z "$questions" ]; then
    questions="$work/questions.txt"
    printf '%s\n' "日本でいちばん高い山と、その高さを教えてください。" "What is 17 times 24?" "これからの流行りを3つ挙げてください。" "Explain in two sentences why the sky is blue." > "$questions"
  fi
  for form in ${WRITE:-at-once sampled thinking}; do
    tokens=${TOKENS:-300}
    [ "$form" = thinking ] && tokens=${THINKING_TOKENS:-600}
    index=-1
    while IFS= read -r question; do
      [ -n "$question" ] || continue
      case "$question" in \#*) continue ;; esac
      index=$((index + 1))
      if [ -n "${PICK:-}" ] && [[ ",$PICK," != *",$index,"* ]]; then continue; fi
      echo "page: $form, question $index"
      node tests/page-27b.mjs "$work/page" write "$question" --as "$work/entries.json:$form" --tokens "$tokens" ${ANYWAY:+--anyway} < /dev/null || status=1
    done < "$questions"
  done
fi
exit $status
