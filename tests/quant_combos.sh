#!/usr/bin/env bash
# tests/quant_combos.sh (T.513): one model of the list in the forms the page can hold it in, side by side by one measure,
# for the owner's choice of lighter models for phones. The forms:
#   int8, int6: the entry's own file converted as the page converts it (6 bits: what ?bits=6 asks for);
#   <a GGUF of the entry's repository> (a Q4_K_M, an IQ4_XS ...): that file with llama.cpp's blocks kept as they are
#     where the page has kernels for them (Conversion(blocks=True), as an entry with conversion.blocks), the rest int8.
# For each form, on the same runner (tests/small_models.sh's measures, T.511.5):
#   1. the checkpoint's bytes;
#   2. the nats per character of the beginning of Wikipedia's 富士山 and of its Mount Fuji (CHARACTERS characters each,
#      3000; tests/start_check.ts from the page's BOS, windows of 512; the perplexity beside them);
#   3. the dozen questions of tests/answers.ts, greedy and the entry's sampling with SEEDS ("1 2"; "" for greedy alone),
#      TOKENS tokens (400);
# and then 4. the speed of the forms in turn, ROUNDS rounds (3), the order turned round each round: tests/page-27b.ts
#   speed (forward.js and its software threads on a shared memory): a token at a time on THREADS (1,4), POSITIONS (64)
#   positions from the BOS, and a prompt of PROMPT tokens (256) in blocks of 16 on the most threads. One process a
#   form a round (a process holds one model): the forms are read against each other within the job, on its one CPU.
# STEPS="text answers speed" (the default): which of 2 to 4 to run.
#
#   node tests/ci.ts run tests.yml only_extra=true extra="npm ci && make kernels python && bash tests/quant_combos.sh hf-shisa-v2.1c-lfm2-350m int8 int6 shisa-v2.1c-lfm2-350m.Q4_K_M.gguf" --grep "quant_combos|start_check|answers|page:" --ref <branch>
#
# Exit: the number of forms that failed.
set -uo pipefail
cd "$(dirname "$0")/.."
id=$1
shift
room=".tmp/quant-combos"
if [ -d /mnt ] && [ -n "${GITHUB_ACTIONS:-}" ]; then
  sudo mkdir -p /mnt/quant-combos && sudo chown "$USER" /mnt/quant-combos
  room=/mnt/quant-combos
fi
mkdir -p "$room"
steps=" ${STEPS:-text answers speed} "
echo "quant_combos $id: $(node -e 'const os = require("os"); console.log(`${os.cpus()[0].model}, ${os.cpus().length} logical cores, ${process.arch}`)')"
if [[ $steps == *" text "* ]]; then
  for language in ja en; do
    title=$([ $language = ja ] && echo 富士山 || echo "Mount Fuji")
    node tests/wikipedia.ts $language "$room/whole-$language.txt" "$title"
    node -e 'const fs = require("fs"); const [from, to, n] = process.argv.slice(1); fs.writeFileSync(to, fs.readFileSync(from, "utf8").slice(0, Number(n)));' \
      "$room/whole-$language.txt" "$room/text-$language.txt" "${CHARACTERS:-3000}"
    echo "quant_combos: the $language text is $(node -e 'console.log([...require("fs").readFileSync(process.argv[1], "utf8")].length)' "$room/text-$language.txt") characters, sha256 $(sha256sum "$room/text-$language.txt" | cut -c1-16)"
  done
fi
# the entry's files (its own GGUF or safetensors, the original's vocabulary beside it), once
began=$SECONDS
folder=$(python3 tests/hf_fetch.py "$id" "$room/downloads" | tail -1)
python3 tests/write_options.py "$id" "$room/small" > "$room/page.json"
echo "quant_combos $id: the entry's files in $((SECONDS - began)) s: $folder"
failed=0
made=()
for form in "$@"; do
  out="$room/$form"
  began=$SECONDS
  (
    set -e
    case $form in
      int8|int6) python3 tests/perplexity_prepare.py "$folder" "$out" "$form" --entry "$id" | cut -c1-300 ;;
      *.gguf)
        # the other GGUF beside the same configuration and vocabulary (a folder of its own: tests/conducting.py's listed()
        # takes the first .gguf of a folder)
        place="$room/blocks-${form%.gguf}"
        mkdir -p "$place"
        repo=$(node -e 'import("./src/models.ts").then(({ MODELS }) => { const m = MODELS.find((m) => m.id === process.argv[1]); console.log(`${m.hf.repo}/resolve/${m.hf.revision}`); })' "$id")
        python3 -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from fetching import download; download(sys.argv[1], Path(sys.argv[2]))' \
          "https://huggingface.co/$repo/$form" "$place/$form"
        dir=$( [ -d "$folder" ] && echo "$folder" || dirname "$folder" )
        for file in "$dir"/*; do
          case $file in *.gguf) ;; *) ln -sf "$(realpath "$file")" "$place/" ;; esac
        done
        python3 tests/perplexity_prepare.py "$place" "$out" int8 --entry "$id" --blocks | cut -c1-300
        rm -f "$place/$form" ;;
      *) echo "quant_combos: $form is no form (int8, int6 or a .gguf)"; exit 1 ;;
    esac
    echo "quant_combos $id $form: the checkpoint is $(stat -c %s "$out.bin") bytes, made in $((SECONDS - began)) s"
    # the page's JSON with this form's options (the entry's over the conversion's: perplexity_prepare.py --entry)
    node -e 'const fs = require("fs"); const [page, options, to] = process.argv.slice(1);
      fs.writeFileSync(to, JSON.stringify({ ...JSON.parse(fs.readFileSync(page, "utf8")), options: JSON.parse(fs.readFileSync(options, "utf8")) }));' \
      "$room/page.json" "$out.json" "$out.page.json"
    if [[ $steps == *" text "* ]]; then
      for language in ja en; do
        echo "quant_combos $id $form: the $language text"
        node tests/start_check.ts "$out" "$out.page.json" --starts bos --tokens 100000 --window 512 --text "$room/text-$language.txt"
      done
    fi
    if [[ $steps == *" answers "* ]]; then
      GREEDY=1 SEEDS="${SEEDS-1 2}" node tests/answers.ts "$out" "$out.page.json" "${TOKENS:-400}"
    fi
  )
  if [ $? -ne 0 ]; then
    failed=$((failed + 1))
    echo "quant_combos $id $form: FAILED"
  else
    made+=("$form")
  fi
  echo "quant_combos $id $form: $((SECONDS - began)) s"
done
rm -rf "$room/downloads"
if [[ $steps == *" speed "* ]]; then
  for round in $(seq 1 "${ROUNDS:-3}"); do
    order=("${made[@]}")
    if [ $((round % 2)) -eq 0 ]; then
      order=()
      for ((i = ${#made[@]} - 1; i >= 0; i--)); do order+=("${made[$i]}"); done
    fi
    for form in "${order[@]}"; do
      echo "quant_combos $id $form: speed, round $round"
      node tests/page-27b.ts "$room/$form" speed --entry "$id" --threads "${THREADS:-1,4}" --positions "${POSITIONS:-64}" --rounds 1 --prompts "${PROMPT:-256}" \
        || echo "quant_combos $id $form: page-27b.ts speed said FAILED (read its lines)"
    done
  done
fi
echo "quant_combos $id: $# forms, $failed failed"
exit "$failed"
