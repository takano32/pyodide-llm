#!/usr/bin/env bash
# t260_answers_ci.sh (the review of T260; a throwaway branch's): what the list's LFM2 entries write for twelve questions (six
# Japanese, six English) as the page's engine samples them at the entry's own temperature, SEEDS of them each, for a person to
# read (tests/answers.mjs through tests/write.sh).
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/t260_answers_ci.sh 250 hf-lfm2.5-230m hf-lfm2.5-350m" --ref t260-review-probe --grep "answers "
set -uo pipefail
tokens="$1"; shift
pip install --quiet numpy tokenizers regex sentencepiece pytest
npm ci --silent
make kernels > /dev/null
echo "answers: $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
SEEDS="${SEEDS:-1 2}" TOKENS="$tokens" WRITER=answers.mjs GREEDY=1 bash tests/write.sh "$@"
