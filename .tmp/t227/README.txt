The scripts of T227's review (Sonnet max, 2026-10-01). Kept on the branch t227-review-probes; they run from a checkout of
t227-review with these files in .tmp/t227/ (git add -f: .tmp/ is ignored), Node 24, no npm install.

fixtures.mjs   the fixtures of tests/bench.mjs as a module (it runs the test's asserts again):
                 sed -e 's#"\.\./src/bench\.js"#"../../src/bench.js"#' -e 's#new URL(`\.\./src/pages/#new URL(`../../src/pages/#' \
                     -e 's#new URL("\.\./src/pages/#new URL("../../src/pages/#' -e 's#new URL("\.\./public/#new URL("../../public/#' \
                     -e 's#new URL("\.\./\.github/#new URL("../../.github/#' -e 's#^console.log("ok");##' tests/bench.mjs > .tmp/t227/fixtures-body.mjs
                 cat .tmp/t227/fixtures-body.mjs .tmp/t227/exports.mjs > .tmp/t227/fixtures.mjs
                 (redo both after tests/bench.mjs changes)
e1-budget.mjs  how many warnings the summary keeps within the login URL's 7,000, with T225's real WRONG rows (strings of CI run
               36867111944, "two-down"), quiet and noisy page path
e2-variants.mjs  the same with the GPU line of the summary cut to a count, and with the warnings in severity order
e3-false-negatives.mjs  failures the page writes in none of warnings()'s words (the model page's path, the CPU section)
e4-escaping.mjs  a device error with line breaks through checkVerdict() and warnings() (the bug of 7989cbd; needs old-bench.js for e7)
e5-parse.mjs   parseReport() and reportsTable() on the six shapes of an issue's body
e6-sentence.mjs, e7-old.mjs  one bullet listed whole or by its sentence; the old warnings() on the test's refused shader
               (e7 needs the code before the review: git show 4dfef30:src/bench.js > .tmp/t227/old-bench.js)
mutate.mjs     breaks T227's code in src/bench.js 36 ways against a copy of tests/bench.mjs (sandbox mut/: cp -r src public .github, tests/bench.mjs)
mutate-page.mjs  breaks the page's wiring 8 ways against the unit test's reading of the page
make-page.mjs  cuts the <script> of src/pages/benchmark.astro into page-under-test.ts (Node strips its types)
page-harness.mjs, page-scenarios.mjs, p1.mjs, p2.mjs, p3.mjs  the page's own script run in Node on a fake DOM and fake workers that answer
               the fixtures and the failures to try (node p2.mjs [part of a scenario's name]; WATCHDOG=1 for the five silent minutes)
