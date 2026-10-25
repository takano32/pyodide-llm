### T.437 [整理][その他] T.367.3 の小さな気づき — 状態: 完了（2026-10-25。前の状態: 未着手、2026-10-11。規模 小）
- `.claude/agents/` の定義（`perf-auditor`・`shader-reviewer`・`shader-writer`）が `public/forward.js` などの古い置き場を言う。
- AGENTS.md の古い節（T.349〜T.353 の決まり、落とし穴）が `public/` と `?v=` のまま（新しい節「実行時の JavaScript を束ねた後の決まり」を上に置いた）: 次の蒸留で。
- `compare-engines` は 7 回では 1〜2% を決められない（同じコードの 2 つのビルドが arm64 で 1.4% 動いた）: 1〜2% を言うときは 15 回以上か、対照を毎回並べる。
- **済み（T.367.5）**: `src/page/list.ts` の `Promise.resolve({…})` の包みと `kept?.`（`kept.js` が動的な import だった名残り）を外した。`keptModels().catch(() => [])` は残した（名残りでない: 置き場を並べられないときの守り）。`*-modules-check.mjs` の頭の説明を今の形に書き直した。`compare-engines` の「15 回以上か、対照を毎回」は `tests/compare-builds.sh` にした（21 回、対照を毎回）。
- `tests/worker-check.mjs` が Node の `TimeoutNegativeWarning` を 1 回出す。鍵の合わない検査が失敗でなく期限まで待つ。`bench-check --dist chrome` は手元の Chrome の WebGPU の節を回す（`--run` で外せる）。
- `gpu-prompt.yml` の `gpu-check.mjs` は GPU の Worker のソースを配る（束でない）: 束の GPU の Worker を GPU つきで踏むのは、本番の `models.yml query=gpuTest=on` だけ。
- **結果（2026-10-25）**: (1) `.claude/agents/` の 4 つ（`perf-auditor`・`reviewer`・`shader-reviewer`・`shader-writer`）の古い置き場（`public/forward.js`・`helper.js`・`jobs.js`・`shaders.js`・`gpu.js`、`forward.js's CPU path`）を `src/runtime/` の今のファイルに。(2) AGENTS.md の古い節は蒸留（T.569）で書き直されていて、残る `?v=` の 2 つは Service Worker の登録の話で正しい。(5) `TimeoutNegativeWarning` は `tests/worker-harness.mjs` の繰り返しのタイマーが、時計を 2 度読んで数 µs 負の待ちを渡していた（Node は 1 ms にする: 動きは同じ）: 0 で抑えた。`worker-check` に警告が出ない、harness を使う 4 つの検査も通る。(5) の後半（`bench-check --dist chrome` が手元の WebGPU の節を回す）と (6) は覚え書きのまま。CI を省いた。訳: 定義の文と試験の台の 1 行で、手元で回した。
- レビュー（Sonnet low、2026-10-25、T.437）: must-fix なし。置き換えた名前は全部実在し古い名前は残っていない、0 にしても次の `fire` が残りを取り直すので早くは鳴らない（`worker-check` 33 件、警告なし）。エージェントの一覧の文は会話の始めに読んだ写しで、新しい会話から今の名前になる。
