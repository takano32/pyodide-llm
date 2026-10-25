### T.435 [整理][その他] T.367.2 の小さな気づき — 状態: 完了（2026-10-25。前の状態: 未着手、2026-10-11。規模 小）
- `src/python/engine/__init__.py` と `convert/__init__.py` のコメントが「public/llama2_….py is the window」のまま（T.367.2 は Python を 1 バイトも変えない証明のために残した）。次に Python に触る回に。
- `python_archive.py` の `create_system = 3` を消しても Linux では試験が落ちない（Windows で作ったときだけ違う。CI は Linux）。
- `tests/tree-check.mjs` の作り物の木と `tests/tree.mjs` の頭の説明が「全部 `public/`」を今の形と書いている。`tests/imports.mjs` の「待たない `import(new URL(…))` を変数に持つ形」は `python.js` の行のためだった: まだ使う所があるか未確認。
- T.367.6 が捨てる古い写しに、`.py?v=` の 34 個と `python.js?v=` が加わった。
- ローカルの `dist/` には `public/__pycache__/` の `.pyc` 34 個が入っていた（pytest が書いたもの）。本番の `dist/` に入っていたかは未確認（CI は pytest の後にビルドする: 入っていた見込み）。いまは起きない。
- **結果（2026-10-25）**: (1) 直した: 2 つの `__init__.py` の頭のコメントを `src/python/…` に（zip を作り直した）。(2) 足さない: `tests/python-archive-check.mjs` はもう全部の項目の `system` 3 と mode 644 を見ている（Linux では Python の既定も 3 なので、行を消しても落ちないのは Windows で作らないかぎり同じバイトだから。CI は Linux）。(3) この回では触らない: `tests/tree.mjs`・`tree-check.mjs`・`imports.mjs` は「全部 `public/`」の形（と `?v=` つきの動的 import）も読む。網が相手にする最も古い木 `f91d83f9` はもう `src/runtime/`・`src/python/` の形なので（`public/` の `.js` は `coi.js` と `coi-test/` の 2 つだけ: レビューが確かめた）、その読み手は要らないかもしれない。消すのは T.587。(4) 古い写しの話は T.449 で。CI を省いた。訳: コメントだけ。
- レビュー（Sonnet low、2026-10-25、T.435）: must-fix なし。(1) は正しい、(2) の読みも正しい（Linux の zipfile の `create_system` の既定は 3: 確かめた）。(3) は取りまとめ役の読みの誤り（網の相手の木はもう今の形）を指摘: 理由の文を直し、消すのを T.587 に。
