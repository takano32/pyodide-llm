### T.421 [整理][遠隔試験] `perplexity_prepare.py` を `--entry` なしで呼ぶ手順の options が黙って変わった — 状態: 完了（2026-10-11、持ち主が (a) に決めた「オススメでよさそう」: 今の形のまま。**T.374.4（本線 9ff1e49）の前と後で、`--entry` なしで測った perplexity の数字は並べない**（頭のトークンと止まりが違う）。2026-10-11、T.374.4 のレビューから。規模 小）
- T.374.4 から、`--entry` なしの変換は tokenizer_config.json と chat_template.jinja を読む（`?hf=` で開いたページと同じ `bos`・止まり・`specials`・`lowercase`）。チェックポイントのバイトは同じ。
- `--entry` なしで呼ぶ所: `.github/workflows/int4.yml`（66 行）・`draft.yml`（52 行）（perplexity を測る: 頭のトークンが変わるので前の数字と並べられない）、`tests/write.sh`（29 行）、`tests/gpu-real.sh`（CPU・GPU・NumPy を同じ options で比べるので害は無い）、`tests/page_qwen35.sh`（引数があるときだけ `--entry`）。
- 決めること: (a) 今の形のまま、台帳に「T.374.4 の前後の数字は並べられない」と書く（取りまとめの勧め: ページの `?hf=` と同じになった）、(b) 呼ぶ所に `--entry` を渡す、(c) `listed()` の答え手に 1 行足して前の動きに戻す。
- `int4.yml` と `draft.yml` は numpy だけを入れるので jinja2 が無く、自前の書式の読み手が走る（開発機と options が違いうる）: 一緒に見る。
