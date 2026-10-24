### T.428 [整理][遠隔試験] HF のモデルを変換するワークフローに jinja2 が入っていなかった — 状態: 完了（2026-10-15、T.527: 直しは本線に（986c2d38）。残りは無い）。前の見出し: 反映済み（2026-10-11、持ち主「jinja2 はホンモノ使っていいって話にしなかったっけ？」。規模 小）
- T.397.1 で「書式は jinja2 で読む」と決め、`deploy.yml` と `tests.yml` の pip には足したが、ネイティブの Python で HF のモデルを変換するほかのワークフローは numpy だけだった: `draft.yml`・`int4.yml`（`perplexity_prepare.py`）、`gpu-prompt.yml` の 2 つのジョブ（`gpu-real.sh`）、`browsers.yml` の `fixed_outputs.py`。そこでは自前の読み手が走り、jinja2 だけが読める書式のモデルは開発機・ページと options が違いえた。4 つに `jinja2` を足した。
- 足していないもの（変換器の書式の読みを通らない）: `gguf.yml`（独立の参照）、`bench.yml`・`models.yml`・`preview.yml`・`threads.yml`（`make models` のサイトのモデルだけ）。
- CI を省いた。訳: pip の 1 語で、`tests/workflows-check.mjs` は手元で通る。`browsers.yml` の固定値は jinja2 のある所（手元の 6 つと tests.yml 経由の run 38102116435 の 16 個）で通っている。次の週次の `huggingface` のジョブで落ちたら、この判断を誤りとして記録する。
- **決まり**: HF のモデルをネイティブで変換する手順を足すときは、pip に jinja2 を入れる。
