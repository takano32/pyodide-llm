### T.593 [追加][モデル] ELYZA-Shortcut-1.0-Qwen-7B — 状態: 完了（2026-10-30、T.591 から。T.511 が足す ELYZA に挙げていた。規模 小）
- elyza/ELYZA-Shortcut-1.0-Qwen-7B @ d33448648d9b1b7d69cbadb3f1a8d6fb4c8a01ee（qwen2、7.62B、apache-2.0、書式あり）。Q8_0: mmnga @ fc1fb43597cab7c9edbe71e41c109fadb83d3815（8.10 GB）。
- 考える形（ELYZA-Thinking）の答えを、考えずに出すよう学ばせたもの。BOS は Qwen2.5 の項目の形で `start_check`。
- **実装（2026-10-31）**: `hf-japanese.ts` に項目 `hf-elyza-shortcut-1.0-qwen-7b`（mmnga の Q8_0、語彙は原本、`QWEN3_OWN_BOS` と `QWEN25`: 一覧の Qwen2.5 の項目と同じ組）、標本抽出はカードの例の 0.6・0.95。`licenses.ts` に 2 行（Apache-2.0、カードの本文も同じ）。名前はモデルの名が作り手（ELYZA）を言うので重ねない。
- **確かめ**: `format_check.py` 9/9。GGUF は `gguf.yml` の run 41204186106 で原本と（339 のテンソル、最も近い量子化とのずれ 0、相対の最大 0.00653、語彙の違い 0）。ページのエンジン（int8 8,568,526,876 バイト、64 ビット）で tests.yml の run 41204223798: 頭のトークンの perplexity（1500 トークン）は なし 9.494・`<|endoftext|>` +0.88%・`<|im_start|>` +3.37%。差は小さいので一覧の Qwen2.5 の項目と同じ `<|endoftext|>` のまま。`answers.ts`（6 問 × greedy と種 2、400 トークン）は 12/12 止まり、日本語の問い 6/6 に日本語（かな 48〜70%）、繰り返し 0、`<unk>` 0。富士山・流行り 3 つ・17 × 24 = 408 の途中の計算・英語の 3 問とも崩れない。CI の 4 論理コアで 1 本の答えに 18〜180 秒。
- CI を回した。訳: 7B は手元で動かさない（AGENTS.md）。
- レビュー（Sonnet low、2026-10-31）: 直すもの無し。`download` は HF の長さ 8,098,525,440 と一致、`bos`・`template` は Qwen2.5 の項目と同じ組、標本抽出と note の言語の語とライセンスはカードと合い、`models-check` は ok。
- 本番（デプロイ 41217540614 の後、`models.yml` の run 41219704791）: Chromium の 4 本で準備完了 115.8 秒、256 トークンを 3.7 tok/s。
