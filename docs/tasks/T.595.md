### T.595 [追加][モデル] llm-jp-3 7.2B instruct3 — 状態: 完了（2026-10-30、T.591 から。規模 小）
- llm-jp/llm-jp-3-7.2b-instruct3 @ cdd4c7f3296fdc7785423a864bd9a86ce4c15915。Q8_0: mmnga @ 368515ca…（7.751 GB）。一覧の llm-jp-3 3.7B と 8B（llm-jp-4）の間。
- **実装（2026-10-31）**: `hf-japanese.ts` に項目 `hf-llm-jp-3-7.2b-instruct3`（mmnga の Q8_0、語彙は原本、一覧の llm-jp-3 3.7B instruct3 と同じ組: `llmJp`・`LLM_JP_INSTRUCT`・`sampled(1.1)`）。`licenses.ts` に 2 行（Apache-2.0、mmnga のカードも同じ）。`format_check.py` の `SYSTEM` に id を足した（カードの例が system の文を渡す: ほかの llm-jp の instruct と同じ）。
- **確かめ**: `format_check.py` 9/9。GGUF は `gguf.yml` の run 41227255238 で原本と（291 のテンソル、相対の最大 0.00582、最も近い量子化とのずれ 0、語彙の違い 0: GGUF の後ろの 10 は詰め物。mmnga の imatrix は Q8_0 には効かない）。ページのエンジン（int8 8,204,337,180 バイト、64 ビット）で tests.yml の run 41227255494: 頭のトークンの perplexity（1500 トークン）は なし 10.898・`<s>`(1) +0.87%。差は小さいので本物の書式と同じ BOS 1 のまま。`answers.ts`（6 問 × greedy と種 2、400 トークン）は 12/12 止まり、日本語の問い 6/6 に日本語（かな 52〜82%）、繰り返し 0、`<unk>` 0。富士山・英語の 2 問は崩れない。流行りの問いは greedy で「未来の予測はできません」と断ってから例を挙げる。17 × 24 は日本語も英語も誤る（2206・40 など: モデルの力）。答えは短め（平均 101 トークン）、CI の 4 論理コアで 1 本に 8〜85 秒。
- CI を回した。訳: 7B は手元で動かさない（AGENTS.md）。
- レビュー（Sonnet low、2026-10-31）: 直すもの無し。リビジョンと `download` は HF と一致、ライセンスは両方のカードが Apache-2.0、項目は 3.7B instruct3 と同じ組、`SYSTEM` の id、note の 8.2 GB と台帳の数字は run のログと合う。
- 本番（デプロイ 41231902623 の後、`models.yml` の run 41232639406）: Chromium の 4 本で準備完了 107.5 秒、217 トークンを 3.8 tok/s。
