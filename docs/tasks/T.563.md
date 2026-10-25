### T.563 [整理][WebGPU] GPU に置く決まりの古いコメント — 状態: 完了（2026-10-25、Opus medium。前の状態: 未着手、2026-10-16、T.261.4.2・T.545.2 のレビューの気づき）
- `src/runtime/forward/alone.ts:191` の「a token's fused layer does not take yet」は古くなった。`gpuside.ts` の `tokensUnfit()` の説明は Llama の流れだけを言う（窓・GELU・後ろの norm・畳み込みを足す）。
- **結果**: コメントだけ（`node tests/same-syntax.mjs`: the same code）。`tokensUnfit()` の頭に、答えのステップが今は取る Gemma 3 の窓・GELU・後ろの norm、LFM2 の畳み込み、Qwen3.5 の linear の層、llama.cpp の塊、表ごとの詰め方を足した。`aloneFactsUnfit()` の Gemma 3 の注を、ステップも GPU が取る今の形（GPU だけには GELU の gate と窓が無く、要りもしない）に直した。CI と軽い組を省いた。訳: 同じコード。
- レビュー（Sonnet low、2026-10-25）: `tokensUnfit()` の注は本体と合う。(1) の文は 2 行にまたがっていて取りまとめ役の grep が見落としていた（`alone.ts` の `aloneFactsUnfit()` の中）: 直した。ほかに古い所は無い。
