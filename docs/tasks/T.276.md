### T.276 [性能][Bonsai] Ternary Bonsai の 1.7B・4B・8B を素の RoPE に — 状態: **完了**（2026-10-11、持ち主「オススメで」。T.235 の番号で実装した: 本線 7430fa2、記録は T.235 の「素の RoPE にした結果」）
- レビュー 2 本（T.235・T.246）の勧めは素の RoPE: 3 つの大きさとも perplexity が 2〜4% 低い（20 の測り全部）。config・GGUF・transformers・fork は yarn。ブランチ `t235-plain-rope` は 3 値にする前のものなので、`ternaryBonsai()` に合わせ直す。覆す条件: Prism ML が yarn で学習したと分かる、文脈を 8192 より上げる。
