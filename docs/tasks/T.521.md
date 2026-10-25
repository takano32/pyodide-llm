### T.521 [整理][Gemma4] T.329.2.2 のレビューの残り — 状態: 完了（2026-10-25、本線に入れた。前の状態: 進行中（2026-10-25、Opus medium）、未着手（2026-10-15、レビューの気づき））
- **結果**: (1) `worker-sink-check` に、保存した Gemma 4 の 2 回目の訪問で `loadConverted()` が最初の部品の見出しの後に `skipping.range` を表の範囲にすることを見る節（壊し方「範囲を置かない」が落ちる）。(2) `keeper()` が開けなかったときの警告を「…: it is not kept, and a model's table of every layer's inputs is held in the memory」に（コンソールだけの文。`silent-check`・`worker-conduct-check` の写しも）。(3) `keeper()` の `holding()` が書いた数を 0 に戻す: 別のトークナイザで開き直すたびに `weightsBuffer()` が呼ぶので、書き直した表を二重に数えない（`kept-check`: 半分書いて開き直し全部書くと whole。壊し方「戻さない」が落ちる）。軽い組 497 秒。CI を省いた。訳: Node の偽物で回る試験と、コンソールの文だけ。
- レビュー（Sonnet low、2026-10-25）: must-fix なし。`holding()` を呼ぶのは `weightsBuffer()` の 1 か所で最初の書き込みより前、開き直しは毎回通る。前の試行が失敗していれば書き込みは何もせず `finish()` が断るのは前と同じ。警告の文は本当のこと（keeper が無ければ `hole` は null で表はメモリに）。試験は空振りしていない。
- 2 回目の訪問で表の部品を飛ばさず読む退行（2.6 GB を余計に読むだけで結果は同じ）を落とす試験が無い: `kept.parts` の読んだ範囲を数える試験を `kept-check` か `worker-check` に。
- `keeper()` が失敗したときのコンソールの警告が「a model on the GPU alone is not kept」のまま（今は表のあるモデルでも出うる）。文はコンソールだけ。
- 同じ変換の中で別のトークナイザで開き直すと `keep.holding` の `held` が二重に数えられ、`whole()` が偽になる（読み直しに落ちるだけで壊れない）。
