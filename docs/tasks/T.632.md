### T.632 [修正][計測実行] start_check.ts が GNU screen の WINDOW=0 を窓の長さに読む — 状態: 完了（2026-10-31、T.596.1 の手元の計測で見つけた。規模 小）
- `tests/start_check.ts` は旗を環境変数でも受け、`--window` を `WINDOW` で読んでいた。GNU screen はシェルに `WINDOW`（その窓の番号、最初の窓で 0）を置くので、screen の中で回すと窓が 0 になり、Python の `range()` が「arg 3 must not be zero」で落ちた（CI では起きない）。
- 直し: 窓の環境変数を `START_WINDOW` に替え（ほかに `WINDOW` を渡す道具は無い: `write.sh` は環境をそのまま渡すだけ）、64 トークンより短い窓（数でないものも）は名指しの文で断る（窓は 64 未満を数えないので、前は 0 の窓がこの文でなく落ち方で言っていた）。
- 確かめ: screen の中（`WINDOW=0`）で `node tests/start_check.ts tiny-lm --tokens 200 --starts none,bos` が表を出す。`START_WINDOW=10` は文で断る。CI を省いた。訳: 手元の道具の小さな直しで、手元で回せた。
- レビュー（Sonnet low、2026-10-31）: 直すもの無し。ほかに start_check へ `WINDOW` を渡す所は無く（tests/ のほかの `WINDOW` は別の検査の局所の名前）、頭のコメントと 64 の線は Python の `len(w) >= 64` と合う。
