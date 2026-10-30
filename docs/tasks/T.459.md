### T.459 [整理][遠隔試験] `forward-modules-check` は取り出さずに読む鍵を見ない — 状態: 完了（2026-10-11、T.378.2 から。規模 小）
- まとめた引数を `kernels.plain` のように分割代入なしで読む所の綴りは検査に掛からない。
- **2026-10-30**: 部品の関数の中で、`createForward()` が自分で作って渡したオブジェクトの鍵を `kernels.plain` の形で読む所も集める: 作った鍵にあること（無ければ「…reads kernels.plian, which createForward() made without it」）と、読まれた鍵として数えること（作ったのに誰も読まない鍵は前からの検査が言う）。ほかの部品のオブジェクトは型が持つので数えない。渡す物の一覧に `softwareThreads.kernels` が加わった（いまこの形で読むのはここだけ）。
- 壊し方 2 つ（`.tmp/` の写しから）: `threads.ts` の `kernels.plain` を `kernels.plian` に（新しい検査が落とす。tsc も言う）、`createForward()` の `kernels` に誰も読まない鍵 `spare` を足す（新しい検査が落とし、前の検査は通した）。CI を省いた。訳: 検査 1 つの直しで、手元で回した。
- レビュー（Sonnet low、2026-10-30）: OK。小さい should 2 つを直した（同じタスクのコミット）: `kernels?.x`（babel の `OptionalMemberExpression`）も読みとして見る（壊し方 `kernels?.blokcs` が落ちる）、読まれた鍵は何度読まれても 1 つと数える（出力の数 123 → 122）。名前を隠す局所と代入の左辺は今の部品に無い。
