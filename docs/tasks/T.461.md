### T.461 [バグ][その他] 組み立ての途中で投げた `Llama` が forward.js のエンジンを手放さない — 状態: 完了（2026-10-11、T.378.3 から。前からの動き。規模 小）
- `external()` の物（GPU の Worker を開いていればそれも）が `release()` されないまま残る。T.378.3 は「次の読み込みに持ち主として残らない」までを直した。
- **2026-10-30**: `worker/weights.ts` の `builtOrLetGo()`: `Llama` を作る 2 か所（ふつうの置き場と GPU だけの置き場）が `external()` の物を渡して作り、`Llama` が投げたら、それまでに `start()` が作ったエンジンを `release()` して投げ直す。次の読み込みはその GPU の Worker の終わりを、組み立ての前に手放した GPU だけのモデルの Worker と同じ `gpuOnlyEnding` で待つ（`loadBegan()`）。
- 検査: `worker-parts-check` の `load.js` の代役の節に「`Llama` が投げる読み込みは失敗のまま、そのエンジンの `release()` が 1 回呼ばれる」。壊し方: `release()` を呼ばない形で落ちる。代役の既定のエンジンに `release()` を足した（本物のエンジンは必ず持つ: 前の節の T.380.2 の読み込みが、投げる `Llama` でこの道を通る）。
- 手元: 軽い組（443 秒）・ビルドして実ブラウザ（Chromium の e2e、stories260K）。CI を省いた。訳: 失敗の道の後始末だけで、代役の試験が踏み、ふつうの読み込みは e2e が踏んだ。
- レビュー（Sonnet low、2026-10-30）: ほぼ OK。直したもの（同じタスクのコミット）: (1) `release()` が同期に投げると Python のエラーの文が隠れていた: 包んで警告 1 行（`silent-check` の `SAID` に足した）、投げ直すのは Python のエラー。試験に「`release()` が投げても読み込みの失敗は Python の文で、警告が 1 行」（守りを外すと落ちる）。(2) GPU だけの置き場の `destroy()` が `gpuOnlyEnding` を上書きしていた: 前の Promise と並べて待つ（どちらも同じ Worker の終わりを待つので害は無かったが、続く形に）。GPU の Worker に stop が 2 回届くのは無害。
