### T.460 [バグ][その他] 読み込みの秒が次の読み込みに持ち越される — 状態: 完了（2026-10-11、T.378.3 から。規模 小）
- 変換の後に読んだサイトのモデルが、前の変換の `seconds.convert` を言う。ページは出さないが、e2e の JSON の行には出うる。読み込みの始まりで消す。
- **2026-10-30**: `worker/clock.ts` に出来事 `loadStarted()`（`download`・`construct`・`convert` を消す。`pyodide` は訪問に 1 回なので残す）、`load()` の頭で呼ぶ（GPU だけのモデルの読み直しの再帰でも置き直す）。
- 検査: `worker-parts-check` に `clock.js` だけの節（変換の 4 つの秒の後に `loadStarted()` と 2 つ: `pyodide` と新しい 2 つだけ）と、`load.js` の代役の節に「前の変換の秒（`took('convert', 9)`）を置いてから読んだ次のモデルの準備完了が `convert` を言わない」。壊し方: `delete seconds.convert` を外す・`load()` の `loadStarted()` を外す、どちらも落ちる。
- 手元: 軽い組（448 秒）・型検査・`worker-modules-check`・`worker-check`・`silent-check`・ビルドして実ブラウザ（Chromium の e2e、stories260K）。CI を省いた。訳: 秒の記録だけで、読み込みの道は手元の e2e が踏んだ。
- レビュー（Sonnet low、2026-10-30）: OK。ページが読むのは `pyodide`・`download`・`construct` だけで、どの読み込みの道（サイトのモデル・変換・保存から）も準備完了の前に `download` と `construct` を置き直す。再帰の 2 回目も同じ。`overlapped` は時計の記録を読まない。
