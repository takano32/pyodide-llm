### T.455 [整理][WebGPU] GPU の側に時計が 2 つある — 状態: 完了（2026-10-11、T.378.2 から。規模 小）
- `gpuSide()` はブロックとステップを自分の `now` で計り、CPU のステップはエンジンの `performance.now()` で計る。エンジンの試験用の `clock` はスレッドの検索にしか届かない。1 つの時計を渡す。
- **2026-10-30**: `createForward()` が自分の `clock` を `gpuSide()` の `options.now` に渡し、CPU のステップ（GPU と比べるもの）も `clock` で計る。`now` は省けない引数に（既定の `performance.now()` を外した）。`forward-modules-check` の `NOT_HANDED` は空に（検査が「渡すようになった」と言って落ちた: 仕組みは残す）。`tests/fakes/token-clock.mjs` の頭に、GPU の側も同じ時計を対で読むことを 1 行。
- 検査: `gpu-default-check` に「時計の読みを数える `clock` と速い偽の GPU で 2 つのプロンプトを流すと、2 つ目は GPU で、時計が読まれる」（37 回）。壊し方: `now: clock` を外すと 0 回で落ちる。
- 手元: 軽い組（475 秒）・`gpu-default-check`・`thread-search-check`・`gpu-part-check`・`gpu-choice-check`・`threads-part-check`・型検査。CI を省いた。訳: 触ったのは時計の渡し方だけで、偽の GPU の Worker の試験が全部踏む。速さは測っていない: 足したのは GPU がステップを取るモデルの CPU のステップごとに閉包 1 回の呼び出し（WebGPU の道で手元で回せず、`performance.now()` の呼び出しを閉包が包むだけ）。
- レビュー（Sonnet high、shader-reviewer、2026-10-30）: OK。tokenClock の下の検査はどれも `forward()` だけを呼び、読みは 0 回か対なので、対のずれで通ってはいない。should 1 つ: GPU の側は条件つきで 1 回だけ読む所がある（`gpuside.ts` の計らないブロック、GPU の失敗と早い戻り）のに、`token-clock.mjs` に足した行が「対で読む」と強く言っていた: 文を直した（同じタスクのコミット）。
