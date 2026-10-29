### T.590 [バグ][遠隔試験] `preview.yml` の画面撮りが `?bench=1` の検査で落ちる — 状態: 完了（2026-10-29、T.372 の画面撮りで見つけた。T.555 から。規模 小）
- `tests/screenshots.mjs` は `?bench=1` の行を「1 行目が everything、2 行目が without the kernels」と位置で読んでいた。T.555 から最初のラウンドは頭に 2 回と最後に 1 回走る（「everything」「everything, before」「without the kernels」「everything, after」）ので、2 行目は「everything, before」で、スマホ幅（メモリを言わない）の「without the kernels を訳つきで飛ばす」の検査が落ちた（本線の run 40920004409 も同じ）。最後に通った `preview.yml` は 2026-10-13（T.555 の前）で、それから誰も回していなかった。
- 直し: 行を名前で引く。手元の Chromium でスマホ幅と PC 幅の両方が通った。CI を省いた。訳: 道具の 1 行で、手元で全部の場面を回した（確かめは T.372 の `preview.yml` がする）。
- レビュー（Sonnet low、2026-10-29）: 直すべき点なし。名前は `rounds.js` の `ROUNDS` と合い、検査の言いたいことは前と同じ。任意の 2 つを入れた: 台帳とコメントの行の並びを正しく書いた、名前の行が見つからないときは落ちる文にその名前と `rounds.js` を言う。
