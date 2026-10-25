### T.448 [整理][その他] T.367.5 の小さな気づき — 状態: 完了（2026-10-25。前の状態: 未着手、2026-10-11。規模 小）
- `tests/imports.mjs` は束ねる前の配置（`?v=`・窓口の一覧）の読み手を持ったまま: 使う所を数えて、無ければ消す。
- `gpu/` と `benchmark/gpu/` のコメントの「shaders.js's X」は窓口の名前で部品を言う（モデルの Worker はもう窓口を読まない）。
- `worker-source.mjs` は「相手が export していない名前」を今も自分で断る（ビルドも止める）。
- `/benchmark/` の GPU の Worker も部品を直に取れる（約 150 行。`shaders` の塊と `LATER` の 1 行が消え、窓口 `shaders.js` は道具だけの物になる: 残すかを決める）。
- `tests/dist-check.mjs` の `keysSaid` は `globalThis.onmessage` を `try`/`finally` なしで置いて消す。規則 8 は `node:` で始まる import しか見ない（素の `worker_threads` は通る見込み）。
- `models.yml` の `query=gpuTest=on` を `long=` と `steps=` なしで頼むと、SwiftShader のステップで 256 トークンが期限に入らない（run 38132198682。2 回目）: 道具が `gpuTest` のときの長さを自分で短くするか、頼みを断る。
- 束ねた GPU の Worker を、GPT-2 か NeoX のモデルで 1 回通す（`LAYER_NORM`・`GELU` の枝）。
- **結果（2026-10-25）**: 直した: (a) `tests/dist-check.mjs` の `keysSaid` は `onmessage` の代役を `try`/`finally` で外す、(b) 規則 8 は `node:` の名前のほかに Node の組み込みの素の名前（`worker_threads` など、`node:module` の `builtinModules`）の import も見る（`tests/dist-rules-check.mjs` に素の名前の場合: 前の規則では落ちる）、(c) `gpu/`・`benchmark/gpu/` のコメントの「shaders.ts's X」を、その名前を出す部品のファイル（`shaders/<部品>.ts's X`）に（26 か所と手で 2 つ。ベンチの `device.ts` の 1 つは本当に窓口を読むのでそのまま。`same-syntax` が同じコード）。そのまま: `imports.mjs` の古い形の読み手は T.587、`worker-source.mjs` が出していない名前を断るのは今も要る守り、ベンチの GPU の Worker が窓口を読むのは道具の都合で残す（取りまとめ役の決め）、`models.yml` の `gpuTest` の長さは T.577 の後も残る道具の頼み方の話（頼むときに `long=` か `steps=` を付ける）。CI を省いた。訳: 検査の道具とコメントで、手元で回した。
- レビュー（Sonnet low、2026-10-25、T.448）: must-fix なし。新しい型は今の dist の 42 ファイルで誤って当たらない、(c) は 16 件を抜き取って部品の export と一致。勧めを入れた: 誤って当たらない側の対照（文字列の中の Node の名前、import でない `from:`）を `dist-rules-check` に 1 つ。
