### T.468 [整理][遠隔試験] ページの知らせの受け口を、要素を引数で受ける関数に分けて Node で回す — 状態: 完了（2026-10-11、T.378.5 から。規模 中）
- 「どの知らせでどの出来事を言うか」（`receiver.ts`・`choose.ts`・`composer.ts`・`run.ts`）は要素を最上位で取るので Node の試験に無く、実ブラウザだけが見る（別の読み込みの知らせを捨てる条件を外しても Node の検査は通る）。受け口を「知らせ → 言う出来事と描くもの」の関数に。`answerBegan()` が書いている間の 2 回目を断らないこと（守りは `composer.ts`）も一緒に。（T.475 から: ステータス行の文字列を組む所（`draw.ts`、要素を読む）も、行を組む純関数に分ける。）
- **子タスクの番号**（2026-10-30、取りまとめ役の決め: 1 つずつ確かめて本線に入れられる単位が 3 つ）: T.468.1 知らせを読むかと仕事の終わりかを純関数に — 完了。T.468.2 ステータス行の文字列（進みの行・準備完了の行）を組む純関数（T.475 から） — 完了。T.468.3 `answerBegan()` が書いている間の 2 回目を断る — 完了。受け口の全部を「知らせ → 出来事と描くもの」の関数にする形は取らない（取りまとめ役の決め: 3 つで、Node が見なかった判断（捨てる・終わり・行の文）は全部 Node の検査に入った。残りは要素に書くことと保存で、それを値に写すと受け口と同じ長さの写しになる）。

#### T.468.1 知らせを読むかと、仕事の終わりか — 状態: 完了（2026-10-30）
- 新しい純な部品 `src/page/heard.ts`: `ofThisLoad(data, loads)`（別の読み込みの遅い知らせは捨てる）と `endsWork(data)`（`ready`・`bench`・`paths`・`done`・`error`: 契約 `contracts/messages.ts` の頭の言うとおり `paths` も。モデルのページには届かないので動きは同じ）。`receiver.ts` はこの 2 つを呼ぶ。
- 検査: `page-parts-check` に、読み込みの番号の 3 つの場合と、契約の `FromWorker` の全部の種類のうち終わりがちょうど 5 つ（契約に種類が増えれば聞かれる）。`receiver.ts` は Node が import できないので、受け口の頭で捨て、終わりでボタンを返すことを文で見る。壊し方 3 つ（番号の条件を外す・`token` を終わりにする・受け口の捨てを外す）がどれも落ちる。
- 手元: 軽い組（450 秒）・型検査・`page-modules-check`・`contracts-check`・ビルドして実ブラウザ（Chromium の e2e と `switch-in-browser.mjs`: 変換の途中で替えたモデルの遅い知らせを捨てる道）。CI を省いた。訳: ページの部品の小さな切り出しで、手元の Chromium が踏んだ。
- レビュー（Sonnet low、2026-10-30）: OK。前の条件と `ofThisLoad()` は undefined・null・数のどれでも同じ。`paths` を足しても、モデルのページは送らず受けない（`/benchmark/` は自分の受け口）。文の検査は、捨てる所の前にコメントしか許さない（意図どおり）、終わりの所は字下げ 2 の `if` があることだけを見る（中身は見ない: 許す範囲）。

#### T.468.2 ステータス行の文字列を組む純関数 — 状態: 完了（2026-10-30）
- 新しい純な部品 `src/page/lines.ts`: `readyStatus()`（準備完了の行: 名前・動き方・2 本以上の本数・GPU の言葉・`?gpuTest=on`・Pyodide の版・古いページの `SLOWER` を最後に）と `progressLine()`（取得と変換の行: 届いた MB・速さ・変換の割合、ローカルのファイルは割合だけ。残りの時間は今までどおり `remaining.ts`）。`draw.ts` の `showStatus()` と `receiver.ts` の `progress` が呼ぶ（式はそのまま移した）。
- 検査: `page-parts-check` に 8 つ（行の部品を 1 つずつ）、壊し方（本数を 1 本でも言う）が落ちる。`updated-check` は `draw.ts` の式を文で持っていたのを、`readyStatus()` を Node で呼んで「古いページでは前と同じ行の後に `SLOWER`、そうでなければ言わない」と見る形に（`draw.ts` が `outdated` を渡すことは文で）。`SLOWER` を読むのは `lines.ts` だけ。
- 網の `page` は違う（予定）: 関数とその鍵（`converted`・`gpuNote`・`gpuTest`・`outdated`・`perSecond`・`received`・`total`）と展開が増え、文と数は同じ（`literal` の違いは無い）。
- 手元: 軽い組（443 秒）・型検査・`page-modules-check`・`bench.mjs`・ビルドして実ブラウザ（Chromium の e2e、`updated-in-browser.mjs` のモデルのページの 24 の場合: 古いページのステータス行を読む）。CI を省いた。訳: ページの部品の切り出しで、手元の Chromium が踏んだ。
- レビュー（Sonnet low、2026-10-30）: OK。移した式はどの入力でも同じ文（`Progress` に `name` は無く、`type`・`load` は読まれない）。`updated-check` の `.ts` の import は bun でも通る。

#### T.468.3 書いている間の 2 回目の `answerBegan()` を断る — 状態: 完了（2026-10-30）
- `answer.ts` の `answerBegan()` は、答えを書いている間（`generating`）は何も変えずに `undefined` を返す（守りは `composer.ts` にもあり、そちらは前のまま。`composer.ts` は答えが無ければ送らない）。
- 検査: `page-parts-check` の `answer.ts` の節に「書いている間の 2 回目は何も変えず、始まらず、種も引かない」。前の場合は答えを続けて始めていたので、ページと同じく間に `idle()`（ワーカーの終わりの知らせ）を置いた。壊し方（守りを外す）が落ちる。
- 手元: 軽い組（445 秒）・型検査・`page-modules-check`・ビルドして実ブラウザ（Chromium の e2e）。CI を省いた。
- レビュー（Sonnet low、2026-10-30）: OK。`answerBegan()` を呼ぶのは `composer.ts` の 1 か所で、直前に `generating` を見て await も無いので、ふつうの訪問で断られることは無い（`generating` を偽にするのは仕事の終わりの知らせの `idle()` で、読み込みの失敗も `error` で戻る）。断られたなら吹き出しが残る順だが届かない。親の決めも妥当。
