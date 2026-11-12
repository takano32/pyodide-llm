# worker の知らせ・計画の鍵・GPU の計画の鍵を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### worker の知らせ・計画の鍵・GPU の計画の鍵を足す

1. 型の 1 か所: 知らせは `contracts/messages.ts`、計画は `contracts/plan.ts`、GPU の Worker とは `contracts/gpu/protocol.ts`、`/benchmark/` の節は `contracts/benchmark.ts`。
2. 投げる所は `postMessage({ … } satisfies 型)`、ページは `receiver.ts` の表 `HEARD` に 1 行（T643.2: 知らせの種類で閉じた表。読まない知らせも `() => {}` の行で「読まない」と言う。行が `"away"` を返すとページは去る: 後は何もしない）と、`heard.ts` の `ENDS`（**含む一覧**: 仕事の終わりの知らせかを決める）。`/benchmark/` の Worker に答えでない知らせを足すときは `ask()` の除く条件に。
   **ページが worker に頼む知らせ（`ToWorker`）は、`worker.ts` の表 `REPLIES` に 1 行**（T641.14: 知らせの種類で閉じた表で、行は知らせと `heard`（始めた読み込みの `signal` を置く: 取り消された読み込みは catch が黙る）を受ける。表に無い種類が来たら error で言う）。書き忘れは tsc、表の鍵と `ToWorker` の突き合わせは `worker-modules-check`（`tests/contracts.ts` の `tabled()`）。
3. 忘れると: 型・書く側 → `worker-modules-check`・`plan-keys-check`、綴り → tsc、`satisfies` → `*-modules-check`。ページが行を書かない・知らない行を書く → tsc（`HEARD` の型）と `page-parts-check`（表の鍵を `FromWorker` と）。知らせの鍵をどのページも読まない・ページが読む鍵が型に無い → `page-parts-check`（T641.15: `HEARD` の行から `src/page/` の関数へ渡る先をたどり、`paths` は `src/bench/path.ts` の 2 つへ。たどるのは `tests/contracts.ts` の `heardOff()`。ページの外（試験が `window.__ready` から読むなど）だけが読む鍵は `ELSEWHERE` に訳を、たどれない渡し方は `HANDED` に）。新しい種類を `ENDS` に入れるかどうかは `page-parts-check` が教える（T641.18: 終わりの知らせ `ENDS` と途中の知らせ `MIDDLE` の 2 つ目の写しを持ち、合わせて `FromWorker` の全種類で重ならないこと、`heard.ts` の `endsWork()` が `ENDS` だけを終わりと読むことを見る: 種類を足したらどちらかに 1 つ）。読まない鍵をわざと足すときは `plan-keys-check` の `held()` の `unread` に訳を。
4. worker が Python の新しい欄を呼ぶなら `contracts/python.ts`（`contracts-check`）。
