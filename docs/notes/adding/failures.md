# 失敗の種類・GPU の訳・黙った catch・後から取るファイル

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### 失敗の種類・GPU の訳・黙った catch・後から取るファイル

- **失敗の種類**: `failure.ts` の `KINDS` に名前と「誰が作り誰が読むか」→ `new Failure("名前", 文)`（やり直さないなら `{ again: false }`。**種類が `refused` になりうる `new Failure(` はどれも `again` を書く**: 取り直しの輪が `again === false` だけで止まるので、書かないと 4xx を取り直す。書き忘れは `failure-check` が文から名指す（T641.16）。書かない訳のあるものは検査の `UNSAID` に訳つきで: いまは `failureOf()` だけ。ほかの種類の付け忘れは誰も言わない）→ `kindOf(error) === "名前"`。ページに伝えるなら知らせ `error` に鍵と `receiver.ts`。`tests/failure-check.ts`（作る所と読む所の両方が要る）。
- **GPU に置かない訳**: `forward/reasons.ts` の `REASONS` に項目 → `REASONS.名前` → `tests/reasons-check.ts` の `SAID` に文を丸ごと（**文面は持ち主**）。GPU だけに置けない事実なら `aloneFactsUnfit()` と `sizes-check`。
- **黙った catch**: 言うなら `silent-check` の `SAID`（Node で走らない所は次の文も `THEN`）、残すなら `SILENT` に [関数, catch の文, 訳]。失敗させて「言った」を見る試験と、失敗の無い場合に 0 行を。
- **後から取るファイル**: 作る物は `built.ts` の表と `built()`（本文を読むのは呼ぶ側）、モジュールは `named(import.meta.url, () => import("./x.ts"))`、Worker は `onerror` に `unstarted(event, import.meta.url, 今の文)`。`tests/updated-check.ts` の `LATER` に 1 行（`said`・`slower`・`quiet`）。**フォールバックを持つ所**は `void wentWithout(error[, import.meta.url])`・`void workerWentWithout(event, import.meta.url)`（待たない・投げない）。フォールバックを持つファイルと 1 トークンの道のファイルは `updated.ts` を知らない。`dist-check` も。
- **訪問者に見える文**（エラー・ステータス行・門の文）は実装の前に持ち主に見せる。
