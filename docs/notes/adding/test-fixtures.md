# 試験の作り物・偽物・道具を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### 試験の作り物・偽物・道具を足す

- **作り物のモデル**は `tests/made_up.py` から: `checkpoint(header, form, dtype, tensor=値を決める関数)`（製品の行を歩く）。値は `named()`・`uniform()`・`drawn()`、GPU の検査は `quantized()`。HF の形は家族ごとに手書き（`hugging_face()`・`qwen35_model()`・`lfm2_model()` など）。**値の決め方に手を入れると乱数の順が変わり全部の作り物が変わる**（gpu-check の線と `tests/fixtures/qwen35-logits.json` はこの作り物で決めてある）: 新しい決め方は新しい関数に。
- **偽物**: 2 つ以上の検査が使うなら `tests/fakes/` に（頭に 3 つ書く）と `tests/fakes-check.ts`。1 つしか使わないならその検査に。
- **ブラウザを動かす道具**: 作業のフォルダは `scratch(接頭)`（`.tmp/` の下に作り終わりに消す。`os.tmpdir()` に作らない: `serving-check` が断る）。サーバは `serving()`、終わりは `closed(() => browser.close())` と `leave(終了コード)`。初回の訪問は 1 回再読み込みされるので待ちを張り直す。
- **道具の取得**: `tests/fetching.ts`・`tests/fetching.py`。自分で `fetch()` と取り直しの輪を書かない。
- **参照の道具**: 線・判定・行・長い文脈・変換は `tests/reference.py` から名前で取り、道具には `made_up()`・状態の読み方・`HELD`・`real()` を書く。`tests/reference_dry.py` に代役と場合。弱い誤りを足すと `WEAK` が言う。
- **前後を測る道具**: 相手の版は `tests/other-tree.ts` から（`git fetch --depth` を書かない: 開発機の履歴を切った）。測る前に両側の誤差を見る。
- **CI だけの道具・手動のワークフロー**: 作り物で通る段（`STAGES=dry`）を足す。触ったらその入力の形で 1 回 dispatch。ジョブと apt の手順に期限、道具の期限は包むジョブより短く。新しい検査は `tests/suite.sh` の `light` か `full` に `part` で。
