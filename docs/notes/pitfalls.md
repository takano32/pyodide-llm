# 落とし穴の細部

AGENTS.md の「落とし穴（実際に踏んだもの）」節から移した（T658、2026-11-12）。要点は AGENTS.md に。 AGENTS.md の同じ項には要点だけ（無い項は全文をここに）。

## Service Worker・隔離・回線（「落とし穴」の項から）

- **部品の取得の取り直し。** Windows の Firefox は本文の途中で「TypeError: Error in input stream」を投げることがある。サイトのモデルの部品は 2 回まで取り直す。5xx と 408 も取り直し、ほかの 4xx は 1 回。**1 つの部品が最終的に失敗したら、ほかの接続も止める**（`innerAbort()`）。メモリが断った書き込みと HF の 429 は取り直さない。
- **Pyodide の読み込みの期限は「進まない時間」で決める（T118・T129）。** 読み込みの間だけ Worker の `fetch` の本文が届いたバイトを数え、30 秒何も届かなければフォールバックに入る（`watchArrivals()`）。(1) **`integrity` を付けた fetch は本文を読み終えるまで解決しない**ので、**NumPy の wheel は `loadPackage("numpy", { checkIntegrity: false })` で取る**（2026-09-28、持ち主の判断）。(2) **静けさは「回ったチック 30 回」で数える**（ページが止められた後の最初のチックが「30 秒止まった」と言っていた）。(3) 深いバッファの遅い回線では、モデルの部品が回線を埋めてフォールバックに入る（隔離のない 1 コアで準備完了する。モデルの部品も数える案は持ち主の判断待ち: ブランチ `t129-review-model-bytes`）。(4) 包んだ Response は `url`・`redirected`・`type` を失う。確かめは `slow.yml`。
- **Playwright の下のオフライン**: Firefox は `setOffline(true)` の後も localhost に届く（手元から配るサイトで「オフライン」を言う道具はサーバにも接続を切らせる）。**Playwright を上げるとブラウザの版が全部動く**ので、上げる回はブラウザを使うワークフローをブランチで一通り回す。道具の側のバグで落ちたら、issue と PR の番号を「覆す条件」と一緒に台帳に書く。

## メモリ・Worker・保存（「落とし穴」の項から）

- **WebKit はメモリの線を越えたページを終わらせる（T506）。** 30 秒ごとにページのプロセスを見て、線（機械のメモリが 16 GB 以下なら 8 GB）以上なら終わらせる（`MemoryPressureHandler.cpp` の `shrinkOrDie()`）。例外もエラーも出ない。だから 8 GB 級のモデルは 16 GB 以下の機械では落ちる（CI のランナーは 15.6 GiB）。**取得の間は捨てる配列を JavaScriptCore が数 GB ためる**ので、線にはモデルより早く届く。**Playwright の WebKit はプロセスが終わっても `crash` を言わない**（`E2E_SILENCE` で落とす）。Playwright のふつうのコンテキストの WebKit は OPFS をメモリに持つ。**ページが黙って消えたら、機械のメモリとプロセスの大きさを 2 秒ごとに書く**（コミット `3eeaf162` の `tests/t506_watch.sh` の形）。
- **Playwright の通常のコンテキストは OPFS の割り当てが約 3〜5 GiB で、`write()` が例外でなく短い数を返す**（`keep()` は断る）。**大きい変換の保存と 2 回目の訪問を見るには `E2E_TWICE`**（ディスクの profile は 10 GiB 以上書ける）。Playwright の WebKit は保存が読み込み直しをまたがないことがある（本物の Safari は未確認）。保存できなければ Worker は理由を準備完了の内訳に出し、モデルはそのまま動く。Cache API では 503MB が容量不足になった（`estimate()` は足りると言っていた）。

## カーネルと正しさの試験（「落とし穴」の項から）

- **64 ビットのメモリの番地の上の 32 ビットを、エンジンが落とすことがある**（**V8 の Liftoff の arm64 だけ**: `v128.load*_splat` と `v128.load32_lane` を 4 GiB を越える番地で。V8 14.3 = Chrome 143 で直った。Node 24 の V8 13.6 が当たる: `node --no-liftoff` では起きない）。検査は手組みの wasm で聞き、誤るエンジンでは `--no-liftoff` で聞き直す。**聞き直す wasm と後で読むカーネルはバイトの違うものにする**（V8 はコンパイル済みの関数をバイトで覚えている）。

## WebGPU（「落とし穴」の項から）

- **lavapipe の subgroup の幅は `LP_NATIVE_VECTOR_WIDTH` で変わる**（128 → 4、256 → 8、512 → 16）。幅に依る誤りは幅 4 では通るので、幅 16 でも回す。**1024（幅 32）は使えない**（16 レーンぶんしか集めない）。32 レーンの端末の形は模擬で確かめただけ。
- **GPU の Worker の読み込みの見張りは段ごとに期限をつける**（SwiftShader は 1 つのシェーダのコンパイルに 10〜90 秒、head 128 の flash attention は静かな a1-free で 124〜126 秒・head 64 は 39〜40 秒: 期限まで 1.45 倍しかなく、機械が混むと越える。時計で 5 秒ごとに「生きている」と送る形は、何も進んでいなくても送るので外した: 段が終わるたびに知らせる）。1 つの段は 180 秒まで（超えたら CPU で準備完了）。**CI の `models.yml`（`gpuTest=on`）で head 128 のモデルがこれで CPU に戻ったら既知として 1 回だけ走らせ直す**。あきらめた頼みのコンパイルは止められない。

## CI・道具・git（「落とし穴」の項から）

- **投げられるものは Error とは限らない**（T242: Pyodide は標準ライブラリ `python_stdlib.zip` の取得に失敗しても先へ進み、`loadPyodide()` が Emscripten の `ExitStatus` で reject する。`String()` にすると「[object Object]」）。`failure.ts` の `told()` が名前と文で言い、Pyodide の段が Error でない値で終わったら「…one of its files may not have arrived」にする。引き金は Windows の WebKit で jsDelivr への取得がまとめて失敗すること。wasm の取得が失敗した回は 30 秒黙ってから「got nothing from the network」。取り直しは足していない（方針 2 に触るので持ち主の判断）。
- **手元のブラウザ。** a1-free では Playwright の Chromium で `tests/e2e.ts` が走る（入れ方は docs/notes/dev-setup.md。WebKit は動かない。`~/.cache/ms-playwright` のブラウザが `playwright-core` の版と合わないと止まる: 入れると `$HOME` に書くので先に言う）。本物の Chrome は Chrome DevTools MCP でも開ける。**7B 級は手元で動かさない**（`models.yml` で）。動かす前に峰の見込みが `free -m` の available に収まるかを見る。`TMPDIR` の長さは効かない（T512 からプロファイルはリポジトリの `.tmp/` に: 159 字の `TMPDIR` でも `tests/e2e.ts` が通った、T412）。画面の PNG は同じビルドでも数個のピクセルが違う。
- **開発機のメモリ。** a1-free は 11.9GB とスワップ 4GB で、持ち主の別のセッションや docker と分け合う（何も走らせていないときの available は約 9.9GB）。峰の読めない計測は `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` で包む。**軽い組（`suite.sh light`）は `MemoryMax=3G` では OOM で止まる**（2026-10-31、T624.2: 約 6 分目、ページの部品の試験の後で峰が 3 GiB。終わり方は 143 で、訳は `journalctl --user` の「oom-kill」にだけ出る）。`5G` で通った（602 秒）。
- **開発機のリポジトリに古い shallow の印が残っていると、網が履歴を切る**（2026-11-07: `.git/shallow` に 2026-10 の横のブランチのコミットが 8 つ残っていて、`git rev-parse --is-shallow-repository` が真だった。それで `tests/other-tree.ts` が `--before <ブランチ>` を `--depth 1` で取り、取ったコミットを新しい境にした。本線が 3 コミットしか無く見え、push が「behind 2459」で断られた）。直し方: `.git/shallow` の各コミットの親が手元にあることを確かめてから、ファイルを消す（`git fsck --connectivity-only`）。手元は shallow でないこと（`git rev-parse --is-shallow-repository` が false）。

## 計測とモデルの質（「落とし穴」の項から）

- **参照を保存の型（bfloat16）のまま持って積を float32 にするとき、埋め込みも float32 で渡す**（`inputs_embeds`。`input_ids` だと最初の RMSNorm が bfloat16 に丸める）。それでも載らないモデルは `--by-layer`（1 層ずつ）。
- **量子化した行を元のモデルの行と比べるとき、相関だけでは群の尺度の誤りが見えない**（尺度を s + 1 と読んでも相関は 4 桁同じ）: 大きさの比を並べる。低いビットの重みでは最小二乗の倍率は 1 に近くない。相関の線は測った値のすぐ下に。
