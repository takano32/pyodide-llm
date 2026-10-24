### T.356 [整理][CPU] `kernels/kernel.ts`（68 KB）を種類ごとに — 状態: 完了（2026-10-09、レビュー済み: Sonnet low。2026-10-09、ブランチ `t356-kernels`、手元の確かめまで。CI 待ち。規模 中）
- 分け方の案: 行列積、attention、norm と活性化、状態を持つ層（gate・convolve・delta_rule・short_conv）、変換（quantize・widen・ternary_x）、標本抽出。**出る wasm が前とバイト単位で同じか**を先に見る（同じなら確かめはそれで済む。違えば全部の組と `kernels-in-browser.mjs` と速さの前後）。Makefile の依存を足す（落とし穴: 書き忘れると古いカーネルで試験が走る）。
- **結果（2026-10-09、手元の確かめまで）**: `kernels/kernel.ts` は 69.3 KB・1266 行 → **2.7 KB・35 行の窓口**（頭のコメントと、モジュールの export を元の順に名指しする 15 行の `export { … } from "./kernel/…";`）と `kernels/kernel/` の 8 つ。ビルドの入口は今までどおり `kernel.ts`（`build.py` は変えていない）。
  - `matmul.ts` 13.4 KB・219 行: relaxed SIMD を使わない行列積（`matmul_f32`・`matmul_q8`・`matmul_q6`・`matmul_t2`）と、重みの隣で要るもの（`int8_sums`・`six_sums`・`interleave`・`add_columns`）。`six.ts`・`ternary.ts` を読むのはここだけ。
  - `quantize.ts` 14.2 KB・219 行: 数の形を替えるもの。保存の型を float32 に広げる 4 つ（`widen_bf16`・`widen_q8_0`・`widen_pq2_0`・`widen_ptq1_0`）と、float32 を int8・6 ビット・3 値にする 3 つ（`quantize_x`・`quantize6_x`・`ternary_x`）。`quantize_x` は変換器のものでもあり、毎トークンの活性値のものでもある（だからファイルの名前は「変換」にしなかった）。
  - `attention.ts` 9.3 KB・180 行: `rope`・`attention`・`attention_f16`。
  - `halves.ts` 4.7 KB・83 行: float16（`to_f16`・`from_f16`・`finite_f16` と、attention が読む `halves4`・`half`）。
  - `activations.ts` 6.9 KB・146 行: 行列積の間で 1 本のベクトルにすること（`rmsnorm`・`layernorm`・`gelu`・`swiglu`・`add_inplace`・`rotate`・`unrotate`）。
  - `stateful.ts` 7.5 KB・146 行: 状態を持つ層（`gate`・`convolve`・`delta_rule`・`short_conv`）。
  - `sample.ts` 11.6 KB・235 行: `argmax`・`penalize`・`sample`（とその 3 つのグローバル）。
  - `math.ts` 3.9 KB・75 行: 2 つ以上の種類が使うもの（`GS`・`hsum`・`fexp`・`vexp`・`largest`）。カーネルは無い。
  - `kernel_relaxed.ts`（21 KB）・`six.ts`・`ternary.ts`・`ceilings*.ts` は動かしていない。`kernel_relaxed.ts` は自分の `GS` を持ったまま（寄せるのは形を替える回）。
- **分けるだけ**: 最上位の文を、上のコメントと一緒に 1 字も変えずに移した（`.tmp/t356/split.mjs`。前の空でない 1211 行のうち、消えたのは `six`・`ternary` の import の 2 行だけ。足したのは 65 行: 頭のコメント、import 9 行、窓口の 15 行、`math.ts` と `halves.ts` の末尾の `export { … };` の 2 行）。ほかのファイルが使う共通の関数は、文を変えずに末尾の 1 行で出す。
- **ビルドはバイト単位では同じにならなかった。同じなのは関数で、違うのは並びだけ。** 14 のビルドのうち、`kernel_relaxed.ts` の 5 つと `ceilings` の 4 つは前とバイト単位で同じ。`kernel.ts` の 5 つは大きさが同じで sha256 が違う:

  | ファイル | 前（本線 `d7142b9` の `make kernels`） | 後 | |
  |---|---|---|---|
  | `simdkernel.so`（22921 バイト） | `ddc9a3edd2ea9695…` | `6009d6073860af03…` | 関数は同じ、順が違う |
  | `simdkernel_plain.wasm`（22904） | `c5364b93239f69eb…` | `a9b723d7888ce821…` | 同上 |
  | `simdkernel_plain64.wasm`（23313） | `1a84e7ae0d105715…` | `4c4cd652332b37e8…` | 同上 |
  | `simdkernel_shared.wasm`（22907） | `c45797b01037605e…` | `88fd13e3dd4e3f30…` | 同上 |
  | `simdkernel_shared64.wasm`（23316） | `4c7396910edcf26d…` | `129b352939772a15…` | 同上 |
  | `simdkernel_relaxed.wasmlib` | `71cb358effb7e4a8…` | 同じ | バイト単位で同じ |
  | `simdkernel_relaxed_plain.wasm` | `d6d0a7c3db920fe3…` | 同じ | 同上 |
  | `simdkernel_relaxed_plain64.wasm` | `bd86f9fe7cc8d55a…` | 同じ | 同上 |
  | `simdkernel_relaxed_shared.wasm` | `26946e489c078c29…` | 同じ | 同上 |
  | `simdkernel_relaxed_shared64.wasm` | `f1dac3604b22bb42…` | 同じ | 同上 |
  | `ceilings_plain.wasm` | `8428b7fc3c7e154e…` | 同じ | 同上 |
  | `ceilings_relaxed_plain.wasm` | `d8b929822e529b2c…` | 同じ | 同上 |
  | `ceilings_relaxed_shared.wasm` | `cfdebea409df5e20…` | 同じ | 同上 |
  | `ceilings_shared.wasm` | `5ae890e8235c24af…` | 同じ | 同上 |

  - **訳**: Binaryen は同じ回数使われる関数を名前の順に並べ、AssemblyScript の関数の内部の名前はファイルのパスで始まる。前は 35 の export が関数の名前の逆順に並んでいて（`widen_q8_0` が 1 番、`add_columns` が 35 番）、後はファイルの名前の逆順が先に効く。型の節の並びも同じ訳で 7 つ入れ替わる。窓口の export の並びで合わせられるのは export の節だけで、関数の並びは 1 関数 1 ファイルにでもしない限り合わない（種類ごとに分ける、と両立しない）。
  - **証明は `node tests/kernels-same.mjs`**（足した道具）: 両方の木を各自の `build.py` で作り、wasm を Binaryen の文にして、export ごとに関数の命令を比べる（呼び先は export の名前で、export の無い関数は中身で、型は中身で読む）。5 つとも「36 の関数・35 の export、関数は同じで順だけ違う」: export の順、型の集まり、import、グローバル 3 つ、各 export の命令、export の無い 1 つ（自分を呼ぶ関数。`sortNucleus` と見ている）が同じ。`public/` のカーネルが今の木のビルドと同じかも見る。
  - **だから要る確かめ（上の案のとおり）**: 全部の組（`tests.yml full=true`）と `kernels-in-browser.mjs`（WebKit・Chromium・Firefox、x86-64 と arm64）と、同じランナーでの速さの前後。関数の命令は同じなので、速さが変わるとすれば置き場所だけ（見立て。未計測）。
- **Makefile**: `KERNEL_PARTS` に 8 つを名指しして `public/simdkernel.so` の依存に足した。`kernel.ts` と 8 つのそれぞれを `touch` して、9 回とも作り直すのを見た（9 回作り直した後の 14 のハッシュは同じ: ビルドは決まった答えを出す）。
- **試験と道具**:
  - `tests/forward-check.mjs` の「`jobs.js` の `ADDRESSES` はカーネルの usize の引数」は `kernels/kernel.ts` を文として読んでいた。`kernels/kernel/*.ts` を読むようにし、窓口が名指しする名前とファイルが、部品の `export function` と合うことも見る（窓口が落とした名前はどのモジュールにも入らない）。Makefile が `kernel/` の全部のファイルを名指ししていることも見る。
  - 別のコミットのカーネルを `git show <コミット>:kernels/kernel.ts` で 1 ファイルずつ取っていた 6 つ（`attention-compare`・`q8r-bench`・`q8r-bench-threads`・`sample-bench`・`six-bench`・`tile-bench`）は、`tests/other-tree.mjs` に足した `kernelSources()` で相手の木の `kernels/` を丸ごと写す。分ける前のコミット（本線）と今の木を並べて、6 つとも手元で終わりまで走った（速さの数字は読んでいない）。分けた後のコミットを相手にする形は `attention-compare.mjs HEAD --exact` と `sample-bench.mjs --edges --commits HEAD` で見た。`sample-bench --stages` は `kernel/sample.ts` に手を入れて窓口に `stop_at` を足す形にした。`ternary-bench.mjs` は今の木だけをコンパイルするのでそのまま。
- **コメント**（別のコミット、ビルドは 14 とも同じ）: 「no kernel above」、`kernel_relaxed.ts`・`six.ts`・`ternary.ts` が名指しする `sums4`・`interleave`・`matmul_q6` の居場所、`forward.js's stagingFinite` → `public/forward/engine.js`。
- **手元の確かめ**: `make kernels`、`smoke.mjs`、`forward-check.mjs`（サイトの 4 モデル）、`threads-check.mjs`、`ternary-check`（945 通り）・`delta-check`・`conv-check`・`rotate-check`・`attention-check`・`finite-check`、`unchanged.mjs`（「unchanged: ok」）、`npm run build`。
- **わざと壊した**: (1) `kernel/activations.ts` の `rotate` の 1 行（入力の絶対値を取る）: `kernels-same.mjs` は 5 つのビルドで「DIFFERENT: the code of rotate」と、作り直す前の `public/` が今の木のものでないことを言い、作り直すと `rotate-check.mjs` が落ちた（`forward-check.mjs` の既定のモデルは `rotate` を呼ばないので通る）。(2) 窓口から `add_inplace` を落とす: `forward-check.mjs` が「kernels/kernel.ts does not name the kernels of kernels/kernel/」で落ちた。(3) Makefile の依存から `kernel/sample.ts` を落とす: `forward-check.mjs` が「the Makefile's rule for the kernels does not name kernels/kernel/sample.ts」で落ちた。(1) は戻して作り直し、ハッシュが表の「後」に戻るのを見た。
- **気づき（直していない。採番するなら）**: (a) 本線と並べて測る 5 つの道具は頭で `git fetch --depth=1 origin +main:…` を打っていたので、開発機で回すと `origin/main` の履歴が切れた（この回で踏んだ）。レビューで行を外し、`otherTree()` の取りも浅いクローンでだけ `--depth 1`（AGENTS.md の落とし穴）。残りは `encode-bench.mjs` の同じ行（T.357）。(b) `make clean` は `public/simdkernel*` を消すが `public/ceilings*.wasm` は消さない。(c) `kernel_relaxed.ts` は `GS` を自分で持つ（`kernel/math.ts` と 2 か所）。(d) `forward-check.mjs` の既定のモデルは `rotate`・`unrotate` を呼ばない（壊しても通る。落とすのは `rotate-check.mjs` と作り物の回した基底）。
- **見ていないもの**: 実ブラウザ（`kernels-in-browser.mjs`・e2e）、WebGPU、全部の組、arm64 以外の Node、速さ（どれも CI で）。`tests/page_27b.sh` と実物のモデルの道具。本線が分けた後の形になってからの「本線対今の木」の 4 つの道具（相手が新しい形のときの `kernelSources()` は上の 2 つで見た）。`kernels-same.mjs` は Binaryen の `wasm-dis` の文に頼る（呼び出しと型の参照の書き方が変われば読み直しが要る）。

  - **CI（2026-10-09、どれも成功）**: 全部の組と網 run 37943312185（網は全部同じ）。`kernels-in-browser.mjs` を WebKit 27.2・Chromium 156・Firefox 157 で、x86-64 run 37943315611 と arm64 run 37943318897。`kernels-same.mjs` は 2 つの CPU の CI でも「14 個: 9 個は同じバイト、5 個は同じ関数が別の順」（x86-64 run 37943321988、arm64 run 37943325635）。
  - **速さの前後（同じプロセスで本線のビルドと交互、同じ 2 つの run）**: `q8r-bench.mjs` は 3 つの大きさ × 3 巡で、本線と行ごとにビット単位で同じ。比は x86-64（Xeon Platinum 8370C）0.97〜1.01、arm64 0.93〜1.05（8192 × 8192 の 3 巡が 1.02・1.05・0.93 で、巡ごとに向きが変わる: 揺れ）。`sample-bench.mjs` は 0.99〜1.00（引いたトークンは全部同じ）。`attention-compare.mjs` は 1.00（出力は一致）。**関数の並ぶ順が変わっても速さは変わらない**。持ち主の端末は未計測。
  - **レビューの直しの後（run 37948461224、成功）**: 軽い組・網・`kernels-same.mjs`、そして浅い clone の上で直したベンチの道具 5 つ（`q8r-bench`・`sample-bench`・`attention-compare`・`six-bench`・`tile-bench`）が最後まで走った。比は 0.98〜1.00。**気づき（直していない、T.357 へ）**: `tile-bench.mjs` の「main: against float64」が 2.33e+4 と出る（tree は 1.65e-7）。relaxed のビルドは本線とバイト単位で同じなので、本線の側だけ誤差が大きいのは道具の本線の側の組み方の誤りと見える（いつからかは未確認）。速さの比（1.002、最小 0.980）の行は両方が走った値。
  - **本番（2026-10-09、デプロイ run 37949303950）**: 並びの変わったカーネルで、Chromium の 2 回の訪問（tiny-lm・llm-jp-3 150M・Qwen3.5 0.8B・LFM2.5 350M・Ternary Bonsai 1.7B・Pythia 70M）run 37950009388、WebKit run 37950012976、Firefox run 37950016066: どれも成功。
