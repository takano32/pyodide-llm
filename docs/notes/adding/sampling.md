# 標本抽出の規則を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### 標本抽出の規則を足す

1. `engine/sampler.py` の `Sampling` に欄（名前は URL の名前。**既定は「その規則なし」**）。なれない値は `__post_init__` で断る。引く語を絞るだけの設定は `NARROWING` に。
2. `NumpySampler.drawing()` と `KernelSampler.drawing()` に 1 行ずつ（順は top-k → top-p → min-p、どの段も前の段の確率を 1 に直してから）。カーネル `kernels/kernel/sample.ts`、`engine/kernels.py` の ctypes の署名（宣言との食い違いは `tests/kernel_signatures.py`）、JavaScript の模型 `shaders/likecpu.ts`（**3 つは同じ形**: smoke が突き合わせる）。
3. `generate()`・`Llama`・worker は触らない。**GPU の SAMPLE に足さない限り、その設定のある生成は自動で CPU のステップになる**（足すなら `external.py` の `ON_GPU` と `gpu_steps()`、`generateMany` の引数（Python の代役 4 つも）、`contracts/gpu/protocol.ts` の `Sampling`、`samplingSettings()` と WGSL の `Sampling`（空きは無い: `SAMPLING_BYTES` を伸ばす）、シェーダ `SAMPLE`: 鍵が変わる。確かめは `/benchmark/` の `checkSampling()` と gpu-check の `narrowed`）。
4. ページからは項目の `generation` と URL（`src/page/fields.ts` の `LIMITS`: `Sampled` の鍵で閉じた表で、範囲と整数に丸めるかを 1 行に。`contracts/messages.ts` の `Sampled` に鍵を足して `LIMITS` に書き忘れると tsc が落とす。T643.7）。シートの欄にするなら `fields.ts` の `FIELDS` に 1 行（要素の id・設定の鍵・出す文 `said`、要れば見せ方 `shown` と読み方 `taken`: `settings.ts` は表を回すだけ）と `index.astro` の `<input>`・`<output id="<id>-value">`（無いと `page-parts-check` が落とす）。文面は持ち主。
5. 試験: `tests/test_sampling.py` の `DEFAULTS`、カーネルの引数の位置、段の順。決めた順があるものはそのトークンそのものを求める。
