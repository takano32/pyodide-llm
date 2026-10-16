# T545: WebGPU でまだできることの洗い出し（2026-10-16）

**まとめ**: 一覧のうち GPU に置けないモデルは、訳ごとに 6 組（LFM2 の畳み込み、Qwen3.5 系の linear attention、Gemma 3・4 と Spark-X の層の組、llama.cpp の塊、回した基底、float32）。いちばん訪問者に効くのは **LFM2**（既定の Shisa V2.1c LFM2 350M を含む 8 項目。うち 4-bit の 1 つは T547 も要る）で、次に **GPU の標本抽出の top-k・min-p・presence penalty**（これが無いと T261.4 の後も Gemma 3 の答えの歩は CPU に残る）。Qwen3.5 系（0.8B〜9B・NeoHorse・Agents-A1 の 16 項目）は T271 の小段がそのまま当たるので、27B と切り離して先に進める勧めを T271 に足した。新しく採番したのは T545〜T551 の 7 つ、追記は T271・T334・T344 の 3 つ。持ち主に聞くのは 1 つ（T548: メモリを言わないブラウザの生成の歩）。

この回は調べと台帳だけ。コードは変えていない。速さはどれも未計測（勧めの順は、当たる項目の数と訪問者の数の見込みで決めた）。

## 読んだもの

- `src/runtime/forward/gpuside.ts` の `gpuFit()`（`gpuUnfit()`・`tokensUnfit()`）、`forward/alone.ts` の `aloneFactsUnfit()`・`gpuOnlyUnfit()`、`forward/reasons.ts` の `REASONS`、`jobs.ts` の `PACKED` の `gpu` の欄（int8・int6・3 値だけが行を持ち、塊は `gpu: null`）、`shaders/engine.ts` の目録、`forward/choice.ts`（`GPU_BLOCK = 64`・`GPU_TOKENS = 4`）。
- 一覧（`src/models/`）の標本抽出: top-k を持つのは Gemma 3（64）・Gemma 4（64）・Qwen3.5 系（20）・Saluki（20）、presence penalty は Qwen3.5 系と Agents-A1、min-p は Bonsai 2 27B の考える形。
- TODO.md の WebGPU の動いているタスク（T171・T263・T270・T271・T293〜T297・T313・T325・T334・T344・T443〜T499 の整理）と docs/tasks の済んだもの（T146〜T241、T255、T351〜T382）。

## GPU に置けない訳と、当たる一覧の項目

| 訳（`REASONS`） | 当たる一覧の項目 | 仕事 |
|---|---|---|
| `convolution` | LFM2.5 230M・350M・LFM2 700M・1.2B Instruct・1.2B JP・Shisa V2.1c LFM2 350M（**既定**）・その 4-bit・Shisa V2.1 LFM2 1.2B | **T545（新）** |
| `linear` | Qwen3.5 0.8B・2B・4B・9B（考える形とすぐ答える形）・日本語版 Qwen3.5 0.8B v2.5・NeoHorse-1 4B・9B・Agents-A1 4B・Ternary Bonsai 2 27B・Underdog Saluki 27B | T271（追記: 27B と分けて先に） |
| `pair` | Gemma 3 270M・1B | T261.4（別の担当が実装中: 重ねない） |
| `pair`・`aloneGate` など | Gemma 4 E2B・E4B / Spark-X2.5 1.7B・4B | T334・T344（追記: T261.4 の部品を使う） |
| 塊（`gpu: null`、文は T503 で決まった「llama.cpp's blocks are not on the GPU yet」） | Saluki 27B・Shisa V2.1c LFM2 350M (4-bit) | **T547（新）**: どちらも層の側（T271・T545）が先 |
| `rotated` | Bonsai 2 27B | T271 |
| `floats` | float32 のサイトのモデル（stories など） | 採番しない（小さいモデルで GPU の得が見込めない） |

答えの歩だけが CPU に残る訳:

| 訳 | 当たるもの | 仕事 |
|---|---|---|
| top-k・min-p・presence penalty（`ON_GPU` の外） | 今は GPU に置けるモデルには無い。T261.4 の後の Gemma 3、T334 の後の Gemma 4、T271 の後の Qwen3.5 系、URL で渡した訪問 | **T546（新）** |
| `memoryUnsaid`（T205） | Safari・Firefox の全部のモデル（生成の歩） | **T548（新、持ち主に聞く）** |
| `stops`（8 を越える止まり） | 未確認（一覧の項目で当たるものは数えていない） | 採番しない |

GPU だけに置けない（`aloneFactsUnfit()`）: 状態のある層・gate・位置の表・gate の無い FFN・LayerNorm・SiLU でない gate・窓、それに回した基底・塊・int6・3 値で DP4A の無いブラウザ。GPU だけが要るのは 2 重に入らない 3B 以上なので、当たる一覧の項目は Qwen3.5 4B・9B・NeoHorse・Agents-A1（T271 の小段 4）、Spark-X2.5 4B（T344）、Gemma 4（T334）、Pythia 2.8B・6.9B（T294、候補のまま）、int6 になる端末（**T550（新、候補）**）。

## 勧めの順

| 順 | 番号 | 題 | 当たるモデル | 良くなること | 大きさ | 端末の鍵 |
|---|---|---|---|---|---|---|
| 1 | **T545** | LFM2 の畳み込みの層を GPU に | LFM2・LFM2.5 の 5 つ、Shisa LFM2 の 3 つ（既定を含む。4-bit の 1 つは T547 も） | 既定のモデルのプロンプトのブロックと答えの歩が、測って速ければ GPU に。スマホの長いプロンプトに効く見込み（未計測） | 中 | 動く（新しいシェーダ） |
| 2 | **T546** | GPU の標本抽出に top-k・min-p・presence penalty | Gemma 3（T261.4 の後）・Gemma 4・Qwen3.5 系・Saluki | 一覧のカードの設定のまま、答えの歩が GPU に残る | 中 | 動く（SAMPLE の文） |
| 3 | T271（追記） | 27B の前に、Gated DeltaNet の歩とプロンプトを Qwen3.5 の小さいモデルで | Qwen3.5 0.8B〜9B・日本語版 0.8B・NeoHorse・Agents-A1 | 16 項目が GPU に（4B・9B の 10 項目は GPU だけにも） | 大 | 動く |
| 4 | **T547** | llama.cpp の塊を GPU に（Q4_K・Q6_K から） | Shisa 350M (4-bit)（T545 の後）、Saluki 27B（T271 の後） | 4-bit の項目が GPU に。Saluki はパソコンの GPU だけに | 大 | 動く |
| 5 | **T548** | メモリを言わないブラウザの答えの歩 | Firefox（と Safari）の GPU に置ける全部 | Firefox のパソコンで答えの歩が GPU に | 小 | 動かない |
| 6 | **T549** | プロンプトの行列積に subgroup matrix | GPU に置ける全部（その機能のある端末） | プロンプトのブロックが速くなる見込み（未計測） | 中 | 動く（候補が増える） |
| 7 | **T550** | int6 を GPU に詰めたまま（GPU だけにも） | int6 になる端末（`deviceMemory` の小さい Android）の 3B 前後 | GPU の重みが 1.125 → 0.875 バイト / 重み、GPU だけに置ける | 中 | 動く |
| 8 | **T551** | 1 回の submit の歩の数とブロックの大きさを端末で選ぶ | GPU に置ける全部 | submit の固定費の大きい端末（持ち主の Android）で歩が速くなる見込み（未計測） | 小〜中 | 調べてから（`GPU_TOKENS` は JavaScript の定数で、シェーダの文に入るかは未確認） |

順の訳:
- **T545 が先頭**: 既定のモデル（2026-10-15 から Shisa V2.1c LFM2 350M）がいま CPU だけ。初めての訪問者のほとんどが開くモデルで、方針 9（既定で GPU をなるべく使う）にいちばん遠い所。畳み込みは 3 タップの因果の畳み込みとかけ算だけで、状態は層ごとに 2 トークン × dim と小さい（KV のように毎回両側に写しても安い）。ただし「状態を持つ層」を GPU が持つ道（位置 0 で消す・順でない位置を断る・CPU と GPU の状態の写し）はまだ無く、それを作るのがこのタスクの半分。同じ道を T271 の DeltaNet が使う。
- **T546 が 2 番目**: T261.4 で Gemma 3 の層が GPU に置けても、その項目の top-k 64 で答えの歩は自動で CPU に戻る（AGENTS.md の「標本抽出の規則を足す」の 3）。T261.4 の担当はこれを知っておく必要がある（T261.4 の中でやるか、T546 を後に続けるか: 取りまとめ役の決めは「別のタスクで、T261.4 の直後」）。
- **T271 の小段を 27B から切り離す**: T271 は「27B を GPU に」として持ち主の判断待ち（「進めるか、hybrid は GPU だけか CPU だけの 2 択にするか」）。小段 2〜4（DeltaNet の歩・linear の層のプロンプト・Qwen3.5 を GPU だけに）は 27B が無くても一覧の 16 項目に効くので、27B の判断を待たずに進められる形に書き直した。
- **T547 は層の側の後**: 塊の型を GPU で読めても、塊を持つ 2 項目はどちらも層（畳み込み・linear）が先に GPU に要る。
- T548〜T551 は小さいか、効くかがまだ分からないもの。

## 元ネタ（公開の実装）

- 畳み込み（T545）: llama.cpp の `ggml_ssm_conv`（LFM2 の短い畳み込みが使う演算。Vulkan の `ssm_conv.comp`、ggml-webgpu にあるかは未確認）。式そのものは AGENTS.md の LFM2 の項（`convolution_form()` の上）にあり、元ネタなしでも書ける大きさ。
- 標本抽出（T546）: 今の SAMPLE（ソートせず pivot を探す top-p）の形の延長。top-k は「k 番目の大きさ」を同じ二分の探しで、min-p は最大の確率 × p の線、presence penalty は今の繰り返しの罰の窓と同じ置き場。CPU の決まった順（top-k → top-p → min-p、どの段も確率を 1 に直す）は `shaders/likecpu.ts` の模型が持つ。
- DeltaNet（T271）: Prism ML の fork の `gated_delta_net.wgsl`（T271 の本文）。
- 塊（T547）: llama.cpp の ggml-vulkan の K 量子化・i 量子化の読み（`dequant_funcs`・`mul_mat_vec_*`）。ggml-webgpu の対応している型は未確認。
- subgroup matrix（T549）: llama.cpp の ggml-webgpu の行列積（Chrome の実験の機能 `chromium-experimental-subgroup-matrix` を使う形）。Chrome が旗なしで出しているか、Android の Mali で使えるかは未確認: そこを先に見る。

## 手元で見られない所（CI だけ）

どれも WebGPU のシェーダか GPU の道なので、確かめは `gpu-prompt.yml`（Dawn と SwiftShader）と `bench.yml`。手元で回せるのは、1 つのシェーダのコンパイル（Dawn の Node 版と lavapipe、AGENTS.md の WebGPU の落とし穴）、選び方の偽の装置の試験（`gpu-default-check`・`gpu-hybrid-check`・`gpu-worker-parts-check`）、網の `shaders`。T545・T546・T547 の端末での速さは、持ち主の端末でまとめて見る一覧に積む。

## 持ち主に聞くもの

**T548: メモリを言わないブラウザ（Safari・Firefox）で、答えの歩を GPU に置くか**（T205 で止めた: iPhone の `/benchmark/` のモデルの節でタブが落ちたため。訪問者のタブが落ちうるので、聞く側の判断）。

1. **（勧め）Firefox だけ置く、Safari は今のまま**: Firefox の WebGPU はパソコンの版で使われている（Android の Firefox で WebGPU が使えるかは未確認）。メモリの見積もりは今と同じ 4 GB の仮定のまま。落ちた報告は iPhone の Safari のものだけ。
2. **今のまま**（どちらも答えの歩は CPU、プロンプトのブロックだけ GPU）: 何も変わらない。
3. **どちらも置く**: iPhone の Safari でもタブが落ちうる。持ち主の iPhone で確かめてから。

答えを待つ間は、T545・T546 など、この判断に依らない所を進められる。
