# エンジンのシェーダを足す・ステップの行列に詰め方を足す

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### エンジンのシェーダを足す・ステップの行列に詰め方を足す

1. 部品（`shaders/<部品>.ts`）に `export const` で（取った行があれば `/*! */` の許諾文も）。
2. 窓口 `shaders.ts` の import と export（忘れると `shaders-modules-check`）。
3. 目録 `shaders/engine.ts` に行（作る関数なら枝の組ごとに 1 行。**どこに足しても鍵は 1 回変わる**）。
4. Worker で `ENGINE.<名前>` をコンパイルする。
5. 試験: `shaders-catalogue-check.ts` の `ORDER`、新しいコンパイルの式なら `device-key-check.ts` の式の一覧。目録に無い作る関数は `tests/shader-makers.ts`。
6. **ステップの行列に詰め方**: `common.ts` の `Packing`、`fused.ts` の `DP4A_WEIGHTS`、`engine.ts` に `FUSED_OUTPUTS.map(…weights: "<名前>")` の行、`gpu/tokens.ts` の `weights:` の式、`ORDER`。プロンプトのタイルは `prompt.ts` の `TILE_WEIGHTS` に 1 行（束縛・読む関数・行の尺度の workgroup のメモリ・`loadSHMB` の行・トークンの和・レーンの尺度・ステップ 2 の行。塊の型は `tileBlocks()` が `SUPERBLOCKS` の行から作る: T642.9）で、`dp4a({ subgroups, weights })` は表を埋めるだけ。新しい表の名前は `tests/device-key-check.ts` の `TERNARY_BRANCHES`・`Q4K_BRANCHES` にも。`DP4A_WEIGHTS` の行は束縛・読む関数（`unpacking`: SDP8AI も）・行の読み `reads`（int8 と 3 値は `scaledReads(尺度の添字, 語)`、塊は `blockRow()`）で、`fusedDp4aMatVec` は詰め方の名前を比べない（T642.10）。
7. **融合した行列の出力の種類**（`FusedOutput`）は `fused.ts` の表 `FUSED_WRITES` に 1 行（束縛 5 から・書き出し・読む行列の数。`FUSED_OUTPUTS` は表の鍵の順で、目録が歩く: 足すなら表の最後に。T642.10）。`fusedMatVec`・`fusedDp4aMatVec` は出力の名前を比べない。ほかに足す所は `gpu/tokens.ts` の `tokenCodes()` の役 → 出力と `tests/gpu_answers.py` の `fused`。**作る関数の局所の名前（引数も）はほかの部品の最上位の名前と違うものに**（T642.10: 引数を `word` と名付けたら `device-key-check` が `superblocks.ts` の `word` と取り違えて「did not reach the hash」と落ちた）。
8. 確かめは CI（`gpu-prompt.yml`）。1 つのシェーダのコンパイルだけなら手元の Dawn で数秒（`getCompilationInfo()`）。WGSL の決まり: 障壁のある呼び出しが続く分岐の条件は `workgroupUniformLoad` で配った値で。NaN と無限大はビットの整数で見る。float32 → float16 の丸めの向きは装置しだい。束ねる範囲の始まりは 256 の倍数。
