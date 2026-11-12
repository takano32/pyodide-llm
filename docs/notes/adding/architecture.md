# アーキテクチャを足す（`LAYOUTS`）

AGENTS.md の「物を足す手順」節から移した（T658、2026-11-12）。要点は AGENTS.md に。

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

### アーキテクチャを足す（`LAYOUTS`）

1. `engine/layout.py` の `LAYOUTS` に `Layout` を 1 つ。**事実 1 つだけが違うなら `LAYOUTS["llama"]._replace(…)` の 1 行**で、Python のほかの所は触らない。どの枝でもない違いは欄を 1 つ足し（既定は今の記録が変わらない値）、読む所に 1 行。
2. JavaScript は 1 か所: `contracts/form.ts` の `Arch`（忘れると `worker-sink-check`）。`forward/architectures.ts` の行は `python_facts.py` が `LAYOUTS` から書き出す（T628.1: 手で書かない）。`tests/plan-keys-check.ts` の `families` に 1 行。**今ある事実の組み合わせだけが違う記録は `forward/engine.ts` に触らずに動く**。
3. **新しい事実**をページのエンジンが読むなら: `Layout` の欄 → `engine/plan.py` の計画の鍵（同じ名前）→ `contracts/plan.ts` の `Plan` → `engine.ts` が頭で 1 回取って枝で読む（`benchmark/madeup.ts` の計画は Python から書き出されるので付いてくる）。層ごとの事実は `layer_facts()` → `Layer` → `engine.ts`（NumPy の forward も `Llama.layers` で同じ表を読む: Python の側は 1 か所）。**同じ名前で選んでいた 2 つの事実は枝を分ける**。`tests/plan_facts.py` の `cases()` に、その事実だけが食い違う記録を 1 つ。
4. 試験: `test_architectures.py` の `FACTS`、行の関数を書いたなら `test_layout.py` の `ORDER`。作り物は `tests/made_up.py` が表から作る。
5. GPU: 層は「RMSNorm と gate つきの FFN」「LayerNorm と GPT-2 の FFN」の 2 組。RMSNorm の組は gate の GELU（`GEGLU`）・後ろの norm（`RMSNORM_ADD`）・窓（`Params.window`）・局所の表もプロンプトと答えのステップの両方で GPU に置く（Gemma 3）。ほかの活性化（厳密な GELU）・LayerNorm の後ろの norm・Gemma 4 の余り（`Shape.extras`）は `REASONS.pair` で CPU。新しい事実を GPU の層に足すなら、計画（`contracts/gpu/protocol.ts` の `Started`）と `gpu/block.ts`、`gpuFit()` の両方の判定、`tests/gpu-hybrid-check.ts`・`gpu-part-check.ts`、gpu-check の作り物（`tests/gpu-check/models.ts`）。GPU だけに置けるのは `aloneFactsUnfit()` の通す組だけ。LayerNorm の eps は GPU・CPU・NumPy とも 1e-5 固定。
