### T.631 [修正][遠隔試験] gpu-tokens-check の SwiGLU の検査が、どの形も取らない回に落ちる — 状態: 完了（2026-10-31、T.594 の CI の run 41217766959 の軽い組で落ちた。規模 小）
- 落ち方: `tests/gpu-tokens-check.ts:299` が「Cannot read properties of null (reading 'map')」。手元でも 5 回に 1 回落ちた（Node 26）。
- 訳: `gpu/tokenforms.ts` の `chooseTokens()` は、形を確かめる前に `m.tokenForm = { form: null, ms: undefined, forms: null }` を置く。SwiGLU を 10% ずらすと DP4A の形も線を越える回があり（T.628.5 のデプロイでも見えた揺れ）、そのときは形が 1 つも取られずに投げる。試験は `m.tokenForm` の有無で枝を選んでいたので、`forms` が null のまま float の形の判定を読んだ。
- 直し: 枝の条件を `m.tokenForm?.forms` に（形が取られた回だけ判定を読み、取られなかった回は投げた文を読む）。製品は変えない。
- 確かめ: 直した `gpu-tokens-check` を手元で 12 回、どれも通った（直す前は 5 回に 1 回落ちた）。CI を省いた。訳: 触ったのは試験の 1 行で、手元で回る（GPU は偽の装置）。
- レビュー（Sonnet low、2026-10-31）: 条件を `m.tokenForm?.forms` にした直しは正しい。forms を非 null にするのは chooseTokens() の最後の 1 か所だけで、取られた回は float の形の判定を、取られなかった回は「no layer of a token is right … stream」の文を求める。壊れた GPU を通す抜け道は無く、隣に同じ形も残っていない。
