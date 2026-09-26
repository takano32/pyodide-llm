# 開発機の用意

新しい開発機でこのリポジトリの作業を始めるための手順（T139、2026-09-26）。AGENTS.md と TODO.md と、この文書だけで始められるように書いてある。エージェントの記憶（`~/.claude/…/memory/`）は機械ごとで、移ると無くなる。持ち主から受けた決まりは AGENTS.md の「持ち主の指示」にある。

CI（`.github/workflows/deploy.yml`）が毎回やっていることと同じ手順なので、迷ったらそちらを見る。

## 1. 入れるもの

- **git と gh**: `gh auth login`（持ち主がする。push とワークフローの起動に使う）。git の `user.name` と `user.email` は持ち主のもの。
- **Node 24**: `.nvmrc` の版。nvm などで。`npm` が付いてくる。
- **Python 3**（3.14 で確かめた）と NumPy・pytest・tokenizers・regex: `pip install numpy pytest tokenizers regex`（deploy.yml と同じ）。ディストリビューションのパッケージでもよい。
- **wget**: `make models` がモデルを取る。
- **Claude Code**。

## 2. リポジトリと生成物

```sh
git clone https://github.com/takano32/pyodide-llm.git
cd pyodide-llm
npm ci                 # AssemblyScript、Pyodide（Node 用）、playwright-core
make models kernels    # モデルの取得・変換・量子化と WASM SIMD カーネル（バイナリはコミットしない、方針 1）
npm install --no-save pyodide@latest   # smoke は最新の Pyodide で確かめる（ページと同じ）
```

`make models` は数百 MB を取り、数分かかる。`public/models/` と `*.bin` はコミットしない（`.gitignore`）。

## 3. 参照の道具（本物と突き合わせる検査だけ）

`tests/format_check.py`（書式）、sentencepiece の突き合わせ（T126）などは transformers と sentencepiece を使う。ページもデプロイも使わないので、venv に分ける。

```sh
python3 -m venv ~/venvs/reference
~/venvs/reference/bin/pip install -r tests/requirements-reference.txt
~/venvs/reference/bin/python tests/format_check.py ~/tmp/format-check hf-qwen3-0.6b
```

PyTorch は要らない（入れない。transformers は「PyTorch was not found」と言うだけで、語彙と書式には困らない）。

## 4. ブラウザ（手元で確かめるとき）

`npx playwright-core install chromium`（Firefox・WebKit も同じ形）。Linux では `--with-deps` で依存のライブラリも（root が要る）。**メモリを先に見る**: Pyodide とモデルのブラウザは 400MB 以上、1B 級で数 GB（`free -m`）。7〜8B（ヒープ 9.7〜11GB）は CI で。速さは手元で測らない（CI と持ち主の端末で。「開発機の値を既定にしない」）。

## 5. 機械のことを確かめる

AGENTS.md の落とし穴のいくつかは機械しだい。新しい機械では確かめて、AGENTS.md の該当の行を書き直す。

- `/tmp` が tmpfs（メモリ）か: `df -h /tmp`。tmpfs なら大きいもの（モデル、`dist` の写し）は `~/tmp` に置く。
- メモリの枠で計測を包めるか: `systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 true`。使えなければ `ulimit -v` で。
- スワップの有無、メモリの量、コアの数と種類（`lscpu`）: AGENTS.md の「計測環境」の行に。

## 6. 用意できたかの確かめ方

deploy.yml と同じものが全部通れば用意できている。

```sh
python3 -m pytest tests -q
for t in bench summary-check models-check ladder-check kept-check coi-js-check; do node tests/$t.mjs; done
node tests/smoke.mjs
node tests/forward-check.mjs --rounds 1 --positions 128
node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --plain
node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --wide
npm run build
```

新しい機械で初めて通したときは、かかった時間（pytest、smoke、forward-check の tok/s、`make kernels`）を AGENTS.md に書く。前の開発機（ARM の big.LITTLE、Cortex-A78 × 4 + A55 × 4、約 6.6GB）では pytest 約 20 秒、smoke 約 8 秒（Qwen3 の検査を足した後、2026-09-26）。
