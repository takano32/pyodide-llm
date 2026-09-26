# 開発機の用意

新しい開発機でこのリポジトリの作業を始めるための手順（T139、2026-09-26）。AGENTS.md と TODO.md と、この文書だけで始められるように書いてある。エージェントの記憶（`~/.claude/…/memory/`）は機械ごとで、移ると無くなるので写さない。持ち主から受けた決まりは AGENTS.md の「持ち主の指示」にある。

CI（`.github/workflows/deploy.yml`）が毎回やっていることと同じ手順なので、迷ったらそちらを見る。

## 1. 入れるもの

- **git と gh**: `gh auth login`（持ち主がする。push とワークフローの起動に使う）。git の `user.name` と `user.email` は持ち主のもの。
- **Node 24**: `.nvmrc` の版。nvm などで。`npm` が付いてくる。
- **Python 3**（3.14 で確かめた）と NumPy・pytest・tokenizers・regex: `pip install numpy pytest tokenizers regex`（deploy.yml と同じ）。ディストリビューションのパッケージでもよい。Ubuntu では venv に（下）。
- **wget**: `make models` がモデルを取る。
- **Claude Code**。

### Ubuntu 26.04 LTS（OCI の Ampere A1、arm64）の場合

**この一覧はまだ Ubuntu 26.04 で試していない**（2026-09-26 に書いた。新しい機械で最初に通したら、足りなかったもの・要らなかったものを直す）。

2026-09-26 に ssh で見た a1-free（Ubuntu 26.04.1、Neoverse-N1 × 2、11.9GB、スワップ 4GB、`/tmp` は tmpfs 5.9GB）には、git・gh（未ログイン）・Node v24.20.0（nvm）・Python 3.14.4・python3-numpy・python3-pytest・python3-venv・wget・curl・gcc・make がもう入っている。足りないのは tokenizers と regex の入った venv、参照の venv、Playwright のブラウザ、`gh auth login`。エージェントの記憶のフォルダは写さない（中身は 2026-09-26 に AGENTS.md と TODO.md に移した）。

```sh
sudo apt update
sudo apt install -y git gh curl wget ca-certificates build-essential \
    python3 python3-venv python3-dev
```

- `gh` は Ubuntu の universe にある。無い・古いときは GitHub の apt のリポジトリから（https://cli.github.com/）。
- `build-essential` は必須ではない（C の計測に使ったことがある程度。NumPy と tokenizers は arm64 の wheel が出ている）。
- **Python のパッケージ**: Ubuntu の Python はシステムに `pip install` させない（PEP 668）。apt で足りるものは apt で、足りないものだけ下の 2 つの方法で。

#### 決めた形（持ち主の判断、2026-09-26）: apt で足りるものは apt、`tokenizers` と `transformers` だけ venv

毎回の試験（pytest・smoke・forward-check・build）に要るものは 26.04 の apt にある（a1-free には入っている）。sentencepiece の突き合わせ（T126）の分だけ足す:

```sh
sudo apt install python3-numpy python3-pytest python3-regex python3-jinja2 python3-sentencepiece
```

`tokenizers`（pytest の本物との突き合わせ数件。無ければ skip）と `transformers`（`tests/format_check.py` だけ）は 26.04 の apt に無いので、system の site-packages を見る小さな venv に入れる（apt の numpy・jinja2・sentencepiece はそのまま使う。PyTorch は入らない、要らない）:

```sh
python3 -m venv --system-site-packages ~/venvs/reference
~/venvs/reference/bin/pip install tokenizers==0.23.1 transformers==5.12.1
~/venvs/reference/bin/python -m pytest tests -q        # 本物の tokenizers との突き合わせも走る
~/venvs/reference/bin/python tests/format_check.py ~/tmp/format-check hf-qwen3-0.6b
```

<details><summary>調べて採らなかった形: <code>tokenizers</code> を 26.10 からピン留めで（2026-09-26、a1-free で入れずに調べた）</summary>

- 26.10（`stonking`）の universe に `python3-tokenizers` 0.23.1 と `python3-transformers` 5.12.1 がある（Debian の forky・sid も同じ版）。
- `tokenizers` は 26.04 の Python のまま入れられる: 26.10 の Python も 3.14、中身は安定 ABI の `tokenizers.abi3-aarch64-linux-gnu.so`、glibc は 2.34 まで。展開しただけの 26.10 の `.deb` を 26.04 の Python 3.14 で import できた。優先度 100 のピン留めと `--no-install-recommends` なら、26.10 から来るのはそれ 1 つ。
- `transformers` は apt では入らない: PyTorch（`python3-torch`・`python3-torchvision`）に依存し、26.04 の上では 26.10 の依存を全部許しても解けない（torchvision → libtorch → openmpi → libucc → ROCm で食い違う）。
- 片方は venv が要るなら両方 venv のほうが単純、というのが持ち主の判断。

</details>

#### venv に全部入れる（apt を使わない機械）



```sh
python3 -m venv ~/venvs/dev
~/venvs/dev/bin/pip install numpy pytest tokenizers regex
# 以降は ~/venvs/dev/bin/python3 を使う（または source ~/venvs/dev/bin/activate）
```

- **Node 24 は nvm で**（`.nvmrc` の版を確実に取るため。Ubuntu の `nodejs` の版は未確認）:

```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
# 新しいシェルで
nvm install    # .nvmrc の版
```

- **ブラウザ**: `npx playwright-core install --with-deps chromium`（`--with-deps` が apt で依存のライブラリを入れる。sudo が要る）。Playwright が Ubuntu 26.04 を正式に扱うかは未確認（扱わなければ依存のライブラリの名前がずれて失敗することがある。そのときは表示された足りないライブラリを apt で）。
- **`systemd-run --user`** は、SSH でログインしたユーザーのセッションで使えるはず（未確認）。使えなければ `loginctl enable-linger $USER` を試す。

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
