# AGENTS

このリポジトリで作業する人と AI エージェントへの引き継ぎ書。会話の文脈なしで読めるように書いてある。
これからやる作業は [TASKS.md](TASKS.md) にある。

**ここに書くのは、次に手を入れる人が誤らないために知っていなければならないことだけ**: 決まり、いまの形、物を足す手順、落とし穴、設計の訳になっている決定と実測。済んだ回に何をしたか（数えた数・壊し方の数・run の ID・その回の形）は TASKS.md・`docs/tasks/T番号.md` と git の履歴に書く。

蒸留の記録: 2026-10-01（約 20 万字 → 14 万字）、2026-10-08（約 31 万字 → 14 万字）、2026-10-13（T500、約 42.7 万字 → 15.5 万字）、2026-10-24（T569、約 21.2 万字 → 13 万字。モデルごとの実測と、経緯の語りを畳んだ）、2026-11-12（T658、約 19.9 万字 → 約 9.6 万字。削らずに細部を `docs/notes/adding/`・`docs/notes/learned/`・`docs/notes/layout.md`・`docs/notes/tests-and-tools.md`・`docs/notes/owner-instructions-history.md`・`docs/notes/pitfalls.md` に移し、ここからそこを指す）。削った細かい数字と経緯は、蒸留の前の全文（5 回目の前は `git show 2fe042ad:AGENTS.md`、4 回目の前は `git show f1ee5020:AGENTS.md`、3 回目の前は `git show f5ed72d5:AGENTS.md`）と、`docs/notes/` のタスクごとの記録と、`docs/tasks/` にある。

**前の版の節の名前**: コードのコメントと過去の記録が「リファクタリングの決まり」とその中の「…とき」の項（「層の部品を作るときに選ぶ」「カーネルを登録簿から引くとき」「計画を値にして渡すとき」「失敗を捕まえて黙るとき」など）や、落とし穴の「形式を変えるとき」「家族を足すとき」を指していたら、それは「設計の決まり（いまの形）」と「物を足す手順」の同じ話の項。「現在の構成」の 3 つの試験の表（「試験の土台と、分け合う道具」「検査」「全部の組・CI・手で回す道具」）は「試験と道具」の 1 つの表に、「GGUF の型（取り込みの型）・…」は「GGUF の型・…」にした。ほかの節は同じ名前のまま。「物を足す手順」の各項はいまは索引で、コメントが指す項の中の番号（「モデルを一覧に足す」の 3 など）は `docs/notes/adding/` の同じ見出しの全文の番号。

## 作業の決まり

- 着手前にこのファイルの「方針」「設計の決まり」「落とし穴」「検証手順」を読む。細部は節ごとのリンク先を、触る所だけ読む。物を足すなら「物を足す手順」のその項を。新しい開発機では先に [docs/notes/dev-setup.md](docs/notes/dev-setup.md)。
- **役割と進め方**: (1) **実装はすべて Opus 5.5 の medium**（2026-10-01、持ち主）。(2) **レビューは別の会話の Sonnet 5.5 の high**（2026-11-12、持ち主「レビューは Sonnet high で行うようにしよう」。WebGPU の `shader-reviewer` も high。2026-10-09〜11-11 は WebGPU のほかは low。`.claude/agents/` の定義で固定。新規のサブエージェントは別の会話として数える）。(3) **直列に、同期的に**: タスクは 1 つずつ、実装 → 確かめ（手元か CI）→ レビュー → 直し → 本線 → デプロイ → 次（2026-10-08。サブエージェントを並べない）。レビュー担当には CI の run の ID とログか、手元の試験の結果を渡す。進め方は [docs/notes/review-by-opus.md](docs/notes/review-by-opus.md)、観点は TASKS.md の冒頭の 5 つ。**レビューだけのタスク**（T638 の形: タスクの差分でなく、本線のいまのコードを領域ごとに見て、リファクタリングの案をタスクの形で出す）は `refactor-reviewer`（Fable 5.1 の max）を領域ごとに並べて回す（2026-11-04、持ち主「新しくレビューだけのタスクを追加して、それに Fable max を使いたい」）。**Fable はタスクを書き出すだけで、リファクタリングはしない**（2026-11-04、持ち主「Fable はタスクを書き出して あとで Opus が処理する方針に変えたい」）: 製品のコードは変えず、報告は `.tmp/` に台帳の形で。案に番号を付けて台帳に積むのは取りまとめ役で、実装は後でふつうのタスクとして Opus 5.5 の medium が台帳の順に（上の (1)〜(3) のとおり）。タスクごとのレビューは上の (2) のまま。過去の記録の「レビュー（Opus xhigh / Fable / Sonnet max・low、…）」はその頃の形。
- **タスクの状態**は `未着手` → `進行中` → `反映済み`（本線に入れてデプロイした）→ 完了。「レビュー（Sonnet high、日付）: …」の 1 行が付いて本線に入っていれば完了。レビューの直しはそのタスクの番号のコミットで。**済んだタスクは `docs/tasks/T番号.md` に移す**（2026-10-15。子タスクは親のファイルに。状態を「完了」にしたら `python tests/ledger.py split` が移し、TASKS.md の頭の目次と `docs/tasks/README.md` を見出しから作り直す。目次は手で書かない: 軽い組の `ledger.py --check` がずれを落とす。古い 1 行ずつの一覧は `docs/tasks/done-list.md`）。見た目のタスク（`[描画]`）は `preview.yml` で画面を撮って持ち主が見る。
- **手元の試験で足りるなら CI を省く**（2026-10-10）。触った所を手元の試験（pytest・smoke・守りの網・軽い組・作り物と小さいモデル・実ブラウザの Chrome）が全部踏むなら、それで本線に入れる。**判断は書く**: 報告と TASKS.md に「CI を省いた。訳: …」か「CI を回した。訳: …は手元で見られない」の 1 行。CI が要るのは WebGPU の道、GB 級のモデルの枝、Firefox・WebKit だけの違い、1 トークン・1 ブロックの道の速さ、x86-64 だけのカーネルで、そのときも要る 1 本だけ。省いて入れたものが毎晩の組で落ちたら、その判断を誤りとして記録して線を直す。
- **自明な変更に CI を回さない**（2026-10-10）: 記録と文書だけ・コメント・手元で回せた道具の小さな直し・手元で通ったレビューの小さな直し。**1 つのブランチの CI は、実装が済んだ時に 1 回**。`models.yml` はページに届くものが変わったデプロイの後だけ。記録だけの push はデプロイも走らない（`deploy.yml` の `paths-ignore`: `*.md` と `docs/`）。
- **記録**: 新しく分かった決定・実測値・落とし穴・手順の変更は、**同じコミットでこのファイルに足す**（語りは TASKS.md に。ここは「いまの形」を書き直す形で、「T○○ から」の追記を重ねない）。git の履歴で分かることは書かない。推測で数値を書かない。測っていないものは「未計測」。
- **コミットはタスク（子タスクなら `T359.4: …`）ごとに分け、別のタスクの変更を混ぜない**。英語の命令形の件名。1 つのタスクを実装・検証・文書まで終えたらコミットして push。持ち主が「実装はまだしない」と言う間は TASKS.md だけをコミット。外向きで取り消しにくい操作（リポジトリ名の変更、公開 gist の作成など）は実行の前に持ち主に確認する。

### 持ち主の指示（エージェントの記憶から移した、T139）

エージェントの記憶は開発機にしか無く、機械を移ると無くなる（T139）。**記憶に頼らない**: 会話で持ち主から受けた決まりはここに、決定・実測・落とし穴はこのファイルの各節に、タスクの状態は TASKS.md に書く。終わった時限の決まり（2026-11-05〜11-06 の並行の回）・置き換えられた項（T188）・経緯と持ち主の言葉の全文は [docs/notes/owner-instructions-history.md](docs/notes/owner-instructions-history.md)。

- **持ち主とは日本語で話す**。訪問者に見える日本語は、ふだんの言葉で（技術者の略語を避ける。「Python が層を順に呼び」は却下、「処理は Python で書かれており」は採用）、飾りの形容詞を足さず、読点は 1 文に 1 つ程度。文面は実装の前に見せる。**返事は短く、日本語で、読点は少なめ**。持ち主の手を借りること（端末での確認、文面の選択）は最後にまとめる。
- **言葉**: T93 のソフトウェアの Worker は「**ソフトウェアスレッド**」と呼ぶ（「助手」は論理コアと取り違えられた）。「4 本」は取りまとめ役 + ソフトウェアスレッド 3 本で、ハードウェアは「論理コア」。予備の道は「**フォールバック**」と書く。タスクの中の 1 つずつは「**子タスク**」（T359 はタスク、T359.6 は子タスク。英語では task・subtask）。**「歩」を使わない**（2026-10-16、持ち主「歩じゃ通じないからね？」）: 1 トークンを書く 1 回は「（答えの）ステップ」、プロンプトをまとめて読む所は「プロンプトのブロック」。
- **速さよりきれいなコード**: 少し遅くてもコードをきれいに保ち、きれいな形の中から速いものを選ぶ。効かなかった最適化は外す。残る小さな遅れは隠さず書く。**速くなるなら、差が小さくても原則として入れる**（2026-09-27、持ち主「小さい値でも速くなったら原則的に入れよう」: どの端末でも遅くならず正しさが保てれば）。止めるのは、コードがとても複雑になるときだけ。
- **分けたことで付く多少の遅れは受け入れる。大事なのはメンテナンスできること**（2026-10-11、持ち主「分割してるんだから多少は遅くなっても普通でしょって言ってんの。メンテナンスできるようにしたいわけ」。読み込みと変換にも、推論にも当てる）: 部品に分ける・状態に持ち主を置く・型を付ける回は、形のよさ（単独に試験できる、足す物が自分のファイルで済む）で決める。速さは 1 回測って数字を報告と台帳に書くが、**1〜2% ほどの差や、対照の幅と重なる差のために、測り直し・原因探し・形の作り直しをしない**（「1〜2%」は取りまとめ役の読みで、持ち主の言葉は「多少」）。はっきり大きく遅くなったとき（対照の幅のずっと外）は、数字を持ち主に見せて決めてもらう（止まって待たずに先へ進む）。速くするためだけのタスク（カーネル、シェーダ）の決まりは変わらない。
- **開発機の値を既定にしない**: スレッドの本数などは端末ごとに実行時に測って選ぶ（forward.js の検索）。開発機の数字は AGENTS.md に計測として書くだけ。
- **子タスク・孫タスク**: 1 つずつ CI・レビュー・デプロイする単位に `T番号.子タスクの番号` を付け、状態を子タスクごとに書く（TASKS.md のタスクの頭の「子タスクの番号」）。子タスクの中に 1 つずつ確かめて本線に入れられる単位が 2 つ以上あれば孫タスク `T番号.子.孫`（例は T397.1.2）。それより深くは分けない（要るなら持ち主に聞く）。1 回のデプロイにまとめることに意味があるもの（T369）は分けない。**分けるかどうかは持ち主を待たずに決める**（取りかかる時に自分で決めて台帳に書き、後の報告で伝える）。**子タスクが 1 つしか無いなら子の番号を採らない**。**調べで出た案は、調べの子タスクにしない**（作るものが別なら自分の番号）。孫タスクも、状態・コミットの件名・ブランチは自分の番号で。設計のタスクは、出した案が次のタスクの子タスクを決める。
- **番号は `T番号`・`T番号.子`・`T番号.子.孫` と書く**（2026-11-05、T646）。**数字と点のほかの字を使わない**（2026-10-10、持ち主「なんで勝手に数字以外を使ったの？」）。台帳・コミットの件名・ブランチ名（`t646-…`）・コメント・`docs/tasks/T番号.md` のどれも。T646 で全部の番号を書き直し、T647 で履歴を書き換えた（古い SHA と新しい SHA の対応は `docs/notes/t647-commit-map-2026-11-05.txt`）。数字の T でない語（モデルの家族の T5、NVIDIA の T4、古い記録の中の検査の札 `T1`〜`T3`）は番号ではない。
- **前にやるべきことは採番する**: あるタスクの前に要る仕事は、それぞれ T 番号を付けて台帳に書き、取りかかる順を台帳の頭に書く（寄せ集めのタスクにしない）。
- **ブランチで試さず、出して戻す**: 本番の前にブランチで試す形はとらない（持ち主はブランチをスマホで試せない）。危ない変更は URL の旗（`?offline=on` など）の後ろに置いて本線に入れ、デプロイして `models.yml` で本番を確かめ、持ち主の端末で問題が出たら戻す。戻しが 1 つで済むようにコミットはタスクごと。**例外は見た目**（`preview.yml` はブランチを撮る）。
- **統合したブランチは消す**: 本線に入れたら、リモートとローカルのブランチを消す（`git push origin --delete <branch>`、`git branch -D <branch>`）。
- **持ち主が決めたら、蒸し返さずにすぐ従う**。ただし取り消しやすくしておく（決めるのは持ち主で、数字は材料として出す）。意見を聞かれたら先に考えを出し、持ち主が「実装はまだしない」と言う間は台帳だけを直す。
- **直列に戻す**（2026-11-06 の夕方、持ち主「減りが速いので直列にしてね」: 使える量の減りが速い）: 「役割と進め方」の (3) のとおりタスクを 1 つずつ（実装 → 確かめ → レビュー → 本線 → デプロイ → 次）。実装とレビューの担当を同時に何本も起こさない。走っている CI は待ってよい。ブランチ名の変更のように、ブランチが 1 本でないと事故る操作の前には、ほかの作業を全部本線に入れてブランチを消す。
- **a1-free ではメモリを遠慮せずに使う**（2026-09-26、持ち主「メモリたくさんあるから活用せよ」）: 重いものも、見込みの峰が `free -m` の available に収まれば回す（速さを測るものだけは、ほかが CPU を使っていないときに）。
- **`/benchmark/` の文面は詳しい説明でよい**（2026-09-27）: 訪問者向けのモデルのページと違い、計測のページは何を測ったか・どう読むかを省かずに書く。言い切れないことは言い切らない（T173 の案 A）。
- **最適にする的は持ち主の Android**（2026-09-27、持ち主「さっきの Android で最適になるようなのがいいな、わたしの端末だ」）: Xiaomi 13T Pro（Chrome、Arm Valhall の GPU、8 論理コア）。端末で測って選ぶ仕組み（T148 など）はそのままにし、決めごとで迷ったらこの端末で速いほうを取る。この端末の数字は /benchmark/ の報告（TASKS.md の T190・T191）。
- **タスクの見出しは `### T番号 [種類][分野] 題`**（2026-09-28）。分野は WebGPU・CPU・計測実行（ベンチ）・変換（取得と変換と GGUF）・解析分割（トークナイザ）・遠隔試験（CI）・文書・モデル・Bonsai・MLX（MLX の形で配られるモデルを動かす仕事）・Gemma4・Neohorse・Agents A1・Spark X・Saluki（その系統のモデルを動かす仕事）・その他（1 件しかない分野）。新しいタスクもこのどれかを付け、合わなければ持ち主に聞く。1 つの見出しに分野を 2 つ付けてよい（T232 の [Bonsai][WebGPU]）。
- **llama.cpp の中身はなるべく Python に取り込み、速さに関わる所だけカーネルに。使えるモデルが狭まるなら TypeScript で妥協する**（2026-10-13）: 読み方・変換・トークナイザ・書式・標本抽出の決まり・モデルの形は Python で書く（形と式を取る。行は写さない）。速さに関わる計算は自前の wasm のカーネル。**層を回す所を Python に戻すと 1 コア・4 GB までのモデルに限られる**（Pyodide の中のメモリはスレッドで分け合えず 32 ビット）ので、そこは今の TypeScript の forward のままでよい。llama.cpp そのものを wasm にして呼ぶ形は取らない（T504 の読み）。
- **なるべく Python、次に TypeScript。JavaScript はなるべく使わない**（2026-10-31、持ち主「JavaScript なくしてなるべく Python にするんだよ」）: (1) **これから書く試験と道具は Python を本体に**し、ブラウザか Node で走らせるしかない所（製品の JavaScript を走らせる台、Pyodide を呼ぶ橋）は **TypeScript（`.ts`）** で書く。`.mjs` で新しく書かない（Node 24 も bun も `.ts` をそのまま走らせる: 消すだけの構文で、`enum` などは書かない）。例: 計算は `tests/gpu_answers.py`、橋は `tests/fakes/gpu-answers.ts`。(2) **製品も、使えるモデルを狭めずに Python に移せる所は移す**（調べは T628）。製品の `src/` はもう TypeScript（T480）で、`.js` は `public/coi.js` だけ（束ねない Service Worker: `// @ts-check` のまま）。(3) **Python に移さないのは、移すとモデルが狭まる所と、ブラウザでしか動かない所**: 1 トークンの forward（層の輪とスレッド）、WebGPU の Worker とシェーダ、ページの画面と Worker の始まり、wasm のカーネル（AssemblyScript）。迷ったら、狭まるかどうかを測ってから決める。
- **文面と方針の判断は最後にまとめて、途中で要る判断は選ばせる**（2026-10-13）: 訪問者に見える文面と、急がない方針の判断は TASKS.md に積んで、作業の区切りにまとめて 1 回で出す。作業を先へ進めるのに要る判断は、選択肢（2〜4 個、勧めを先頭に、それぞれ何が起きるかを 1 行）で持ち主に選んでもらう。答えを待つ間は、その判断に依らない所を進める。
- **重要な質問のほかは、聞かずに自動で進める**（2026-10-15、持ち主「重要な質問以外は自動で進めてほしいなあ」。上の行をこれで絞る）。**取りまとめ役の読み**: 勧めの案があり、取り消しやすく、訪問者への害が小さい判断（実装の形、試験の線、置き場、今ある文の使い回し、勧めのある小さな文面の追加、標本抽出の既定、子タスクの順など）は、勧めの案で進めて TASKS.md に「取りまとめ役の決め（訳）」と書き、区切りのまとめで一覧にして見せる。**聞くのは重要なものだけ**: 取り消しにくい・外向きの操作、ライセンスと法の話、方針の番号（方針 1〜10）に触る決め、訪問者に大きな費用か害が出る決め（取得やメモリが大きく増える、既定のモデルが変わる、今あるモデルの答えが変わる）、勧めの案が立てられないもの。迷ったら進めて、まとめで見せる側に倒す。
- **不明点が無ければ、タスクが尽きるまで続けて作業する**（2026-10-24、持ち主「不明点がなければタスクが尽きるまで作業する、また、これを基本的なルールとして追記」）。1 つのタスクを本線に入れてデプロイしたら、報告して止まらずに台帳の頭の順で次に取りかかる。止まるのは、上の行の「重要な質問」に当たるとき・台帳に積んだタスクが尽きたときだけ。区切りの報告は短く書いてそのまま続ける。
- **タスクがなくなるまで、止まらずに次のタスクに着手し続ける**（2026-11-12、持ち主「無限にタスクがなくなるまでタスクに着手するようにしてね」「タスクなくなったの？」: 取りまとめ役が列を絞りすぎて、手が空いているのに次を起こさなかった）。列が 1 つ空いたら、その応答の中で台帳から次を起こす。台帳の頭の順が待ちで詰まったら（前のタスクと同じ所に触る・CI を待っている）、その下の未着手・進行中の残りから、いま起こせるものを探して起こす。持ち主の判断が要るタスクは飛ばして区切りのまとめに積み、それで止まらない。「いま起こせるものが無い」と言うのは、台帳の未着手と進行中を全部見て、どれも判断待ちか走っているタスクとぶつかると確かめた後だけ。
- **並べるのは、安全に並べられるタスクだけ**（2026-11-12、持ち主「安全に並列に進められるタスクだけ並列に進める」「レビューが詰まってるあいだ、安全だから並列数増やせそうな気がするけど、無理なの?無理はしなくていい」。下の「直列に戻す」をこれで緩める）: 触るファイルと表が重ならないタスクを、担当ごとに別のワークツリーで。同じ表に行を足すタスク（家族と家族、一覧と一覧）は前のものが本線に入ってから。レビューは実装の列と並べてよい。本数は開発機の CPU（2 コア）とメモリで決める（取りまとめ役の読み: CI の待ちが中心の担当は数えず、手元で重い試験を回す担当は 2〜3 本まで）。使える量の減りが速いと持ち主が言ったら直列に戻す。**2026-11-12 の夜に直列に戻した**（持ち主「上限が心配だから順次、直列にしよう」: 並べて 8 本まで増やした後）。いまは下の「直列に戻す」のとおり 1 つずつで、並べるのは持ち主がまた言ったときだけ。戻すときは、走っている担当のうち始めたばかりのものを止め（後で続きから起こせる）、深く進んだものは終わらせてから 1 本にする。
- **「次に取りかかる」と書いたら、同じ応答の中で次を起こす**（2026-11-06、持ち主「なんで取り掛からなかったの？」）: 区切りの報告は、次の担当を起こす呼び出しか CI の待ちと同じ応答で出す。起こすものが無い（台帳が尽きた・持ち主の答えを待つ）ときだけ、そう書いて止まる。
- **手元の道具は bun でも回す。CI とデプロイは Node 24 のまま**（2026-10-25、方針 3 は変えない）。軽い組は `RUNTIME=bun … bash tests/suite.sh light`（bun で回せない試験は `suite.sh` の `BUN_LEAVES` が訳つきで外す）。道具を書くときは両方で走る形に: `node:module` の `stripTypeScriptTypes` は `tests/untyped.ts` の `blanked()` で、部品をまっさらに読み直すのは `tests/fresh.ts` で（bun は `file:` の URL の問いを落として同じモジュールを返すので、bun では絶対の道に問いを付ける）、`readdirSync` の順に頼らない、`process.execPath --check` を使わない。**bun の未対応を見つけたら上流（oven-sh/bun）に PR を出すのが目標**: 置き場は `~/GitHub/bun`（`origin` が上流、`my` がフォーク）。**手元では建てない**（bun のビルドは clang と rustc の LLVM の大きい版が同じであることを求め、手元の Rust は LLVM 23）。PR は再現と issue の文から。違いの一覧と建て方は docs/notes/t584-bun-2026-10-25.md、上流に出す候補は TASKS.md の T585。
- **持ち主の端末での計測は、実装が全部終わってからまとめて 1 回**（2026-10-01、持ち主「…そんなに何回も計測頼めないよ？」）。タスクごとに「この端末で押してください」と頼まない。端末でしか分からないことは TASKS.md の「持ち主の端末でまとめて見るもの」の 1 つの一覧に積み、実装は端末の数字を待たずに先へ進む（CI と式で決められる所まで決め、覆す条件を書く）。一覧を頼むのは、並んでいる実装が本線に入り終わったときに、端末ごとに 1 回で済む手順にまとめて。
- **手元でやることと CI でやること（2026-10-09）。手元で軽く確かめてから CI に出す。**
  - **手元でやる**: 編集、`node --check`・`bash -n`、pytest（`.venv`）、smoke、`npm run build`、作り物と小さいモデルの試験（forward-check・threads-check・worker の試験・守りの網など）。**ページや Worker を変えたら、実ブラウザで通しで見る**: ビルドしたページを Playwright の Chromium（`tests/e2e.ts`）か **Chrome DevTools MCP**（持ち主「Chrome DevTools MCP も活用してね」）で開き、準備完了・答え・止めるボタン・設定・モデルの切り替え・コンソールのエラーを見る。見た目は前後の画面を撮って比べる。モデルはサイトの小さいもの（stories260K・tiny-lm・llm-jp-3 150M まで）で、先に `free -m` を見る。
  - **CI だけ**: WebGPU（lavapipe・SwiftShader・Dawn を手元で回さない）、GB 級のモデル、Firefox と WebKit、速さの判断（手元の数字で決めない）、全部の組。
  - **サブエージェントへの頼みには、この 2 つをここから写す**（記憶から書かない: 2026-10-09 に要約から「手元にブラウザは無い」と書き、頼み 8 本に写し続けた）。頼みに書く決まりは、ほかも AGENTS.md の文から写す: 待ち方（run の ID と期限、`pgrep -f`・`pkill -f` を使わない、`run_in_background` の中で `&` を使わない）、作業ファイルの置き場（リポジトリの `.tmp/`、`$HOME` と `/tmp` に置かない）、本線に触らないこと。
- **WebGPU の試験は CI で回し、ログから読む**（2026-09-27、持ち主「…必要な情報は CI で細かいログ出してそっちから解析しない？」）。開発機（a1-free、ARM の 2 コア）で lavapipe・SwiftShader・Dawn の gpu-check や bench-check の GPU の節、実物のモデルの GPU の確かめを回さない。WebGPU の確かめはブランチを push して `gpu-prompt.yml`（Dawn と SwiftShader）・`bench.yml`・`preview.yml` を CI で回し、判断に要る細かい数字（層ごとの誤差、形ごとの線と比、段ごとの秒）を CI のログに出して、そこから読む。
- **`$HOME` に黙ってファイルを作らない**（2026-09-26、持ち主「$HOME には無言でファイル作らんでほしいなあ」）。作業ファイルはリポジトリの中の `.tmp/`、venv は `.venv/`（どちらも `.gitignore` にある）。pytest の `tmp_path` も `.tmp/` に（`tests/conftest.py` が `PYTEST_DEBUG_TEMPROOT` を向ける）。`$HOME` の下に何か要るとき（道具が入れるキャッシュなど）は先に言う。`~/tmp` は持ち主がほかでも使うので、使ったら自分の作ったものだけを消す。サブエージェントへの頼みにも書く。
- **失敗で持ち主に迷惑をかけたら**（2026-09-20、`/tmp` に大きいものを写して機械を落とした）、謝るのは 1 回、作業ツリーの状態を正確に伝え、続ける前に確認する。
- 「c,p」「c.p」はコミットして push の意味。作業ツリーに 2 つのタスクの変更が混ざったら、まとめてコミットせず、タスクごとの中身をステージしてから（例: そのタスクだけの版のファイルを作って `git add`）コミットする。

## プロジェクトの目的と方針

ブラウザ上の WebAssembly 版 Python（Pyodide）だけで言語モデルを動かす**実験**。実用のエンジンではなく、「WASM Python でどこまでできるか」を測って公開することに価値を置く。JavaScript + WASM 版は別プロジェクトで作る（ここではやらない）。

決定済みの方針（変えるときは持ち主に確認する）:

1. **バイナリをリポジトリに入れない。** モデルのダウンロード・変換・量子化・分割はすべてビルド時（`make models`）に行う。
2. **Pyodide は常に最新版**を実行時に解決して読み込む（`src/runtime/worker/pyodide.ts`）。手で上げるバージョン定数は置かない。CDN が落ちていたら動かなくてよい（信頼性のための仕組みは足さない）。Service Worker（`public/coi.js`）は COOP/COEP を足すために置き、登録に失敗しても 1 コアで動く（T93）。**オフラインでも使えるように、その Service Worker で Pyodide と NumPy の応答を保持する**（T111、既定。`?offline=off` で外せる。版が URL に入ったファイルは写しがあればネットに聞かない）。
3. **Node は 24 LTS**（`.nvmrc`）。ページは Astro、チャット風 UI。
4. **既定モデルは Shisa V2.1c LFM2 350M**（2026-10-15、持ち主が T511.5 の比べを見て決めた: 問いに答える小さいモデルのうち事実の当たりがいちばん多い。初めての訪問は 379 MB を Hugging Face から取って変換する。T511.6）。**置き場は `src/models/listed.ts` の `DEFAULT_MODEL` の 1 か所**（窓口 `src/models.ts` が出す。一覧に無い id なら読み込みの時に投げる）（ページの始まり・覚えたものも URL も無い訪問・保存の消えた大きいモデルの代わり・「Other repository…」から戻す値が読む。`models-check` が既定の id と 500 MB 以下を見る。e2e はモデル ID `default` で名指さない訪問を開く）。一覧の並び（T128）は変えず、tiny-lm は一覧の先頭のまま（イントロのボタンが選ぶ）。選んだモデルは次回も覚えている（`localStorage`。`?model=` などの URL が優先。HF から 500MB 超を取り直すモデルは無確認では始めない: 線は `src/models/memory.ts` の `UNASKED_BYTES`、聞くかは `asksBefore()` の 1 か所で、ページの選ぶ所・覚えたモデル・保存の確かめが読む。T643.6）。既定は llm-jp → tiny-lm → llm-jp → tiny-lm → Shisa と、質と軽さの間で変わってきた。**一覧の並び**: `MODELS` を作るときに並べ替えて決める（ファイルの中の順ではない）: 各グループの中は「日本語が使える（項目の `japanese`: note の「日本語」と食い違えば `models-check` が落とす）軽い → 重い、そのあと英語だけの軽い → 重い」（T128、大きさは `modelBytes()`。`tests/models-check.ts` が順を見る）。英語だけのモデルは、はじめは全部見せて、あとで減らす（持ち主の判断）。
5. 大きいモデルは **int8 で配布**し、量子化前の原本も選べるようにする。**int8 より小さい重みは品質が 3〜5% 落ちても採る**（T98。4 ビットは届かず 6 ビットに決まった。ただし**コードをきれいに保つことを速さより優先する**）。**既定値でモデルを手加減しない**（T56）: 生成の長さの既定はコンテキストいっぱい（`steps: 0`）、コンテキストはモデル本来の値（ページの上限 4096 まで）。
6. GitHub Pages（静的ホスティング、HTTP ヘッダ変更不可）で動くこと。**スレッドは Service Worker で取れたときだけ使い、取れない端末でも同じ速さの 1 コア版で動く**（T93）。**Safari で動かない機能も足してよい**（T101 の Memory64: 4GB を超えるモデルは Chrome と Firefox だけ）。
7. コミットは関心ごとに分け、英語の命令形の件名。main に push するとデプロイされる（2026-09-26 までの既定のブランチは master: T140）。
8. **ライセンスで一覧から外さない**（2026-09-25）。非商用のモデルも、原本がゲート付きのモデルの写しも入れる。ライセンス名はモデルカードから写す（`tests/models-check.ts`）。
9. **既定で GPU をなるべく使う**（2026-09-26）: WebGPU がある端末では、オプションなしで GPU を使う。端末ごとに測って、GPU が CPU より遅ければ自動で CPU に戻す。**オプションで出し入れする形にしない**。**測って選ぶのは重みを 2 重に持てるモデルだけで、2 重に入らない大きいモデルは読み込みの前に GPU だけと決め、CPU の側は `/benchmark/` の CPU の節の読みの速さから見積もる**（2026-09-27、T156 の案 B）。
10. **ライセンスは MPL-2.0**（2026-09-27、T174）。ほかのプロジェクトのコードを写すときは、そのファイルのコメントに元の著作権の行と許諾文の全文を残す。GPL などのコピーレフトの行は写さない（形だけを取る、T168 の clpeak）。

## 現在の構成

場所の決まり: Python のソースは `src/python/`、実行時の TypeScript（worker・forward・GPU・シェーダ・ベンチの Worker）は `src/runtime/`（束ねられる）、ページのスクリプトは `src/page/`・`src/benchmark/`、作る物（カーネルと Python の zip）は `src/runtime/built/`（`.gitignore`）、`public/` はそのまま配る物（`coi.js`・`coi-test/`・`models/`）だけ。窓口（`llama2_numpy.py`・`llama2_convert.py`・`worker.ts`・`forward.ts`・`shaders.ts`・`gpu.ts`・`benchmark/gpu.ts`・`src/models.ts`・`src/bench.ts`）は部品を読み込んで名前を出すだけで、中身は隣のフォルダにある。**道具は場所を書かず `tests/tree.ts`・`tests/tree.py` に聞く**。

### 製品

1 行 1 つの役割。部品の一覧と中身の細部は [docs/notes/layout.md](docs/notes/layout.md)（表の全文）。

| ファイル | 役割 |
|---|---|
| `src/python/llama2_numpy.py` と `src/python/engine/` | 推論エンジン（NumPy の参照と、ページの forward への橋）。窓口は外が読む名前だけを出す。llama2.c の legacy 形式（7 個の int の見出し + テンソル）を読む。dtype は float32・float16・int8・int6・ternary・binary。部品は下から `proxy`・`packing`・`dtypes`・`layout`・`tokenizer`・`layers`・`kernels`・`checkpoint`・`tensors`・`sampler`・`plan`・`external`・`generation`・`model` |
| `src/python/engine/marian.py`・`translation.py`・`translating.py` | Firefox Translations（Marian）の読み手 / 翻訳を NumPy とカーネルで（fxtranslate と同じ数をビット単位で） / ページの worker が持つ形（`Translation`） |
| `src/python/llama2_convert.py` と `src/python/convert/` | HF のチェックポイント → legacy 形式と tokenizer.bin（ビルドとブラウザの両方）。部品は `checkpoint`（`Writer`）・`template`・`readers`（`SOURCES`）・`superblocks`・`grids`・`sources`・`config`・`plan`・`families/`（`FAMILIES`）・`stream`・`gguf`・`tokenizer`・`vocabulary`・`beginning`・`conversion`・`conduct`（変換の段取りのジェネレータ）。**窓口に書いた値は部品に届かない**（試験が差し替える `KV_START`・`Tokenizer` は `engine.model` に書く） |
| `python_archive.py` | Pyodide に置く Python を zip 2 つに（`engine.zip`・`converter.zip`。同じソースからは同じバイト） |
| `convert_hf.py`・`quantize.py`・`fetch_translations.py`・`Makefile` | ビルドの入口。`make models`・`make kernels`・`make python`・`make run`。モデルを作り直す依存は `MODEL_PYTHON` |
| `kernels/` | WASM SIMD カーネル（AssemblyScript）。`blocks.ts`・`blocks_relaxed.ts` は別のモジュール（llama.cpp の塊）。制約と実測は `kernels/README.md` |
| `src/runtime/worker.ts` と `worker/` | モデルの Web Worker。持ち主の `runtime.ts`（Pyodide・カーネル・forward・URL の旗・WebGPU）・`request.ts`・`loaded.ts`・`pool.ts`・`alone.ts`・`weights.ts` と、`pyodide.ts`・`sources.ts`（llama2.c のチェックポイントの取り込み元の表 `SOURCES`）・`ranges.ts`（HF の Range）・`conduct.ts`（段取りに答える輪と答え手）・`convert.ts`・`load.ts`・`translate.ts`・`timing.ts`・`clock.ts`・`outdated.ts` |
| `src/runtime/forward.ts` と `forward/` | 1 トークンの forward（JavaScript が自前の `WebAssembly.Memory` の上でカーネルを呼ぶ）。`engine.ts`（`createForward()`）・`tensors.ts`・`blocks.ts`・`staging.ts`・層の種類の表 `kinds.ts`（`LAYER_KINDS`）と `attention.ts`・`linear.ts`・`convolution.ts`・`ffn.ts`・`cache.ts`（KV）・`perlayer.ts`・`control.ts`・`threads.ts`・`gpufit.ts`・`gpuside.ts`・`choice.ts`・`paths.ts`・`sizes.ts`・`architectures.ts`・`memory.ts`（`footprint()`）・`alone.ts`・`reasons.ts` |
| `src/runtime/jobs.ts`・`helper.ts` | 制御領域の語・`ADDRESSES`・`WORK`・`PACKED`・`runner()` / ソフトウェアスレッド |
| `src/runtime/gpu.ts` と `gpu/` | モデルのページの GPU の Worker。`device.ts`・`weights.ts`・`forms.ts`・`block.ts`（プロンプトのブロック）・`tokenattention.ts`・`tokens.ts`・`tokenforms.ts`（生成のステップ）・`reference.ts`（読み込みの検査の JavaScript の参照: 装置を触らない）・`requests.ts`・`start.ts`、層の種類の表 `kinds.ts`（`GPU_KINDS`）、詰め方ごとの行の形の表 `packings.ts`（`PACKINGS`） |
| `src/runtime/shaders.ts` と `shaders/` | WGSL。エンジンの `common`・`prompt`・`superblocks`・`steps`・`attention`（attention の検査の作り物と JavaScript の答えも）・`fused`・`run`・`sample`・`rotate`、目録 `engine.ts`（`ENGINE_SHADERS`・`TERNARY_SHADERS`・`ENGINE`）、端末の鍵 `key.ts`、ベンチだけの `bench`・`matvec`・`stages`、CPU の標本抽出の模型 `likecpu.ts`、float16 の読み書き `halves.ts`（GPU の Worker とシェーダの部品はこれだけを読む） |
| `src/runtime/contracts/` | 境目の型だけ: `plan.ts`・`messages.ts`・`gpu/protocol.ts`・`form.ts`・`conduct.ts`・`python.ts`・`benchmark.ts` |
| `src/runtime/failure.ts`・`updated.ts`・`built.ts`・`kept.ts` | 失敗の運び方 / 「サイトが更新された」の判定と文 / 作る物の表 `BUILT` / 変換したモデルの保存（OPFS、無ければ Cache API。置き場は同じ形 `Place` の 2 つの実装 `kept/opfs.ts`・`kept/cache.ts`） |
| `src/pages/index.astro` と `src/page/` | チャット風のページ。持ち主の `model.ts`・`answer.ts`、純な部品（Node が import できる）`remaining.ts`・`repository.ts`・`thought.ts`・`folder.ts`・`heard.ts`・`fields.ts`（生成の設定の表 `LIMITS`・`FIELDS`）・`parameters.ts`（URL の名前の表 `NAMES`）・`choosing.ts`・`lines.ts`（ステータス行などの言葉の全部: `draw.ts`・`receiver.ts` は描くだけ）、ほかに `dom`・`markdown`・`address`・`remembered`・`opening`・`list`・`settings`・`draw`・`isolation`・`receiver`・`choose`・`composer`、最後に最初の読み込みの `first` |
| `src/pages/benchmark.astro`・`src/benchmark/`・`src/runtime/benchmark/` | `/benchmark/`（`?run=all`・`?model=`・`?size=`・`?large`・`?full`） |
| `src/models.ts` と `src/models/` | モデル一覧。窓口は名前を出すだけで、組み立て・並べ替え・`DEFAULT_MODEL`・`sources()` は `listed.ts`。`formats.ts`・`sampling.ts`・`builders.ts`（項目の型 `Entry = Site | Remote | Translating | Linked`）・`filled.ts`・`licenses.ts`・`memory.ts` と一覧の `built.ts`・`translations.ts`・`hf-japanese.ts`・`hf-japanese-large.ts`・`hf-classics.ts`・`hf-families.ts` |
| `src/bench.ts` と `src/bench/` | ベンチの報告の組み立て（節の表と判定。ページの部品は聞いて描くだけ） |
| `public/coi.js` | サイト全体の Service Worker。**触る前に落とし穴の「Service Worker」の項を読む** |
| `astro.config.mjs`・`tsconfig.*.json`・`package.json` | 束の設定 / 型検査の 3 つのプログラム / `npm run build`・`dev`・`types` |
| `README.md` と `docs/` | 訪問者向け（英語）。日本語の内部の文書は `docs/notes/`、済んだタスクは `docs/tasks/` |

### 試験と道具

**探し方**: 検査は `tests/` に、名前が中身を言う（`<部品>-check.ts`・`<部品>-parts-check.ts`・`<部品>-modules-check.ts`・`test_<部品>.py`）。軽い組と全部の組の中身は `tests/suite.sh` の `part` の行が一覧。ここには、知らないと使い方を誤るものだけを書く。検査ごとの細部は [docs/notes/tests-and-tools.md](docs/notes/tests-and-tools.md)（表の全文）。

| ファイル | 役割 |
|---|---|
| `tests/suite.sh` | `light`（デプロイと `tests.yml` の既定。build と `dist-check` はワークフローがその後に）と `full`（毎晩と `full=true`）。`node` は全部 `tests/limited.ts`（期限）越し。手元では `PATH=<リポジトリ>/.venv/bin:$PATH bash tests/suite.sh light` |
| `tests/unchanged.ts` | 守りの網（「設計の決まり」）。相手の木は `tests/other-tree.ts` の `otherTree()` |
| `tests/tree.ts`・`tree.py`・`tree.json` | 木のどこに何があるかの 1 か所（`runtime()`・`python()`・`built()`・`served()`、在るかを聞くなら `{ maybe: true }`）。`x.js` と頼んでも `x.ts` が見つかる |
| ソースを文として読む道具 | `imports.ts`（`windowed()`）・`names.ts`・`untyped.ts`・`contracts.ts`・`worker-source.ts`・`shaders-source.ts`（`oneSource()`）・`shader-makers.ts` |
| 台と代役 | `worker-harness.ts`（worker を Node の vm に。`run()` は strict）・`engine.ts`（ページと同じエンジン `kernel_llama()`）・`plans.ts`（素の Python の `forward_plan()`: `planOf()`）・`bundle.ts`（`ENGINE_FROM=dist`）・`tests/fakes/`（2 つ以上の検査が使う偽物）・`fresh.ts`・`bun-preload.ts` |
| 作り物 | `made_up.py`（作り物のモデルの 1 か所）・`made_up_tokenizers.py`・`make_*.py`・`engine_plans.py`・`plan_facts.py`（事実の食い違う作り物）・`conduct_hub.py` |
| 取得・段取り・ブラウザ | `fetching.py`・`fetching.ts`（道具の取得の 1 か所）・`gguf_fetch.py`・`hf_fetch.py` / `conducting.py`・`conducting.ts` / `serving.ts`（`dist/` を `/pyodide-llm/` の下で）・`leave.ts`（`scratch()`・`closed()`・`leave()`） |
| `tests/test_*.py`（`conftest.py`） | エンジンと変換器。順の写し `test_layout.py` の `ORDER`、事実の写し `test_architectures.py`・`test_families.py` の `FACTS` |
| worker の検査 | `worker-check`・`worker-memory-check`・`worker-sink-check`（`FORM` が変換器から `footprint()` まで届くこと）・`worker-parts-check`・`worker-init-check`・`worker-conduct-check`・`worker-fetches-check`（変換が頼むものの列: わざと変えたら `--write` して差分を読む）・`worker-outdated-check` |
| forward の部品の検査 | `threads-part-check`・`gpu-part-check`・`gpu-alone-check`・`layer-parts-check`・`kernel-registry-check`（`WORK`・`PACKED` を Python の `DTYPES` と）・`sizes-check`・`plan-facts-check`・`plan-keys-check`（計画と GPU の計画の鍵を両端で）・`memory-check`・`frame-check`・`thread-search-check`・`paths-check`・`finite-check`・`unfinite-check` |
| GPU の Worker とシェーダの検査 | `gpu-worker-parts-check`（偽の装置）・`gpu-forms-check`・`gpu-tokens-check`・`gpu-start-check`（答えの計算は `tests/gpu_answers.py`）・`gpu-reference-check`・`gpu-choice-check`・`gpu-default-check`（`--quick` は軽い組）・`gpu-hybrid-check`・`layer-check`・`shaders-catalogue-check`・`device-key-check`・`shaders-modules-check` |
| 失敗・ページ・道具の検査 | `failure-check`・`reasons-check`（`SAID`）・`silent-check`（`SILENT`）・`updated-check`（`LATER`）/ `page-parts-check`・`models-check`・`bos-check`・`bench.ts`・`kept-check`・`coi-js-check`・`python-archive-check`・`contracts-check` / `tree-check`・`imports-check`・`fakes-check`・`fetching-check`・`serving-check`・`conducting-check`・`workflows-check`・`dist-rules-check`・`gpu-check-dry-check`・`reference_dry.py`（CI だけの道具を代役で端から端まで） |
| 回線の要る軽い組の検査 | `markdown-check.ts`（最新の marked を jsDelivr から）・`repository-check.ts` |
| 塊の型の試験 | `test_blocks.py`・`saluki_form.py`・`test_kept.py`・`test_widened.py` / `tests/blocks-forward-check.ts`・`blocks-check.ts`（`blocks-model.ts`） |
| Saluki の道具 | `reference_saluki.sh`・`page_saluki.sh`（段は各スクリプトの頭）・`saluki-kernel-bench.ts`（塊のカーネルの速さを測る唯一の道具） |
| `tests/ci.ts`・`summary.ts`・`reports.ts` | **CI を待つのはこれで、いつも裏で**: `run <ワークフロー> key=value… [--ref <ブランチ>]`・`wait <ID …>`・`deploy`・`nightly`（`--minutes`・`--grep`。終わり方 0 成功・1 失敗・2 期限・3 待つ run が無い・4 ほかが待っている）/ 結果の表 / 訪問者のベンチの Issue |
| `tests/smoke.ts` | Node の最新 Pyodide でエンジン・カーネル・モデル（約 10 秒）。Pyodide の 32 ビットの NumPy でしか出ない誤り |
| `tests/types.ts` | 型検査と、tsc が言わない規則（`AMBIENT`・`ANY_MOST`） |
| `tests/forward-check.ts`・`threads-check.ts` | forward を NumPy と / 本数を変えてもビット単位で同じ・途中で止まるスレッド |
| `tests/kernel_signatures.py` | Python の ctypes の署名を AssemblyScript の宣言と突き合わせる（軽い組） |
| `tests/kernels-in-browser.ts` | 各ブラウザの wasm のエンジンでカーネルの呼びを重ねて（**新しいカーネルはここに足す**） |
| `tests/int8-check.ts`・`ternary-check.ts`・`blocks-check.ts` | カーネルを JavaScript の模型とビット単位で。**int8 の積の和の順か群のレーンの割り方を変えたら、模型の `q8r()`・`q8()` も直す** |
| `tests/dist-check.ts`・`dist-same.ts`・`same-syntax.ts`・`said-texts.ts`・`names-follow-check.ts` | ビルドした `dist/` の決まり / 「同じ」を言う道具（「設計の決まり」） |
| `tests/e2e.ts` | 実ブラウザ。環境変数 `E2E_TIMEOUT`・`E2E_RESULTS`・`E2E_ARTIFACTS`・`E2E_TWICE`（2 回目の訪問、ディスクの profile）・`E2E_THEN`・`E2E_OFFLINE`・`E2E_QUOTA`・`E2E_CONSOLE`・`E2E_SILENCE`（答えなくなったページを落とす秒、既定 120）。モデル ID に `local`・`hf`・`hf:<owner>/<repo>@<revision>`・`default`。一覧に無い id はブラウザの前に落とす |
| 試験の組に無い実ブラウザの道具 | `gemma4-in-browser.ts`・`blocks-in-browser.ts`・`coi-off-in-browser.ts`・`switch-in-browser.ts`・`markdown-in-browser.ts`: **触ったら手で、`npm run build` の後**（何に触ったら回すかは細部の表） |
| CI だけの実ブラウザの道具 | `saluki_in_browser.sh`・`repository_in_browser.sh`・`gemma4_in_browser.sh`（デプロイされていないブランチの確かめ） |
| `tests/gpu-check.ts` と `gpu-check/` | GPU の正しさ（CI だけ）。`--engine dawn`・`--forms`・`--nan`・`real=<一覧の ID>`・`--part k/n`、丸めを替えた装置は `GPU_ROUNDING=`。期限は `--minutes`（既定 90） |
| 速さの道具 | `compare-engines.ts`・`compare-builds.sh`・`threads-compare.ts`・`abba-convert.sh`（前後を同じプロセスで交互に）/ `profile.ts`・`overhead.ts`・`profile-convert.ts` / `*-bench.ts` / `ternary-browser-bench.ts`（relaxed 無しの 3 値・1 ビットのカーネルを各ブラウザのエンジンで 1 本、`--before <コミット>` のカーネルと交互に。**唯一 WebKit の速さを測れる**。案は `ternary-plain-forms.ts`、組には入らない。`--relaxed` で relaxed の `matmul_t2r` を案 `ternary-relaxed-forms.ts` と: T288） |
| モデルの質の道具 | `perplexity.ts`・`start_check.py`/`.ts`・`chat_nll.py`・`degenerate.ts`・`answers.ts`（12 の問い）・`write.sh`（人が読む）・`rope_compare.sh` |
| 変換と書式の道具 | `format_check.py`・`template_corpus.py`・`conversions_compare.py`（**変換器を変えたら**一覧の全項目の前後）・`fixed_outputs.py`・`gguf_check.py`・`tokenizer_json_check.py`・`plamo_tokenizer_check.py` |
| transformers・llama.cpp との比べ（CI だけ） | `reference_<家族>.py`（と `.sh`）・`reference.py`・`reference_dry.py` / `page_27b.sh`・`page_saluki.sh`・`reference_saluki.sh`（`STAGES=dry` は毎晩、本物は頼んだときだけ、参照は `REFERENCES=<run の ID>` で前の run から）/ `page_<家族>.sh`。線は「transformers 自身の 2 つの道の差の 10 倍」で、弱い誤りを毎回入れる |
| `.github/workflows/` | `deploy.yml`（main への push）、`tests.yml`（どのブランチでも。`full`・`extra`・`runner`・`only_extra`・`minutes`・`updated`。毎晩 03:17 JST に全部の組）、`gpu-prompt.yml`、`browsers.yml`、`models.yml`（名前を渡したモデルだけを本番で）、`bench.yml`、`preview.yml`（ブランチの画面）、`gguf.yml`、`slow.yml`、`fetch.yml`、`threads.yml`、`coi.yml`、`draft.yml`。apt を走らせる手順は `tests/apt-timeouts.sh` と手順の期限 |

## 設計の決まり（いまの形）

リファクタリングの組（T346〜T391、2026-10-09〜10-13、持ち主「デザインパターンを活用して抽象化と責務の分割でファイルが扱いやすいサイズになることを重視したい」）が残した形と決まり。

細部を移した項は [docs/notes/learned/design.md](docs/notes/learned/design.md) に全文がある。

### 物差し

- **抽象化の目的は疎結合（2026-10-10、持ち主「テストしやすく、機能追加、モデルの追加などがしやすいように疎結合にするんだよ」）。** 3 つで測り、報告とレビューに書く: (1) **試験しやすいか**（部品をほかの部品の代役だけで単独に試験できるか。できない部品は何に結び付いているかを書く）、(2) **機能を足しやすいか**、(3) **モデルを足しやすいか**（作り物を 1 つ捨ての木に本当に足して、触るファイルと場所を前後で数える）。**数が減っても、共有の流れの中を直す形なら結び付きは残っている**: 足す物が「表の 1 行と自分のファイル」で済むかを見る。
- **設計のためなら動きが変わってもよい（2026-10-09、持ち主「美しい設計にするためにはリファクタリングの前後で挙動が変わっても構わないよ」）。** (1) 同じ動きに保つために形を曲げない。(2) **予定した違いを実装の前に書き、守りの網が言う予定に無い違いは誤りとして直す**（網は門でなく報告）。(3) 見つかったバグは採番して後で直す（形の変更と混ぜない）。(4) **訪問者に 1 回の費用がかかる変更はまとめる**: 変換の出力や options が変わるものは `CONVERTER` を上げる回に、シェーダの文が変わるもの（端末の鍵が変わる）は 1 つのタスクに。(5) 正しさの試験（NumPy・transformers との比べ、GPU の検査の線、ビット単位の一致）はゆるめない。
- **速さ**: 1 トークン・1 ブロック・1 submit の道に何かを足す回は、前後を同じプロセスで交互に 1 回測って数字を書く。部品に分ける・持ち主を置く・型を付ける回は形のよさで決め、1〜2% や対照の幅と重なる差のために測り直さない（「持ち主の指示」の 2026-10-11 の 2 項）。**遅くしない形**: 事実は作るときに 1 回ローカルに取る（1 トークンの道では表も値の欄も引かない）、渡すだけの関数を挟まない、部品は頭で 1 回ほどく。
- **目安の大きさは 1 ファイル 50 KB・800 行以下**（越えても落とさない: `node tests/unchanged.ts sizes` が言う）。意味の切れ目でない所で切らない。
- **パターンは、分岐が 2 つ以上の実装を持つ所にだけ**（アーキテクチャの違いは表、dtype と取り込みの型は登録簿、取り込み元は Adapter、外への窓口は Facade）。**実装が 1 つしか無い所に抽象の層を足さない**。
- **外から見える名前は窓口に残す**。**試験だけが読む名前は窓口に出さず、試験が部品から直に読む**（T641.10: `forward.ts` の窓口は worker とページとベンチが読む名前だけ。`--forward` で別の窓口を受ける道具は、その隣の `forward/` の部品から）。分けるだけの回（中を 1 字も変えずに動かす）と、形を替える回を別のコミットにする。**元の著作権と許諾文は、その行を持つ全部のファイルに残す**（方針 10）。

### 守りの網と、「同じ」を言う道具

- **守りの網は `node tests/unchanged.ts [--before <コミット>] [記録…]`**（既定は `origin/main` と。手元で約 2 分、CI では `tests.yml` の `extra=`）。記録は 12: `shaders`（窓口の全 export の文、作る関数が枝を全部通る引数で出す WGSL、作り物のアダプタの `deviceKey()`。`--built` で束も）、`models`（`src/models.ts` の全 export）、`calls`（Python の置き場と、何もしないカーネルへの呼び出しの列と番地: 全アーキテクチャ × dtype × メモリの種類 × relaxed）、`choices`（forward が端末なしで決めるもの）、`exports`（窓口の export と署名）、`layouts`（チェックポイントについて木が言うこと）、`python`（単体試験が行う変換・logits・ID を試験ごとにハッシュ）、`page`、`bench`、`gpuworker`、`gpusteps`（GPU の Worker が偽の装置（`tests/fakes/webgpu.ts`）に頼むこと: 作り物の 8 つのモデル（Llama・Qwen3・Gemma 3・LFM2・Qwen3.5・GPT-2・GPT-NeoX・回した基底の Llama。Llama と Qwen3.5 と回した基底は GPU だけでも）と、状態を層の束に割った Qwen3.5 の作る置き場とシェーダ、プロンプトのブロック・ステップの鍵と値の上げ（`keysIn()`）・全部の形の答えのステップをディスパッチと書き込みのバイトごとに。置き場は作った順でなく初めて出た所で名付けるので、部品を動かしても同じ物を同じ順に束ねれば同じ）、`sizes`。
- **網の決まり**: (1) 相手にできるのは `f91d83f9`（T379.2、2026-10-12）から後の木だけ。(2) 相手の木にはビルドの産物（`make models` のファイル）をこの木からリンクする。pytest は `PYTHONHASHSEED=0`。(3) **網は単体試験が踏む所までしか見ない**: 触る所に試験が無ければ、先に試験を足す別のコミットを。(4) `python` は両方の木の試験を回すので、製品のコードを動かす回では pytest に触らない（新しい検査は Node の道具に。触るしかなければ訳をレビューに渡す）。(5) **シェーダの部品に触るブランチは、文を変えるつもりが無くても網の `shaders` を回す**。(6) 記録が辞書をまるごとハッシュしている所に鍵を足すと全部「違う」と出る: 足した物は別の記録に。
- **型を足すだけ・名前を替えるだけの回の証明は 2 つ重ねる**: `node tests/dist-same.ts <前のビルドの dist> dist` が「the same code」と、`node tests/same-syntax.ts [--before <コミット>]`（型を空白にしたソースの構文木が同じ）。**束の比べだけでは足りない**（束ね手は `{ ms }` と `{ ms: ms }` を同じ文に刷る）。同じなら CI も速さも要らない。
- **ほかの「同じ」**: 文を動かすだけの回は `tests/said-texts.ts`、カーネルは `tests/kernels-same.ts`（ファイルを分けると wasm の関数の順が変わる。小さな関数を寄せると、数は同じでも命令が変わることがある: T644.5 で塊のカーネルの float16 の尺度の読み（load16_u と extend16_s）を `half()`（load16_s）に寄せると命令が変わったので分け、T644.12 で前後を測って寄せた（塊の 8 モジュールの 22〜29 か所が命令 1 つ減り、速さは前後の幾何平均で 0.996〜1.008 倍）。命令の変わる寄せは、前後のカーネルの出のバイトの比べと速さの前後で決める）、変換器は `tests/conversions_compare.py`。
- **比べを作るときの教訓**: 2 つの木を比べる道具はファイルでなく木を取る。網の道具は相手の木にも当たるので、消す名前を道具が直に呼んでいないかを見る。**同じコードの 2 回の間で揺れる行を先に知る**。

### 木と束

- **隣は静的な import**（`import { a } from "./x.ts";`、本当の拡張子で）。窓口は部品を import して名前を出すだけ。**後から取るモジュールは `import("./文字の名前.ts")`**、**Worker は `new Worker(new URL("./文字の名前.ts", import.meta.url), { type: "module" })`**、**作る物は表 `src/runtime/built.ts` の `BUILT` に 1 つずつ文字で**書いて `built(name, signal)` で取る。`src/runtime/` に `?url`・`?worker`・`import.meta.env`・`__BUILD__` を書かない（Node が読めない）。**変数の入った `new URL` を書かない**（Vite がフォルダごと `dist/` に写す）。
- **塊の名前は `_astro/[name].[hash].js`**（`public/coi.js` の `family()` が分ける。`/benchmark/` の入口は `benchmark-` を付ける）。別の worker のグラフが同じ名前の塊を出すと互いの写しを消すので、**後から取るモジュールや入口を足す・消す変更は `npm run build && node tests/dist-check.ts`**。「更新された」の線は「名指される塊が替われば名指す塊の名前も替わる」に乗るので、**束ね手か名前の形（`astro.config.mjs`）を替える回は `node tests/names-follow-check.ts`**（全部の組: 名指される 14 の塊を 1 つずつ替えて約 3 分。作る物（wasm・zip）は替えない）。worker の鍵は窓口でなく `shaders/key.ts` から取る。
- **束は関数を刷り直す**（コメントは消え、定数の式は畳まれる）。残るコメントは `/*! … */` と `/** … */` だけ。だから (1) **ほかのプロジェクトから取った行の著作権と許諾文は `/*! … */` に書く**（WGSL の文字列の中には書かない。通知とその文の間に何も挟まない。`dist-check` が塊ごとに数を数える）。(2) 型だけの宣言の上に `/** */` を書かない。(3) 配られるファイルの文を書き換える道具は束の文に当てる。(4) 速さの証明は束そのもの（`ENGINE_FROM=dist`、`compare-engines --bundle`）。
- **worker の入口と、後から取る `forward` の塊の両方が読むモジュールを作らない**（束ね手がそれを worker の入口に置き、`forward` が worker を import する）。だから `failure.ts`・`updated.ts` は forward の部品が読まず、`forward/reasons.ts` は worker が読まない（`reasons-check`）。**エンジンのシェーダの部品はベンチの部品から何も取らない**（向きは「エンジンが出し、ベンチが取る」）。**ブラウザだけのファイルに Node の枝を書かない**。
- **ページが実行時に取るライブラリ**（marked と DOMPurify）は束に入れず、書式のあるモデルの準備完了の時に jsDelivr から読む。描く範囲を変えるなら [docs/notes/learned/design.md](docs/notes/learned/design.md) の項を読む。
- **サイトの根はページが渡す**（`init` と `load` の知らせの `root`）。
- **Python は zip 2 つで配る**: 部品はパッケージのフォルダの下に置けば入る（フォルダの外の `.py` は `python-archive-check` が落とす）。部品はパッケージの名前で import する（相対の import は使わない）。Python を変えたら `make python`（軽い組が古い zip を落とす）。
- **道具は木の場所を書かない**: `tests/tree.ts`（Python は `from tree import python_folder`）。**実行時のフォルダの下に、目印（`forward.ts`・`llama2_numpy.py`・`simdkernel.so`・`coi.js`）と同じ名前のファイルを置かない**。ソースを文として読む道具は `untyped(文, 木が持つ本当の名前)` を通す。ファイルを終わりの `.js` で数える道具を書かない。
- **デプロイをまたいで開いていたページ**: 自分の版のファイルが無ければ「The site was updated while this page was open. Reload the page to go on.」（持ち主の文、`updated.ts` の `UPDATED`）。線は、作る物の取得が 404、または状態の分からない失敗のとき名指す側のモジュールを HEAD で聞いて 404。回線の切れ・5xx などは今の文のまま。**フォールバックのある所は動きを変えず**、遅い道に入った訳がこれのときだけ準備完了のステータス行に「the site was updated: reload for full speed」（`SLOWER`）。実ブラウザは `tests/updated-in-browser.ts`（GPU だけのモデルの GPU の Worker が始まらない場合は、その場合だけ SwiftShader の WebGPU の Chromium を別に開き `?gpuTest=only` で: CPU で読み直して答える。Firefox と WebKit では何も見ない: T491）。
- **Service Worker の登録・`?coi=off` の外し方・1 回の読み込み直しは、2 つのページとも `src/page/isolation.ts` の `isolating(how)` の 1 か所**。**`isolation.ts` は何も import しない**。2 つのページの違い（`how.fallback`・`how.offline`）と `dist-check` の数え方は [docs/notes/learned/design.md](docs/notes/learned/design.md)。
- **`coi.js` の `forgotten()`** は「このサイトで、`v=` があり、この登録の `v=` と違う写し」だけを捨てる。ほかのサイトの応答・隔離のヘッダ・fetch の受け口は `coi-js-check` が文のハッシュで持つ（変えると落ちて、持ち主の端末が先と言う）。
- **開発サーバ**（`npm run dev`）は自分で COOP・COEP を送り、Python の zip 2 つは `astro.config.mjs` のプラグイン `python-archives` が作る（細部は design.md）。端末でない所の `astro dev` はデーモンで残る（`npx astro dev stop`）。

### 型

- **3 つのプログラム**: `tsconfig.page.json`（`src/page/`・`src/benchmark/`）、`tsconfig.worker.json`（`src/runtime/**/*.ts`、`@webgpu/types`）、`tsconfig.sw.json`（`public/coi.js` を `// @ts-check` で）。決まりは `tsconfig.base.json`（`strict`、消すだけの構文: `enum`・`namespace`・引数の `private x` は書けない、`verbatimModuleSyntax`）。`tsconfig.json` は置かない。回すのは `npm run types`。**npm の `pyodide` の型は読まない**（worker が呼ぶ欄だけを `contracts/python.ts` に）。
- **型だけの import は `import type { X } from …`**（`import { type X }` は Node が空の import として相手を走らせる）。型だけの `typeof import(…)` も書かない（`import type * as X`）。
- **窓口の受け口**: worker・GPU の Worker・`/benchmark/` の GPU の Worker の窓口は、最上位で `onmessage` を置くより前に何も待たない（`tests/receivers-check.ts`、T483）。**モデルの worker の部品は import の時に何も始めない**（T641.8: アダプタを頼むのと古い Cache の削除も `runtime.ts` の `init()` の頭で。`init()` は最初の読み込みが `runtimeAsked()` で呼ぶので、読み込みが待つ `adapterAsked` はその前に無い。`worker-init-check` が見る）。`tsconfig.base.json` は `moduleDetection: force`（import の無いファイルもモジュール: 大域の名前に取り違えない）。
- **`types.ts` が言うもの**: 宣言も import もせずに使う名前は `AMBIENT` のどれか（**新しいグローバルは `AMBIENT` に 1 つ足す**）、書かれた `any` は `ANY_MOST` まで。**`*-modules-check.ts` に残してあるもの**: import の輪と届かないファイル、import した名前を隠す局所、2 つの部品の同じ最上位の名前、持ち主の検査、型の余り、`satisfies` の書き忘れ。**規則を消すかどうかは、壊し方を本当に入れて決める**。
- **型の書き方**: (1) 型は境目に書き、中は推論に任せる。部品が渡し合う物の型は読む側に書き、作る側が変数に当てるか戻り値の型を書く（**変数を通ったオブジェクトは余分な鍵を調べられない**）。(2) **型は閉じて書く**（索引の型は綴りの誤りを通す）。表の鍵を型にする（`keyof typeof ADDRESSES`）。(3) **worker の `postMessage()` には `postMessage({ … } satisfies Ready)`**。受け口は `MessageEvent<ToWorker>`。(4) **`!` は「引数で受けていない状態を当てにしている」の印**。(5) **文を変えずに型を当てる**（後から埋まる `let` は `let x!: T`）。関数を cast しない。`Object.assign`・`Object.fromEntries`・spread の中の配列は鍵を見ないので `satisfies` か戻り値の型を。caught value は `unknown`。(6) 文を読む試験（`tests/bench/`、`test_converter_version.py`）があるので、式の中に注釈を書かない所がある。
- **型は片側しか守らない**（Python と試験の代役は読まれない: 試験の `.ts` は tsc のプログラムに入っていない）。だから線の両端を突き合わせる検査が型の鍵も読む（`plan-keys-check`・`worker-modules-check`・`worker-sink-check`・`contracts-check`: 「型にあって誰も書かない」「書いているのに型に無い」の両方で落ちる）。**教えないもの**: 値の型、同じ型の引数の取り違え、forward.ts がカーネルに渡す引数の数と順（Python の ctypes の署名は `tests/kernel_signatures.py` が AssemblyScript の宣言と突き合わせる: T639.11）。

### 状態の持ち主

分け合うオブジェクトは無い。状態は持ち主の部品の変数で、**外は起きたことを関数で言い（出来事）、読むのは `export let` の名前か getter**。setter は無い。一緒にしか正しくない 2 つは 1 つの出来事で置く。1 回だけ渡る事実は引数にする。

| 持ち主 | 持つもの |
|---|---|
| `worker/runtime.ts` | 訪問で 1 回決まるもの（Pyodide・エンジンの Python・カーネル・forward・URL の旗・アダプタ） |
| `worker/request.ts` | ページがいまの読み込みに頼んだもの（何も import しない） |
| `worker/loaded.ts` | 読み込んだモデルとエンジン（`adopted()`・`letGo()`） |
| `worker/pool.ts`・`worker/alone.ts`・`worker/weights.ts` | 使い回すメモリといまのモデルのメモリ（`pooled()`・`weightsInPython()`）/ GPU だけのモデル・CPU に戻したモデル・読み込み中のモデルの鍵（`aloneBegan()`）/ 層ごとの埋め込みの表を読むファイル（`rowsNow`）・表をメモリに持つモデル・塊を広げる読み込み。向きは weights → alone → pool |
| `forward/threads.ts`・`forward/gpuside.ts` | スレッドと本数の検索 / GPU の Worker・どちらに回すか・測り直し |
| `forward/control.ts`・`forward/cache.ts` | 制御領域の語と待ち / KV のキャッシュ |
| `gpu/device.ts` | `model`・`stopping`・`lost` と、モデルの記録（端末の事実 `m.features`（`half`: shader-f16、`shuffles`: 装置の subgroups だけ（プロンプトの DP4A のタイル: `subgroup_id` を読まない）、`subgroups`: 装置の subgroups と WGSL の `subgroup_id`（attention と答えのステップの行列）、`packed`: packed int8 dot）と、モデル自身のシェーダ `m.own`（3 値と 1 ビットは `"ternary"`）と、形を選ぶ計測の時計 `m.now`（T642.4）は `openDevice()` が 1 回作る: `gpu/` の部品は `navigator.gpu` の WGSL の機能を直に読まない（読むのは `shaders/key.ts` の `deviceFeatures()` の 1 か所で、記録も端末の鍵もそれから。鍵と同じ形のプロンプトの形の一覧 `devicePromptForms(features, limits, own)` は記録の `features` を受ける。T642.3・T642.16。`/benchmark/` の GPU の節はまだ自分で読む）。`m.weights`・`m.block`・`m.forms`・`m.tokens`: **持ち主のファイルがリテラルで作り、その鍵が宣言**。別の部品の記録には代入しない。層の種類が作る物は種類のファイルの閉包の中にあり、`m.block.kinds`・`m.tokens.kinds` に置くのは `block.ts`・`tokens.ts`） |
| `src/page/model.ts`・`answer.ts` | ページが持つモデルと読み込みの進み / 書いている答え（描く間合い: 最初はすぐ、その後 8 トークンか 32 ms ごと） |
| `src/benchmark/running.ts` | 走っている run と、節ごとの結果 `results`（T643.14: 出来事 `sectionEnded()`・`gpuRewritten()` のたびに新しいオブジェクト。節の要素 `parts` は `section.ts` が自分で作り、節の Worker の終わりは `endWorkers()`、`window.__benchmark` に書くのは `dom.ts` の `told()` の 1 か所で、`results` はそこの getter が今のものを読む） |

- **ページの持ち主**（`model.ts`・`answer.ts`）は、ほかの部品を import せず、出来事でない関数も定数も持てない（`page-modules-check`）: 計算は純な部品に置き、出来上がった値を渡す。時計と乱数は引数で（Node がそのまま import して試験できる）。描く関数は状態を書かない。**持ち主に変数を足すと、部品の試験の「最初の状態」の断言が落ちる**（2 つ目の写し）。`opening.ts` は `model` を読む部品より先に走る（`index.astro` の import の順）。
- **部品に渡すのは数個の物**。部品は頭で 1 回ほどき、動く値だけを毎回オブジェクト越しに読む。
- **名前の教訓**: **分け合う物や取り込む名前は、同じ名前の局所が無いことを先に見る**（局所が黙って隠す。`node --check` は通る）。メモリの見え方 `F`・`U`・`H` を作るときに取って持たない（メモリが伸びると切り離される: 1 ステップの頭で取る）。フィールドや `at` のような名前は綴りを誤っても何も言わない。名前の置き換えは構文を知っている道具で（`sed` はコメントと文の中も替える）。**ファイルの名前を一括で替えるときは、替えたファイルを名指す所だけを替える**（T629.1: `.mjs` → `.ts` の置換が jsDelivr の `pyodide.mjs` まで替え、本番が約 4 時間モデルを読めなかった。それを期待する検査も同じ置換で替わったので通った）。置換が `src/` に触ったら実ブラウザで通しで見る。

### Python のエンジンと変換器

- **事実は表の 1 か所**: ファイルの行と大きさは `engine/layout.py`（行の関数、`placed()`、`kind_of()`）、アーキテクチャの事実は `LAYOUTS`（`Layout`: `rows`・`stateful`・`rope`・`yarn_magnitude`・`layer_norm`・`gated_ffn`・`rotatable`・`unturned`・`activation`・`local`。引くのは `layout_of()`、表に無い名前はどの入口も断る）、dtype は `DTYPES`、取り込みの型は `SOURCES`、家族は `FAMILIES`、トークナイザの種類（`bpe`・`unigram`・`bytebpe`・`plamo`）は `engine/tokenizer.py` の `TOKENIZER_KINDS`（T639.4: encode と decode の関数・ダミーの空白を置くか・U+2581 を空白と読むか・種類の索引を作る関数。表に無い種類は断る）。tokenizer.bin を歩くのは `engine/tokenizer.py` の `pieces_of()`（語片を 1 つずつ）と `charsmap_after()`（語片の後ろの正規化）の 1 か所（T639.7: `Tokenizer` と `check_tokenizer()` が読む）。**共通の流れは名前を比べず、表を引く**。行で言えることは欄にしない。**同じ集合に見える 2 つの事実を 1 つの欄にしない**。登録簿は読む物より下に置く。
- **断りの順は訪問者に見える**: 検査を寄せるときに順を変えない。
- **`Llama` は部品の合成**: 速い部品をもらったインスタンスは同じ名前のインスタンス属性に 1 回だけ置く（`ExternalForward`・`KernelSampler`）。NumPy の forward は `Llama` 自身のまま。**層ごとの事実は NumPy の forward も計画と同じ表を読む**（T639.2: `Llama.layers` は `layer_facts()` の値で、作るときに 1 回。種類・置き場・幅・head・鍵と値・RoPE・FFN を層ごとに。道具が層を壊すときは `llama.layers` を差し替える: `reference_llama.py` の `unturned_faults()`）。部品は `Llama` を持たない（`test_parts.py` が弱い参照で見る）。**`Llama.__init__` は見出しから先に `Dims` を作り、数（head・q と鍵と値の幅・状態を持つ層の数）はそこから読む**（T639.8: 同じ数を `Llama` で計算し直さない。作る段は `engine/model.py` の関数 `switches()`・`suited()`・`turned_by()`・`numpy_caches()`・`numpy_states()`・`backend_text()` で、どれも渡された値だけを読む）。**pytest の中の `kernels=` は黙って NumPy になる**（本物のカーネルは smoke）。
- **`Llama(checkpoint, tokenizer, **options)` の options は 4 つの表のどれか 1 つに 1 行**（T639.1、`engine/model.py` の `OPTIONS`）: 形は `FORM`（`form_of()` が既定ごと読む）、読み方はトークナイザの `SETTINGS`（Llama の名前 → `Tokenizer` の引数の名前、既定は `Tokenizer` のもの）、計画は `Settings` の欄（Llama が作る `MADE` を除く: `PLANNED`、既定は `Settings` のもの）、残りは Llama 自身の `OWN`。どれにも無い名前は `TypeError` で名指して断る。**設定を足すなら表の 1 行と読む所**（署名には書かない）。表が名前を分け合わないことと、2 つ目の写し `READING` は `tests/test_parts.py`。
- **計画は値で渡して持たない**: `forward_plan(dims, dtype, read, …)` は見出しと形だけで作れる。部品は計画を `start()` に渡すだけ。
- **標本抽出の設定は 1 つの値 `Sampling`**: 作る所で検査し、生成の始めに 1 回ほどく（`drawing(sampling, rng)`）。`generate()` は `**settings` を値に渡すだけ。**知らない設定は「できない」の側に倒す**ので、**設定の既定の値は「その規則なし」にする**。
- **`generate()` の輪は 1 つ**（T639.12）: プロンプトのブロック・プロンプトの 1 トークン・答えのステップのどれも、選んだトークンを輪の中の同じ 1 か所で文にし、history と stats の数え（`tokens`・`sampled`・`prompt_tokens`・考え）に入れる。sampler と GPU のステップは、ブロックが済んだ位置（ブロックが無ければ 0）で 1 回作る（`test_parts.py` がブロックとの順を見る）。**その 1 か所を局所の関数にしない**: 1 トークンの道に呼び出しを挟まない（`test_parts.py` の T359 の試験が、`generate()` の枠が呼ぶエンジンの関数を数える）。
- **考えの上限（T520）は `generate()` の引数 `thinking`**（0 は上限なし）。**印の種類は表 `THOUGHT_MARKS`**（ページの写しは `src/page/thought.ts` の `MARKS`）。種類ごとの細部は [docs/notes/learned/design.md](docs/notes/learned/design.md)。
- **変換の段取りは Python のジェネレータ**（`convert/conduct.py`）。線は「何を頼むかは Python、どう取るか・どこに置くかは JavaScript」。頼み（`text`・`bytes`・`range`・`size`・`stream`）を yield して答えを受け、`("done", 変換)` か `("missing", 場所, 名前)` で終わる。**輪（`worker/conduct.ts` の `answered()`）は置き場を知らず、答え手 `{ answers, missing }` から答える**。決まり: (1) **失敗は境目を渡さない**（404・無い名前だけが答え `undefined`。ほかの失敗と取り消しは投げ、`finally` がジェネレータを閉じる）。(2) **「無い」は `undefined`**（`null` は `jsnull`）。(3) ストリームの部品は変換の `feed` に直に渡す。(4) **同じバイトを 2 度頼まない**。(5) 後始末は 1 つの `finally` で全部呼ぶ。(6) Pyodide の前に走る所（ページの門、`kept.ts`、`footprint()`）は JavaScript に写しを 1 つ残し、検査が Python の値と比べて**どちらを直すかを文で言う**（`folder.ts` の `TOKENIZERS`、`forward/architectures.ts`、`benchmark/madeup.ts`）。
- **Pyodide の事実**: **JavaScript から来た値（dict・配列・Uint8Array の JsProxy）をほどくのは `engine/proxy.py` の `plain()` の 1 か所**（T639.5: エンジンも変換器も。変換器は前から `engine` を import するので、`converter.zip` の物を `engine.zip` に入れずに済む）。**変換器で JavaScript の値を受けるのは境目の 2 つだけ**（T640.10）: 段取りの答え（`convert/conduct.py` の `joined()`: Uint8Array を bytearray に 1 回写す）と、`Conversion` の 2 つの口（段取りが答えのまま渡す `__init__` の tokenizer と、worker がストリームの部品を直に渡す `Stream.feed()`）。その内側（`gguf_read()`・`rope_freqs_values()`・`joined_shards()` など）は Python の値だけを受け、`plain()` を呼ばない。**辞書か配列の option を足したら、ページの道（smoke の `Llama.callKwargs` に JavaScript の物を渡す形）で 1 回通す**（T649.3: T649.1 の `breaks` は pytest と全符号位置の比べを通り、ページで初めて「'pyodide.ffi.JsProxy' object is not iterable」で落ちた）。ジェネレータの PyProxy は JavaScript のイテレータ（`next(答え)` が `send`、`return()` が `close`）。`run_sync()` はふつうの呼び出しから落ちる。`unpackArchive()` は Node の `Buffer` を断る。lock にあるパッケージは版ごとに確かめる（314.0.7: jinja2・sentencepiece・safetensors・regex はあり、`tokenizers` は無い）。

### ページのエンジン（forward）

- **線**: 層が何であるかを決めるのは Python の表で、ページのエンジンは計画（`contracts/plan.ts` の `Plan`）が言う事実を読むだけ。`createForward()` は**アーキテクチャ・層の種類・dtype の名前を 1 つも比べない**。事実の置き場は 3 つ: 層ごとのものは `layers[l]`（`kind`・置き場・`rope`・`window`・`local`・Gemma 4 の `head`・`wide`・`kept` など）、モデルで 1 つのものは `Layout` の欄と同じ名前の鍵、行の有無で言えるものは鍵にしない（`floats()`・`matrix()` は計画に無いテンソルに 0・null を返す）。
- **窓のある層（`window`）は attention の層のまま**: キャッシュの番地を窓の最初の位置から渡す（**attention のカーネルは窓を知らない**）。**鍵と値は窓の分だけ輪に持つ**（T510.1、`forward/cache.ts`）: 周期 C = 窓 + 16 の輪に、位置 p を p mod C と p mod C + C の 2 か所に書く。どの窓も輪の上の続いた範囲で、数は全位置を持つ形とビット単位で同じ。輪にするのは 2C が文脈より短いときだけ（`memory.ts` の `ringPositions()`）。輪の層は位置 0 か「書いた最後の位置 − 16」より後からしか受けない。輪の「書いた最後の位置」は輪の記録 `Ring.latest`（層と GPU のブロックの両方が書く）。**attention の層の閉包は輪も全位置の塊も 1 つ**（T641.12、`forward/attention.ts`）: 違うのは場所の引き方だけで、走りの始めの `begun()`（鍵と値の頭: 輪のものか、キャッシュが伸びると動く塊のもの。輪は `Ring.latest` の検査と更新も）、位置の場所 `slotOf()`、輪の 2 か所目までの `twice`（塊は 0）。
- **層の種類は表 `LAYER_KINDS`**（`forward/kinds.ts`: 名前 → `{ frame, make(core, count, facts), state, cached, layers, widths, staged }`）。**キャッシュの置き場は種類の `cached(sizes, form, layers)`**（T641.6: 層ごとの鍵の幅（他の層の鍵と値を読む層は 0）と窓。`footprint()` はこれと `ringPositions()` で数え、エンジンの attention は同じ式 `placeOf()` を計画の事実に当てて `cache.hold()` に渡す。`plan-keys-check` が form から言う置き場と計画の事実を突き合わせる）。**表の順が、種類が取るメモリの順**（新しい種類は表の最後に）。FFN と norm は作るときに 1 つ選ぶ。KV のキャッシュは別の部品（メモリの最後）。**知らないものは作るときに断る・できない側に倒す**: 状態を持つ層のあるモデルは GPU に置かない（**例外は、作った層が状態の置き場 `staging` を言う種類**: LFM2 の畳み込みと Qwen3.5 の linear の層）。
- **仕事の種類は表 `WORK`、詰め方は表 `PACKED`**（`jobs.ts`）。番号は表の中の位置で `KIND.<名前>`。表を引くのは作るときに 1 回。**1 トークンの仕事（count 1）は使わない out stride を 9 つ目の引数として呼びに渡す**（T612: 頭打ちした attention の頭打ちの値。仕事の 8 つの引数は attention で全部埋まっている）。**GPU の側は `PACKED` の行の `gpu` を読み、そこに無い詰め方は何も書かなくても CPU に残る**。
- **読み込みの前の見積もり**（`footprint()`・`gpuBytes()`・`gpuOnlyUnfit()`: Pyodide なしで呼ばれる）はエンジンと同じ表から答える: `Sizes`、種類の `state()`、`ARCHITECTURES`（T628.1 から写しでなく、Python の `LAYOUTS` から `python_facts.py` が `src/runtime/built/facts/` に書き出したもの。`plan-keys-check` は今も計画と突き合わせる）、`PACKED`。**Python の事実を Pyodide の前の TypeScript が読むなら、写しを手で書かず `python_facts.py` に書き出しを足す**（節ごとに 1 つのモジュール `built/facts/<節>.ts` の `PYTHON_<節>`: worker の入口と forward の塊が同じモジュールを読むと束が 2 つの forward を出す（T628.3 で `dist-check` が落とした。軽い組は `dist-check` を回さない: 書き出しに触ったら `npm run build && node tests/dist-check.ts`）。`make kernels` と `python_archive.py` が作る。走らせる python3 に NumPy が要る（エンジンの部品を import する）。**`src/models.ts` も読む**（`src/models/memory.ts` の比）ので、`make kernels` を回さないで一覧を読むワークフローのジョブ（`tests/e2e.ts` を回すもの・`gguf.yml` の `plan`）は先に NumPy を入れて `python python_facts.py` を回す（T628.8: NumPy のほかは要らない）。作った `.ts` は束に入り、`dist-check` は名前で取る物と数えない）。`kept.ts` の dtype の名前と `src/models/memory.ts` の比も `built/facts/` から（T628.3。比は検査が今も `PACKED` と比べる）。
- **層ごとの埋め込み（Gemma 4 の `per_layer`）は `forward/perlayer.ts`**、表はメモリに置かず保存から読む（持ち主、2026-10-15）。**ハンドルは 1 つのファイルに 1 つ**で、持ち主は `weights.ts` の `rowsNow`、閉じるのは `letGo()`。細部は design.md。
- **行ごと・層ごとに型の違うモデル（llama.cpp の塊のまま持つモデル）**: 塊のカーネルは別のモジュールで、**塊の行のあるモデルのエンジンだけ**が同じメモリの上に載せる。深いモデルは同じ丸めの NumPy とも最大の logit の 1e-2 までしか合わないので、`tests/blocks-forward-check.ts` は呼び出しを 1 つずつ計画とカーネルの模型に照らす。計画の形・見積もり・worker の細部は [docs/notes/learned/design.md](docs/notes/learned/design.md)。
- **手で書いた計画を残さない**: 試験の計画は素の Python の `forward_plan()` で（`tests/plans.ts`）。手書きの計画は無い（T628.5: Pyodide の無い `/benchmark/` の作り物の `benchmark/madeup.ts` も、`python_facts.py` がビルドの時に `forward_plan()` を直に呼んで作る（T640.9: `Llama()` を途中で止める細工を外した。設定は `Llama` が int8 のファイルに渡すものを書く: `int8`・`KV_START`・head 全体の RoPE の表）。検査は今も値ごとに比べる）。
- **GPU に置かない・やめた訳は値で、文は `forward/reasons.ts` の `REASONS` の 1 か所**（文面は持ち主のもの）。訳を文の中身で見分けない。

### シェーダと端末の鍵

- **エンジンの GPU の Worker が走らせうる WGSL は目録 `shaders/engine.ts` の行**（`ENGINE_SHADERS`・3 値だけの `TERNARY_SHADERS`）。行は文か、作る関数と 1 つの枝の引数。読むのは 3 つ: 鍵（`deviceKey()` は行の `text()` を順に畳む。ほかにその端末が作れるプロンプトのタイル、アダプタの情報、userAgent、WGSL の機能）、Worker（`ENGINE.<名前>` で取る。部品から import しない）、試験。**鍵に入るのは端末に渡す WGSL の文だけ**。
- **行の順も文の 1 字も鍵**: 変わると全部の端末が形を選び直す。だから今あるシェーダの文は要らない限り変えず、**鍵を動かす直しは 1 回のデプロイにまとめる**。いまは訪問者が少ないので、鍵が 1 回変わることは止める理由にしない（2026-10-09、持ち主。増えたかは持ち主が言う）。**同じ文の写しを断片にまとめるのは、同じ文を出す限りいつでもよい**。
- **作る関数の行は「枝を選ぶ引数の全部の組 × 数は 1 組」**。枝の名前は文字の合併型（`FusedOutput`・`FusedInput`・`Packing`）で、既定の引数は持たない。**`@compute` の行は作る関数の自分の文に残す**。断片は部品の末尾の `export { … }` で隣に渡し、窓口からは出さない。**断片の名前はどの export とも、ほかの部品の最上位の名前とも違うものに**。テンプレート文字列の中には型もコメントも足さない。
- **見張り**: `shaders-catalogue-check`（順 `ORDER`、枝の組、宣言）、`device-key-check`（どの切れ端に 1 字足しても鍵が動く）、網の `shaders`、`gpu-worker-parts-check`、束の鍵は `dist-check`。
- **一様バッファ（WGSL の `struct Params`・`Norm`）のバイトは、その構造体を持つ部品の作り手で**（細部は design.md）。GPU の Worker で `new ArrayBuffer` に構造体の欄を手で並べない（並びの誤りは tsc も名前も言わない）。
- **形（タイル・attention・ステップの層）は読み込みの時に JavaScript と比べて測って選ぶ**。検査の比べは「線の内なら通す」の形に書く（NaN は線を越えても越えなくてもいない）。

### 失敗の運び方

- **JavaScript が投げるものは 3 つ**: (a) 素の `new Error("文")` は読む人への文。(b) 種類か「やり直しても同じ」かサーバの答えを誰かが読むなら `new Failure(種類, 文, { again, status, cause })`（`failure.ts` の `KINDS`。**種類で見分け、クラスで見分けない**: 束ごとに写しがある）。**エラーに後からプロパティを書かない**。(c) よそが投げたもの（回線の TypeError、PythonError、`JsException`、Error でない ExitStatus）を読むのは **`failureOf()` の 1 か所**。取り消しは `signal.aborted` で見分ける。**Python を通ると種類は落ちる**ので、状態に聞く（`gpuOnlyLost()`）。
- **失敗を捕まえて黙る catch** を書いたら 3 つのどれか（`silent-check`）: **言う**（`console.warn` で「何が失敗したか (訳): 何が代わりになったか」の 1 行）、**投げ直す**、**残す**（表 `SILENT` に訳つき）。訪問者に見える文は catch のために作らない。**`console.error` は使わない**（e2e が落ちる）。**正常な訪問でコンソールに 1 行も増やさない**。取り消しで拒否される Promise の `.catch(() => {})` は外さない。
- **Worker の知らせで「空いた」と読むのは、仕事の終わりの知らせだけ**（`ready`・`bench`・`done`・`error`: 含む一覧）。ページが何かを覚える知らせには読み込みの id（`load`）を付ける。
- **エンジンができる前の GPU の Worker の失敗**は `worker/alone.ts` が聞いて、エンジンを作った直後に渡す。固まった Worker は 60 秒で読み込みの失敗。

### 試験の作り

- **代役が本物より単純な所は、その差を踏む入力を本物で 1 つ回す**。偽物の頭には「何の代役か・本物と同じ性質・本物より単純な所」を書き、偽物そのものの性質に試験を置く。**寄せるのは「顔」で、「答え方」は検査に残す**。
- **比べの相手が同じ表を読むと、表の誤りが入力にも入って通る**: HF の形の作り手は手書きのまま。**表の事実は、試験に 2 つ目の写しを置く**（`ORDER`・`FACTS`）。
- **性質の試験と壊し方**: 検査を足したら、その検査が落ちる壊し方を 1 つ回す。壊し方が通ったら、壊した所が効いているか・検査が本当に踏んでいるかを先に見る。壊し方は捨てのワークツリーで 1 本ずつ。どの名前が落ちたかを読む。
- **作り物の寸法が 1 つの値に潰れていると、添字の誤りは見えない**: 実物が踏む数（グループ 2 つ以上、周回 2 回以上、表 2 つ、幅が別々、4 つずつの道の残り 0〜3）を数える。数が小さいと大きさの取り違えは 64 バイトの切り上げに隠れる。後ろの norm の無い深い作り物は値の広がりを小さく（0.1）。
- **CI でしか走らない道具は、先に手元で端から端まで回す代役を作る**（`gpu-check-dry-check`、`reference_dry.py`）。判定を関数にして作り物の数で回す（**NaN は `max()` と「線を越えたか」を素通りする**: `not (大きさ <= 線)` で数える）。
- **Node で走らない所**（最上位で要素を読むページの部品）は文の検査と実ブラウザだけが見る。`runtime.ts` の `init()` は Pyodide と forward の取り方（`STARTS`）を引数で受け、`worker-init-check` が偽の Pyodide で回す（T462）。ページの関数は要素を読まないものに分ければ Node が import できる。vm の文脈の物と Node の物は `assert.deepEqual` で「違う」と出る（文字列にして比べる）。
- **落とし穴**: 「A は B より前」を `indexOf(A) < indexOf(B)` で書くと A が無いとき通る。名前を文で数える検査はコメントにも当たる。数百 KB の bytes を `assert a == b` で比べない。嘘をつく代役には、頼まれてはならない頼みで落ちる口を付ける。文脈に文字列で書く代入は strict で。

### 計測の読み方

- **前後は同じプロセスで交互に、順を毎回裏返して**（`compare-engines.ts`、スレッドつきは `THREADS=2,4 bash tests/compare-builds.sh`、プロンプトの塊は `PROMPT=1 THREADS=2,4 bash tests/compare-builds.sh`（`threads-compare.ts --prompt`: 1 トークンの道の道具はプロンプトの塊を踏まない。モデルはファイルでも渡せる: int6 は `quantize.py`、3 値は `tests/make_ternary.py <out> ternary large`）、変換は `tests/abba-convert.sh`）。測る前に 2 つが同じ答えかを見る。**CPU の名前を出して CPU ごとに読む**（GitHub のランナーは回ごとに CPU が違う）。x86-64 と arm64 の両方で。
- **対照を並べる**: 同じコードの 2 つのビルドの比が雑音（1.5% 動くことがある）。**1〜2% の差は読まない**。1 回の run の「± の外」で決めない。
- **疑った費用は、それだけを回して µs で測る**（差の差で測らない）。プロファイラの下の計測は呼び出しの数の違いを大きく見せる（`PROFILE=0`）。
- 速さは手元の数字で決めない。静かな機械で（`/proc/loadavg`）。ソフトウェアのアダプタ（lavapipe・SwiftShader）の速さでシェーダの費用を決めない。

## 物を足す手順

どの手順も「触る所」と「忘れると誰が教えるか」まで書く。**教えないと書いてある所は、黙って別の動きになる**。試験の側の 2 つ目の写し（`FACTS`・`ORDER`・`KINDS` など）はわざと 2 回書いてあるもので、足したら一緒に直す。

各項の全文は `docs/notes/adding/` にある。**物を足す前に、その項の全文を読む**（ここは索引）。

### モデルを一覧に足す（`src/models/`）

一覧に項目を足す・書式や BOS を替えるとき。**`bos` と `template` は組**（片方だけだとエラーにならず文が悪くなる）、GGUF は先に `gguf_check.py tensors` を 1 つずつ、書式は `format_check.py` で本物と突き合わせ、書かせて人が読む。全文は [docs/notes/adding/models-list.md](docs/notes/adding/models-list.md)。

### 家族（config.json の `model_type`）を足す（変換器）

変換器に新しい `model_type` を読ませるとき。`convert/families/` に `Family` を 1 つと `FAMILIES` に 1 行、試験は `test_families.py` の `FACTS` の 1 行。**`_replace` は GGUF の名前も引き継ぐ**。llama.cpp が値や並びを変えて書くテンソルは家族の `gguf_marks(config)` の表（印は `convert/gguf.py` の `MARKS`。名前を `endswith` で選ばない: 書き忘れるとエラーにならず値が違う）。全文は [docs/notes/adding/family.md](docs/notes/adding/family.md)。

### アーキテクチャを足す（`LAYOUTS`）

forward の計算が違うモデルのとき。`engine/layout.py` の `LAYOUTS` に `Layout` を 1 つ、JavaScript は `contracts/form.ts` の `Arch` の 1 か所（`forward/architectures.ts` は書き出し）。新しい事実は計画の鍵まで同じ名前で通す。全文は [docs/notes/adding/architecture.md](docs/notes/adding/architecture.md)。

### ファイルの形式を変える・行を足す

legacy 形式の行を足す・動かすとき。行は `layout.py` のそのレイアウトの関数だけで、**行を動かしたらファイルの形式の変更: `CONVERTER`**（入れ替えは `ORDER` と本物のファイルだけが落とす）。**層によって形の違う行は、形ごとに別の名前の束にする**（1 層だけの束は変換器が束と読まない）。設定は変換器から `footprint()` まで 1 つの辞書で渡る（`worker-sink-check`）。全文は [docs/notes/adding/file-format.md](docs/notes/adding/file-format.md)。

### 層の種類（状態を持つ層）を足す

LFM2 の畳み込みや Qwen3.5 の linear のような層のとき。Python は `Stateful` と `STATEFUL_LAYERS`、JavaScript は `forward/<種類>.ts` と `forward/kinds.ts` の表の**最後**に 1 行。状態を読んで書く段をスレッドに出すなら、読む置き場と書く置き場を分ける（`linear.ts` は層の数 + 1 の置き場を回す: T285）。全文は [docs/notes/adding/layer-kind.md](docs/notes/adding/layer-kind.md)。

### dtype（詰め方）を足す

ファイルの dtype を足すとき（塊の型は「GGUF の型」）。`engine/dtypes.py` の `DTYPES` に 1 行と `jobs.ts` の `PACKED` に 1 行、同じ見出しでほかの dtype と大きさが重ならないこと。全文は [docs/notes/adding/dtype.md](docs/notes/adding/dtype.md)。

### カーネル・仕事の種類を足す

wasm のカーネルと段の仕事のとき。静的データを置けない、relaxed SIMD は別ファイル、`WORK` の最後に 1 行、**`tests/kernels-in-browser.ts` に足す**、ctypes の署名は `kernel_signatures.py` が見る。**塊の型（`BLOCKS`）を足すなら、型の番号で分ける並びが 4 か所（`rowsOf`・`tileOf`・`matmul_sbw_tile`・`sb_row`）で、1 行忘れるとその型は何も書かない**（`blocks-check` が落とす）。全文は [docs/notes/adding/kernel.md](docs/notes/adding/kernel.md)。

### GGUF の型・取り込み元の種類・トークナイザの候補を足す

GGUF の型・段取りの取り込み元・置き場・トークナイザの候補・語彙の読み手・量子化の札のとき。読み手は変換器の側に置く、同じ番号で塊の大きさが 2 通りある型は読まない、元の実装から写したら許諾文の全文。トークナイザの候補を足すと、ページの `folder.ts` の `TOKENIZERS`（`python_facts.py` が書き出す）の門の文が名前から組まれて変わる: **訪問者に見える文なので持ち主に見せてから**。全文は [docs/notes/adding/gguf-sources-tokenizers.md](docs/notes/adding/gguf-sources-tokenizers.md)。

### 標本抽出の規則を足す

`Sampling` の欄を足すとき。**既定は「その規則なし」**、kernel・NumPy・`likecpu.ts` の 3 つは同じ形、GPU の SAMPLE に足さない限りその設定の生成は CPU。全文は [docs/notes/adding/sampling.md](docs/notes/adding/sampling.md)。

### エンジンのシェーダを足す・ステップの行列に詰め方を足す

WGSL を足すとき。目録 `shaders/engine.ts` に行（どこに足しても鍵は 1 回変わる）、作る関数の局所の名前はほかの部品の最上位の名前と違うものに。確かめは CI。全文は [docs/notes/adding/shaders.md](docs/notes/adding/shaders.md)。

### worker の知らせ・計画の鍵・GPU の計画の鍵を足す

知らせ・計画の鍵のとき。型は `contracts/` の 1 か所、ページは `receiver.ts` の `HEARD` に 1 行と `heard.ts` の `ENDS`（含む一覧）、worker は `REPLIES` に 1 行。全文は [docs/notes/adding/messages-plan-keys.md](docs/notes/adding/messages-plan-keys.md)。

### 失敗の種類・GPU の訳・黙った catch・後から取るファイル

失敗の種類は `KINDS`（`refused` になりうる `new Failure(` は `again` を書く）、GPU の訳は `REASONS`（文面は持ち主）、黙った catch は `silent-check`、後から取るファイルは `LATER`。全文は [docs/notes/adding/failures.md](docs/notes/adding/failures.md)。

### 実行時のファイル・Python の部品・ページの部品を足す

部品を足すとき。ページの URL の名前は `parameters.ts` の `NAMES`、ページの保存は `remembered.ts` の `recall`・`keep` だけ、Python を変えたら `make python`。全文は [docs/notes/adding/runtime-python-page-parts.md](docs/notes/adding/runtime-python-page-parts.md)。

### 試験の作り物・偽物・道具を足す

作り物は `tests/made_up.py` から（**値の決め方に手を入れると全部の作り物が変わる**: 新しい関数に）、偽物は `tests/fakes/`、作業のフォルダは `scratch()`、取得は `tests/fetching.*`。全文は [docs/notes/adding/test-fixtures.md](docs/notes/adding/test-fixtures.md)。

## これまでに分かったこと（決定・実測・理由）

計測環境: 2026-09-26 までの「この開発機」は ARM big.LITTLE（Cortex-A78×4 + A55×4）、メモリ約 6.6GB・スワップなし。2026-09-26 からは a1-free（OCI の Ampere A1、Neoverse-N1 × 2、メモリ 11.9GB・スワップ 4GB、Ubuntu 26.04.1）。どちらも Node 24。**変更の前後を 2 つの機械の数字で比べない**。モデルごとの数字の表・run の ID は `docs/tasks/T番号.md` と `docs/notes/` にある。

ここには、次に手を入れる人が知らないと誤る決まりと数字だけを書く。各項の全文（経緯・実測の表・モデルごとの話）は `docs/notes/learned/` の分野ごとのファイルにある。

### 土台

全文は [docs/notes/learned/foundation.md](docs/notes/learned/foundation.md)。

- **a1-free の手元の時間。** pytest 5〜10 秒、smoke 12 秒、forward-check 52 秒、`make kernels` 24 秒、`make models` 2 分 11 秒、build 9 秒。huggingface.co から 5.4 MB/s。
- **スレッドで伸びるのは約 2 倍、頭打ちはメモリ帯域。** **大小のコアに均等に割ると 2 つの大コアより遅い**。 バッチ 1 の int8 は 1 バイトあたり 1 積和。単スレッドのネイティブ（NEON）と WASM SIMD カーネルはほぼ同じ。
- **なぜ Pyodide か。** WASM の上の Python で NumPy と動的リンク（ctypes）の両方があるのは Pyodide だけだった。数値の処理を全部カーネルにして、ネイティブの llama2.c `-Ofast` と並んだ。
- **エンジンの正しさ。** llama2.c の C 実装と 5 プロンプト × 256 トークンでバイト単位一致。
- **速さ（Chromium、カーネルあり）**: tiny-lm 約 270〜295、stories15M 約 300（int8）、llm-jp-3-150m 75〜79 tok/s（256 トークン）。カーネルなしは約 40・50・8.5。位置とともに遅くなる: 文書の tok/s は 256 トークンでの値。小さいモデルは greedy だと反復するので、日本語モデルは temperature 0.7 / top-p 0.9 / 繰り返しペナルティ。
- **大きさと速さ（Pythia の梯子、T84）**: 5 段（70M〜1.4B）を 1 つのジョブで測るのは `models.yml`（`browsers.yml` は段を 2 つのランナーに分ける）。毎秒読む重み（int8 MB × tok/s）は x86-64 の EPYC 9V74 で 25〜29 GB/s、arm64 で 32〜43 GB/s（4 本）。準備完了のうち変換は 3〜9%。表は `kernels/README.md`、道具は `tests/ladder.ts`。
- **int8 の品質は原本と区別できない**（stories15M +0.04%）。int4 は +16.8% で不可。活性値も 8 ビットに量子化する経路は原本と ±0.1%、7 ビット（relaxed）で +0.39%、速度は 3 割増し。
- **1 トークンの内訳**（`tests/profile.ts`）: llm-jp-3 150M は行列積 92%（分類器 46%）・attention 3%（位置 2000 で 48%、4000 で 65%）。
- **tokenizer.bin は llama2.c 本家のもの**（llama2.py 付属のものは語彙が 204 個重複）。**プロンプトの先頭に空白を付ける**（sentencepiece のダミープレフィックス。付けないと perplexity が 8.5% 悪化）。
- **int8 より小さい重み**: 6 ビット（`int6`）は大きさが int8 の 7/9、速さは約 0.45 倍。だから**ページは int8 が入らないときだけ 6 ビット**（`weightsFor()`・`automaticBits()`、`?bits=6` / `?bits=8`）。**64 ビットのメモリのあるブラウザ（Chrome・Firefox）では int8 のまま 64 ビットに置く**（`automaticDtype()`）。
- **`navigator.deviceMemory`** は Chromium 系だけが言い、近い 2 のべきに丸める（6 GB のスマホは 4）。コードは `>= 8` を「8GB 以上」と読む。言わないブラウザは 4 GB と見る。
- **GPT-2 の外れ値の列**（T92）: 最終 norm の重みの最大/中央値が 4 以上のときだけ（門なしは効かずに約 3% 遅い）。
- **`?without=kernels,int8,relaxed,sampler` で最適化を 1 つずつ外せる**（知らない名前は `ValueError`）。ステータス行は何が動いたかを必ず言う。
- **OS 5 種 × ブラウザ 5 種で動く**（`browsers.yml`）。relaxed SIMD の無いブラウザは `matmul_q8`（3 値・1 ビットは `matmul_t2`・`matmul_b1`）で動く。Playwright の WebKit は 26.4 で無く 27.2 で有る（T247・T290）。本物の Safari 27 が持つかは未確認（持ち主の端末で: TASKS.md）で、`?without=relaxed` がその代わり。Windows ARM の Playwright の Chromium と WebKit は x86-64 のエミュレーションで 3 分の 1。ランナーの速さは OS をまたいで比べない。
- **ダウンロード。** 分割並列は約 1.8 倍速い。モデルの取得は Pyodide の読み込みより先に始まる。**準備完了の内訳の秒数は、それぞれが実際に動いていた時間だけ**。NumPy の wheel は版を解決した直後に `pyodide-lock.json` から URL を組んで先読みする。
- **カーネルにアドレスを渡す配列は Python 側で必ず保持する**。Python のサイドモジュールは `simdkernel.so` の 1 つで、relaxed のものは作らない（relaxed は forward.ts が自分でコンパイルする）。
- **gist**: https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901 （大きな成果が出たら `10-` を更新する）。

### HF からの取得と変換

全文は [docs/notes/learned/hf-and-conversion.md](docs/notes/learned/hf-and-conversion.md)。

- **変換器はファイルの順に受け取る**（`Stream`: 日本からは 1 回の要求に 1〜3 秒）。**最初の部品は 8 MiB で、その速さで残りを決める**。律速は取得で、変換のスレッド化は取りやめた。 量子化はカーネル `quantize_x`（NumPy とバイト単位で同じ）、広げもカーネル（`widen_*`）で、変換器には `readers` の 1 つで渡す。HF は日本からも米国の CDN に 302 で振り向け、署名つきの URL は約 1 時間で切れる。
- **読めるもの**: Llama 構造（RoPE の縮めは llama3・linear・yarn。Mistral と Granite 4.2 は Llama として読む）・Qwen2・Qwen3・Qwen3.5・LFM2・Gemma 2・Gemma 3・Gemma 4・Spark・PLaMo 3・BitNet b1.58（safetensors だけ: 行列は U8 に 4 値ずつ詰めて行列ごとの `weight_scale`。公式の GGUF は bitnet.cpp の型 I2_S で読まない）・GPT-2・GPT-NeoX、safetensors（分割も）か GGUF（Q8_0・3 値・1 ビットの Q1_0・K / i 量子化の塊）、Unigram か byte-level BPE か sentencepiece の BPE を書き出した形の `tokenizer.json` か sentencepiece のモデル。
- **byte-level BPE は、スコアに −順位を書けば HF と同じ結果**（tokenizer.bin の形式は変えない）。前分割は手書きで、知っている型だけを受け付ける。
- **回転の並べ替えは q と k だけ**（Qwen2 の bias も wq・wk と同じ並べ替えが要る）。**ファイルから分からない設定は、変換器が渡す経路もテストする**（`Conversion.options`）。**RoPE の表を作る式は `rope_frequencies()` の 1 か所だけ**。**`footprint()` には形を渡す**（渡さないと Qwen3 の KV を 45% 少なく数える）。**RMSNorm の eps はモデルのものを使う**。 Qwen3 の書式: 考える形は本物の書式、すぐ答える形は空の `<think>\n\n</think>\n\n` を手で書く。Mistral は `normalize()` が llama に読み替える（`"head_dim": null` は「無い」）。分割された safetensors は見出しを 1 本にまとめる（`joined_shards()`）。
- **chat_template は jinja2 だけで、transformers の `apply_chat_template` と同じ環境で描く**（`convert/template.py` の `rendered()`。jinja2 を import できない変換器は書式なしで黙って続けず `ImportError`）。`src/models.ts` の `template` が優先。**書式の読み方に触ったら `tests/template_corpus.py`**。 worker は変換するときに `loadPackage("jinja2")` を NumPy と同じ扱いで待ち（`worker/pyodide.ts` の `templatePackage()`）、30 秒何も届かなければ NumPy と同じ失敗で変換が止まる（方針 2）。保存した変換を読むだけの訪問は頼まない。トークナイザの名指さないトークン（空の `pad_token`）は jinja2 に渡さない（`x is defined` が変わる）。
- **GGUF を足すときは必ず `gguf_check.py tensors` を先に通す。作り手ではなく 1 つずつ**。語彙・config・書式は原本のリポジトリ（`hf.vocabulary`）から、重みだけ GGUF から。**前分割の名前は、実物の GGUF と原本の `tokenizer.json` を突き合わせたものだけを `GGUF_PRETOKENIZERS` に**。**GGUF の中の書式は原本と違うことがある**。 **Q8_0 は損なく int8 に戻る**。ページの `?hf=` とシートは GGUF も開く。フォルダは safetensors だけ。**GGUF だけの変換**（`Conversion.from_gguf()`）は分かる所は読み、分からない所は前と同じ文で断る（BOS を言わない GGUF は書式の頭が 1 つのトークンのときだけ、EOS を言わない GGUF は断る、点数の無い語彙と Unigram の点数は断る）。llama.cpp の書き方（Llama の q・k は回して入っている、GPT-2 の Conv1D は転置、NeoX の `query_key_value` は [q; k; v]、2023 年の古い NeoX の GGUF は使わない）は全文に。
- **一覧に無いリポジトリの入口**（`src/page/repository.ts`）: HF の API にファイルの一覧を聞いて、safetensors か読める GGUF のいちばん上（`QUANTIZATIONS`: 頭の 4 つの順は持ち主のもの）を選ぶ。**段取りには何も足していない**。
- **保存の名前は取り込み元から作り、項目の id を読まない**（`keptName()`）。取り込み元が変わると保存の名前が変わり、Worker は古い名前の保存を消す（`replaced()`）。1 つの変換を分け合う項目は `shares` に両方の id。
- **頭のトークン（BOS）・止まり・書式の規則は「書式を読めたモデルは、本物の `apply_chat_template` と同じ ID を送る」**（持ち主、2026-10-10）。変換器は 1 ターンを描いた文の頭を見る（`convert/beginning.py` の `turn_head()`）。**頭のトークンは止まりに入れない**。**書式が答えの後に書くトークンも止まり**（`answer_end()`）。**`bos` と `template` は組**。**変換器の `bos` か書式の出し方を変えるときは `tests/conversions_compare.py` で全項目の送る ID を前後で比べる**（`format_check.py` は BOS が 1 つ多くても通す）。モデルごとの BOS は `start_check` で測って項目の options に。 変換器は (1) トークナイザの BOS で始まるならそれが `bos`、(2) 別の「1 つに読まれるトークン」（`<|im_start|>`）で始まるならそれが `bos` で書式はその次から、(3) 文字で始まるなら BOS が前。**(2) をしない場合**: sentencepiece 系は最初の文にだけダミーの空白を付けるので、頭のトークンを外すと次の文の読みが変わる。**止まり**はモデルのファイルが「答えの終わり」と言うもの（BOS・config.json の `eos_token_id`・トークナイザの `eos_token`）で、BOS がどこにも名指しされていないモデルは当て推量の 1 を止まりに入れない。`generation_config.json` は取っていない。
- **特殊トークンの後のダミーの空白（`prefixed`）と、最初の文の前のダミーの空白（`spaced`）は別の物**（`tests/test_prefixed.py`）。`do_lower_case` は options の `lowercase`。 `prefixed` は無ければ最初の文にだけ、`"every"` は特殊トークンの後のどの文にも、`"wanting"` は空白で始まらない文にだけ、`"never"` はどの文の前にも付けない。**「▁」をどこにも足さない tokenizer.json は、変換器が `prefixed: "never"` を言う**。`spaced` を読むのは unigram の道だけ。**書式から何かを外す・足す変更は、sentencepiece 系と byte-level の両方の項目で送る ID を比べる。**
- **変換の transform** は `None` か `TRANSFORMS`（`convert/plan.py`）の名前で始まる段か段の並びで、知らない名前はどの読み手も拒否する。
- **実物のモデルの固定値（T86）**: 合成モデルでは「動くが文章が壊れる」を捕まえられない。**取り込み元を替えたら、前の道を通る固定値が残っているかを見る**。`--write` は `--only` と一緒には使えない。
- **yarn の RoPE は cos と sin に 0.1 ln(factor) + 1 を掛ける**（`rope_magnitude()`。落としてもエラーにならない）。設定と factor は transformers 5.16.1 のとおり（`YARN_KEYS`・`yarn_contexts()`）、ほかの鍵（`attn_factor`）は断る。**ページの文脈（4096）では yarn のほうが素の RoPE より perplexity が高い**ので、一覧の Ternary Bonsai は `rope_scaling: {}`（`PLAIN_ROPE`。`null` は使わない: `jsnull`）。

### トークナイザ

全文は [docs/notes/learned/tokenizers.md](docs/notes/learned/tokenizers.md)。

- **全符号位置を本物と比べる試験は `EVERY_CODE_POINT=1`**（全部の組だけ）。前分割は文字を種類の 1 字に写して標準の `re` で走らせる（`CharClasses`）。
- **語彙に無いバイトは落とす**（本物と同じく語に入れない）。
- **`encode()` の BPE は対を heap に入れ、合併した所の両隣だけ入れ直す**（`merged()`: byte-level BPE も同じ輪で、点数の線だけが違う。`bpe` は −1e10 より上、`bytebpe` は `UNMATCHABLE`（−1e8）より上を合わせる）。unigram は語片を辞書で引き、試す長さを `reach` まで。
- **前分割の文字の種類は本物に合わせてある**（\x1c〜\x1f は空白でない、ſ は `(?i:'s)` で s と読む、`gpt2-digits` の数は `\p{N}`）。Python の Unicode が割り当てていない文字は比べから外す。
- **Mistral の Tekken**（`tekken`）は字の大文字・小文字を見分ける前分割で、字の種類の表 `TekkenClasses` の上で `STAGES` の 1 段。GPT-4o の型（`gpt4o`: Phi-4 mini・kanana-2）は Tekken の型の語の後ろに `(?i:'s|…)?` が付いたもので、表は `GptFourClasses`。**字の種類の 1 字は、`(?i)` の縮約が名指す字（s・t・r・e・v・m・l・d）と違う字に**（Tekken の小文字の類「l」のままだと `'ll` が類を字の l と読んだ: 小文字は「w」）。**Split の Removed と invert の組は、型が全部の字に当たる名前（`WHOLE`）だけ Isolated と同じ**と読む。
- **sentencepiece の正規化は model の charsmap そのもので**（`Charsmap` が double array を辿る）。文の中の U+2581「▁」は空白（byte-level BPE を除く）。
- **sentencepiece の BPE を書き出した tokenizer.json（Gemma 3・Gemma 4 など）は、語片の点数に「最初の merge の順位 × −1」**。変換器が merge の割り方と同じ点数の並びを確かめて、破れていれば断る。**この読み手に触ったら `tests/tokenizer_json_check.py --planes 17` を手で**。GGUF の語彙（`add_space_prefix: false`）は今も断る。
- **PLaMo の `tokenizer.jsonl`**: 点数は `round(score·1e4)` の整数で足す、同点は最初の語片が長いほう、語片の無い字は 1 字 1e7。**本物の道具を numba 抜きの素の Python で回さない**。
- **置き換え（options の `replaced`）**: 語彙に無い文字（改行）をカードが別の文字列（`<NL>`）で書くモデルのために、encode の頭で置き換え、`generate()` が出る文を `Unreplaced` で戻す。
- **added tokens の `rstrip`・`lstrip` は設定 `stripped`**（T258: `{"right": [...], "left": [...]}`。変換器は印の立つ tokenizer.json だけが言い、無ければ鍵を書かない）: 本物は印の右（左）の空白を印に含める（Phi-3 の `<|user|>\n` は 1 トークン）。空白は Unicode の White_Space（Python の `isspace` から \x1c〜\x1f を除く）。**書式の頭の印を BOS にしたモデルは、書式の残りが BOS の後ろから始まる**ので `generate()` が `encode(…, after=bos)` を呼ぶ: 道具でページの送る ID を作るときも BOS を `after` に渡す（`format_check.py`・`conversions_compare.py`）。**一覧で今あるもののうち印の立つ tokenizer.json は Shisa V2.1 unphi4 14B だけ**で、その項目は `stripped: {}`（読まない）で T258 の前の送る ID のまま（持ち主の答え待ち: TASKS.md の T258.1）。
- **special でない追加のトークンも specials に**（本物は前分割の前にどこでも 1 つに読む）。 並びは長い順・同じ長さは文字の順。specials の多いモデル（Gemma 3 の 6,406 個）は切り分けの型を組ごとに 1 回だけ作る（`Tokenizer.cut()`）。型は trie で、そこに書かれた最も長い special を取る（`longest_written()`、T509: 1 つの選択だった頃は `<` のたびに全部を試し、HTML 4,000 字で Pyodide の中 873 ms）。
- **`bpe` の合わせる線は −1e10 のまま**（T509）: sentencepiece のモデル自身が NORMAL の語片に −1e9 を付けることがあり（Mistral と Llama 2 の語彙の空白の並び）、本物はそれも合わせる。線を変換器の `UNMATCHABLE` に上げると、一覧の 17 項目で空白の並びの読みが本物と違う。

### スレッドとメモリ

全文は [docs/notes/learned/threads-and-memory.md](docs/notes/learned/threads-and-memory.md)。

- **GitHub Pages のままでも cross-origin isolated にできる**: Service Worker が COOP と COEP（`require-corp`。`credentialless` は WebKit で効かない）を足す。
- **ソフトウェアスレッドは既定でオン**。行列積の行を塊に刻んで取り合う形（1 スレッドあたり塊 2 つ以上）。**ソフトウェアスレッドごとに自分の番号で待たせ、名指しで起こす**。**取りまとめ役を Pyodide の居るメインスレッドに置かない**。
- **`WebAssembly.Memory` はモデルごとに作らず使い回す**（Chromium は 1 ページに 3 つ目を断った）。 最大はモデルから決め（4 × ファイル + 1GB）、次のモデルが収まる限り使い回す（`pooledWeights()`）。後に別のモデルが続かない Worker は +1 GiB を予約しない（ページが init の `ahead` にこれからの読み込みを並べる）。**最初から 4GB の最大で作ってはいけない**（スマホで Pyodide の予約が入らず黙って固まった）。
- **段のやり直し**: 止められたスレッドの段は取りまとめ役が 10 秒でやり直す。**状態を読んで書く段は、読む所と書く所を分ける**。
- **プロンプトのまとめ処理**（`forwardMany`、最大 16 トークン）は結果がビット単位で同じ。段の仕事の並びは `jobs.ts` の `WORK`。 1 本ではほとんど効かず、スレッドと組むと効く（4 本で 2.91 倍）。活性値は「トークンごとの枠」に置く。
- **KV の型を決めるのは `footprint()` の中の 1 か所**（worker は実際に取れたメモリで問い直して `external({ halfKeys })` で渡す: 渡さないと位置 2048 で 4 GiB を越える）。**規則: 共有で GQA でなければ float16、それ以外は float32。ただし float32 で 32 ビットに入らない共有メモリ・もとから 64 ビットが要るモデルは float16。共有でないメモリでは float16 にすれば 32 ビットに入るときだけ float16**。GPU の K と V は float16。**試験の旗の後ろだけで通っていた道が既定の道になるときは、その道の速さを測る**。
- **KV は 256 から倍々に、その場で伸ばす**（`grow()`）。 `footprint()` がヘッダから後ろに置くものの上限を出し（relaxed の補正・外れ値の列・活性値の枠・RoPE の表・状態・KV）、`tests/memory-check.ts` が実物の確保と比べる（上回り約 1 MiB）。float16 の KV は読む量が半分だが広げに 2.4〜2.9 倍かかり、GQA のモデルでは損。**作ったときに足りないと分かる共有メモリは共有でないメモリに替える**。
- **64 ビットのメモリの上限は 16 GiB**（越えるモデルは `pastWide()` が取得の前に断る）。64 ビットの上のカーネルは約 1 割遅いので 4GB を超えるモデルにだけ。**ソフトウェアスレッドへの仕事の表は float64**（32 ビットに番地を置くと上の桁が切れた）。 AssemblyScript の Wasm64 の出力は `kernels/wasm64.ts`（asc の transform）が直す。
- **スレッドの本数の検索**: 判定は本数ごとの下側の中央値。 1 回の比べは best・候補・候補・best の 4 つの塊。前の訪問の本数はその訪問の最初の生成で検索し直す（その後は 8 回ごと）。鍵はモデルごと（`threads:<モデル>:<論理コア>:<メモリ>:<UA>`）。Worker 間の `postMessage` の往復は 0.1〜0.3ms。**覚える本数は GPU の準備と並ばずに測った答えだけ**。作り物の速さは実物の本数を言わない。試験は偽の時計（`createForward` の `clock`）。
- **変換したモデルの保存は OPFS**（`keptName()`、古い版の保存は `replaced()` が消す）。 2 回目の準備完了は llm-jp-3 440M で 18.4 → 4.4 秒。

### CPU のカーネル

全文は [docs/notes/learned/cpu-kernels.md](docs/notes/learned/cpu-kernels.md)。

- **1 トークンの生成は帯域の壁**（`matmul_q8r` が DRAM の読み出しの 97%）。上限のループは `kernels/ceilings*.ts`。 a1-free の V8 の上限は読み出しだけ 11.8 GB/s、持ち主の Android は 1 トークン 1 本 12.9〜18.9 GB/s・4 本 27.6〜28.7（表は `docs/notes/t158-cpu-audit-2026-09-27.md`）。**上限のループは検算で見張る**（戻り値を捨てたループは消える）。**ループの定数は呼び出しから渡す**。
- **カーネルの形の決まり。** (1) x86-64 と arm64 の両方で測り、的の arm で決める（x86 で速い形が arm で 0.55〜0.6 倍のことがある）。(2) 4 つずつの道と残りの道の試験は残りが 0〜3 のどれも通る幅で。(3) ビット単位で同じと言う変更は厳密な比べも回す（`attention-compare.ts --exact`）。
- **サンプリングの決まり。** kernel（`sample`・`penalize`）・NumPy・JavaScript の模型（`walkLikeCpu`）は同じ形（smoke が見る）。(1) top-p の足切りの線は「いちばん大きい確率」で抑える。(2) **最大の logit が有限でない logits からは引かず、エラーで止める**（kernel は −1、エンジンは `ValueError`（`NOT_FINITE`））。(3) 順は「top-k → top-p → min-p」、presence penalty は書いた語のうち直近 64 語から。**同じ確率の語は添字の小さいほうが先**。greedy は top-k と min-p を見ない。 GPU の SAMPLE にもある。
- **forward.ts の外の費用**: llm-jp-3 150M で 3.5〜4.1%、tiny-lm で 16〜20%、その多くが `sample`。Python から JS を呼ぶ 1 回は 1.3〜1.7 µs。
- **3 値（`ternary`）と 1 ビット（`binary`）の変換は丸めない**（なれない値は断る）。 3 値は 128 個ごとのグループに 32 バイトと float16 のスケール（並びは「値の全部、そのあとスケールの全部」）、1 ビットは 16 バイト（Q1_0 と同じ 18 バイト、群のバイトはカーネルの並び）。速さは同じで、1 ビットの得はメモリ。binary は同じ重みなら 3 値の道の logits とビット単位で同じで、カーネルは 3 値と 1 つ（型引数）。**GPU では binary を 3 値の符号として持ち、シェーダの選びは詰め方の名前でなく `asTernary()` で**（名前で比べると binary が int8 の文で掛けられる）。
- **llama.cpp の塊のカーネル**: 重み = F·S·W − Fm·M、群の和は整数で厳密。`matmul_sbw_tile` は relaxed SIMD の決まっていない動きに乗るので `signed_dots` が真の所でだけ呼ぶ。表はチェックポイントの `codebooks` の行から実行時に作る（ファイルの形式にカーネルの都合を入れない）。
- **relaxed の 3 値の形は今の `matmul_t2r` のまま**（T288）: int16 の和の形は AMD の Zen 3・4 で 1.33〜1.36 倍だが arm64 の V8 で 0.63〜0.65 倍、BitNet 流の表は arm64 の V8 で 0.94〜1.01 倍で重みの並びが変わる。数字は `kernels/README.md`。
- **小さいモデル（350M〜E2B）では int8 がいちばん速い**（6 ビットは 0.52〜0.74 倍、Q4_K_M の塊は 0.76〜0.88 倍）。塊の得は取得とメモリだけ。
- **attention の点数の頭打ち**（Gemma 2）は別の export `attention_capped`・`attention_f16_capped` で、今までの `attention` は命令単位で同じ。GPU には置かない（シェーダに足すと端末の鍵が動く）。
- **回した基底**（Bonsai 2 27B）は重みの側には畳めない。外れ値の列は使わない。

### WebGPU

全文は [docs/notes/learned/webgpu.md](docs/notes/learned/webgpu.md)（llama.cpp の塊を GPU に置く形・状態を持つ層・Gemma 3 の細部もここ）。

- **生成の勝ち目は GPU と CPU の帯域の比で頭打ち**（持ち主の Android で 1.39 倍）。だからプロンプトから GPU に置き、生成は融合とまとめ読みで。
- **橋渡しは GPU の Worker と共有メモリ**: 置くのは層の行列（詰めた値とスケール、チェックポイントと同じ並び）と norm で、埋め込みの行は CPU が引き、RoPE の角はブロックの位置の行だけを送る。重みは WebAssembly のメモリと GPU の両方に置く（スマホと Apple は同じ物理メモリなので 2 倍）。モデルを替えると GPU の Worker は全部のバッファと device を手放して閉じる。 頼みは `postMessage`、答えは制御領域の語で、`forwardMany` は `Atomics.wait` で待つ（`Atomics.waitAsync` は Firefox 141〜144 に無い）。10 秒印が動かなければ CPU でやり直す。**GPU は自分の K と V を持つ**。**SwiftShader の速さは意味が無い**。
- **形は公開の実装から取り、端末で測って選ぶ**（持ち主「既存実装の最高なやつパクらん？」）。覚えた形は端末の鍵が同じときだけ使う。**計測の輪は `gpu/timing.ts` の 2 つ**、**計測が読む時計はモデルの記録の `m.now`**: `gpu/` の部品に `performance.now()` を直に書かない。
- **既定で GPU（方針 9）**: GPU を待たずに準備完了。**フォールバックのアダプタ（`isFallbackAdapter`）は断る**（試験だけが `?gpuTest=` で許す）。`requestAdapter` は `powerPreference` を付けない。プロンプトはブロックごとに GPU か CPU か（GPU が CPU の 0.95 倍より速いブロックだけ）。試験は `tests/gpu-default-check.ts`。
- **生成も GPU に**: **乱数は Python の生成器からステップの数だけ先に引く**。 Python の `generate()` が `GPU_TOKENS`（4）ステップずつ `generateMany` に渡し、GPU は 1 回の submit で ID と K・V を読み戻す。サンプリングは CPU と同じ乱数で同じトークン（同じ確率の語が境にまたがると GPU はその全部を取る）。LayerNorm は行列の読みに畳めない。外れ値の列のあるモデルの分類器は float で掛ける。メモリを言わない端末（Safari・Firefox）でも生成を GPU に置く（T548、持ち主の決め。持ち主の iPhone で落ちたら Safari だけ戻す）。GPU が断れば（`undefined`）そのステップは CPU。**束ねる範囲の始まりは 256 の倍数が要る**。
- **2 重に入らないモデルは GPU だけに**: **読み込みの前に決める**（`weightsPlace()`）。メモリを言わない端末と 6 ビットは GPU だけにしない。 2 重が `deviceMemory` の半分に入れば両方に置いて測る（8 と言う端末は 6.5 GiB まで: 2B 以下は 2 重、3B 以上は GPU だけ）。CPU の速さは `/benchmark/` の CPU の節の値から見積もる。GPU だけのモデルは表も K と V も CPU に持たない（**分類器が別のモデルの見出しの語彙は負の数**）。**GPU が失敗したら CPU で読み直す**。**SwiftShader で GPU のステップを回す e2e は 256 トークンに 1500 秒を越える**（数トークンで: `query=gpuTest=on&steps=40`。**`steps` はプロンプトも数える**）。
- **次のモデルは GPU の Worker の後始末を待ってから読む**（`release()` が `ended` を待つ）。
- **有限でない値は浮動小数で比べない**（lavapipe は `x != x` を畳む）: ビットの整数で見る。 `generateMany` は GPU の返した ID が語彙の範囲にあるかを見る。forward は書き戻しの float16 の指数のビットを見て、有限でない K か V のある頼みを断る。
- **float32 → float16 の丸めは装置しだい**（WGSL は向きを決めない）。**実装に任された丸めの後の値は、参照が自分で作らずに GPU から受け取り、許される隣の値かを別に見る**（`heldHalves`・`heldFloats`）。0 の向きの装置は `tests/rounding.ts`。
- **gpu-check の線は自分の目盛りで**（E16・H16・Q8、層ごとに）。**8 ビットの活性値の形は層ごとの小さな誤りを見分けられない**ので、形に依らないシェーダの誤りは float の形が捕まえる。**線を決める前に、わざと壊した GPU がどこに出るかを NumPy で出せる**。 **深い実物の DP4A は 8 ビットの丸めの境で層ごと・run ごとに違う層が線に近づく**（3 倍では正しい GPU が 1.39 倍まで出た）。CPU の誤差の比の線は誤った GPU を通した。
- **3 値を GPU に**: 重みは広げずに置き、形は DP4A だけ。 シェーダが読む所で packed int8 に戻す。答えのステップは塊と同じく役の鍵の詰め方で引く。**読み込みの検査 `tiledOff()` は NaN を通していた**（比べは「線の内なら通す」の形に）。3 値の作り物は dim 128・分類器が共有でスケールの添字の誤りの多くを通した（`synthetic-ternary-untied` を足した）。packed int8 dot の無いブラウザは CPU。新しい dtype を GPU だけに置けるようにするときは保存の `mayKeep` も見る。
- **llama.cpp の塊を GPU に**: 型は `shaders/superblocks.ts` の表 `SUPERBLOCKS` の行。 塊のまま GPU に置き、尺度のバッファを持たない。IQ 型はチェックポイントの `codebooks` の行をそのまま GPU に 1 回上げる。モデルの `own` は `"blocks"`（型を足すと塊のモデルの鍵が 1 回動く）、モデルは持つ型の文だけをコンパイルする。**型の WGSL が使う断片の定数（`K_BITS`・`IQ_*` の類）を新しく足したら、`tests/device-key-check.ts` の `Q4K_BRANCHES` にも名前を**（無いと「did not reach the hash」）。**活性値の和を float16 にしない**。**行列を束ねるのは詰め方が同じときだけ**（`tokensLayout()`）。型を足す手順は全文に。
- **状態を持つ層も GPU に**（Qwen3.5・LFM2）: **linear の層の状態は「層の束」のバッファに置く**（端末が束縛できる大きさ `m.limit` まで）。 道は `docs/notes/adding/layer-kind.md` の 5。**1 つの層の状態だけで端末の束縛の上限を越えるモデルは置かない**。
- **Gemma 3 の GPU の K と V は全位置を持つ**（CPU の窓の輪は CPU だけ）。 gate の `GEGLU`、後ろの norm は `RMSNORM_ADD`、窓は `Params.window`、局所の層は 2 つ目の角の表。読み込みの検査は窓の層のあるモデルでは短い窓でも比べる（窓 ±1 は見えにくい）。**gpu-check は窓の層の鍵と値を、CPU の輪がまだ持つ位置だけで比べる**。

### `/benchmark/`

全文は [docs/notes/learned/benchmark.md](docs/notes/learned/benchmark.md)。

- **ページの部品は聞いて描くだけ**: 表や判定を足すなら `src/bench/` に置く（DOM・Worker・`localStorage` を読まない）。**節の要素は `benchmark.astro` に手で書く**: 節を足す・題を変えるなら `src/bench/sections.ts` の `SECTIONS`・`TITLES` と一緒に（ずれは `tests/bench.ts` が落とす）。
- **GPU の節の段は表**（`gpuPlan()`・`STEPS`）: 段を足すなら契約の `ToBenchGpu` と `BenchGpuResults`、`STEPS`、`gpuPlan()` に 1 行ずつ。
- **報告の URL は 7,000 文字まで**（超えると return_to が黙って落ちる）。**ページに失敗を書くときは WRONG・failed・unsteady・skipped のどれかの語を入れる**（`warnings()` が拾う）。**Worker に答えでない知らせを足すときは、`ask()` の除く条件に入れる**。
- **GPU の節の計測は n と 2n の差で取り、形を交互に測る**。**ビット一致を求めてはいけない**（WGSL は並べ替えを許す）。端末の機能はエンジンの `deviceFeatures()` で読む（組ごとの形は `tests/bench-features-check.ts`）。行列 × ベクトルの 2 つはエンジンの文そのもので、**T624.2 より前の報告のこの 2 つの数とは比べない**。
- **3 値の行（T270）**: GPU の節の最後の段 `ternary`（行列 × ベクトルの 2 つの形・タイル・層を int8 と交互に）と CPU の節の最後のカーネルの表。符号を 0・1・2 のまま −Σa を引く形はベンチだけの文 `TERNARY_SUMS_MATVEC`（目録に足さない: 端末の鍵が動く）。どちらが速いかは端末の報告で決める。ベンチのワーカーで `fusedDp4aMatVec` に `"ternary"` を頼んでよいのは `gpu/ternary.ts` だけ。
- **メモリを言わない端末は NumPy のラウンドを飛ばす**（iPhone でタブを落とした）。
- **節は止まっても次へ進む**（節の Worker が 5 分何も言わなければ failed）。節の進みの知らせは `{stage, at, of}`。実行中は Screen Wake Lock を頼み、隠れていた秒は報告の頭に 1 行。
- **「GPU ÷ CPU」は CPU の節の実測と比べる**。デバイスを失った後・フォールバックのアダプタ・WRONG の形では比を出さない。**モデルの節はページの本当の経路を測る**。
- **ラウンドのずれ**: 最後 ÷ 2 回目の速さが線 `DRIFT_LINE` 0.9（仮）を下回ると「unsteady」で警告に入る。小さいモデルでは比は雑音。
- **Page memory の節**は 64 MiB ずつ伸ばし各段の前に `sessionStorage` に印を書く（iOS は例外を出さずにタブを落とす）。Run all には入れない。
- **CI はふだん軽く、毎晩全部**。schedule の run は main のワークフローの版で走り、失敗の通知は cron の行を最後に変えた人に行く。

### 足したモデルと、そのとき分かったこと

モデルごとの数字・書かせた文・transformers との一致は `docs/tasks/T番号.md` と `docs/notes/`、各モデルの項の全文は [docs/notes/learned/models.md](docs/notes/learned/models.md)。

- **一般の教訓**: (1) **調べを通っても、書かせると壊れるものがある**: 突き合わせと書かせて読むのと、両方が要る。(2) **GGUF のリポジトリの名前でなく、カードの `base_model` を読む**。(3) **頭のトークンの効きは、同じ元のモデルの後訓練でも大きさ違いでも 1 つずつ違う**。**自分の greedy の答えで測ると、その答えを書いた文脈のほうが必ず良く出る**: 「本物」が 2 つあるモデルは、書かれたのでない答えで測って決める。(4) **int8 の損は小さいモデルほど大きい**。(5) **eps に敏感なモデルがある**（LFM2・Gemma 3・Spark）。(6) **カードの標本抽出の組をそのまま信じない**: 設定を替えたら同じ種・同じ長さで書かせて読む。(7) **考える形が考えを終える数は、書かせる長さで決まる**: 長さと種をそろえた対照を同じ回に。(8) 名前に「(thinking)」を入れると `format_check.py` が `enable_thinking` を渡す。(9) **「層ごとに選ぶ」を真偽で書くと、もとから 0 のモデルを巻き込む**: 条件は「このモデルの印」で書く。(10) transformers の `generate()` は attention_mask を渡さないと、pad が BOS と同じ id のモデルで BOS を mask する。(11) **倍率の小さな誤りは logits の線でしか見えない**。
- **BOS と書式の家族ごとの形**: Qwen3 の 8B 系・Hermes 3・SmolLM2 Instruct・Qwen3.5・Granite・SmolLM3・Spark は BOS を書式の最初のトークン（`<|im_start|>` など）にして書式をその次から。BOS を言わない Qwen3 の派生（Shisa・CAT-Thinking・Bonsai）は変換器が 1 にするので項目の options で書く。LINE japanese-large-lm は大きさで向きが逆。sarashina2.2 の instruct と CAT-Translate は `bos: 9` と書式を組で、llm-jp-4 8B は `bos: 10`（`<|start|>`）。**Agents-A1 の既定の system の文が 1 字違っても落とす自動の試験は無い**: 文に触ったら `format_check.py` を手で。書式の長いモデル（Agents-A1・SmolLM3）は e2e が 512 トークン、Lumi v2 は 768。
- **モデルごと**（1 行ずつ。全文は models.md）:
  - Ternary Bonsai（1.7B・4B・8B・27B）と 1-bit の Bonsai: 3 値・1 ビットのまま持つ。一覧は素の RoPE。27B は参照の fork と丸めの段の下に入らないので、符号 1 つの誤りは NumPy の比べが守る。 原本に BOS が無い（項目の options で 151643 と止まり）。語彙・config・書式は `-unpacked` の原本から。27B は 64 ビットのメモリの CPU で、見張りは毎週 `browsers.yml` の `ternary-27b`、毎晩 `STAGES=dry`、参照は `REFERENCES=<run の ID>` で前の run から。1-bit の Bonsai 27B は binary でも 64 ビット（`wide: true`）。
  - BitNet b1.58 2B4T（T283）: Llama に wo の前と w2 の前の RMSNorm（行 `rms_att_sub_weight`・`rms_ffn_sub_weight`）と ReLU の 2 乗の gate（`relu2`、カーネル `reglu2`）。原本の safetensors は行列を U8 に 4 値ずつ詰め、行列ごとの `weight_scale` を掛ける（段 `planes` と `by`）。行列は形の `kinds` で 3 値のまま、埋め込みはファイルの dtype（一覧は int8）。**`v`・gate・up の scale は後ろの sub norm が eps の分しか見せない**（試験は scale を小さくして見る）。モデルは活性値をトークンの最大値で 8 ビットに丸めるが、エンジンは丸めない（ページは 32 ずつ: 実物で丸めない transformers と平均 0.015、モデル自身の丸めは平均 0.25 動かす）。丸めた形は 256 段の境で値が転ぶので、比べは最大でなく平均で。**2B の transformers（float32 で約 10 GB）とエンジンの NumPy（約 9 GB）を 1 つのプロセスに持つと CI のランナー（16 GB）を失う**: `reference_bitnet.py` は段ごとのプロセス。BOS `<|begin_of_text|>` は無しより −2.05%（`start_check.py --english`）。公式の GGUF は bitnet.cpp の I2_S（型 36）で読まない。GPU には置かない（`REASONS.pair`）。細部は docs/notes/t283-bitnet-2026-11-12.md。
  - Qwen3.5: linear の層（Gated DeltaNet）は位置 0 で状態を消し「次の位置」だけを受ける。gate の小さな行列は int8 のファイルでも float32。 llama.cpp の GGUF は値の head が鍵の head より多いモデルで値の head を並べ替えて書く（`untiled()` で戻す）。**transformers はデルタ則を型に依らず float32 で計算する**。BOS は `<|im_start|>`（`<|endoftext|>` を前に置くと素の文で +20〜95%）。作り物の decay は符号を反す（そのままだと状態が無限大）。NeoHorse-1・Agents-A1 は Qwen3.5 の後訓練で、頭のトークンの効きは測り直す。**大きいモデルを道具で回すときの落とし穴**（2 GiB 越えのファイル・1 プロセス 2 回の載せ）は全文に。
  - SmolLM3: RoPE を掛けない層（`unturned`）。作り物は「最後の層が掛けない層」の並びに。 llama.cpp は掛けない層を決め打ちで書かないので、並びは原本の config から。書式は手書き（system のターンを必ず書く）。
  - Granite 4.2: score の倍率を q の行列に畳む。ほかの 3 つの倍率が 1 でないモデルは断る。 GGUF の道でも変換器が 1 回だけ掛ける（`gguf_agrees()` が 4 つの倍率を config と比べる）。
  - MiniCPM5: 前分割が 2 段（`minicpm5`）。OpenBMB の GGUF は `llama-bpe` と言う。
  - LFM2・LFM2.5: 畳み込みの層。**FFN の大きさは config の値ではない**（`lfm2_config()`）。量子化の損は BOS の 1 本のベクトルで決まる。 `conv_bias` が真のモデルは断る。状態に触るのが取りまとめ役だけなら 2 枚持ちは要らない。gpu-check の DP4A の「最初の層を Q8 と」と「尺度」の検査は、最初の層が attention でないモデルでは外す。BOS は 1（350M は BOS なしだと perplexity 15 倍）、止まりは 1 と 7。
  - Gemma 3（4B 以上は画像つきのまるごとのモデル）: 局所の層・`(1 + w)` の norm・√dim を変換器が畳む。窓の誤り（±1）は 512 位置を越えないと見えない。 **config の書き方が 3 つある**（`gemma3_config()` が直す）。**transformers のキャッシュは窓の層の鍵と値を直近 window − 1 位置しか持たない**。止まり `[1, 2, 106]`。4B 以上の全位置の層の RoPE は linear の縮めで、局所の層は縮めない。1 層のモデルは変換できないので、窓の単体試験は 2 層の作り物で。
  - Gemma 4 の E2B・E4B: KV の共有と層ごとの埋め込み。**FORM の鍵の名前は `Dims` の属性と `footprint()` の options の鍵とぶつからないものに**。表はメモリに置かず保存から読む。 **llama.cpp の GGUF は値を HF のまま書き、回す割合は `rope_freqs` の表だけが言う**。**ページの道は小さなずれを大きくする**（比べには対照 `page_gemma4.sh` の `against` を並べる）。`loaded` は表を引いたメモリの数。
  - Gemma 2: 点数と logits の頭打ち。**llama.cpp は 2B の点数も 1/√256 で割り、transformers と違う**。 transformers との比べは eager の attention でないと頭打ちが効かない（`attn_implementation="eager"`）。google の原本はゲート付きで、写しから。GPU には置かない。
  - Spark-X2.5: erf の GELU、head ごとの出力の gate（head の値を**後ろの head から**広げる）。 transformers に型が無く、モデル自身の `modeling_spark.py` で比べる。q と k は層ごとに違う幅で並べ替える（`Family.turns`）。erf の GELU と tanh の形の取り違えは最尤では見えない。GPU には置かない。
  - PLaMo 3（Translate 0.8B・2B・8B・610M Fin Instruct）: **モデル自身のコードは config.json の `layer_types`・`rope_parameters` を読まず、平らな鍵から作る**。重みは PFN の Q4_K_M を塊のまま。 norm は c + w（後ろの 2 つは c が 1 でない）を変換器が畳む。**PFN の GGUF の最初のリビジョンの LFS の写しは 403**。書式は向きごとに手書きで、`format_check.py` は `reference_of()` がリポジトリ自身のトークナイザを読む（numba が要る）。PLaMo Community License の表示の決まり（名前・「Built with PLaMo」・契約の写し）は `models-check` が見る。
  - Underdog Saluki 27B: 10 種類の K / i 量子化の塊のまま持つ。**素の llama.cpp の logits は道と CPU で 0.3〜0.8 動く**: 細い線の相手は活性値を丸めない llama.cpp とエンジンの NumPy。 チャットのターンの印の後の logits は最後の 1 ビットで大きく動く（落とし穴の「計測とモデルの質」）。塊でない道は広げる変換になり、カーネルの無い読み込みは持ち主の文で断られる。
  - 日本語と英語の古典のモデル: **transformers 5 は `LlamaTokenizerFast` で byte-level のモデルに sentencepiece 流のトークナイザを作ることがある**（`format_check.py` が 0/9 のとき参照を疑う）。 rinna japanese-gpt2 xsmall は greedy で `<unk>` をくり返す。h2o-danube3 は本物が BOS もダミーの空白も置かないので入れなかった。置き場（`deviceMemory` 8）: 1.8B までは両方に置いて測り、3B 以上は GPU だけ。
  - Phi-3 mini（T258.1）: `qkv_proj` と `gate_up_proj` を `rows` で割る（PLaMo 3 と同じ）。アーキテクチャ `phi3` は Llama の回す割合つき（`PARTLY`、Phi-3 mini は頭の全部）で、表に Llama と同じく倍率を載せる。transformers は `sliding_window`（2047）を掛けるので文脈を窓に切る。**tokenizer.model には書式の印が無い**（added tokens は tokenizer.json だけ）: 一覧の項目は `tokenizer.json` を名指す。書式の印は `rstrip`（上の「トークナイザ」）。GGUF は読まない（llama.cpp の phi3 は行列を 1 つのまま持つ）。Phi-4 mini（T258.2）は頭の 4 分の 3 を回し、RoPE は longrope: **短い因子が全部 1 でも、attention factor（文脈 ÷ 元の文脈から: 131072 ÷ 4096 で 1.190）が cos と sin に掛かる**（落としてもエラーにならず、score が 1.42 倍平らになる）。transformers は文が元の文脈を越えると全部を長い因子で回すので、文脈を元の文脈に切り、エンジンには短い因子・attention factor・元の文脈だけを渡す（`phi3_longrope()`。longrope を読む家族は `Family.scalings` が言う）。
  - rinna japanese-gpt-neox 3.6B instruction-ppo: 改行を `<NL>` に（`replaced`）。**transformers 5 の `T5Tokenizer` はカードのトークナイザではない**: 参照は本物の sentencepiece。
  - 調べの記録（T81・T249〜T252・T517・T511）の置き場は全文に。

## 落とし穴（実際に踏んだもの）

物を足すときの落とし穴は「物を足す手順」に、設計の決まりに畳んだものは「設計の決まり」にある。ここは、道具・ブラウザ・CI・Pyodide・計測で次の人が踏みうるもの。

細部を移した項は [docs/notes/pitfalls.md](docs/notes/pitfalls.md) に全文がある。

### ページとブラウザ

- **GitHub Pages は全ファイルを 10 分キャッシュさせる。** worker と、worker が読むもの（部品・カーネル・Python の zip）は中身のハッシュ付きの名前（`_astro/<名前>.<ハッシュ>.<拡張子>`）で出るので混ざらない。`?v=<ビルド>` が残るのは Service Worker の登録（`coi.js?v=`）だけ。新しいファイルは `src/runtime/` に置いて静的に import するか、`src/runtime/built.ts` の表に 1 行。
- **GitHub Pages は `.bin` も gzip で送り、Range 要求には gzip ストリームの断片を返す。** Content-Length は展開後のサイズではなく、範囲分割はできない。進捗とバッファ確保には `src/models.ts` の `bytes` を使う。**モデルのサイズが変わったら `bytes` も直す**（合わないと Worker がエラーで止まる）。
- **`loadPyodide()` は wasm の取得に失敗してもエラーにならず固まる。** 例外を前提にしたフォールバックは機能しない。
- **ページに `<meta charset>` がないと日本語が化けてモデルに渡る。**
- **フォームの中に `<input>` を足さない**（Enter で暗黙の送信になる。生成設定のパネルはフォームの外、フォームの中のボタンは `type="button"`）。入力欄を `<input>` に戻すなら、停止は本物の押下（`event.detail > 0`）だけで。
- **ページにアイコンが無いと、ブラウザは `/favicon.ico` を取りにいく**（本物の Edge はその 404 をコンソールのエラーにし、e2e が落ちた。Playwright の Chromium は取りにいかない）。アイコンは data URI の SVG。
- **本番を `tests/e2e.ts` で確かめるときは、デプロイ済みのページと同じ版のスクリプトを使う**（push の前に本番を見るなら `git show HEAD:tests/e2e.ts`）。
- **ロードを中止しても、その後始末は数ターン後に走る。** Worker は前のロードの完了（`unloaded`）を待ってから次を始める（縮まないメモリがモデル 1 個ぶん膨らむ）。中止で reject される Promise には `catch` を付ける。
- **準備完了のステータス行は何が動いたかを言う**（SIMD kernels・relaxed SIMD・64-bit memory・llama.cpp's blocks・GPU の訳）: 文面は持ち主のもの。
- **`stats` の `prompt_tokens` は `tokens - sampled` では求まらない**（停止トークンは数えないので 1 ずれる。専用のカウンタで数える）。
- **Worker の知らせで「空いた」と読むのは仕事の終わりの知らせだけ**（除く一覧で読んでいたとき、文の途中のスレッドの知らせで 2 本目の文が走った）。**知らせを足すときは含む一覧に入れるかだけを決める**（決めないと `page-parts-check` が落ちる: T641.18）。`tests/e2e.ts` は答えの間のスレッドの知らせの数を出す（**0 なら見張りは空振り**）。
- **Worker は生成中にメッセージを受け取れない**ので、`generate()` は 50ms ごとに `MessageChannel` で 1 回イベントループに返す（`setTimeout` は 4ms の下限で速いモデルが約 5% 遅くなった）。
- **チャットテンプレートの中の特殊トークンは、文字として渡すと壊れる**（`</s>` がばらばらになる）。エンジンの `specials` で「書かれていたらそのトークン」にし、特殊トークンの直後にはダミーの空白を付けない（`prefixed` が別に言わない限り）。
- **Playwright で huggingface.co の応答を差し替えても、範囲取得の経路はテストできない**（Worker からの要求では 206 の `Content-Range` が消える）。HF から取る経路は実物のモデルで確かめる（塊のモデルの `blocks-in-browser.ts` は自分で全部に答えるので通る）。

### Service Worker・隔離・回線

- **Service Worker のうち、ほかのサイトから来る応答と隔離のヘッダ（COOP・COEP）の扱いに触る変更は、CI で通っても出さない。持ち主の端末（iPhone の Safari、Android と PC の Chrome、日本の回線）で確かめてから**（2026-10-09 に範囲を絞った: このサイトのどのファイルを写しに取るかのような、地域にも端末にも依らない変更はふつうに出す）。2026-09-25、クロスオリジンの応答を素通しにする変更を米国の CI だけで確かめて出したら、iOS で NumPy が終わらず、HF の取得が始まらなくなった（素通しにすると隔離の検査が CDN の本物のヘッダに落ち、それは地域とブラウザで違う）。**持ち主が「最近まで動いていた」と言ったら、その日の変更を順に疑う**。
- **Service Worker は毎回登録し直す**（「隔離されていれば登録しない」にしたら、古い版のまま更新されず止まった）。同じ URL なら何も起きず、`?v=` が違えば入れ替わる。**Service Worker は訪問者のブラウザに残る**: 壊れた版は入れ替わるまで効き続ける。
- **`?coi=off` はモデルのページでも `/benchmark/` でもサイト全体の Service Worker の登録を外す**（次のふつうの訪問が登録し直す）。Service Worker を通ってヘッダつきで来たページはそのままではモデルの worker が始まらないので、隔離されていれば 1 回読み込み直す（T507・T582）。**隔離したページとヘッダの無いページは `sessionStorage` を分け合わない**ので、読み込み直しの印は navigation の `type` で見る。
- **ページは初回の訪問で 1 回再読み込みされる。** Playwright の待ちは「Execution context was destroyed」になるので張り直す。
- **隔離されていないページには `SharedArrayBuffer` という名前そのものが無い**（Node には常にある）。名前を使う前に `typeof SharedArrayBuffer !== "undefined"`。`tests/engine.ts` は `shared: false` のときこの名前を消す。
- **クロスオリジンの応答のヘッダは、`Access-Control-Expose-Headers` に無いと読めない**（CDN は `*` で見せ、**WebKit は `*` を効かせない**: `Content-Range` が読めなかった）。大きさは HEAD の `Content-Length` から取る。
- **Range 要求にファイル全体（200）が返ることがある**（WebKit と HF の CDN、T112）。本文をそのまま流すと変換器が位置のずれたバイトを読み、モデルは読み込めるのに壊れた文を出した。`fetchRange` は 200 なら要求した範囲だけを流れながら切り出し、残りは取り消して console に記録する（丸ごと読んでから切り出していた頃は部品ごとにファイル全体を取っていた）。頼んだより短い 206 は `inOrder` がエラーにする。
- **本文が途中で短く終わっても、エラーになるとは限らない。長さを確かめる**（短いまま Cache API に保存されると、次からずっとそれを読む）。
- **Pyodide の読み込みの期限は「進まない時間」で決める**（30 秒何も届かなければフォールバック）。**NumPy の wheel は `loadPackage("numpy", { checkIntegrity: false })` で取る**（`integrity` を付けた fetch は本文を読み終えるまで解決しない）。細部は [docs/notes/pitfalls.md](docs/notes/pitfalls.md)。

### メモリ・Worker・保存

- **WebKit はメモリの線を越えたページを終わらせる（T506）**: 例外もエラーも出ず、8 GB 級のモデルは 16 GB 以下の機械では落ちる。**Playwright の WebKit はプロセスが終わっても `crash` を言わない**（`E2E_SILENCE` で落とす）。細部は pitfalls.md。
- **モデルを手放すときは `gc.collect()` を呼ぶ**。エンジンの部品は `Llama` を持たない（`tests/test_parts.py`）: **`Llama` か部品に、`Llama` を掴むクロージャや属性を足すとこの試験が落ちる**。
- **Pyodide の PyProxy は、壊すまで Python のオブジェクトを生かし続ける**（属性を読むたびに新しい PyProxy ができる。ジェネレータが返すタプルも）。**Python から JS の関数に渡した PyProxy は、呼び出しが終わると壊される**（持つなら `copy()`）。JS から Python の配列に書くときは呼ぶたびに `getBuffer()` を取り直す。**Pyodide は自分の `WebAssembly.Memory` を公開していない**（重みを別のメモリに置く理由）。
- **JS から Python に返す「無い」は `undefined`**（`null` は `pyodide.ffi.jsnull` で `is None` が偽。Node の試験は通っていた）。**逆向きも同じで、Python の `None` はページに `undefined` で届く**（T614: stats の `thought` を `=== null` で見ていて「Thought for undefined tokens」と出た。型は `number | null | undefined` と書く）。
- **メモリ不足は 3 つの形で届く。** Python の中なら `MemoryError`、JavaScript の中なら `RangeError`（Firefox は `InternalError: out of memory`。「call stack」の RangeError は除く）、**Python から呼ばれた JavaScript の中なら `PythonError`（`JsException`）**。読むのは `failureOf()`。**試すときに本当に大きな確保をしない**（上限を超える大きさなら何も確保せずに例外）。float16 の原本は NumPy が float32 に広げるので約 3 倍要る。
- **スマホでは wasm のメモリの予約の余地が小さい**（最大 4GB で予約したら Pyodide が入らず固まった）。CI のランナーでは出ないので、メモリの予約に触る変更は持ち主のスマホで。**後に別のモデルが続かない Worker の共有メモリの最大は要るぶんだけに**（T242: Windows の WebKit が 2 つ目の大きな共有メモリでプロセスごと落ちた。余白は `footprint()` の上回りの 1〜2 MiB だけなので、`footprint()` が足りなくなればベンチに先に出る）。
- **モジュール Worker の先頭で本当に待つ `await` を置くと、最初のメッセージを取りこぼす**（Node の worker_threads は溜めるので手元では出ない。本番を約 40 分止めた）。受け取りの Promise を await の前に作る。
- **Worker の中で作った Worker を、作った側が `Atomics.wait` で待たない**（中の Worker が進まない: Chromium で 60 秒何も受け取らず落ちた。Node の worker_threads の Dawn では通った）。待ちは `Atomics.waitAsync` か async のループの `await` で（`gpuOnlyWeights` の `room()`）。プロンプトのブロックとステップの GPU の待ちは、GPU の Worker の準備ができた後だけなので当たらない。
- **OPFS の同期のアクセスハンドルは 1 つのファイルに 1 つ**。使わないかもしれない道のために開いたものは、別の道が同じ所に書く前に閉じる（残ると `keep()` が開けず、後の `drop()` がフォルダごと消した）。
- **`Atomics.wait` で待つ取りまとめ役は、止められたスレッドを永遠に待ちうる**（ページはベンチを 1 回だけ頼む、エンジンごとに制御領域を 0 にする、段のやり直し T120）。
- **Playwright の通常のコンテキストは OPFS の割り当てが約 3〜5 GiB**: 大きい変換の保存と 2 回目の訪問を見るには `E2E_TWICE`（細部は pitfalls.md）。
- **時計で「止まった」を数えると、ページごと止まった後で誤る**（`performance.now()` は進む）。頼んだより大きく待った間は数えない。**非同期に起こすもの（`spawn()`）は、待つ間に持ち主が手放した場合も確かめる**。
- **`createForward` の中で動くものが使う `const` は、関数の頭かモジュールに置く**（後ろで宣言した定数を途中で読んで ReferenceError）。
- **int8 のファイルはコンテキストを変えてもサイズが変わらない**（Cache API のキー（URL + サイズ）が同じになるので `MODEL_CACHE` を上げた）。モデルの中身を変えてもサイズが同じなら `src/runtime/worker/sources.ts` の `MODEL_CACHE` を上げる。
- **変換器の出すもの（チェックポイントのバイトか options）を変えたら、`src/python/convert/version.py` の `CONVERTER` を上げる**（T628.4 から変換器の持ち物で、ページの `kept.ts` の `CONVERTER` は `python_facts.py` が書き出す）。**上げるときは、待っている変換器の直しを同じ回に入れる**（2026-10-15、持ち主「変換器の版が変わりそうなタスクは優先しようよ」: 訪問者の取り直しを 1 回で済ませる）。上げないと前の保存が古い options のまま使われる。上げれば古い版の保存は使われずに消える。
- **サイトは約 557 MiB**（T653: llm-jp-3 150M の float16 と stories42M の float32 の原本、450 MiB を外して HF から取る形にした）。**約 1,007 MiB（10 進で 1.056 GB）だったとき、GitHub Pages のデプロイが Pages の段の 10 分を越えて落ちることが続いた**（2026-11-06 の夕方、14 回のうち 7 回。GitHub の状態ページに障害は無く、訳は未確認: T652）。同じ大きさで 2026-11-05 には通っていた（デプロイ 42078436685）。`dist-check` が `dist/` の全部を 1 GiB（`SITE_BYTES`）で落とすが、1 GiB の手前でも遅く落ちうるので、モデルを足すときはサイトの大きさを持ち主に見せる。**`dist/models/` の部品は Makefile の `CHECKPOINTS` だけ**（`dist-check`: 外したチェックポイントの部品が手元の `public/models/` に残っていると落とす。`public/models/.done` は Makefile にも依るので、`make models` が作り直す）。

### Python・Pyodide・トークナイザ・変換

- **Pyodide の NumPy は BLAS も SIMD も無効**（スカラー WASM 相当）。
- **Pyodide の NumPy の整数は 32 ビット。大きな積は巻く**（`np.prod((64, 27648, 5120))` は 469762048）。**大きさや置き場を数える所に NumPy の整数を使わない**（`math.prod`）。ネイティブで通る試験は素通りするので、smoke が Pyodide の中で 32B と 27B の大きさを見る。
- **smoke の Python は 1 つの名前空間**: 節を足すときは前の節の名前（`values`・`rng` など）を書き換えない。
- **Pyodide の `loadPackage()` は、取れなかったパッケージをコンソールに書くだけで解決することがある**（取れたかは `pyimport` で）。「あるか」の答えを 1 回で覚える形（`convert/template.py` の `jinja_environment()` は `functools.cache`）は試験で覚えを消す（`jinja_environment.cache_clear()`: T640.11）。
- **Jinja は元の文の最後の改行を 1 つ落とす**。試験のテンプレートで改行を書かせるなら `{{ '…\n' }}`。
- **transformers 5 が保存した config.json は、RoPE の設定を `rope_parameters` 1 つに入れる**（`rope_theta` などがトップレベルから消える。読まないと黙って文が壊れる）。`normalize()` が古い名前に直す。config の読み方を足すときは新しい版の書き方も見る。
- **書式の頭の特殊トークンを外すと、後ろの文の読み方が変わることがある**（sentencepiece の「最初の文にだけダミーの空白」）。両方の家族で送る ID を比べる。
- **`arch`・`dtype`・形を受け取る関数を足したら呼び出し側を全部探す**（`tests/test_convert_hf.py` がビルドとページの 2 つの経路の一致を見る）。
- **int8 のモデルをカーネルなしで 2 つ同時に載せない**（float32 に広げる。前の開発機が 2 回落ちた）。ネイティブでモデルを載せる前に `free -m`。
- **Python のファイルをわざと壊して戻すときは `PYTHONDONTWRITEBYTECODE=1` で**（同じ秒に戻すと壊した版の `.pyc` が使われ続ける）。`__pycache__` を消してからやり直す。
- **トークナイザの試験は往復（encode → decode）では足りない**（どの区切り方でも元の文に戻る）。定義どおりの素朴な形と ID の列で比べる。
- **トークナイザの確かめていない前提**: `<0xNN>` の語片が無いときバイトは ID `byte + 3`。`<0x` で始まり `>` で終わる 6 バイトの語片は生のバイトとして復号する（rinna の `spiece.model` はバイトの語片が無く、改行が壊れていた: いまは `collapse`・`unknown` と model の charsmap）。新しい語彙を足すときは、本物の sentencepiece と BMP の全文字で比べる。トークナイザの種類・停止トークンはファイルから分からないので、ローカルのモデルでは設定の JSON で渡す。

### カーネルと正しさの試験

- **活性値を丸めるカーネルの「同じ丸めの参照」は、深いモデルでは端から端まで合わない**（T363.5）。上流の float32 の和の順の違いで境の値が隣の整数に丸まり、層ごとに広がる（8 層の作り物で最大の logit の 6e-3〜1.3e-2）。**1 層の型の取り違えも同じ大きさで、logits の線では落とせない**: 呼び出しを `wrap` で受けて、引数を計画に、積をカーネルの模型にビット単位で照らす。
- **エンジンの最適化の段は、呼んだ回数で変わる。Node の V8 だけの試験は、ほかのエンジンの誤りを見ない**（T231: JavaScriptCore は `interleave()` を約 2000 回呼ぶと 3 つの連なったシャッフルを誤って畳み、3 値のモデルが無意味な文を書いた）。`tests/kernels-in-browser.ts`（呼びを 1・400・4000・30000 回と重ねて各エンジンで）と、e2e が無意味な文を落とす。
- **64 ビットのメモリの番地の上の 32 ビットを、エンジンが落とすことがある**（V8 の Liftoff の arm64 だけ、Node 24 が当たる。細部は pitfalls.md）。
- **AssemblyScript は、インライン関数に定数で渡した引数の分岐を畳まない**（全部の枝がコンパイルされ、3 分かかり、relaxed を切ったビルドが使わない枝で断られた）。**型引数で渡す**（`f<XXS2>(…)` と `idof<K>()`）。**定数のベクトルを表から使う所ごとに読むと、V8 は毎回ロードする**: よく使う小さな定数はレジスタのベクトルのシフトで作る。
- **検査の「壊す位置」が、見たい道を本当に踏むかを見る**（壊す値がどれもグループの最大を変えるか前の 12 個にあり、最後の 4 個を見ない誤りが通った）。
- **CPU の float16 の読みは NaN を有限の数にする**（`halves4`・`half`）。だから forward は GPU の書いた K と V を写す前に指数のビットで見る（`stagingFinite()`、カーネル `finite_f16`）。試験で NaN を見るときは書き戻しの置き場の float16 のビットを読む。
- **量子化の損は、全部の文が共有する 1 つのベクトルで決まることがある**（T272）。最初のトークン（BOS）の状態は文に依らない定数で、その丸めの誤差は全位置に同じ向きに効く: perplexity は数 % 動くが、KL は小さく、答えの最尤は 99% 一致する。**Wikipedia の perplexity の 1 つの数字を「このモデルの損」と読まない**: (1) グループの始まりをずらした別の引きで広がりを見る、(2) 位置・層・行列・head の順に丸めを 1 つずつ外して絞る、(3) KL と最尤の一致を並べる、(4) ランダムな雑音で再現できるかを見る。
- **試験の線は、通してはいけない誤りを並べてから決める**（誤りを 1 つずつ入れて同じ参照の数と比べる）。logits だけでなく各層の状態（再帰の状態・畳み込みの行・鍵と値）も比べる。**参照どうしの差も CPU で変わる**ので、線は固定の値でなくその差の倍数に。固定の線は深いモデルで外れる。**試験の道具そのものにも誤りを入れて、落ちるのを見る**。
- **「止まるスレッド」の試験は、どの段で止まるかを決める**（最初に起こされた段は起動の競争で決まる）。状態を読んで書く段には `tests/threads-late-helper.ts`。
- **本物の GPU のアダプタが要る分岐は、CI のどの試験も通らない**（「hybrid は GPU に置かない」の行を消しても全部通った）。偽の GPU の Worker の工場を渡して「呼ばれない」を見る（`tests/gpu-hybrid-check.ts`）。
- **引きを 2 つの実装で比べる試験は、確率がほぼ 0 の語どうしを許す**（大きい順に歩く引きで乱数が 1 のすぐ下のとき、どこで止まるかは実装しだい）。
- **同じ入力を何度も通して測るときは、その計算が入力を書き換えないかを見る**（繰り返しの罰を logits にその場で掛けていて、測った回は語彙の 93〜99% が床の上だった）。サンプリングだけを測るときは罰を 1 に。

### WebGPU

- **WGSL のシェーダの誤りは「Invalid ShaderModule」とだけ出る**。どこが悪いかは `createShaderModule({ code }).getCompilationInfo()`。
- **装置のエラーの文は改行を持つ**（Dawn は「 - While validating …」の行が続く）。1 行の段落や表のセルには `tableCell()` で 1 行にしてから入れる。
- **WGSL の一様性の解析**: 障壁のある呼び出しが続く分岐の条件は、一様と分かる値でなければならない（`atomicLoad` の値も workgroup のメモリの値も一様と見なされない）。配るのは `workgroupUniformLoad`。
- **Dawn の Node 版（npm の webgpu）では、`create([])` の返す instance を持ち続ける**（書き捨てると GC で消え、次の非同期の呼び出しで bus error）。`navigator.gpu` に持つ。
- **GPU のバッファは 16 バイト単位に丸めて作っている**（`buffer()`）。読み戻しは要るバイトだけ（GPT-2 の logits の後ろの 0 が最大になって、正しい形を断った）。
- **ソフトウェアのアダプタの速さの比で、シェーダの費用を決めない**: **版を決まった順に回すと、順番で速さが分かれる**（同じ文でも）。順を毎回混ぜ、同じ文をもう 1 回入れた対照の比を並べ、その外のときだけ読む（`tests/sample-gpu-bench.ts`）。
- **GPU の Worker に試験の口を足さずに、GPU の持つものを壊す**（gpu-check は Worker を作る関数を包めば頼みが全部見える。重みはエンジンを作る前に共有メモリを書き換えて準備の後に戻す。**ただし 1 層目と分類器は読み込みの検査が比べるので壊せない**）。forward.ts は 1 回断ると GPU を止めるので、何通りも試すときは試験が自分で頼みを送る。
- **WebGPU は手元で回さない（持ち主の指示）。例外は 1 つのシェーダのコンパイル**: Dawn の Node 版（`.tmp/` の下に入れる）と lavapipe（`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`）で shader-f16 と subgroups のある装置になる（Playwright の SwiftShader には shader-f16 が無い）。
- **GPU の Worker の読み込みの見張りは段ごとに期限をつける**（1 つの段は 180 秒まで）。**CI の `models.yml`（`gpuTest=on`）で head 128 のモデルがこれで CPU に戻ったら既知として 1 回だけ走らせ直す**（細部は pitfalls.md）。
- **形を選ぶ計測は小さいモデルでは submit の待ちに埋もれる**（n と 2n の差で、交互に測る。lavapipe の作り物では差が負にもなる）。
- **SwiftShader の WebGPU のアダプタは、まれに取れない**（8 回に 1 回ほど `requestAdapter()` が null）。CI で 1 回落ちたら、もう 1 回走らせる。
- **正しい理由で CPU に置いたことを通す検査は、その理由をモデルの形から計算して比べる**（理由の文だけで通すと、誤って CPU に置く壊れも通る）。

### CI・道具・git

- **「この変更とは関係ない落ち方」で済ませない（2026-10-10、持ち主「原因は突き止めるべきなんじゃないか？」）。** たまに落ちる・止まるものは原因を調べ、分からなければ「次に起きたら分かる」ように道具を直して採番する。分かった 2 つ: (1) Node のタイマーは `performance.now()` で見て早く鳴ることがある（**時間を比べる検査の文には測った数字を出す**）。(2) **Node は、重い仕事の直後の `process.exit()` の中でまれに止まる**（約 70 回に 1 回）: 終わる前に 200 ms 待つ（`tests/leave.ts` の `leave()` が待つ: T580）。`tests/suite.sh` の `node` は全部 `tests/limited.ts`（期限。止まった所のスタックと生かしているハンドルを出す）を通る。**道具の期限は包むジョブの期限より短く**。たまに止まるものは手元で数十〜数百回回して再現を試す（`/proc/<pid>/task/*/` の `comm`・`stat`・`wchan`）。
- **待ちは ID で、期限つきで、裏で（T142）。** 遅く見えたのは終わらない待ちだった: `pgrep -f` がループ自身のコマンド行に一致して回り続けた、`pkill -f` が打ったシェル自身を止めた、同じものを待つループが重なった、「最近の N 件の run」から探す待ちが新しい run で外れた。**決まり**: CI は `tests/ci.ts` で run の ID を待つ。手元の計算は PID（`while kill -0 $pid`）か、最後に必ず書く印のファイル（`trap` で）で待つ。`pgrep -f` と `pkill -f` を名前で使わない。どの待ちにも期限。同じものを 2 回待たない。サブエージェントへの頼みにもこの決まりを書く。
- **エージェントの `run_in_background` の中で、さらに `&` で裏に回さない**（外のシェルが終わった時点で中の計算が止められた）。印のファイルは同じ命令の最後に書く。
- **GitHub の API は全部の会話で 1 時間 5000 回を分け合う**。待つ run は 1 回の `ci.ts wait <ID …>` にまとめ、詰まったら数分あける。
- **数回に 1 回の失敗を「このコミットから」と読まない**（26% で落ちるものが 2 回続けて通る見込みは 55%）。**通った版と落ちた版をそれぞれ 12 回以上回してから比べる**（bench.yml の `times=` と `build=`）。直しの確かめも直した版と戻した版を同じ時に。**`actions/checkout` の `ref` に短いハッシュは渡せない**。Astro のページの版（`__BUILD__`）は `GITHUB_SHA`。
- **並べて走らせた CI の run は、ログの中の ID で突き合わせる**（`ci.ts` の出力の頭の「started … run <ID>」）。
- **CI のブラウザと apt の入れる手順には、手順の期限と apt の time-out を付ける**（apt が止まってジョブの 120 分を使った）。apt を走らせる手順は `timeout-minutes: 20` と `bash tests/apt-timeouts.sh`（`tests/workflows-check.ts` が見る）。手順の期限が止めたら 1 回走らせ直す。
- **デプロイが落ちた後の直しが記録と文書だけなら、デプロイは走らない**（`paths-ignore`）。2026-10-24、台帳の目次が古くて T571 のデプロイが軽い組で落ち、目次だけを直して push しても本番は古いままだった: `gh workflow run deploy.yml --ref main` で回す。台帳を直したら push の前に `python tests/ledger.py --check`。
- **`setup-python` の `3.x` は出たばかりの版になる**（T630: 2026-10 に 3.15・Unicode 17 になり、本物の tokenizers（Unicode 16）と全符号位置で比べる試験が 4,657 字で落ちた）。全部の組の `tests.yml` は 3.14 に留めてある。毎晩が急に落ちたら、まず道具の版（Python・tokenizers・Playwright）が動いていないかを見る。
- **手動のワークフローの入力を空で渡しても「空」にはならない**（既定の値になる）。**空を「全部」の印にしない**（`only=all`）。手順が飛ばされたジョブも success になるので、分数か手順の結論を見る。
- **ワークフローやシェルの中の `'`**: `run:` の `node -e '…'` の中にアポストロフィを書かない（コメントにも。`workflows-check` が `run:` を `bash -n` で読む）。`case` の中の `"${x:-it's}"` は後の引用符を飲み込み、`bash -n` は通る。**手動のワークフローに触ったら、その入力の形で 1 回 dispatch する。**
- **取得は長さを確かめる**（Python の `read(量)` は接続が早く閉じても黙って短いものを返す）。**CI の取得が変換の途中で「ファイルが短い」と落ちたら、まず取得の長さを疑う**。`download(optional=True)` は 4xx を全部「無い」にするので、段取りの答え手は 404 だけを「無い」と読む。
- **止まったテストの出力をパイプ越しに待たない**（`| cut` がバッファに溜めて何も出ない）。ファイルに書いてから読む。
- **ブラウザを動かす道具は、終わりを待たずに自分で終える。Node の道具は、出力がパイプを抜けてから終える**（Windows の WebKit の `browser.close()` は返らないことがある。大きなログの直後の `process.exit()` はパイプの入口で切れ、落ちたのに FAILED が出なかった）。**落ちたのに訳が出ていないログは、まず切れていないかを見る**。`tests/leave.ts` の `closed()` と `leave()`。
- **投げられるものは Error とは限らない**（Pyodide は Emscripten の `ExitStatus` で reject する）: `failure.ts` の `told()` が名前と文で言う（細部は pitfalls.md）。
- **CI のランナーから Wikipedia や huggingface.co への接続も切れる**（道具の取得は `tests/fetching.*` の 1 か所で取り直す）。
- **Playwright の Firefox で速度を測らない**（wasm がベースラインコンパイラだけで 13 分の 1）。速度は `tests/stock-firefox.ts` で CI で。
- **ワークツリーの `node_modules` を本線へのシンボリックリンクにしない**（`make kernels` の `npm ci` がリンクの先を消して作り直した）。ワークツリーでは自分の `node_modules` を `npm ci` で。`make models` の作るファイルはリンクでよい。**ワークツリーで smoke を回すなら、そこで `npm ci`・`make kernels`・`make models` を回す**（本線のモデルのファイルは古いことがある）。
- **開発機の `/tmp` は tmpfs（メモリ）**（a1-free は 5.9GB）。大きい作業ファイルはリポジトリの `.tmp/` に。前の開発機で `dist` を 2 つ `/tmp` にコピーして持ち主のマシンが落ちた。
- **手元のブラウザ。** a1-free では Playwright の Chromium で `tests/e2e.ts` が走る（WebKit は動かない。入れ方は docs/notes/dev-setup.md）。**7B 級は手元で動かさない**（`models.yml` で）。細部は pitfalls.md。
- **開発機のメモリ。** 峰の読めない計測は `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` で包む。**軽い組は `MemoryMax=3G` では OOM で止まる**（`5G` で通った）。細部は pitfalls.md。
- **手元は shallow でないこと**（`git rev-parse --is-shallow-repository` が false）: 古い `.git/shallow` の印が残ると網が履歴を切る（直し方は pitfalls.md）。
- **`git rm` は削除をすぐ登録する**（別のタスクのコミットに入った）。コミットの前に `git diff --cached --stat`。止められたシェルコマンドは途中まで実行されていることがある: `git status` で確かめる。
- **走っている `tests/suite.sh` を書き換えない**（bash はスクリプトを少しずつ読む）。
- **smoke の Python の節が落ちると、CI のログに訳が出ない**（例外の行として縮めたソースの全部が出て、ログが途中で切れる）。手元で `node tests/smoke.ts > .tmp/…/smoke.txt 2>&1` を回して、その行を除いて読む。

### 計測とモデルの質

- **チャットのターンの印の後の logits は、活性値を丸めるモデルでは最後の 1 ビットで大きく動くことがある**（Saluki 27B: 同じ丸めを 100 万分の 1 ずらすだけで logits が 13 動く。確率 3e-15 の語で、最尤は同じ）。誤りと決める前に: (1) 参照を自分でゆすって同じ位置・同じ大きさが出るかを見る（`reference_saluki_numpy.py` の `NUDGED`）、(2) 丸めた後の整数の積をビット単位で模型と比べる（`page-27b.ts --exact`）、(3) 線は全部の語の最大でなく、KL と「確率のある語」の上に引く。素の文だけで線を決めると書式つきの文で落ちる。
- **性能の比較は、新旧を同じプロセスで交互に走らせる**: `node tests/compare-engines.ts [モデル ID …] [--ref origin/main] [--rounds 5] [--tokens 128] [--greedy] [--bundle]`（順は 1 ラウンドごとに裏返す。「one side twice」がその場の雑音。測る前に同じ種から同じ文を書くことを見る）。**1% 未満の差が同じ向きにそろったら、両側を同じコミットにした対照（`--ref HEAD`）を同じ run に並べる**。1〜2% を言うなら `tests/compare-builds.sh`（21 回、対照を毎回）。CI では `node tests/ci.ts run tests.yml extra="node tests/compare-engines.ts tiny-lm llm-jp-3-150m --rounds 7" --ref <ブランチ>`（arm64 は `runner=ubuntu-24.04-arm`）。
- **小さな費用を、大きな時間の差で測らない**（Python から JS を呼ぶ 1 回が 9〜122 µs と出た: ループで測ると 1.3〜1.7 µs）。**forward の直後に走るものと、ループで回したものを比べない**（キャッシュが追い出されている）。
- **GitHub の ubuntu-latest のランナーは run ごとに CPU が違う**（EPYC 7763・9V74・9V45、Xeon 6973P-C・Platinum 8573C・8370C。llm-jp-3 150M の 1 トークンは 8.1〜13.7 ms と割れ、同じカーネルの変更が CPU で 0.88〜1.18 倍と割れた）。**速さは同じ回の中の比か同じプロセスの交互で読み、CPU の名前を出して（`os.cpus()[0].model`。arm64 は「unknown」）CPU ごとに読む**。1 本で遅い形が既定の本数でも遅いとは限らない。2 本は 4 vCPU のランナーでは揺れて読めない。
- **カーネルの速さは、行列をメモリから読む形で測る**（同じ行列を繰り返すとキャッシュの速さ: `tests/tile-bench.ts` の形で、1B と 3B〜8B の幅も）。
- **速さの計測は、同じ機械でほかの担当が Chromium や build を回している間に取らない**（load が 2 を越えると 36% 遅く出た）。測る直前に `cat /proc/loadavg` を見て、1 未満のときだけ測り、前後の load を添える。
- **`tests/perplexity.ts` の 1500 トークンは、文のファイルの最初の 1500 トークン** = 最初の記事（Mount Fuji・富士山）の冒頭だけ。記事ごとに回すなら `wikipedia.ts <言語> <ファイル> <題>`。1 つの記事の差は ±0.3% の揺れ。
- **緩い比べを足した所には、その緩さが通す壊し方を 1 つ入れて落ちるかを見る**（`format_check.py` は前に 1 つ多い BOS を `LENIENT` と `UNMEASURED` の項目でだけ通す。前は既定で通していて、損の大きい家族が一覧に入ってから見つかった: T312）。
- **頭のトークンの効きは、素の文・手書きの答え・モデル自身の答えの 3 つで測る**（独立した文の尤もらしさは、モデルが書く文の変化を測らない: Qwen3.5 0.8B で `<|endoftext|>` は手書きの答えでは変わらず、自身の答えの分布は KL 0.115 動いた）。**数トークンの 1 つの文や 1 つの物差しで「効かない」と決めない。**
- **サンプルした 1 つの答えの事実の誤りは、引きの 1 つかもしれない。greedy と、種を変えた引きの数で言う**（`TEMPERATURE=` を付けた `tests/answers.ts`）。変換の誤りを疑う前に greedy と transformers との logits を見る。
- **答えが日本語かは、かなで数える**（漢字で数えると中国語の答えが日本語に見えた）。繰り返しの印は考えの中の言い換えにも当たるので、本文を読んで決める。
- **「読めない」「できない」の理由は、走らせて最初の断りを見て書く**。

## 検証手順

1. エンジン・`quantize.py` を変えたら、まず `.venv/bin/python -m pytest tests -q`（リポジトリの中の venv、docs/notes/dev-setup.md。system の python3 では tokenizers が無く、本物の tokenizers と比べる試験が skip になる）と `node tests/smoke.ts`（どちらもデプロイでも走る）。加えてネイティブの Python で回帰確認する: stories15M（float32）で `Once upon a time` の greedy 出力が `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.` で始まること。stories260K なら `...She loved to play outside in the park.`。余裕があれば llama2.c の `run.c` を `gcc -O2` でビルドして全文一致を見る。
2. ページや Worker を変えたら、**実ブラウザで通しで確認する**: `npm run build && node tests/e2e.ts [モデル ID] [chromium|firefox]`（準備はスクリプト冒頭のコメントと docs/notes/dev-setup.md）。「Run が有効になる → Enter → 回答の下に tok/s が出る」まで待ち、スマホ幅でページ自体がスクロールしないことと、決定的なモデルでは出力の冒頭も確かめる。
3. push 後は `node tests/ci.ts deploy` でデプロイを待ち、本番に対して同じ確認をする: `node tests/e2e.ts stories260K chromium https://takano32.github.io/pyodide-llm/`。
4. **WebGPU（`src/runtime/gpu.ts`・`gpu/`・`shaders.ts`・`shaders/`・forward の GPU の道）を変えたら、ブランチを push して CI で**（開発機では回さない）。**既定は軽い組なので、判断の前は `full=true` で**: `node tests/ci.ts run gpu-prompt.yml [full=true] [models="synthetic tiny-lm"] [real="hf-qwen2.5-0.5b-instruct hf-qwen3-0.6b"] [forms=TF.js] --ref <ブランチ>` を裏で。読むのは各回の「keys and values … — FAILED」の有無、層ごとの表（NumPy と E16）、`- A: … of the line`、`seconds of`。**実物（`real=`）は `engines=dawn` で Dawn のジョブだけを回す**（SwiftShader のジョブは 1B 級の実物で 90 分の期限まで走る。SwiftShader で回すなら `forms=` を付ける: 360M 以上は形を選ぶ計測が 180 秒を越える）。
5. **カーネル（`kernels/*.ts`）か、それを呼ぶ `forward.ts`・`jobs.ts` の道を変えたら、ブラウザのエンジンでも**（Node の試験は V8 の最初の段だけを見る）: `node tests/ci.ts run tests.yml only_extra=true extra="npm ci && make kernels && npx playwright-core install --with-deps webkit && node tests/kernels-in-browser.ts webkit" --ref <ブランチ>`（Chromium と Firefox は `chromium firefox`、arm64 は `runner=ubuntu-24.04-arm`、macOS は `runner=macos-latest`）。**CPU はランナーしだい**（WebKit の x86-64 の誤りは AMD の EPYC に当たった回で出た）。新しいカーネルは `tests/kernels-in-browser.ts` に足す。
6. **27B と Saluki の計算が変わりうる変更**（行列積・回した基底・3 値と塊のカーネル・linear の層・GGUF の読み・一覧の項目・`footprint()`・**GPU に置く門**（T588: T271.1 が門を外して `dry` の GPU の期待が 1 日落ちた））は、ブランチを push して `tests/page_27b.sh`・`tests/page_saluki.sh` を CI で（頼んだときだけ。参照を前の run から `REFERENCES=<run の ID>` で取ると短い）。**先に `STAGES=dry`** が通ること。長い走りは短い走りで道具を先に通す。数が動いたら、線でなく原因を見る（符号 1 つの誤りは丸めの下で見えない）。
7. **変換器の出すもの（options・tokenizer.bin）か、一覧の項目の `bos`・書式・`specials` を変えたら**: `.venv/bin/python tests/conversions_compare.py <取得のフォルダ>` で一覧の全項目の前後（変わった鍵、tokenizer.bin、一覧として・`?hf=` として送る ID）を出して、変わった項目を 1 つずつ読む。それから `tests/format_check.py <同じフォルダ>` と `--hf`。書式の読み手に触ったら `tests/template_corpus.py`。どれも手元の `.venv` で回る（全項目で約 10 分ずつ）。**加えて CI で、実物の小さいモデルを transformers と比べる参照を 1 本**: `node tests/ci.ts run tests.yml extra="bash tests/reference_qwen35.sh --only=real" --ref <ブランチ>`（Qwen3.5 0.8B、約 4 分。T424: 頼んだときだけ回す道具は、T369 の回に変換器の `bos` が変わって本線で落ちたまま気づかれなかった）。
