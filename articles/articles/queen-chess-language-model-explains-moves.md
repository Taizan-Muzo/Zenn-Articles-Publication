---
title: "沈黙の名人に言葉を与える──Queenが4Bでグランドマスター級の棋力と説明を両立する仕組み"
emoji: "♛"
type: "tech"
topics: ["機械学習", "LLM", "自然言語処理", "知識蒸留", "強化学習"]
published: true
---

## TL;DR

チェスエンジンは超人レベルの棋力を持つが、自分の指し手を説明できない。一方、LLMはもっともらしい説明は生成できるが、棋力が弱すぎて説明の価値が限られる。Princeton大学の研究チームは、無搜索で超人級の棋力を持つエンジン**Leela**を「沈黙のエキスパートencoder」として凍結し、4B程度の言語モデルにgated cross-attentionで橋渡しした上で、**Bellman更新の自然言語アナロジー**に相当する反復探索蒸留を7ラウンド回すことで、棋力**Elo 1782 → 2697**（+915）を達成した。GPT-5.6-Sol（2071）やGemini-3.1-Pro（2201）を大きく上回り、GM中央値（2730）に迫る水準。しかも説明の質を支える主変（PV）の無誤謬率は全モデル中最高である。

**論文**: [Language Models that Play Chess and Explain Their Moves](https://arxiv.org/abs/2610.03695) (arXiv:2610.03695)
**著者**: Adithya Bhaskar, Jeffrey Cheng, Danqi Chen (Princeton Language and Intelligence)
**リソース**: [モデル](https://huggingface.co/collections/princeton-nlp/queen-chess-models) / [コード](https://github.com/queen-project/queen)

![fig1](/images/queen-chess-language-model-explains-moves/fig1.png)

---

## 背景：棋力なくして説明なし

1950年にShannonが計算量を見積もって以来、チェスはAI研究の試金石であり続けてきた。現代のTransformerベースエンジン（Leela Chess ZeroやDeepMindのBT5）は、探索なしでもほぼ超人級の指し手を出せる。だがその出力は、合法手の確率分布と勝率評価だけだ。**なぜその手が良いのかを言葉で語れない**。著者らはこれを「沈黙のエキスパート（silent expert）」と呼ぶ。

逆に言語モデル側はどうか。チェスの解説を生成させる研究は複数あるが、その多くは与えられた指し手を条件にコメントするか、 isolatedなパズル局面を扱うにとどまり、そもそも指し手自体が弱い。GPT-5.6-SolやGemini-3.1-ProのようなフロンティアLLMでも、実戦棋力はElo 2000強で、LMらしい流暢な説明を生成しても「弱い手の良い説明」になってしまう。

論文の立場は明快だ。**説明の価値は、それが説明する指し手の質に上限を付けられる**。だから棋力が前提になる。では、どうやって棋力を言語モデルに持ち込むのか。ここで鍵になるのが、Leelaの潜在表現にすでにチェスの概念がエンコードされているという先行知見（Jenner et al., 2024）だ。ゼロから概念を学ばせるのではなく、**すでに概念を持つモデルの表現を読み取れるようにする**発想である。

---

## 方法その1：Flamingo型アーキテクチャで潜在表現を橋渡しする

Queenの入力は「(FEN, テキスト)」のペアだ。FENでエンコードした盤面をLeela（BT5, 240M, 15層）に入力し、そのhidden statesを言語デコーダSmolLM3-3Bに供給する。

![fig2](/images/queen-chess-language-model-explains-moves/fig2.png)

橋渡しの部分はFlamingoスタイルの**gated cross-attention**である。特徴的なのは2点。

- **encoderの中間層を捨てない**。Flamingoの元実装は最終hidden stateのみを使うが、Queenはencoderの第$i$層とdecoderの第$2i$層をペアリングし、16個のcross-attentionブロックをdecoder層0, 2, …, 30の直前に挿入する。盤面理解は「駒の配置」から「戦術的含意」へと層を追うごとに抽象化されていくので、初期段階の表現も残しておく価値がある、という判断だ。
- **ゲート$\tanh(\alpha)$の$\alpha$を0で初期化する**。訓練開始時点ではcross-attentionが恒等写像となり、盤面情報は「ノイズ」として最初から注入されない。学習が進むにつれて$\alpha$が開いていく。

さらに、tokenizerのセル名（square名）表記の揺れや「bishop」「knight」の非チェス的意味の悪影響を避けるため、64マスと12種の駒に**専用トークン**を新設している。

ここまでのポイントは、**encoderもdecoderも凍結し、bridge（約470M）と新トークンのembeddingだけを訓練する**ことだ。言語能力を壊さずに、盤面概念だけを取り込む設計になっている。

---

## 方法その2：QAカリキュラムでencoderの「読み方」を教える

bridgeをいきなり訓練しても、decoderはLeelaの表現から何を読み取ればいいか分からない。そこで4段階の質問応答カリキュラムを回す。この段階の成果物は**Pawn（Position AWare Network）**と呼ばれる。

1. **Static-Current**：現在局面の静的な事実。指定マスの駒、駒の数、material points など
2. **Dynamic-Current**：現在局面の動的な事実。合法手の一覧、取れる駒、checkの応じ方、詰み判定など
3. **Static-Future**：与えられた手順を踏んだ**先の局面**を再構成した上での静的質問
4. **Dynamic-Future**：未来局面の動的な推論。手順後の合法手や攻撃者の列挙など

前段のデータは後続段階で10%程度リプレイして忘却を防ぐ。検証セットの精度はStatic系で99.97%、Dynamic-Futureでも96.07%に達し、Leelaの潜在表現をかなり信頼性高く「翻訳」できるようになったことが分かる。なお消融実験では、encoderを外して盤面を辞書形式のテキストで直接与えるとElo 514まで転落し、カリキュラムを省略しても同様に崩壊する。**encoderとカリキュラムの双方が強い初期化に不可欠**という結果だった。

---

## 方法その3：Bellman更新の自然言語アナロジーによる反復探索蒸留

ここが本論文の最も面白い部分だ。

![fig3](/images/queen-chess-language-model-explains-moves/fig3.png)

AlphaZeroはMCTSで得た改良された評価をネットワークに蒸留し、探索なしで同等の評価を再現できるようにした。探索ツリーをalpha-beta剪定に単純化すると、これは**Bellman value iteration**にほかならない。

$$V_{k+1}(s) = \max_{a}\,[\, R(s,a) + \gamma\, V_k(s\,')\,]$$

Queenはここで大胆な置き換えを行う。**スカラー価値$V$を自然言語の説明$E$に置き換える**のだ。

$$E_{k+1}(s) = \mathrm{Distill}\,[\, \mathrm{Consolidate}\,(\{E_k(s\,'_a)\}_{a \in \mathrm{Top3}})\,]$$

各ラウンドの流れは次の5ステップに整理できる。

- **① Sample**：約40万局面をサンプリングする。自模型とStockfishの対戦、Lichessの人間の対局、パズルの3ソースだ
- **② Generate**：Pawn-$k$が各根局面について有望な候補手3つを選び、説明を生成。3手すべてがmistake（Stockfish比で勝率10%以上の悪化）なら、最悪の1手をStockfishの最善手で差し替える
- **③ Recurse**：子局面ごとに説明を生成させ、子の指し手予測がmistakeならその子を新たな根として再帰下降する（最大5回）。子はまだ誤らないが根では既に誤る、その境界まで降りるイメージだ
- **④ Consolidate**：Qwen3.8-27Bに「新しい内容は導入しない」と指示した上で3つの子の説明を統合する。評価値の数値部分だけはStockfish（100K nodes）の真の値で上書きする
- **⑤ Train**：統合結果と元の説明のPVを最初の分岐点で比較し、**改善した場合のみ**採用。残った約25万例でPawn-$k$からSFTしてPawn-$(k+1)$を作る

種の初期化にはGPT-5.6-Sol (low) の生成した約8,400例を使うが、面白いのはここが必須でない点だ（後述のHCE消融）。

7回の反復でEloは 1782 → 2024 → 2187 → 2346 → 2539 → 2434 → 2559 → **2697** と推移する。途中に一時的な落ち込み（P4→P5）があるものの、全体としては+915の着実な上昇だ。

---

## 実験結果

![fig4](/images/queen-chess-language-model-explains-moves/fig4.png)

### Accuracy：実戦棋力

8つの異強度エンジン（LichessスケールでElo 1978〜2865をアンカー設定）と各32局を対戦させた結果がこちら。

| モデル | Elo |
|---|---:|
| **Queen (4B)** | **2697** |
| Gemini-3.1-Pro | 2201 |
| GPT-5.6-Sol | 2071 |
| GPT-5.6-Luna | 1822 |
| C1-4B（同規模の先行手法） | 514（32局全敗） |
| GM中央値（Lichess blitz） | 2730 |

フロンティアLLMに対する勝率はGeminiで94.6%、Solで97.4%。パラメータ数は3桁小さいにもかかわらず、というのは強烈だ。Leela本体（2987）との差は、著者らが**verbalization debt**と呼ぶ「潜在知識を言語化するコスト」の現れだろう。

### Substantiation：説明の裏付け

推奨手だけでなく、予測した主変（PV）全体に誤りがないかをStockfish 1M nodesで検証した。指標は「PV全体にmistakeがない割合（NMR）」と「最初の一手がmistakeでない割合（FNMR）」。

| モデル | 戦術 NMR | 戦術 FNMR | 一般 NMR | 一般 FNMR |
|---|---:|---:|---:|---:|
| **Queen** | **68.8** | **91.6** | **66.6** | **97.1** |
| Gemini-3.1-Pro | 66.9 | 82.5 | 63.0 | 89.0 |
| GPT-5.6-Sol | 65.1 | 81.9 | 61.5 | 86.8 |

すべての指標で首位。教師であるはずのSol自身を大きく上回っている点が、反復蒸留の効果を雄弁に物語る。

### Coherence：説明の「語り」

GPT-5.6-Sol (high) をジャッジに使った1–5点評価では、構造的一貫性3.51（Solと同点、Geminiの3.55に接近）、流暢さ4.45と健闘する。一方で**概念的一貫性は2.76と明確に低い**。フォークやピンといった戦略的母題の言及に幻覚が残り、反復蒸留では「見たことのない母題を言語化する力」までは教えられない、という正直な限界の報告だ。

---

## 考察

個人的に最も重要だと思ったのは、**Leelaの単なる複製ではない**という検証だ。複数のほぼ最善手が存在する局面1000で見ると、Queenは**53.8%**の局面でLeelaのpolicyと異なる手を選んでいる。encoderの表現は「答え」ではなく「手がかり」として使われているわけで、この設計の本質を突いている。

また、フロンティアLLMに頼らない訓練可能性も示されている。GPT-5.6-Solによる種の初期化を、Stockfishの手作り評価（HCE）特徴から組み立てたテンプレート説明に置き換えても、Elo軌跡は2115 → 2497（P4時点）とむしろ高いスタートを切る。特許的なLMの蒸留が必須でないというのは、閉じた環境でこのレシピを再現できる可能性を意味する。

一般化の観点も見逃せない。このレシピの前提は「**沈黙のエキスパートencoderが訓練可能な領域**」というだけで、ゲーム、ロボティクス、computer useといった状態を持つ環境にそのままapplyできる。エキスパートモデルが強い決定を支え、LMがそれを人間可読な説明とさらなる推論の足場に変換する——この分業は、AGI的エージェント設計の一つの型になるかもしれない。

---

## 関連研究

- **Leela Chess Zero / BT5**（Monroe et al., 2026）：Queenのencoder。Transformerベースのチェスエンジンで、探索なしでほぼ超人級
- **Jenner et al., 2024**：Lc0の潜在表現に駒配置・合法手・戦術的継続がエンコードされていることを示した先行研究。Queenの動機の直接の源泉
- **C1-4B**（Tang et al., 2026）：チェス特化の蒸留で説明を生成する同規模モデル。Queenとの対比で、単一モデルで棋力を獲得することの難しさを浮き彫りにする
- **LLAMIA**（I et al., 2026）：同時期の類似のencoder-decoder構成。モデルが未公開のため比較対象外
- **AlphaZero**（Silver et al., 2017）と**Bellman value iteration**（Bellman, 1957）：反復探索蒸留の理論的雛形

---

## まとめ

- 沈黙のエキスパートencoder（Leela）を凍結したままFlamingo型gated cross-attentionで言語モデルに接続し、QAカリキュラムで表現の読み方を訓練した
- **Bellman更新の自然言語アナロジー**に相当する反復探索蒸留を7ラウンド回し、Elo 1782 → 2697（+915）を達成。4B規模でGPT-5.6-Sol・Gemini-3.1-Proを大きく上回り、GM中央値に迫った
- 説明の裏付け（PVの無誤謬率）は全モデル中最高。一方で概念的母題の言語化には課題が残り、これが今後の研究課題
- 「エキスパートモデルの表現＋言語モデルの語り」という分業レシピは、チェス以外の状態的環境（ロボティクス、computer use）への一般化が期待される

言語モデルに専門能力を持たせる文脈では「ツールとして呼ぶ」と「重みに焼き込む」の二択が定番だったが、この論文は第三の道──**潜在表現の空間で接続する**──を示したとも読める。説明が次の学習サイクルの教師データになる閉ループの設計も含めて、まだ続報を追いたいテーマだ。

---

## 参考

- Bhaskar, A., Cheng, J., & Chen, D. (2026). [Language Models that Play Chess and Explain Their Moves](https://arxiv.org/abs/2610.03695). arXiv:2610.03695
- [Queen Project Website](https://queen-project.github.io/) / [GitHub](https://github.com/queen-project/queen) / [Hugging Face Models](https://huggingface.co/collections/princeton-nlp/queen-chess-models)
- Jenner, E. et al. (2024). [I Will Not Get Lost in Your Hidden States: probing Leela Chess Zero](https://arxiv.org/abs/2407.01857)
- Silver, D. et al. (2017). Mastering the game of Go without human knowledge. *Nature*
