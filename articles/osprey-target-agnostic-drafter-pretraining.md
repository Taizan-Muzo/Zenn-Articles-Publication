---
title: "加速する側だけが世間知らず──Ospreyが投機デコードのドラフター事前学習を再利用可能にする仕組み"
emoji: "🦅"
type: "tech"
topics: ["LLM", "推論高速化", "Speculative Decoding", "事前学習", "EMNLP"]
published: false
---

## TL;DR

投機デコード（speculative decoding）は、軽量な**drafter**が未来のトークンを予測し、巨大な**target**がそれを一括で照合するという、無損失で推論を加速する鉄板技術だ。しかしその速度向上はもろい。drafter は単一 target・狭い分布に向けて訓練されるため、チャットからコードへ、単言語から多言語へ、という workload シフトで受入率が崩壊する。例えば EAGLE-3 を math データだけで適応すると、math 内では平均受入長（AL）5.06 なのに code では 1.86 に落ちる。

これは現代 LLM 開発の逆説だ。target モデルは大規模事前学習で広い汎化を獲得するのに、それを加速する drafter はその恩恵をほとんど受けられない。理由は既存の drafter 手法がターゲット依存だから──target の隠れ状態を食い、target のロジットで蒸留するので、事前学習も target ごとにやり直しになる。

Osprey はこの逆説を解く。**Qwen3-4B を2層に剪定し、FineWeb で次トークン予測を行う「target-agnostic な事前学習」を1回だけ行う**。できた shallow NTP backbone は、語彙整列・zero-initialized QKV 拡張・EAGLE-3 方式の on-policy 蒸留によって各 target に軽く適応されるだけで、Qwen3-8B / Llama-3.3-70B-Instruct / 229B の MiniMax-M2.5 に転用できる。

結果は target 間で AL をそれぞれ +16.1%、+21.2%、+22.7% 改善。特に OOD や多言語で強く、かつ10k steps の backbone でほぼ全恩恵が得られる。サービング量がある程度あれば数億トークンで事前学習コストを回収する。

> **Core insight**: 投機デコードの drafter も、target 固有の蒸留に加えて「汎用言語モデルとしての事前学習」を資産化できる。加速側だけが世間知らずである必要はない。

---

## 背景：速いのに、なぜ世間知らずなのか

### 投機デコードの強みと脆さ

LLM 推論は、巨大なパラメータを1トークンずつ前向きに計算するため遅い。そこで投機デコードでは、軽量な drafter がまとめて未来トークンを「下書き」し、target がそれを1回の forward で受理・棄却する。受理された分だけ並列に進められるので、**正確性は損なわずに2〜5倍の加速**が実現できる（Leviathan et al., 2023; Chen et al., 2023; Xia et al., 2024）。

だが現実の加速幅は、**drafter の予測が target とどれだけ一致するか**で決まる。多くの drafter はある程度特殊なデータ分布や特定モデル向けに訓練される。すると、ユーザーが話題を変えたり、言語を変えたり、長文生成を要求したりしたときに、drafter の予測は target から外れ、受入率が急落する。

本論文の Figure 2 に載っている例が象徴的だ。EAGLE-3 を math データで訓練して Qwen3-8B を加速すると、math 内では AL 5.06 を出すが、code 評価では 1.86 に落ちる。加速がほとんど解消する。

![Osprey の問題意識と cross-domain 評価](/images/osprey-target-agnostic-drafter-pretraining/fig2.png)

### 大モデルは「読書」しているのに、下書き役は読まされていない

近年の LLM 開発では、target モデルは広く多様なデータで pre-train され、ドメイン・タスク・文体を跨いで汎化することが価値の中心になっている。にもかかわらず、それを加速する drafter は random init から narrow な分布で訓練されることが多い。すなわち、投機デコードは「小モデルのレイテンシ利点だけを受け継ぎ、大モデルの汎化利点は受け継がない」状態だった。

自然な対処は、drafter も pre-train することだ。しかし既存の EAGLE 系列は target の隠れ状態を入力に使い、target のロジットで蒸留するため、**事前学習すら target ごとにやり直さなければならない**。それではコストが高すぎて現実的ではない。

Osprey はここで方向を転換する。既存の pre-trained small LM（Qwen3-4B など）を出発点とし、浅い drafter に削ぎ落としてから**target 非依存の NTP 事前学習**を行う。そして各 target では軽い変換と蒸留だけ行う。これにより、drafter 本体に蓄えられた広い言語知識を複数 target で共有する。

---

## 方法詳解：4段階パイプライン

Osprey の訓練は、target-agnostic 相（1 回だけ）と per-target 相（target ごと）に分かれる。Figure 1 が全体像だ。

![Osprey の4段階パイプライン](/images/osprey-target-agnostic-drafter-pretraining/fig1.png)

### Stage 1: Prune

出発点は市販の small LM、主に Qwen3-4B だ。先頭の N 層だけを残し、埋め込み層・最終 RMSNorm・LM head は保持する。本論文の主要 backbone は N=2（約591M パラメータ）の2層構成だ。浅くする必要がある──drafter は autoregressive に何度も forward を走らせるので、深いと投機の意味がなくなる。

ただし急激な層削除は次トークン予測能力を大きく損なう。そこで Stage 2 で回復する。

### Stage 2: Pre-train（target-agnostic な NTP 復元）

剪定された浅いモデルを FineWeb の100B token サンプルで、**標準的な次トークン予測**で訓練し直す。実験では5エポック、約462B token、ピーク学習率 1e-4 で55k steps まで走らせる。この段階で剪定によって失われた言語モデリング能力が回復する。

重要な点は、この段階では target モデルに一切触れないことだ。得られるのは「浅い汎用 NTP backbone」であり、これが Qwen3 / Llama / MiniMax など異なる target に後で使い回される。

また、本論文は計算効率の観点から「compute-optimal」を意図的に超える訓練量を使う。これは後の実験で「10k steps（84B tokens）の backbone でほぼ同じ性能」になることが示され、大規模運用時の損益分岐を大きく緩める。

### Stage 3: Convert（語彙・隠れ状態の移植）

Stage 2 で作った backbone を特定 target に適合させる。ここでは3つの構造変更を行う。

![Stage 3 の zero-init QKV 拡張と語彙整列](/images/osprey-target-agnostic-drafter-pretraining/fig3.png)

#### 多層 target 隠れ状態の注入

EAGLE-3 と同じく、target の低層・中層・高層の隠れ状態を連結し、新しく学習する射影行列 $W_c$ で drafter の隠れ次元に射影する。これを $c_{t-1}$ として drafter の各 attention 層に供給する。

#### Zero-initialized QKV 拡張

新しい情報を入れる際の最大の難題は、**事前学習済みの計算を壊さないこと**だ。Osprey では、attention の Q/K/V 射影行列を次のように拡張する。

$$
\widetilde{W}_{qkv} = [\,0 \;|\; W_{qkv}\,]
$$

$c_{t-1}$ とトークン側の表現 $z_t$ を連結して入力すると、初期化時には

$$
\widetilde{W}_{qkv} \begin{bmatrix} c_{t-1} \\ z_t \end{bmatrix} = W_{qkv} z_t
$$

となり、target 情報が $0$ で乗算されるので drafter の振る舞いは pre-train 直後と**完全に同一**に保たれる。その後の適応訓練でゼロ列が学習され、target の情報を取り込んでいく。

#### 語彙整列

Source tokenizer と target tokenizer は異なる可能性がある。埋め込み行列と LM head は target 語彙サイズにリサイズし、ルールは単純だ：

- 共通トークン → pretrained 行をそのままコピー
- target 専用トークン → 文字列を source tokenizer で分割し、subtoken 埋め込みの平均で初期化

これにより、語彙を変えても事前学習で得られた言語知識をなるべく保持する。

### Stage 4: Distill（on-policy TTT 蒸留）

変換後の drafter を、凍結した target からの蒸留で微調整する。損失関数は EAGLE-3 の Training-Time Test（TTT）フレームワークを踏襲する。トークン側は教師強制するが、drafter の隠れ状態は autoregressive に展開するので、推論時とのギャップを抑える。

損失は前方の drafting 位置ほど重視する割引付きクロスエントロピー：

$$
\mathcal{L}_{\mathrm{adapt}} = \sum_{k=0}^{K_{\mathrm{train}}-1} \lambda^k \, \mathrm{CE}\bigl(p_\theta^{(k)}, q_\phi^{(k)}\bigr), \quad \lambda = 0.8
$$

この stage のみが target 依存で、かつそれほど重くない。これが Osprey の本質的な分離だ。

---

## 実験結果

### Qwen3-8B：5×5 クロスドメイン評価

5 つのドメイン（chat, code, commonsense, finance, math）で、それぞれのドメインで適応した drafter を他の4ドメインでも評価する25組の行列を作成した。Osprey の全25組平均 AL は **3.686**。EAGLE-3 from-scratch の **3.176** と比べ **+16.1%** だ。

細分すると、in-domain（ID）では 4.465 vs 3.971（+12.4%）、out-of-domain（OOD）では 3.493 vs 2.977（+17.3%）。OOD/ID 比も 0.782 vs 0.750 と改善している。つまり Osprey の優位は、**分布がずれた環境で最大になる**。

### 大規模 target への転送

同じ1個の2層 backbone を、Qwen3 系列以外の2つの巨大 target に適応した。

**Llama-3.3-70B-Instruct**
Osprey の平均 AL は 2.97。再現した EAGLE-3 比で +21.2%、公式 EAGLE-3 checkpoint 比で +27.5% 改善。スループットも 183.7 tokens/s と、再現版の 155.8、公式の 150.3 をそれぞれ上回る。

**MiniMax-M2.5（229B MoE、FP8 推論）**
Osprey は平均 AL 3.232 vs EAGLE-3 2.634（+22.7%）、スループット 219.8 vs 187.1 tokens/s（+17.5%）。特に LiveCodeBench で +50.7%、HumanEval で +38.7%、MATH-500 で +24.6% と大きく伸びた。

### 多言語転送

MiniMax-M2.5 には code データだけで適応したにもかかわらず、多言語 benchmark でも EAGLE-3 を上回る。MGSM（11言語）で +12.4%、Global-MMLU（42言語）で +23.8%、質問と同じ言語で答える native 設定でも +15.4% だ。事前学習された広い言語知識が、対話応答のような通常の chat 分布以外でも活きている。

![スケール間・多言語での転送と損益分岐](/images/osprey-target-agnostic-drafter-pretraining/fig4.png)

### アブレーション

**学習率の頑健性**
適応時の学習率を 1e-4 から 1e-5 に下げると、Osprey は AL が 3.686 → 3.578 とわずか -0.11 しか落ちない。一方、from-scratch の EAGLE-3 は 3.176 → 2.110 と -1.07 も落ちる。Pretrained な初期化が強い prior を与え、ハイパーパラメータの変動に強い。

**Stage 2 の効果**
剪定直後に distillation へ進む場合と、Stage 2 pre-train を挟む場合を比較。全体 AL は 3.605 → 3.790（+0.185）。ID、OOD 両方で一貫して改善し、pre-train が後続の target 適応に有用な prior を与えていることが示された。

**パラメータ統制比較**
Osprey の優位は単に容量が大きいからではない。2層 EAGLE-3 from-scratch（body 487M）と Osprey（body 265M、1層 EAGLE-3 269M とほぼ同等）を同じ条件で比較すると、前者は1層版より AL +0.094 に留まるが、Osprey は +0.405 も上回る。優位性は事前学習言語 prior にある。

**事前学習量の飽和**
10k steps まで事前学習した backbone と 55k steps のものを比較すると、適応後の全体 AL は 3.552 vs 3.578 とほとんど変わらない。つまり **10k steps（約84B tokens、576 H100-hours）で backbone のほぼ全利益が得られる**。

**損益分岐**
MiniMax-M2.5 の例では、Osprey は10億 tokens 生成あたり約880 H100-hours を節約する。10k steps backbone のコスト 576 H100-hours は **0.65B tokens** で回収できる。55k steps 版（3,040 H100-hours）でも **3.5B tokens** で回収する。複数 target で使い回せばさらに償却が速まる。

---

## 考察：事前学習と target 適応の分離が効く理由

Osprey の成功は、**一般言語モデリング能力**と**target 条件付け**を分離した点にある。既存の EAGLE-3 系は、distillation 時に target のロジットと隠れ状態を同時に学ぼうとする。これは強力だが、target forward のコストが支配的で、広いコーパスを使うことが難しい。結果として drafter は教師分布を模倣するのが精一杯で、分布外では予測が乱れる。

Osprey では、広いコーパスでの言語モデリングを事前に済ませ、target 適応ではあくまでその言語 prior を target の局部分布に矯正するだけだ。だから学習率を下げても壊れにくく、OOD でも予測が安定する。

また、zero-init QKV 拡張は技術的な鍵だ。target 情報を無理やり concat するのではなく、初期時には既存の pre-train 振る舞いを完全に保存しながら、訓練を通じて target 情報を徐々に取り込めるようにする。これが「pre-train した知識を失わずに target には適応できる」という、一見相反する二つの要求を両立している。

語彙整列も同様に、drafter 側の重みを初期値として使うことで、target tokenizer に合わせた語彙変更後も言語知識を保持する。target 側の embedding を使う方法では、そもそも source の言語 prior が失われてしまう。

ただし限界も率直に書かれている。Stage 2 の upfront コストは低トラフィックやワンオフな deployment では回収できない。結果は SGLang の固定設定・3つの target モデルでのみ報告されている。seed も single run 主体なので小さな差は慎重に見るべきだ、と論文本体でも述べている。

---

## 関連研究

**投機デコードのドラフター進化**: EAGLE（Li et al., 2024）が target の penultimate 特徴でドラフトを打ち、EAGLE-2 が動的な draft tree、EAGLE-3 が多層隠れ状態と TTT を導入した。Osprey は EAGLE-3 の TTT フレームワークを流用しつつ、drafter 本体を random init ではなく pre-trained shallow LM から始める点で異なる。

**蒸留の系譜**: DistillSpec（Zhou et al., 2024）は on-policy white-box distillation で投機受入率を高めた。Osprey は蒸留目的関数そのものを新設するのではなく、**学生の初期化を強める**ことで同じ distillation 予算をより効果的に使う。

**再利用・オンライン化の試み**: OmniDraft（Ramakrishnan et al., 2025）は cross-vocabulary 検証と n-gram cache で異なる target 間をつなぐ。Online Speculative Decoding（Liu et al., 2024b）は観測トラフィックから継続的に微調整する。Osprey はこれらと直交し、**広範な web コーパスを使った1回の事前学習**で汎用 backbone を作る方向を突き進めた。

**self-speculative 手法**: LayerSkip（Elhoushi et al., 2024）は target 自体の浅い層を early-exit として訓練する。これは drafter を別途必要としないが、target 側の変更を強いる。Osprey は独立した shallow drafter を使うため target を無改造で使える。

**近傍テーマとの関係**: 本ブログでは Uno（離散拡散による AR 無損失高速化）や LILA（構造化プルーニング）、BiToK-SD（自己蒸留のトークン選択）など、効率化を扱ってきた。Osprey は異なる軸──**推論サービング時の speculative decoding**──に座るが、共通して語られるのは「学習済み知識をどう圧縮・転用するか」という問題意識だ。

---

## まとめ

Osprey は投機デコードの drafter を「毎回 target ごとに narrow な分布で育てる」設計から、「汎用 small LM を浅く剪定し、1回の target-agnostic 事前学習で reusable backbone にする」設計へと転換した。4段階パイプラインのうち、計算の重い pre-train は1回だけ、軽い convert と distill を target ごとに繰り返すだけで、Qwen3-8B から Llama-3.3-70B、229B MoE の MiniMax-M2.5 まで転用できる。

数値的には EAGLE-3 比で AL +16.1%〜22.7%、OOD・多言語でさらに大きな優位を示し、10k steps の backbone でほぼ全恩恵が得られる。これは LLM 推論を加速する仕組み自体が、加速対象である大モデルの汎化力をようやく共有し始めた、と言えるだろう。

---

## 参考

- Bie, F., Jian, Y., Yu, Y., Zhou, Z., Shao, Z., Athiwaratkun, B., Song, S. L., Xu, C., Wu, X., & Zhang, T. (2026). [Osprey: Target-agnostic Pre-training Makes Stronger Drafters in Speculative Decoding](https://arxiv.org/abs/2609.09338). arXiv:2609.09338 [cs.CL]. Accepted at EMNLP 2026.
- コード・チェックポイント: [https://github.com/LeanModels/Osprey](https://github.com/LeanModels/Osprey)
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. ICML 2023.
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. arXiv:2302.01318.
- Li, Y., Wei, F., Zhang, C., & Zhang, H. (2025). EAGLE-3: Scaling up inference acceleration of large language models via training-time test. arXiv:2503.01840.
- Elhoushi, M., et al. (2024). LayerSkip: Enabling early exit inference and self-speculative decoding. ACL 2024.
- Zhou, Y., Lyu, K., Rashidi, A. K., Pfister, T., Zaheer, M., Suresh, A. T., Roumeliotis, F., Brunskill, E., & Cui, Y. (2024). DistillSpec: Improving speculative decoding via knowledge distillation. ICLR 2024.
- Xia, H., Yang, Z., Dong, Q., Wang, P., Li, Y., Ge, T., Liu, T., Li, W., & Sui, Z. (2024). Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. ACL 2024 Findings.
