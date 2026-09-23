---
title: "「待って」より「やってる」が推論に効く──Aha-Flow DistillationがFlow Markerの見落しを正す仕組み"
emoji: "🌊"
type: "tech"
topics: ["LLM", "推論", "自己蒸留", "Chain-of-Thought", "FlowMarker"]
published: false
---

## TL;DR

LLMの推論過程に出現する「Wait」「Hmm」といった**Aha Marker**（回溯・修正の合図）ばかりが注目されてきたが、本論文はその補完相手である**Flow Marker**──「I'm doing」「I'm checking」といえば現在の経路を継続確認する発話──を初めて形式化し、推論蒸留に活用する手法 **Aha-Flow Distillation (AFD)** を提案する。Flow-CoT（Aha MarkerをFlow Markerに書き換え、推論内容は保持）を構築し、OPSDの二モード拡張としてAFDを定義。Qwen3-8B/4BでAIME25・HMMT25において一貫した改善を達成した。特に、同じFlow-CoT/Aha-CoT混合でも単純混合より二モード訓練が+0.6上回り、**異質監督の組織化**こそが効くことを示した。

![Aha MomentとFlow Momentの推論軌跡の比較](/images/aha-flow-distillation-reasoning-markers/fig1.png)

## 背景

### Aha Momentという偏り

推論モデルのChain-of-Thought（CoT）において、自己修正や回溯を示す発話──「Wait, actually...」「Hmm, let me reconsider...」──はAha Momentと呼ばれ、GRPOやon-policy distillationなど後段訓練の中心に据えられてきた。DeepSeek-R1の報告以降、この"反思的推論"は推論モデルの代名詞のような扱いを受けている。

しかし、Aha Momentばかりが推論の質を決めるわけではない。過度な回溯は冗長な推論を生み、本来有効だった経路まで見直させてしまう。実際、弱いモデルほど頻繁に方針を切り替え、有望な軌跡を放棄する傾向が報告されている。

### Flow Momentという見落とし

本論文が指摘するのは、Aha Momentの**補完相手**としてのFlow Momentである。現在の推論経路をそのまま継続・確認する発話──「I'm computing the derivative...」「I'm now checking the boundary condition...」──は、軌跡の連続性を支える重要な信号なのに、これまでほぼ無視されてきた。

> 推論軌跡を未分化な文字列として扱う従来の枠組みでは、Aha MarkerとFlow Markerの役割の違いが見えない。しかし、on-policy自己蒸留において**教師分布を条件付ける特権情報の形式**が結果に直結するなら、この区別は看過できない。

### 研究問い

> **後段訓練で用いる推論形式は、最終モデルの性能に影響するか？**

著者らはOn-Policy Self-Distillation（OPSD）を実験基盤に選び、異なる推論スタイルを特権情報として与えたときの影響を体系的に調べる。

## 方法詳解

### Flow-CoTの構築

Flow-CoTは、元のCoTの**推論内容を一切変えずに**、Aha風の談話標識をFlow風に書き換えたものである。書き換えにはDeepSeek-V4-Flash APIを用い、以下の規則に従う：

| 元のAha Marker | 書き換え後のFlow Marker |
|---|---|
| "Okay, let's try to tackle this..." | "I'm tackling this..." |
| "Hmm, this seems complex." | "I'm noticing that this is getting complex." |
| "Wait, actually the formula is..." | "I'm correcting myself: the formula is..." |
| "But wait, let me check..." | "I'm now checking..." |
| "Alternatively, maybe using..." | "I'm trying another approach: ..." |
| "Not directly obvious. Maybe not." | "I'm finding no direct correspondence, so I'm dropping this idea..." |

**重要**: どの行き止まり・自己修正・検証ステップもそのまま保持する。数学的内容は逐語的に保存。段落構造も維持する。変わるのは推論の**語り方**だけである。

つまり、Flow-CoTは意味的に元のCoTと整合しており、主な違いは推論過程の言語化の仕方にある。

### OPSDの復習

On-Policy Self-Distillation（OPSD）では、学生モデルが自身のon-policy生成から学習しつつ、教師が特権参照解を追加で観測して教師分布を形成する。

訓練対 $(x, y^\star)$ について：

1. 学生がon-policy応答 $\hat{y} \sim p_S(\cdot|x)$ をサンプリング
2. 各生成ステップで、教師は特権参照 $y^\star$ を追加で観測し、両モデルは同じ生成済み接頭辞 $\hat{y}_{<n}$ で条件付けられる

$$\mathcal{L}_{\mathrm{OPSD}}(\theta) = \mathbb{E}_{(x,y^{\star})\sim\mathcal{S}}\;\mathbb{E}_{\hat{y}\sim p_{S}(\cdot\mid x)}\sum_{n=1}^{|\hat{y}|} D\!\Bigl(p_{T}\!\left(\cdot\mid x,y^{\star},\hat{y}_{<n}\right)\;\Big\|\;p_{S}\!\left(\cdot\mid x,\hat{y}_{<n}\right)\Bigr)$$

ここで $D$ はJensen-Shannon Divergence（JSD）を用い、極端なトークンが更新を支配するのを防ぐため閾値 $\delta_{\mathrm{clip}}$ で切り詰める。

**核心的非対称性**: 教師だけが $y^\star$ にアクセスできる。ゆえに、**特権情報の形式が教師分布を直接決定し、学生が受け取る監督信号を左右する**。

### Aha-Flow Distillation（AFD）: 二モード訓練

AFDの設計思想は、**学生に要求する推論行動と、教師を条件付ける参照情報を一致させる**ことにある。

![AFDの二モード訓練アーキテクチャ](/images/aha-flow-distillation-reasoning-markers/fig2.png)

各訓練サンプル $x$ について二値モード指示子 $s(x) \in \{0, 1\}$ を定義する：

| | Aha 分枝 ($s(x)=0$) | Flow 分枝 ($s(x)=1$) |
|---|---|---|
| **特権情報** | 簡潔解 $y^\star$ | Flow-CoT $y^\star_{\mathrm{flow}}$ |
| **推論指令** | "step by stepで推論し、最終答を $\boxed{}$ に" | "直接・自信を持って推論し、各ステップを明示、不要な躊躇なく" |

形式的には：

$$(y^{\star}(x),\mathcal{I}(x))=\begin{cases}\bigl(y^{\star},\mathcal{I}_{\mathrm{aha}}\bigr),&s(x)=0,\\[5pt] \bigl(y^{\star}_{\mathrm{flow}},\mathcal{I}_{\mathrm{flow}}\bigr),&s(x)=1.\end{cases}$$

**設計上の要点**:

- **同一分枝内**では学生も教師も同じ指令 $\mathcal{I}(x)$ を受け、教師が追加で特権参照を観測する
- **推論時**には標準反思指令のみを使う。Flow風推論は純粋に訓練信号として機能する
- AFDは共有指令下での単純混合ではなく、**各特権情報を対応する推論指令と対にする**点が本質的に異なる

## 実験結果

### 設定

- **骨格モデル**: Qwen3-8B, Qwen3-4B
- **訓練**: OPSDスキーム、LoRA（$r=64, \alpha=128$）、OpenThoughts-30K
- **評価**: AIME25・HMMT25、各問12サンプル、Avg@12指標
- **推論**: すべてのモデルで同一の汎用反思指令を使用

![AIME25/HMMT25ベンチマーク結果](/images/aha-flow-distillation-reasoning-markers/fig3.png)

### 主結果

**Qwen3-8B**:

| 方法 | AIME25 | HMMT25 | Avg@12 |
|---|---|---|---|
| Base (Instruct) | 65.6 | 43.9 | 54.8 |
| + SFT | 64.2 | 42.9 | 53.6 |
| + GRPO | 68.9 | 46.7 | 57.8 |
| + OPSD† (再現) | 73.1 | 48.6 | 60.8 |
| **+ AFD** | **73.6** | **48.9** | **61.3** |

**Qwen3-4B**:

| 方法 | AIME25 | HMMT25 | Avg@12 |
|---|---|---|---|
| Base (Instruct) | 66.4 | 42.2 | 54.3 |
| + SFT | 62.3 | 43.4 | 52.8 |
| + GRPO | 68.1 | 44.4 | 56.3 |
| + OPSD† (再現) | 69.2 | 45.8 | 57.5 |
| **+ AFD** | **69.7** | **47.5** | **58.6** |

AFDは両モデル規模・両ベンチマークで一貫してOPSD†を上回る。Qwen3-4Bでは+1.1と改善幅がより大きい。**評価時はすべて同じ反思プロンプトを使っている**ため、改善は訓練過程に起因する。

### 消融実験

Qwen3-8Bで、他の設定を統制して消融を行った。

![消融実験の結果](/images/aha-flow-distillation-reasoning-markers/fig4.png)

| 特権情報 | 訓練モード | AIME25 | HMMT25 | Avg@12 |
|---|---|---|---|---|
| ─ | ─ | 65.6 | 43.9 | 54.8 |
| Solution | OPSD | 73.1 | 48.6 | 60.8 |
| Aha-CoT | OPSD | 70.8 | 47.5 | 59.2 |
| Flow-CoT / Aha-CoT | OPSD | 70.8 | 48.1 | 59.5 |
| Flow-CoT / Aha-CoT | AFD | 72.8 | 48.3 | 60.1 |
| **Flow-CoT / Solution** | **AFD** | **73.6** | **48.9** | **61.3** |

（Flow-CoT / Aha-CoTを含む行は50/50混合）

三つの知見が読み取れる：

1. **特権情報の形式がOPSDに実質的影響を及ぼす**: Solution単独（60.8）はAha-CoT（59.2）を明確に上回る。簡潔な解が冗長CoT軌跡より効果的傾向
2. **二モード訓練は独立した寄与を持つ**: 同じFlow-CoT/Aha-CoT混合でも、OPSD 59.5 → AFD 60.1（+0.6）。**信号の組織化**が単なる混合を超える
3. **Flow-CoT/Solution構成が最強**: Aha-CoTをSolutionに差し替えると60.1 → 61.3。Solution-only OPSD（60.8）も凌駕

## 考察

### Flow Markerは何をしているのか

Flow Markerの役割は、推論軌跡の**自己確信を言語化**することにある。「Wait」という発話が軌跡の不連続性を示すなら、「I'm doing」は連続性の確認である。この確認発話が教師分布に組み込まれることで、学生は「推論を続けてよい」という暗黙の正則化効果を受ける──と解釈できる。

### 二モード訓練が単純混合に勝つ理由

同じFlow-CoTとAha-CoTを50/50で与えても、単一のOPSD下では異質な推論スタイルが同じ指令空間で競合し、教師分布が揺れる。AFDでは各特権情報が**対応する指令と対になって**提示されるため、教師分布の条件付けが一貫し、学生が受け取る勾配の分散が抑えられる。これが+0.6の差として現れる。

### なぜSolutionがAha-CoTより強いのか

冗長なAha-CoTを特権情報として与えると、教師が「回溯の仕方」まで分布に反映させ、学生に過剰な自己修正を促す可能性がある。一方、簡潔なSolutionは「答への直接経路」のみを符号化し、推論の冗長性を教師分布から排除する。この直感が、Flow-CoT/Solution構成の強さを説明する──Flow分枝が「継続推進」のスタイルを教え、Aha分枝が「簡潔解答」の内容を教えるという役割分担が成立するからだ。

### 限界

- Flow-CoTの構築に外部API（DeepSeek-V4-Flash）を依存しており、書き換え品質がモデルに依存する
- 評価が数学推論（AIME/HMMT）に限定されており、常識推論やコード生成への一般性は未検証
- Flow Markerの定義がまだ暫定的で、「I'm doing」以外の多様なFlow表現の体系化が不十分

## 関連研究

### 推論軌跡とAha Moment

CoT提示が多段推論を大幅に向上させることはWei et al. (2022)以降広く知られている。Self-consistency（Wang et al., 2022）やTree-of-Thought（Yao et al., 2023）は推論軌跡を探索空間として扱うが、**元認知表現の異なる役割にはほとんど関心を向けていない**。

自己修正の信頼性についてはHuang et al. (2024)が外部フィードバックなしでは限界を示し、RL誘発行動の分析ではGandhi et al. (2025)が検証・回溯・下位目標設定などを離散推論操作として整理した。Bogdan et al. (2025)は重要推論ステップの大部分が自己修正ではないと主張し、Wang et al. (2025)は弱いモデルの過度な方針切替を指摘している──Flow Markerの重要性を示唆する先行知見と言える。

### On-Policy蒸留

知識蒸留（Hinton et al., 2015）からon-policy蒸留（Agarwal et al., 2024）への移行は、訓練分布と生成分布のミスマッチを解消する方向だった。OPSD（Zhao et al., 2026）は外部教師を不要にしたが、多様性低下や過度な確信更新が知られている。Purified OPSD（Shen et al., 2026）や不確実性適応（Ke et al., 2026）は目標関数の修正で対処するが、**本論文は「推論スタイルの組織化」という直交する次元からOPSDを改善する**。

## まとめ

Aha-Flow Distillationは、推論軌跡に潜む**Flow Marker**──現在の経路を継続確認する発話──を初めて形式化し、Aha Markerと対になる推論信号として活用する手法である。Flow-CoTの構築と二モード訓練の導入により、異質な推論監督を単に混ぜるのでなく**推論指令と対にして組織化**することで、OPSDを一貫して改善した。

結果として示されたのは、「どのような推論スタイルを特権情報として与えるか」そして「それをどのように訓練モードに割り当てるか」が、自己蒸留において**無視できない設計選択**であるという事実である。推論の質を語るとき、Aha Momentだけを見るのは半分しか見ていない──Flow Markerもう一度、推論軌跡の中に目を配る必要がある。

## 参考

- Wang, X. & Peng, P. "Aha-Flow Distillation: Flow Markers Matter in LLM Reasoning." arXiv:2609.07036, 2026. ([論文](https://arxiv.org/abs/2609.07036), [コード](https://github.com/Wang-Xiaodong1899/Aha-Flow-Distillation))
- Wei, J. et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
- Guo, D. et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.
- Zhao, J. et al. "On-Policy Self-Distillation for Reasoning." 2026.
- Shen, M. et al. "Purified OPSD." 2026.
- Gandhi, K. et al. "Cognitive Behaviors in Reasoning Models." 2025.
- Bogdan, L. et al. "Reasoning Models Don't Think, They just Rationalize." 2025.
