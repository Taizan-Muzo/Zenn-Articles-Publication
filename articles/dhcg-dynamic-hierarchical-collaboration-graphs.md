---
title: "チームは必要になってから増やす──DHCGが実行フィードバックで協調グラフを動的に育てる仕組み"
emoji: "🕸️"
type: "tech"
topics: ["LLM", "マルチエージェント", "Multi-Agent Reasoning", "POMDP", "推論"]
published: true
---

## TL;DR

LLM マルチエージェントシステム（MAS）は、専門家を集めれば複雑な問題を解ける可能性を持つ。しかし既存の自動構築手法には大きなしわ寄せがあった。①**役割の構成が固定されている**（graph-based MAS）、②**エージェント間の情報依存がテンプレートで決まるため、不要なものが流れ込んだり重要な証拠が欠落したりする**（workflow-based MAS）、③**対話の深さや規模が実行前に決まるため、簡単な問題で無駄に回り、難問で回数が足りなくなる**。

DHCG はこれらを実行時に動的に解く。复旦大学らの研究チームが提案する本手法は、MAS 構築を**部分観測マルコフ決定過程（POMDP）**として定式化し、**Planner** がクエリと逐次の実行フィードバックを見ながら、階層的協調グラフをゼロから育てていく。必要な専門家をその場で生成し、必要な情報だけを下流にルーティングし、十分なら早期に打ち切り、足りなければさらに層を追加する。

Planner の学習には、少数の軌跡から多数の選好対を作る**action-aware preference optimization（APO）**を提案。クロスアクション対（EXPAND vs FINALIZE）と同一アクション対（EXPAND vs EXPAND）を使い分け、下流成果とコストを同時に見る。7 つのベンチマーク（コード・数学・知識推論）で平均 **73.43%** を達成し、single-agent ベースラインより **+13.06**、既存 MAS より **+2.77〜+8.02** ポイント上回った。Planner は 8B の Qwen3 に学習させただけで、GPT-5.4-mini や DeepSeek-V4-Pro を Planner に使った場合と同等の平均スコアを出す。

> **Core insight**: マルチエージェントの強さは「最初に何人集めるか」ではなく、「実行中に何人、どんな専門家が必要かを都度判断できるか」にかかっている。

---

## 背景：マルチエージェントチームを動的に組み上げるという問い

### 静的な役割から動的な構成へ

LLM 単体でも強いが、専門性や視点を分担した複数エージェントが協力すれば、もっと難しい問題に挑める。AutoGen、CAMEL、MetaGPT など初期の MAS フレームワークは、人手で設計した役割と相互作用パターンに依存していた。これはタスク固有の適応性に限界があり、作る労力も大きい。

その後、graph-based MAS（G-Designer、GPTSwarm、AgentPrune など）はエージェント間の通信トポロジーを自動で最適化し、workflow-based MAS（ADAS、AFlow、MaAS など）は実行可能な手続きを探索する方向が登場した。だが、これらもいずれかの意味で**実行前に設計が固定される**か、**決められた部品の範囲内で最適化する**に留まる。

### 既存手法の3つの限界

論文は、既存の動的 MAS 手法が抱える3つの限界を整理する。

1. **Limited Composition（限られた構成）**: あらかじめ定義されたエージェントノードやワークフロー演算子からしか構成できない。実行中に「この問題にはこういう専門家が必要だ」と新しい役割を生やすことができない。
2. **Misaligned Dependencies（ズレた依存関係）**: 通信の依存はテンプレートや大域的な目的で決まることが多い。すると、下流のエージェントにとって無関係な文脈が流れ込んだり、逆に欠かせない中間証拠が届かなかったりする。
3. **Inflexible Scale（硬直した規模）**: 対話の上限ラウンド数や、ワークフローの深さが事前に決まっている。簡単な問いでも決められた回数だけ回し続け、難問では足りずに途中で打ち切られる。

![既存MASの3つの限界とDHCGの新しいパラダイム](/images/dhcg-dynamic-hierarchical-collaboration-graphs/fig1.png)

これらを解くため、著者たちは現実のチーム作りに着想を得る。本気のプロジェクトでは、最初から全員を集めておくのではなく、必要に応じて専門家を呼び、不要になれば締める。DHCG はこの「**実行フィードバック駆動の漸進的グラフ構築**」を MAS に実装する。

---

## 方法詳解：MAS 構築を POMDP として動的に解く

### 階層的協調グラフ

DHCG はマルチエージェント協調を**階層的有向非巡回グラフ（DAG）** $G=(V,E)$ でモデル化する。ノード $v_i \in V$ は1回のエージェント実行を表し、辺 $(v_j, v_i) \in E$ は $v_j$ の出力が $v_i$ に入力として渡される情報依存を表す。

重要な制約として、ノード集合を層に分割する。

$$
V = \bigsqcup_{l=0}^{L} V_l, \qquad E \subseteq \bigcup_{l=1}^{L}(V_{l-1} \times V_l)
$$

つまり、**層内のエージェントは相互に独立**であり並列実行できる。層を跨いだときだけ情報が流れる。これにより「同じ段階での並列協力」と「段階を進むための逐次依存」を区別する。

各ノードは言語モデル方策 $\pi_{\theta_i}$ と命令プロンプト $p_i$ で指定される。元のクエリ $q$ と、前層からルーティングされた入力集合 $x_i = \{ y_j \mid (v_j, v_i) \in E \}$ を受けて、出力 $y_i \sim \pi_{\theta_i}(\cdot \mid q, p_i, x_i)$ を生成する。

### MAS 構築を POMDP に再定式化

DHCG ではグラフを一度に設計しない。Planner が実行中に次々と決定する。これを POMDP として定式化する。

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, T, R, \Omega, O, \gamma \rangle
$$

- $\mathcal{S}$: グラフ構築の潜在状態
- $\mathcal{A}$: Planner の行動空間
- $T$: 遷移関数
- $R$: 最終的なタスク性能と実行コストを含む報酬
- $\Omega$: 観測空間
- $O$: 観測関数
- $\gamma$: 割引率

Planner の観測は、クエリ $q$ と直前に実行された層の役割・出力対の集合からなる。

$$
o_t = O(s_t) = \begin{cases}
(q, \emptyset), & t=0, \\
(q, \mathcal{Z}_{t-1}), & t \geq 1
\end{cases}
$$

ここで $\mathcal{Z}_{t-1} = \{ (\rho_i, y_i) \mid v_i \in V_{t-1} \}$ である。Planner はこの観測に基づいて行動を選択する。

$$
a_t \sim \pi_{\theta}(\cdot \mid o_t), \qquad s_{t+1} \sim T(\cdot \mid s_t, a_t)
$$

### 3 つのモジュール：Planner、Worker、Generator

DHCG は **Planner**、**Worker**、**Generator** の3モジュールで構成される。

![DHCG の3モジュールと動的なグラフ構築ループ](/images/dhcg-dynamic-hierarchical-collaboration-graphs/fig2.png)

#### Step 1: 初期層の構築

初期層では、クエリ $q$ だけから独立した視点を用意する。ここでは互いに独立して推論してもらう。これには明確な理由がある：協調の初期段階で情報を無理に結びつけると、誤りが伝搬・増幅するリスクがあるからである。

Planner はクエリを分析し、必要な初期エージェント数 $n_0$ と互補的な役割を生成する。

$$
V_0 = \{ v_i = (\rho_i, p_i) \}_{i=1}^{n_0} \sim \pi_{\theta}^{\mathrm{init}}(\cdot \mid q)
$$

Workers が並列に実行し、その役割・出力対の集合 $\mathcal{Z}_0$ を Planner に返す。

#### Step 2: 動的計画ループ

各層が終わるたびに、Planner は次のような2種類の行動から選ぶ。

$$
\mathcal{A}_t = \begin{cases}
\mathrm{FINALIZE}(C_{t-1}), & C_{t-1} \subseteq V_{t-1}, \\
\mathrm{EXPAND}(V_t, E_t), & E_t \subseteq V_{t-1} \times V_t
\end{cases}
$$

- **FINALIZE**: 構築を打ち切り、選んだノード集合 $C_{t-1}$ の出力だけを Generator に渡して最終解答を作る。
- **EXPAND**: 新しい層 $V_t$ を追加する。ただし、すべての前層ノードと繋げるのではなく、各新規ノード $v_i^{(t)}$ に対して実際に必要な源ノード $\mathcal{S}_i^{(t)} \subseteq V_{t-1}$ を選ぶ。**疎なルーティング**を行う。

この反復は、最大層数 $L$ に達するか、Planner が FINALIZE を選ぶまで続く。

#### Step 3: 解答合成

Generator は選ばれた役割・出力対だけを見て最終的な答えを生成する。これにより、冗長な中間文脈を排除し、簡潔な解答合成が可能になる。

### Planner の学習：Action-aware Preference Optimization

Planner は役割生成・依存ルーティング・終了判断を同時に行う。これを教師あり学習（SFT）で訓練すると、特定の計画パターンに過学習しやすい。そこで論文では、下流の実行結果から選好対を作り、加重 DPO で学習する **action-aware preference optimization（APO）** を提案する。

![action-aware preference optimization の概要](/images/dhcg-dynamic-hierarchical-collaboration-graphs/fig3.png)

#### 軌跡木の収集と価値推定

まず、Planner の候補行動を複数サンプリングし、再帰的に実行して軌跡の木を作る。各候補行動 $n_t$ の価値を、下流の最終スコアとコストから推定する。

$$
V_t = \begin{cases}
r - \lambda \cdot C_T, & t=T, \\
\gamma \cdot \bar{V}_{t+1} - \lambda \cdot C_t, & t<T
\end{cases}
$$

$r \in [0,1]$ は最終解答スコア、$C_t$ はトークンコスト、$\lambda$ は精度と効率のトレードオフ、$\gamma$ は割引率、$\bar{V}_{t+1}$ は子行動の平均価値である。

#### クロスアクション対と同一アクション対

EXPAND と FINALIZE を比較するとき、コストをそのまま考慮してしまうと「低コストな FINALIZE が勝つ」という偏りが生じる。そこで論文では次のように使い分ける。

- **クロスアクション対**（主に EXPAND vs FINALIZE）: $\lambda=0$ にして、タスク性能だけで比較する。これにより、正しい理由で終了すべきか追加すべきかを学ぶ。
- **同一アクション対**（EXPAND vs EXPAND など）: コスト込みの価値で比較する。同じ「追加」行動の中でも、どれが効率的かを区別する。

価値差が小さい対は破棄し、十分な差がある対だけを残す。

#### 重み付き DPO

各選好対を $(o_i, a_i^{+}, a_i^{-}, w_i)$ と表す。$w_i$ は価値ギャップをクリップしたものであり、平均で正規化して $\tilde{w}_i$ とする。凍結した参照 Planner $\pi_{\mathrm{ref}}$ を使い、DPO 損失を次のように定義する。

$$
\Delta \rho_\theta^{(i)} = \log \frac{\pi_\theta(a_i^{+} \mid o_i)}{\pi_{\mathrm{ref}}(a_i^{+} \mid o_i)} - \log \frac{\pi_\theta(a_i^{-} \mid o_i)}{\pi_{\mathrm{ref}}(a_i^{-} \mid o_i)}
$$

$$
\mathcal{L}_{\mathrm{APO}} = -\frac{1}{N} \sum_{i=1}^{N} \tilde{w}_i \log \sigma\left( \beta \Delta \rho_\theta^{(i)} \right)
$$

クロスアクション対が「広げるか確定するか」の大判断を、同一アクション対が「同じ選択肢の中での質と効率のトレードオフ」を学ばせる。

---

## 実験結果：平均73.43%で既存MASを上回る

### 設定

評価は7つのベンチマークで行う。コード推論は HumanEval と MBPP、数学推論は MATH-500・GSM8K・AIME 2025、知識集約的推論は GPQA Diamond と MMLU-Pro である。

Planner は Qwen3-8B からファインチューニングし、Workers と Generator は Qwen3-8B の生のモデルを使う。比較対象は single-agent（IO、CoT、CoT-SC）、静的 MAS（LLM-Debate、MacNet）、グラフ型 MAS（G-Designer、ARG-Designer）、ワークフロー型 MAS（AFlow、MaAS）のすべてである。すべてのエージェントは同一の LLM バックボーンと同一の推論設定で動かす。

### 全体性能

![7ベンチマーク性能比較と消融実験](/images/dhcg-dynamic-hierarchical-collaboration-graphs/fig4.png)

DHCG は平均 **73.43%** で最高性能を記録した。single-agent ベースライン（IO）より **+13.06**、最強の静的 MAS（LLM-Debate）より **+4.76**、最強のグラフ型 MAS（G-Designer）より **+3.33**、最強のワークフロー型 MAS（MaAS）より **+2.77** ポイント上回った。

| ベンチマーク | IO | CoT | MaAS | DHCG |
| --- | --- | --- | --- | --- |
| MBPP | 63.73 | 70.48 | 82.01 | **84.36** |
| HumanEval | 82.70 | 85.75 | 89.82 | **93.13** |
| MATH-500 | 81.20 | 84.60 | 87.33 | **88.73** |
| AIME | 17.78 | 18.89 | 28.89 | **30.00** |
| GSM8K | 88.09 | 92.58 | 92.89 | **93.93** |
| GPQA | 37.54 | 39.90 | 49.83 | **55.89** |
| MMLU-Pro | 51.57 | 54.29 | 63.86 | **68.00** |

特に GPQA で **+18.35**、MMLU-Pro で **+16.43** ポイント上昇しており、知識集約的なタスクで動的な役割分担が効いていることがわかる。コード生成でも HumanEval で **+10.43**、MBPP で **+20.63** と大きく伸びている。興味深いことに、Planner の訓練データは数学とコードだけであるにもかかわらず、GPQA や MMLU-Pro でも高い性能を出している。

### Pass@5 との経験的上界

5 回サンプリングしたときの Pass@5 でも DHCG は single CoT を上回る。HumanEval では **98.47%**、MATH-500 では **95.20%**、MMLU-Pro では **81.71%** を達成した。これは、異なるサンプリングで多様な協調グラフと推論経路が得られるためである。

### 汎化分析

**Planner の入れ替え**: GPT-5.4-mini を Planner にしても平均 60.76%、DeepSeek-V4-Pro でも 60.78%、訓練済み Qwen3-8B Planner でも 60.79% となった。8B のコンパクトな Planner が、大規模な Planner と同等の計画能力を引き出せることを示している。

**Worker の入れ替え**: 訓練済み Planner を固定し、Workers を DeepSeek-V4-Pro や GPT-5.4-mini に変えても、対応する single CoT ベースラインを上回る。学習したグラフ構築方策が未知の Worker モデルにも転移する。

### 消融実験

Planner 訓練を比較すると、SFT はベース Planner より平均 **-1.21** ポイント低下した。トークン単位の模倣学習は、特定の計画パターンに過学習し、逐次の実行フィードバックへの適応を損なう。これは「SFT してから RL する」という定番パイプラインを疑問視する結果でもある。Vanilla DPO はベース Planner よりわずかに改善し、提案の加重 APO がさらに上回った。

グラフ構築の3つの動的要素をそれぞれ固定化すると、動的スケールの除去が最も大きな打撃となった（平均 **-2.63**、HumanEval で **-4.33**）。これは協調の規模を事前ではなく実行中に決めることが、最も重要であることを示唆している。

| 設定 | MMLU-Pro | GSM8K | HumanEval | 平均 |
| --- | --- | --- | --- | --- |
| Base Planner | 67.14 | 93.55 | 91.35 | 84.01 |
| SFT | 66.29 | 93.81 | 88.30 | 82.80 |
| Vanilla DPO | 66.71 | 93.40 | 92.37 | 84.16 |
| w/o Dyn. Routing | 67.00 | 93.71 | 91.86 | 84.19 |
| w/o Dyn. Scale | 65.62 | 92.76 | 88.80 | 82.39 |
| w/o Dyn. Role | 66.19 | 93.78 | 89.31 | 83.09 |
| DHCG (APO) | **68.00** | **93.93** | **93.13** | **85.02** |

### コストと難易度適応

DHCG はクエリごとに計算資源を配分する。MMLU-Pro では、G-Designer より **50% 以上トークンを削減**しつつ高い精度を達成し、MacNet より **32.16% 少ないトークン**でより正確である。難易度分析では、難問ほど多くの Workers を含む大きなグラフを構築し、易問では小さなグラフで済ませる傾向が確認された。

---

## 考察：何がこの方法を強くしたのか

### 1. 「動的スケール」が最も重要

消融実験で最も大きな性能低下をもたらしたのは「動的スケール」の固定化だった。これは意外かもしれない。なぜならば、多くの議論が「どのエージェントを繋げるか」に集まりがちだが、実は「いつ協調を終えるか・広げるか」が最も効果的な指揮の一部だからである。DHCG は、簡単な問いに大規模なグラフを使わず、難問でこそ深く掘り下げることができる。

### 2. SFT は逆効果という発見

SFT が性能を落とした点にも注目したい。これは単に「模倣データの質が悪い」という話ではない。SFT はトークン単位の尤度最大化であり、Planner が個別の実行フィードバックに応じて行動を変える「意思決定方策」を学ぶのに不向きだからである。DHCG のような逐次意思決定タスクでは、強化学習的な選好最適化の方が、汎化と適応の両面で優位に働く。

### 3. 訓練ドメイン外への転移

Planner を数学・コードのデータだけで訓練したにもかかわらず、GPQA や MMLU-Pro で強いのは興味深い。これは、DHCG が学んだのは「ドメイン固有の役割」ではなく、「どういう状況でどういう協調構造を作るべきか」というメタレベルの計画能力に近いからではないかと推測できる。ただし、エージェントの専門性が問われる極めて特殊なドメインでは、より広い訓練データが必要かもしれない。

### 4. コストと精度の両立

動的にグラフを構築することで、固定したテンプレートよりも無駄な通信を減らせる。MMLU-Pro の例が示すように、精度を上げながらトークンを削減できる。これは MAS が実用化される上で重要な性質である。ただし、 Planner 自身の推論コストも含めた総コスト分析が、より大規模な運用では必要になるだろう。

---

## 関連研究：MAS 構築の系譜

DHCG は静的 MAS、グラフ型 MAS、ワークフロー型 MAS、そして方策最適化型 MAS のいずれとも位置づけが異なる。

**静的 MAS**: AutoGen、CAMEL、MetaGPT は人手で役割とプロトコルを設計する。タスク横断的な汎化には限界がある。

**グラフ型動的 MAS**: GPTSwarm、G-Designer、ARG-Designer、AgentPrune はトポロジーを学習または剪定する。しかし、通常はあらかじめ用意されたノード集合の中で最適化するため、新しい専門家を実行中に生やせない。

**ワークフロー型動的 MAS**: ADAS、AFlow、MaAS は探索や進化、フィードバックで手続きを設計する。ただし、依存関係はしばしばテンプレートまたは大域目的で決まり、下流の情報要求との整合が弱い。

**方策最適化型 MAS**: ScoreFlow は実行スコアから選好を学び、DyFlow は自己対戦で Designer を訓練し、Workflow-R1 や FlowSteer は多ターンのワークフロー構築を最適化する。これらは大筋で DHCG と近いが、DHCG は「階層的グラフの役割・依存・規模を同時に動的に決める」点で異なる。特に、EXPAND と FINALIZE のクロスアクション対を区別して扱う APO は、他の方法では見られない設計である。

---

## まとめ

DHCG は、LLM マルチエージェントシステムの協調グラフを実行中に動的に構築する新しいパラダイムを提案した。MAS 構築を POMDP として定式化し、Planner が役割生成・依存ルーティング・終了判断を逐次行う。初期層では独立した視点を用意し、それ以降は観測された実行フィードバックに基づいて、新しい専門家を追加するか、または構築を打ち切って解答を合成するかを決める。

Planner の訓練には、少数の軌跡から多数の選好対を作る action-aware preference optimization を導入した。クロスアクション対と同一アクション対を使い分けることで、下流のタスク性能と実行コストを同時に考慮できる。実験では7つのベンチマークで平均73.43%を達成し、既存の MAS 手法を2.77〜8.02ポイント上回った。さらに、8B の訓練済み Planner が大規模な Planner と同等の性能を出し、未知の Worker モデルにも転移する。

論文が示唆するのは、マルチエージェント協調の次の段階は「設計図を作る」ことではなく、「実行しながらチームを組み替える能力」を持たせること、という点である。今後、Planner の計画コストとのトレードオフ、より長い文脈を持つタスク、ツール使用との組み合わせなどが、さらなる研究の方向になりそうだ。

---

## 参考

- Ren, J., Yuan, J., Huang, C., Ma, H., Fan, J., & Chen, T. (2026). **DHCG: Dynamic Construction of Hierarchical Collaboration Graphs for LLM-Based Multi-Agent Reasoning.** arXiv preprint arXiv:2610.07835. https://arxiv.org/abs/2610.07835
- Wu, Q., et al. (2024). AutoGen: Enabling next-gen LLM applications via multi-agent conversations. *CoLM*.
- Hong, S., et al. (2024). MetaGPT: Meta programming for a multi-agent collaborative framework. *ICLR*.
- Li, G., et al. (2026). Assemble your crew: Automatic multi-agent communication topology design via autoregressive graph generation. *AAAI*.
- Zhang, J., et al. (2025). AFlow: Automating agentic workflow generation. *ICLR*.
- Zhang, G., et al. (2025). Multi-agent architecture search via agentic supernet. *ICML*.
- Wang, Y., et al. (2025). DyFlow: Dynamic workflow framework for agentic reasoning. arXiv:2509.26062.
