---
title: "SPPO: PPOを超える長距離推論RL"
emoji: "⚡"
type: "tech"
topics: ["LLM", "強化学習", "推論モデル", "ACL2026"]
published: true
---

## TL;DR

- 推論LLMのRL訓練において、標準的なToken-Level PPOは長いChain-of-Thought（CoT）で信用分配が崩壊し、訓練が不安定になる
- SPPO（Sequence-Level PPO）は、推論プロセス全体を1つの「原子アクション」として扱い、**Contextual Bandit問題**に再定式化することでこの課題を解決
- GRPO（N=8）と同等の性能を達成しながら、**5.9倍の訓練速度向上**と**12.8%のGPUメモリ削減**を実現
- 7BモデルでAverage@16スコア **58.56** を記録（Baseモデル 52.49 → +6.07pt）
- ACL 2026 Main Conference採択

![](/images/sppo-sequence-level-ppo/fig1_comparison.png)

## 問題背景

推論LLM（Reasoning LLM）の訓練において、Proximal Policy Optimization（PPO）は中核的な役割を果たしている。DeepSeek-R1の成功以降、Verifiable Reward（検証可能な報酬）を用いた強化学習が推論能力の伸長に不可欠であることは広く認識されている。

しかし、PPOを長いCoT推論タスクに適用しようとすると、2つの深刻な壁にぶつかる。

### 壁1：長距離の信用分配崩壊

標準PPOは各TokenごとにAdvantage（有利度）を計算する。しかしCoT推論では、1つの推論軌跡に数百〜数千Tokenが含まれる。最終的に「答えが正しい（R=1）」という結果だけが得られる場合、どのTokenが正解に貢献し、どれが有害だったのかを特定するのは極めて困難だ。この**Temporal Credit Assignmentの不安定性**が、訓練の安定性を著しく損なう。

### 壁2：Value Modelのメモリコスト

Token-Level PPOでは、Policy Modelと同サイズのValue Model（Critic）が必要だ。7BのPolicy Modelを訓練する場合、さらに7BのCriticをGPUに載せる必要があり、メモリ消費が倍増する。

### 既存の回避策とその限界

GRPO（Group Relative Policy Optimization）はCriticを排除し、同じプロンプトから複数サンプル（N=8）を生成してGroup内の相対比較でAdvantageを推定する。これによりCriticのメモリ問題は解消されるが、**8倍のサンプリング計算量**が訓練スループットの重大なボトルネックになる。

このジレンマ——**安定性 vs 計算効率**——をどう乗り越えるか。ここがSPPOのモチベーションだ。

![](/images/sppo-sequence-level-ppo/fig4_resource_efficiency.png)

## 方法

### コアアイデア：推論をContextual Banditとして扱う

SPPOのキーインサイトはシンプルだが強力だ。

> **推論プロセス全体（プロンプト + CoT）を、1つの「原子アクション」として扱う。**

具体的には、推論タスクをToken-LevelのMarkov Decision Process（MDP）ではなく、**Sequence-Level Contextual Bandit問題**として再定式化する。

- **状態（State）** $s_p$：プロンプト（問題文）— 推論中は不変
- **アクション（Action）** $a$：CoT推論軌跡全体（1つのまとまり）
- **報酬（Reward）** $R \in \{0, 1\}$：最終的な正誤判定

この定式化により、信用分配の問題が根本から消える。Advantageは序列全体に対して1つだけ計算すればよいのだ。

### Advantage推定

$$A(s_p, a) = R - V_\phi(s_p)$$

$V_\phi(s_p)$ はプロンプト $s_p$ に対する成功確率の推定値。$R$ は0 or 1の結果報酬。このスカラーAdvantageを、軌跡内の**全Tokenに一様に伝播**させる。

### Value Modelの学習

Value Modelの損失関数にはBinary Cross-Entropy（BCE）を用いる：

$$L_V(\phi) = -\mathbb{E}[R \log V_\phi(s_p) + (1-R)\log(1-V_\phi(s_p))]$$

これは「このプロンプトの正解確率はいくらか？」という回帰問題として捉え直している。

### Policyの更新

PPOのClipping Objectiveを踏襲しつつ、Token-Level Advantageの代わりにSequence-Level Advantageを用いる：

$$J_{\text{SPPO}}(\theta) = \mathbb{E}\Big[\min\big(r_t(\theta)A(s_p,a),\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A(s_p,a)\big)\Big]$$

ここで $r_t(\theta) = \frac{\pi_\theta(a_t|s_p, a_{<t})}{\pi_{\theta_k}(a_t|s_p, a_{<t})}$ は各TokenのImportance Ratio。

### 解耦Critic設計

SPPOのもう一つの工夫は、**Criticサイズの解耦**だ。7BのPolicy Modelに対して、わずか1.5BのCriticを用いる。Contextual Bandit定式化により、Value Modelはプロンプトの難易度予測だけを行えばよく、Token-Levelの細かい推定が不要なため、小さいモデルでも十分機能する。

## 実験結果

### 数学的推論ベンチマーク（7Bモデル）

DeepSeek-R1-Distill-Qwen-7Bをベースに、5つの数学ベンチマークで評価（Average@16）。

![](/images/sppo-sequence-level-ppo/fig2_benchmarks_7b.png)

| 方法 | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Average |
|------|--------|--------|-------|---------|---------|---------|
| Base Model | 41.25 | 26.67 | 79.38 | 87.20 | 27.94 | 52.49 |
| Token-level PPO | 45.20 | 35.42 | 85.31 | 88.48 | 27.80 | 56.44 |
| RLOO | 46.67 | 32.50 | 86.88 | 90.35 | 28.72 | 57.02 |
| GRPO (N=8) | 47.08 | 35.00 | 86.25 | 90.15 | 28.74 | 57.44 |
| **SPPO** | **50.83** | 35.00 | 86.25 | 90.13 | 28.35 | **58.11** |
| **SPPO + Critic** | **52.29** | 34.58 | **87.19** | 89.88 | **28.86** | **58.56** |

**SPPOは全手法中最高のAverageスコアを記録**。特にAIME24ではBase Modelから+11.04pt、PPOから+7.09ptの大幅改善。

### 訓練効率

![](/images/sppo-sequence-level-ppo/fig3_training_efficiency.png)

- **GRPO比5.9倍の訓練速度**：GRPO（N=8）が40時間かけて到達する性能水準を、SPPOは**約22時間**で到達
- **GPUメモリ12.8%削減**：解耦Critic（7B+1.5B）により、標準設定（7B+7B）と比べて大幅なメモリ節約
- **サンプル効率N=1**：GRPOが1プロンプトあたり8サンプル必要なのに対し、SPPOは1サンプルで済む

### 価値モデルの品質検証

- プロンプトの予測難易度と実証的な解法率（AVG@64）の間に**ピアソン相関 r=0.642**、スピアマン順位相関 0.664を確認
- 小さいCritic（1.5B）でもプロンプトの相対的な難易度を適切にキャプチャできている

### RLVR制御実験

5つの制御タスク（Precision CartPole, MountainCar, Hopper, LunarLander, Pendulum）で検証。特に**長ホライズンタスク（Hopper H=1000、MountainCar）において、標準PPOが失敗する設定でSPPOが安定して解を発見**し、Sequence-Level定式化の優位性を裏付けた。

### 消融研究：なぜBCE損失だけではダメか

論文の重要な消融実験として、「PPOフレームワーク + BCE損失」を試した結果が報告されている。これによりSPPOの性能向上がBCE損失そのものではなく、**Sequence-Level Contextual Bandit定式化（Advantageの均一伝播）に起因する**ことが示された。Token-LevelにBCEを適用しても、信用分配の不安定性は解消されないのだ。

## 考察

### Contextual Bandit再定式化の威力

SPPOの成功は、「問題の定式化を変えるだけで劇的な改善が得られる」という研究の王道を歩んでいる。推論という本来SequentialなプロセスをBandit問題に還元する発想は一見大胆だが、検証可能報酬（Verifiable Reward）が使える文脈では極めて合理的だ。

正誤判定ができるタスクにおいて、途中のTokenごとに細かいCreditを割り当てる必要は本当にあるのか？ SPPOは「ない」という答えを出している。

### 「Tail Effect」の解消

標準PPOの信用分配が長いCoTで壊れる原因として、論文は「Tail Effect」を指摘している。推論軌跡の末尾にあるTokenほど報酬に近いため、Advantageが偏りやすい。SPPOは全Tokenに同じAdvantageを与えるため、このバイアスが原理的に存在しない。

### 実用面でのインパクト

- **N=1**で動くということは、大規模分散訓練のオーバーヘッドが劇的に減る
- **小さなCritic**はGPUメモリの観点からだけでなく、Critic自身の訓練安定性にも貢献するはずだ
- DAPOや他のRLフレームワークにも組み込み可能な汎用的なアプローチ

### 留意点

- 評価は数学推論タスクに限定。コーディング、論理推論、一般推論への一般化は未検証
- GRPOとの比較はN=8固定。Nを変動させた場合のトレードオフ分析があるとさらに説得力が増す
- プロンプト単位のValue予測は効果的だが、より複雑なタスクではSequence-Levelの粒度が粗すぎる可能性もゼロではない

## 関連研究との位置づけ

| 手法 | 課題 | SPPOの立ち位置 |
|------|------|----------------|
| Token-level PPO | 信用分配不安定 + 大きなCritic | → **安定化 + 軽量化** |
| GRPO | 8倍のサンプリングコスト | → **N=1で同等性能** |
| RLOO | Leave-one-out baselineの計算量 | → **よりシンプルな定式化** |
| ReMax | 報酬最大化に特化 | → **汎用的なPPO枠組みとの互換** |

DAPO（Decoupled Clip + Dynamic Sampling等の技術）がRLシステムの「ハードウェア面」を整備したとすれば、SPPOは「アルゴリズム面」での根本的な改善と言える。両者は補完関係にあり、組み合わせることでさらに強力なRL訓練パイプラインが構築できるだろう。

## まとめ

SPPOは、推論LLMのRL訓練における「PPOは不安定」「GRPOは遅い」という二律背反を、**Sequence-Level Contextual Banditへの再定式化**という1つのアイデアで解決した。解耦Critic設計によるメモリ効率化も含め、実用面でもすぐに採用可能な手法だ。

ACL 2026への採択も納得の、シンプルかつ効果的な研究。

## 参考

- 論文: [SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks (arXiv:2604.08865)](https://arxiv.org/abs/2604.08865)
- コード: [https://github.com/sustech-nlp/SPPO](https://github.com/sustech-nlp/SPPO)
- ACL 2026 Main Conference
- 著者: Tianyi Wang, Yixia Li, Long Li, Yibiao Chen, Shaohan Huang, Yun Chen, Peng Li, Yang Liu, Guanhua Chen (SUSTech / Microsoft)
