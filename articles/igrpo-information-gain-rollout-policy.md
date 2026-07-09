---
title: "IGRPO: 情報量ゲイン駆動木構造RolloutでLLM Agentの探索効率を最大化する"
emoji: "🌳"
type: "tech"
topics: ["LLM", "強化学習", "Agent", "検索拡張QA", "推論最適化"]
published: false
---

## TL;DR

LLM Agentが長ホライズンの検索タスクを解く際、**中間状態ごとの情報量（Information Gain）に応じてRollout予算を適応的に配分する**フレームワーク **IGRPO** を提案。情報量の高い探索分岐を優先的に拡張し、低情報量の分岐を漸進的に抑制する木構造Rolloutにより、計算資源の無駄を大幅に削減。理論的に**Rollout過程が教師分布 $\mu_\theta \propto \pi_\theta \exp(\gamma V)$ を誘導すること**を証明し、7つの検索拡張QAベンチマークで一貫して最强ベースラインを上回る（3B平均+3.1pp、7B平均+0.9pp）。

> **論文**: Zhang et al., "Information Gain-based Rollout Policy Optimization" (arXiv:2607.06223, 2026)
> **所属**: 上海交通大学（SJTU）
> **リンク**: [arXiv](https://arxiv.org/abs/2607.06223)

---

## 背景：LLM Agentの検索における計算割当問題

検索拡張型LLM Agentは、外部検索ツールと複数回のインタラクションを通じて問いに答える。各ターンで「何を検索するか」「いつ検索するか」「いつ回答するか」を決定する必要があるが、ここには根本的な**計算割当問題**が潜んでいる。

### 従来アプローチの限界

Chain-based手法（Search-R1, ZeroSearch, IGPOなど）は、1つの問いに対して複数の独立したRolloutを生成する。しかし、**各Rolloutの計算量は均一に割り当てられ**、すでに探索価値の低い中間状態に無駄な計算が費やされる。

Tree-based手法（Tree-GRPO）はRolloutを木構造に拡張し、共有プレフィックスを活用するが、分岐の拡張判断はヒューリスティックまたは不確実性駆動であり、**中間状態の下流有用性を明示的に評価していない**。

![](/images/igrpo-information-gain-rollout-policy/fig1.png)

この図が示す通り、異なる中間状態は情報量において劇的に異なる。IGRPOの核心は、**この差を明示的に測定し、予算配分に反映させる**点にある。

---

## 手法：IGRPOの仕組み

### 1. 答案スコアと情報量ゲイン

まず、中間状態 $h$（これまでの検索履歴）における**正解答案のモデル内尤度**を定義する：

$$s_\theta(h; a) = \pi_\theta(a|h) = \exp\!\left(\frac{1}{L}\sum_{j=1}^{L}\log\pi_\theta(a_j|h, a_{<j})\right)$$

このスコアは、現在の検索状態が正解にどれだけ近づいているかを代理指標として捉える。**情報量ゲイン（IG）**は、親ノードからのスコア増分として定義される：

$$\text{IG}(h) = s_\theta(h; a) - s_\theta(\text{parent}(h); a)$$

根ノードでは $\text{IG}(h_0) = s_\theta(h_0; a)$。**重要な観察**として、正解Rolloutは失敗Rolloutに比べて一貫して高い累積情報量ゲインを示すことが経験的に確認されている。

### 2. 予算認識木構造Rollout

各拡張ステージ $i$ で、アクティブノード集合 $\mathcal{A}_i$ の各ノード $h$ に**ノード価値関数**を計算する：

$$\text{val}(h) = \frac{s_\theta(h;a) + \text{IG}(h)}{2}$$

これを用いて拡張確率を割り当てる：

$$p_i(h) = \frac{\exp(\gamma \cdot \text{val}(h))}{\sum_{h' \in \mathcal{A}_i} \exp(\gamma \cdot \text{val}(h'))}$$

- **高情報量ノード**: 拡張確率が高く、より頻繁にサンプリングされる
- **低情報量ノード**: 拡張確率が低く、計算リソースの消費が抑制される
- **温度 $\gamma$**: 探索-利用のバランスを制御（実験で $\gamma=1$ が最適）

### 3. 理論的基盤：誘導教師分布

IGRPOの最も理論的に深い部分は、**木構造Rollout過程が自然に教師分布を定義する**ことにある。

**定理1**: 各層の拡張予算 $B_t \to \infty$ のとき、サンプリングされる完全Rolloutの分布は以下に収束する：

$$\mu_\theta(o) \propto \pi_\theta(o)\exp(\gamma V(o))$$

ここで $V(o) = \sum_{t=1}^{T-1}\text{val}(h_t)$ は軌跡の累積情報量価値。

**定理2**: この分布は以下の変分問題の唯一の最適解である：

$$\max_\mu \; \mathbb{E}_{o\sim\mu}[V(o)] - \frac{1}{\gamma}\mathbb{D}_{\text{KL}}(\mu \| \pi_\theta)$$

Donsker-Varadhan変分公式により証明される。この結果の意義は大きい——**誘導分布 $\mu_\theta$ は、累積情報量で重み付けられた方策の再重み付き版**であり、現在の方策 $\pi_\theta$ に対する明確な改善方向を提供する。

### 4. 方策最適化目標

教師分布 $\mu_{\theta_\text{old}}$ からサンプリングしたRollout集合に対し、GRPOと同様のクリッピング目的関数で方策を更新する。決定的な違いは**サンプリング分布**——IGRPOは情報量バイアス付きの $\mu_\theta$ からサンプルするため、学習シグナルの質が根本的に向上する。

![](/images/igrpo-information-gain-rollout-policy/fig2.png)

---

## 実験結果

### 全体性能：7ベンチマークで一貫してSOTA

3つの単跳びQA（NQ, TriviaQA, PopQA）と4つの多跳びQA（HotpotQA, 2Wiki, MusiQue, Bamboogle）で評価。

![](/images/igrpo-information-gain-rollout-policy/fig4.png)

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MusiQue | Bamboogle | **Avg** |
|--------|----|----------|-------|----------|-------|---------|-----------|---------|
| Search-R1 | 34.1 | 54.5 | 37.8 | 32.4 | 31.9 | 10.3 | 26.4 | 37.4 |
| GiGPO | 42.0 | 59.5 | 42.4 | 36.9 | 37.0 | 12.6 | 64.1 | 42.3 |
| IGPO | 40.2 | 58.1 | 43.2 | 35.7 | 34.9 | 12.3 | 64.5 | 41.4 |
| Tree-GRPO | 44.5 | 60.4 | 43.7 | 31.1 | 31.7 | 11.9 | 28.0 | 40.8 |
| **IGRPO** | **45.6** | **60.9** | **47.5** | **38.9** | **40.2** | **14.1** | **65.7** | **45.4** |

（Qwen2.5-3B-Instruct、†=訓練データ、⋆=未見データ）

**注目すべき点**：

1. **全7データセットでIGRPOが最高スコア**を達成（3Bモデル）
2. Tree-GRPOは多跳びタスクでBamboogleが28.0%と著しく低下するのに対し、IGRPOは65.7%を維持
3. 未見データ（⋆）でも一貫した改善——過学習の兆候なし
4. 7Bモデルでも同様に平均48.2%で最强を記録

### ポリシーエントロピーの動態分析

![](/images/igrpo-information-gain-rollout-policy/fig3.png)

訓練中のポリシーエントロピーから3つの手法の本質的な違いが浮かび上がる：

- **IGPO**: エントロピーが急速に低下し、過早収束（premature convergence）に陥る。限られた行動セットに過度に集中してしまい、探索が枯渇する
- **Tree-GRPO**: 全訓練期間で高いエントロピーを維持するが、これは情報量を考慮しないランダムな木拡張によるもので、**探索の質が低い**
- **IGRPO**: まず確実な検索行動を学習してエントロピーが適度に低下し、その後有望な中間状態の周辺で探索を広げて微増する——**「学習してから探索する」最適パターン**

### 温度パラメータ $\gamma$ の影響

| $\gamma$ | Avg EM (%) | 解釈 |
|----------|-----------|------|
| 0 | 42.2 | IGバイアスなし → 均等拡張 → IGRPOの利点消失 |
| **1** | **45.4** | **適度な利用-探索バランス** |
| 5 | 44.5 | 過度な利用 → 高IGノードに集中 → 探索不足 |

$\gamma=0$ はTree-GRPOに近い挙動になり、$\gamma$ が大きすぎると計算が少数の高IGノードに偏り、ロバスト性が低下する。$\gamma=1$ が情報量に基づく適応的配分と多様な探索の最適なバランスを達成する。

---

## 考察

### なぜ小さいモデルほど恩恵が大きいのか

3Bモデルでの改善幅（+3.1pp）が7Bモデル（+0.9pp）を上回る傾向は、Agent RL研究で一貫して観察される現象だ。小さいモデルの初期検索方策は信頼性が低く、**「どの中間状態を優先的に探索すべきか」の指導がより重要になる**。IGRPOの情報量ベースの予算配分は、まさにこの脆弱性を補うメカニズムとして機能している。

### Tree-GRPOの失敗から学ぶ

Tree-GRPOは木構造探索を導入した点で先駆的だが、Bamboogleのような複雑な多跳び推論で著しく性能が劣化する。これは**木構造であること自体が十分ではない**ことを示している。分岐の質を判断する基盤（IGRPOでは情報量ゲイン）が不可欠であり、構造的拡張と内容的評価を統合して初めて、木構造Rolloutの潜在能力が発揮される。

### 変分公式の美しさ

定理2の変分表現 $\max_\mu \mathbb{E}[V] - \frac{1}{\gamma}\mathbb{D}_{\text{KL}}(\mu\|\pi_\theta)$ は、RLの报酬最大化とKL正則化の構造と完全に相同である。つまり、**IGRPOのRollout過程自体が「方策改善」の役割を果たしている**——方策更新の前に、サンプリング段階で既に情報量に基づく改善方向が埋め込まれている。この二段構えの最適化構造は、理論的に非常に美しい。

---

## 関連研究

### チェーンベースAgent RL
- **Search-R1 / ZeroSearch**: RLで検索行動を誘発する基礎的なアプローチ
- **StepSearch / GiGPO**: ターンレベルの細粒度信用割当を導入
- **IGPO**: 情報量ゲインを**内発的報酬**として利用（IGRPOは**Rollout配分**に利用する点が異なる）

### ツリーベース探索
- **Tree-GRPO**: 木構造Rolloutの先駆け。共有プレフィックスで計算効率を改善
- **ARPO / AEPO**: エントロピー誘導のRollout生成

### IGRPOの独自性
IGPOが情報量ゲインを**報酬信号**として扱うのに対し、IGRPOは**Rollout予算の配分基準**として利用する。この違いは本質的——IGRPOは学習シグナルの質をサンプリング段階で向上させ、方策最適化に高品質なデータを供給する。

---

## まとめ

IGRPOは、LLM Agentの検索プロセスにおける**中間状態の情報量を明示的に評価し、計算予算を適応的に配分する**ことで、長ホライズン探索タスクの効率を根本的に改善するフレームワークである。

- **情報量ゲイン駆動の木構造Rollout**: 高情報量分岐を優先拡張、低情報量分岐を抑制
- **理論的保証**: Rollout過程が教師分布 $\mu_\theta \propto \pi_\theta \exp(\gamma V)$ を誘導することを証明
- **実験的検証**: 7ベンチマークで一貫してSOTA、3B平均+3.1pp、7B平均+0.9pp
- **エントロピー動態**: 「学習してから探索する」最適パターンを実現

木構造Rolloutの「形」と情報量評価の「中身」を統合したこのアプローチは、Agent RLの計算効率化において新たな標準を提示するものと言える。

---

## 参考

1. Zhang, Y., Xu, F., Ding, J., et al. "Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents." arXiv:2607.06223, 2026.
2. Shao, Z., et al. "GRPO: Group Relative Policy Optimization for Alignment." DeepSeekMath, 2024.
3. Jin, Z., et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." 2025.
4. Feng, J., et al. "GiGPO: Anchor-based Grouped Policy Optimization for Fine-grained Credit Assignment." 2025.
5. Wang, X., et al. "IGPO: Incentivizing Exploration via Information Gain in Reinforcement Learning." 2025.
6. Ji, Z., et al. "Tree-GRPO: Improving Math Reasoning with Tree-structured Sampling." 2025.
