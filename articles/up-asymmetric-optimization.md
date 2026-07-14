---
title: "UP: 非対称最適化でRLの探索-安定性ジレンマを打破"
emoji: "🔓"
type: "tech"
topics: ["LLM", "強化学習", "GRPO", "DAPO", "ポリシー最適化"]
published: false
---

# UP: 非対称最適化でRLの探索-安定性ジレンマを打破

## TL;DR

LLMの推論強化における強化学習（RL）は、**探索（exploration）と安定性（stability）のジレンマ**に直面している。純粋な重要度サンプリング（IS）は勾配爆発を引き起こし、標準的なクリッピングは低確信度の正しい推論経路の学習を構造的に阻害する。

本論文が提案する**UP（Unbounded Positive Asymmetric Optimization）**は、このジレンマを根本から解消する。正のアドバンテージ（正解ロールアウト）に対してはstop-gradientによる**自己アンカリング比**で無制限のREINFORCE等価勾配を解放し、負のアドバンテージ（誤答）に対しては従来のクリッピングを安全弁として残す。UP-DAPOはAIME24でAvg@32 +3.44pp、UP-GRPOは5ベンチ平均で+1.16ppの向上を実現し、Dense/MoE/VLMにわたる汎用性も実証された。

## 背景：RL後トレーニングの探索-安定性ジレンマ

LLMの推論能力を高めるため、GRPOやDAPOなどのRLVR（Verifiable Rewards付きRL）が標準パラダイムとして定着している。これらの手法はサンプル効率を上げるために**重要度サンプリング（IS）**に依存しているが、ISには構造的な問題がある。

### 困難その1：ISによる不安定性

IS比 $r_{i,t}(\theta) = \pi_\theta / \pi_{\text{old}}$ が勾配をスケーリングするため、稀少だが高報酬の推論経路では、行動確率が小さい $\pi_{\text{old}}$ により比が爆発的に大きくなり、学習が発散する。

### 困難その2：クリッピングによる探索の餓死

この不安定性を防ぐため、GRPOはIS比を $[1-\epsilon, 1+\epsilon]$ に、DAPOは $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$ にクリッピングする。しかし、本論文は**Probability Capacity（確率容量）**という概念を形式化し、クリッピングがもたらす構造的問題を浮き彫りにした。

![探索-安定性ジレンマの概念図](/images/up-asymmetric-optimization/fig2_dilemma.png)

正のアドバンテージを持つトークンに対するCapは：

$$\text{Cap} = \min(1, (1+\epsilon_{\text{high}})\cdot\pi_{\text{old}}) - \pi_\theta$$

つまり、**Capは$\pi_{\text{old}}$に線形依存**する。$\pi_{\text{old}} = 0.01$の低確信度トークンでは、$\epsilon_{\text{high}}=0.28$としても絶対増加量はわずか0.0028に過ぎない。$\pi_\theta$が0.0128に達した瞬間にCapはゼロとなり、勾配が消失する。

![DAPO vs UP-DAPOのProbability Capacity比較](/images/up-asymmetric-optimization/fig1_prob_capacity.png)

左図のDAPOでは、異なる$\pi_{\text{old}}$値に対してCapが狭い帯に制限されているのがわかる。一方、右図のUP-DAPOでは、Capが$1 - \pi_\theta$となり、$\pi_{\text{old}}$に依存しない完全な予算が確保される。

## 方法：UPの核心メカニズム

UPは2つの数学的操作でジレンマを解消する。

### 操作1：自己アンカリング比（Self-Anchored Ratio）

正のアドバンテージ（$\hat{A} > 0$）に対し、IS比の代わりにstop-gradient演算子 $\text{sg}(\cdot)$ を用いた**自己アンカリング比**を構成する：

$$\tilde{r}_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\text{sg}(\pi_\theta(o_{i,t} \mid q, o_{i,<t}))}$$

逆伝播時、$\text{sg}(\pi_\theta)$は定数スカラーとして振る舞うが、順伝播値は$\pi_\theta$に等しい。$\nabla x / x = \nabla \log x$の恒等式により、目的関数はREINFORCE勾配に退化する：

$$\nabla_\theta J_{\text{UP}+}(\theta) = \mathbb{E}\left[\hat{A} \cdot \nabla_\theta \log \pi_\theta\right]$$

**この等価性が本質的**である。ISの不安定性の根因である$\pi_{\text{old}}$への依存を完全に排除し、Capは$1 - \pi_\theta$に解放される。

### 操作2：非対称設計

無制限メカニズムは正解ロールアウトに**意図的に制限**される。負のアドバンテージは勾配方向を反転させるため、ここで無制限更新を行うと既存の表現を破壊する激进的な勾配上昇になってしまう。

したがって、UPは$\hat{A} \leq 0$に対してはDAPO/GRPOのクリッピングをそのまま安全弁として保持する：

> **正解ロールアウトでは探索を解放し、誤答ロールアウトでは安全を確保する**

### UP-GxPO：3つのインスタンス化

UPは任意のGxPOアルゴリズムの正ブランチを置き換えるプラグアンドプレイ設計である：

**UP-DAPO（トークンレベル）**：
$$J_{\text{UP-DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_i |o_i|} \sum_i \sum_t \begin{cases} \hat{A}_{i,t} \cdot \log \pi_\theta & \text{if } \hat{A}_{i,t} > 0 \\ \min(r_{i,t} \hat{A}_{i,t},\ \text{clip}(r_{i,t}, 1-\epsilon_l, 1+\epsilon_h) \hat{A}_{i,t}) & \text{if } \hat{A}_{i,t} \leq 0 \end{cases}\right]$$

**UP-GRPO**：正ブランチをREINFORCE等価に置き換え、負ブランチはGRPOの対称クリッピングとKLペナルティを保持。

**UP-GSPO（シーケンスレベル）**：正ブランチは長さ正規化REINFORCE勾配に退化することが証明可能。

## 実験結果

### UP-DAPO：探索と安定性の両立

Qwen3-14B-Base + DAPO-17K-MATHでの評価：

| 指標 | DAPO | UP-DAPO | 向上 |
|------|------|---------|------|
| Peak Avg@32 | 47.71 | **51.15** | +3.44 |
| Peak Maj@32 | 58.36 | **60.88** | +2.52 |
| Peak Best@32 | 80.49 | **81.79** | +1.30 |

UP-DAPOは生成エントロピーを継続的に高く維持し、Best@32の向上から探索容量が真に拡大されたことが確認できる。勾配ノルムとKLダイバージェンスはDAPOと同等かそれ以下で、追加の探索が安定性を犠牲にしない。

### UP-GRPO：多数のRL手法を凌駕

Qwen3-8BでMATH (Levels 3-5)上に訓練し、5つの推論ベンチマークで評価：

![UP-GRPO vs ベースラインの結果比較](/images/up-asymmetric-optimization/fig3_up_grpo_results.png)

| 手法 | AIME24 | MATH500 | 平均(5ベンチ) |
|------|--------|---------|---------------|
| GRPO | 35.73 | 86.00 | 55.79 |
| GSPO | 40.52 | 88.20 | 60.15 |
| **UP-GRPO** | **41.04** | **88.40** | **61.31** |

UP-GRPOは平均61.31%で最強ベースラインGSPO (60.15%)に+1.16ppで勝利し、5ベンチのうち4つで最高または同率最高を記録した。特に重要なのは、UP-GRPOのポリシーエントロピーが訓練中に継続的に上昇し続ける一方、多くのベースラインでは**エントロピー崩壊（entropy collapse）**が起きている点だ。KLダイバージェンスも12手法の中で最低レベルに保たれている。

### 消融実験：自己アンカリングの重要性

![消融実験：勾配ノルムとKLの安定性](/images/up-asymmetric-optimization/fig4_ablation_stability.png)

| バリアント | 結果 |
|------------|------|
| DAPO（ベースライン） | 安定した訓練 |
| DAPO + $\epsilon_{\text{high}} = \infty$（$\pi_{\text{old}}$アンカリング維持） | ~80ステップで勾配ノルムが$10^{13}$に爆発 |
| 対称無制限（正負とも無制限） | 25ステップ内に崩壊 |
| **UP-DAPO** | **安定して訓練完了** |

この結果は、**単にクリッピングを緩和するだけでは不十分**であることを示している。stop-gradientによる自己アンカリングこそが安全な無制限更新を可能にし、負ブランチのクリッピングが不可欠な安全装置である。

### 架構・モダリティ横断の汎用性

**MoE**: Qwen3-30B-A3B-Base + UP-GSPO → AIME24 Avg@32 52.71%→**55.73%** (+3.02pp)

**VLM**: Qwen3-VL-8B-Instruct + UP-GRPO → Geometry3K 59.30%→**62.60%** (+3.30pp)

いずれの設定でもKLはベースラインとほぼ同等で、追加の不安定性は観測されなかった。

## 考察

### なぜ非対称設計が正解なのか

本論文の洞察は単純だが深い：**信頼領域は失敗保護のためであり、成功の制限ではない**。正解ロールアウトの確率を高めたい局面でIS比のクリッピングに阻まれるのは、最適化の観点から明らかに非効率である。

自己アンカリング比の数学的等価性（REINFORCEへの退化）は、UPが新たな目的関数ではなく、IS比という「不必要な中間層」を除去した結果に過ぎないことを示している。$\pi_{\text{old}}$でアンカリングする理由はISの分散削減にあるが、正のアドバンテージではこの分散削減がCapの制限というより大きな代償を払っている。

### MoEにおける構造的問題の解決

UPはMoEモデルにおけるtrain-inference mismatchも本質的に解決する。MoEのRLでは、パラメータ更新がルーティングロジックを動的に変更するため、ロールアウト時と訓練時で活性化される専門家が乖離し、IS比が極めて不安定になる。UPが正のサンプルでIS比を除去することで、このルーティングミスマッチを完全に回避できる。

## 関連研究

- **GRPO/DAPO/GSPO**: UPの基盤となるgroup-based ポリシー最適化。いずれもIS比のクリッピングに依存。
- **REINFORCE++**: IS比を用いないが、分散が大きく性能が制限される（本論文のTable 1でAvg 46.77%と最下位）。
- **Dr. GRPO, CISPO, DPPO, GMPO, SAPO, ASPO**: いずれもGRPOの改良版だ、対称クリッピングの構造的制限は共有している。
- **OPD (On-Policy Distillation)**: 別のアプローチでRLの成果を再利用する手法。UPとは目的関数レベルでの改良という点で直交する。

## まとめ

UPは、RL後トレーニングにおける探索-安定性ジレンマを**非対称な目的関数設計**で解消する汎用的手法である。

- **Probability Capacityの形式化**でクリッピングの構造的限界を特定
- **Stop-gradient自己アンカリング比**で正のアドバンテージに対し安全な無制限勾配を実現
- **非対称設計**で正解ロールアウトの探索解放と誤答の安定性確保を両立
- UP-DAPO/UP-GRPO/UP-GSPOとして、token・sequenceレベルの両方に適用可能
- Dense/MoE/VLM、言語/マルチモーダルにわたり一貫した改善を実証

実装コストはほぼゼロ（数行の分岐追加のみ）で、既存のRLパイプラインにプラグアンドプレイで統合できる点も実用的価値が高い。探索と安定性のトレードオフを「設計の正しさ」で解消するアプローチとして、今後のRL後トレーニングの標準コンポーネントになる可能性がある。

## 参考

- Fan, C., Liu, P., Huang, J., Liu, S., & Lin, Y. (2026). UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma. arXiv:2607.06987.
- Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning. arXiv:2402.03300.
- Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.
- Team, K. (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv:2501.12599.
