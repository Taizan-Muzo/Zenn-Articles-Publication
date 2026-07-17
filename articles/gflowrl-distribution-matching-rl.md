---
title: "GFlowRL：分布マッチングRLを初めて大規模LLMで安定化"
emoji: "🌊"
type: "tech"
topics: ["LLM", "強化学習", "GFlowNet", "後訓練", "推論モデル"]
published: false
---

## TL;DR

GFlowNet（Generative Flow Network）に基づくRLは、報酬最大化RLの**モード崩壊問題**に対する理論上の解として注目されてきた。しかし、既存手法（FlowRLなど）は学習した**分割関数 $Z_\phi$** が勾配不安定性の根本原因となり、大規模モデルやMoEでは発散に終わっていた。

Microsoft Researchの**GFlowRL**は、この分割関数ネットワークを**完全に削除**し、rolloutグループ自身から計算されるバッチ内モンテカルロ推定量で代替する。非対称フローギャップクリッピングと重要度サンプリング補正を加えたこのシンプルな設計により、Dense・MoE双方で安定訓練が初めて実現された。14BモデルでCodeforces 2048（o3-miniに25 Elo差）、7B数学平均+8.44pp vs GRPO、応答多様性3.93（GRPOの3.2倍）という結果を出している。

> **論文**: Liu et al., "GFlowRL: Scaling Distribution-Matching RL to Large Language Models" (arXiv:2607.13394, 2026/07/15)
> **機関**: Microsoft Research
> **コード**: https://github.com/microsoft/gflowrl

---

## 背景：報酬最大化RLのジレンマとGFlowNetの約束

### モード崩壊の根本原因

PPO、GRPO、REINFORCE++といった主流のRL後訓練手法は、本質的に**報酬最大化**を目指す：

$$\mathcal{J}_{\text{RM}}(\theta) = \mathbb{E}_{\mathbf{y} \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y})]$$

高報酬軌道に確率質量を集中させるため、複数ある正解解法のうち1つに収束し、**応答の多様性が失われる**。これは特に数学推論で致命的だ——複数の valid なアプローチ（代数的、幾何学的、帰納的など）が存在する問題では、多様な推論パスを維持することが性能に直結する。

### GFlowNet：分布マッチングのパラダイム

GFlowNetはこの問題に対して根本的に異なるアプローチを提案する。**報酬最大化ではなく、報酬分布へのマッチング**を行う：

$$p^\star(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp(\beta r(\mathbf{x}, \mathbf{y}))$$

逆KLダイバージェンス $D_{\text{KL}}(\pi_\theta \| p^\star)$ を最小化することで、全ての高報酬モードを**網羅的にサンプリング**する方策を学習する。理想論としては美しい。

### 現実の壁：分割関数 $Z_\phi$ の学習失敗

Trajectory Balance（TB）目標では、正規化定数 $Z(\mathbf{x})$ を学習ネットワーク $Z_\phi$ で近似する必要がある。FlowRLではプロンプトの最終隠れ状態上の3層MLPがこの役割を担う。

しかし、ここに**致命的な学習視点の不整合**が生じる：

- **方策 $\pi_\theta$**：事前学習済みLLMから初期化。数十億パラメータが豊かな言語・数学事前知識を符号化しており、少数の更新で機能する
- **分割関数 $Z_\phi$**：ランダム初期化。同じ有限ステップ数の中で複雑な量をゼロから学習しなければならない

結果として、$\log Z_\phi$ は訓練の大部分で**プロンプトに対する有効な乱数関数**として振る舞うことになる。

![勾配安定性の比較：FlowRLは55/421ステップで勾配爆発を起こすのに対し、GFlowRLとGRPOは安定している](/images/gflowrl-distribution-matching-rl/fig1_gradient_stability.png)

---

## GFlowRL：分割関数を「推定」で置き換える

### コアアイデア：バッチ内モンテカルロ推定量

GFlowRLの最大の洞察はシンプルだ：**$Z_\phi$ を学習するのではなく、rolloutから直接推定する**。

GRPO風の訓練では、各プロンプト $\mathbf{x}$ に対して既に $G$ 個の rollout $\{\mathbf{y}^{(1)}, \dots, \mathbf{y}^{(G)}\}$ をサンプリングしている。TB損失の最適点では、$\log Z(\mathbf{x})$ は各 rollout について $\beta r + \log \pi_{\text{ref}} - \log \pi_\theta$ に等しい。したがって、これらのバッチ内平均が自然な推定量になる：

$$\mathcal{Z}_t(\mathbf{x}) = \frac{1}{G} \sum_{i=1}^{G} \left(\beta r(\mathbf{x}, \mathbf{y}^{(i)}) + \log \pi_{\theta_{\text{ref}}}(\mathbf{y}^{(i)}|\mathbf{x}) - \log \pi_{\theta_{\text{old}}}(\mathbf{y}^{(i)}|\mathbf{x})\right)$$

この置き換えには3つの重要な帰結がある：

1. **勾配ノルムがGRPOと同じスケールに戻り、訓練安定性が回復する**
2. **補助ネットワークとその分散訓練におけるオプティマイザ状態が不要になる**
3. **GRPOとほぼ同じインフラで動作する**

### 分布マッチング残差

$\mathcal{Z}_t(\mathbf{x})$ をTB損失に代入すると、rollout $\mathbf{y}^{(i)}$ の**分布マッチング残差**が得られる：

$$\Delta^{(i)}(\theta) = \text{sg}[\mathcal{Z}_t(\mathbf{x})] + \frac{1}{|\mathbf{y}^{(i)}|} \log \frac{\pi_\theta(\mathbf{y}^{(i)}|\mathbf{x})}{\pi_{\theta_{\text{ref}}}(\mathbf{y}^{(i)}|\mathbf{x})} - \beta r(\mathbf{x}, \mathbf{y}^{(i)})$$

- $|\mathbf{y}^{(i)}|$ で**長さ正規化**し、長いシーケンスが損失を支配するのを防ぐ
- $\text{sg}[\cdot]$（stop gradient）で $\mathcal{Z}_t(\mathbf{x})$ は残差の**ベースラインとしてのみ**機能し、勾配を運ばない

### 非対称フローギャップクリッピング

残差 $\Delta^{(i)}(\theta)$ を「フローギャップ」と「方策更新項」に分解：

$$g^{(i)} = \text{sg}[\mathcal{Z}_t(\mathbf{x})] + \frac{1}{|\mathbf{y}^{(i)}|} \log \frac{\pi_{\text{old}}(\mathbf{y}^{(i)}|\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}^{(i)}|\mathbf{x})} - \beta r(\mathbf{x}, \mathbf{y}^{(i)})$$

$$\tilde{g}^{(i)} = \text{clip}(g^{(i)}, -\epsilon_{\text{low}}, +\epsilon_{\text{high}})$$

**非対称**（$\epsilon_{\text{low}} < \epsilon_{\text{high}}$）にすることで、正の補正（確率質量を増やす方向）により大きな空間を与える。これは数学推論で特に重要——訓練初期は正解解法が低サンプリングされやすく、積極的に確率質量を押し上げる必要がある。

![GFlowRLアルゴリズムの全体パイプライン：分割関数ネットワーク不要のシンプルな設計](/images/gflowrl-distribution-matching-rl/fig2_algorithm_pipeline.png)

### 重要度サンプリング補正

rollout方策 $\pi_{\text{old}}$ と最適化中の方策 $\pi_\theta$ の間の**ドリフト**を補正する：

$$w^{(i)} = \min\left(\frac{\pi_\theta(\mathbf{y}^{(i)}|\mathbf{x})}{\pi_{\text{old}}(\mathbf{y}^{(i)}|\mathbf{x})}, 1 + \epsilon\right)$$

### 最終的なGFlowRL損失

$$\mathcal{L}_{\text{GFlowRL}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} w^{(i)} \left(\tilde{g}^{(i)} + \frac{1}{|\mathbf{y}^{(i)}|} \log \frac{\pi_\theta(\mathbf{y}^{(i)}|\mathbf{x})}{\pi_{\text{old}}(\mathbf{y}^{(i)}|\mathbf{x})}\right)^2$$

### 理論的保証

**命題 B.1（不動点最適性）**: フローギャップクリッピングが非活性なとき、GFlowRLはTrajectory Balanceと同一の不動点を持つ。ゼロ損失の自己無矛盾不動点は $\pi_\theta(\mathbf{y}|\mathbf{x}) \propto \pi_{\theta_{\text{ref}}}(\mathbf{y}|\mathbf{x}) \exp(\beta r(\mathbf{x}, \mathbf{y}))$ を満たす。

**注記 B.3**: 不動点ではフローギャップがゼロになるため、クリッピングは最適点近傍で非活性となり、定常分布を乱さない。

---

## 実験結果

### 数学推論（Qwen2.5-7B, Dense）

| 手法 | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | **平均** |
|------|--------|--------|-------|---------|---------|----------|----------|
| Backbone | 4.38 | 2.08 | 30.78 | 54.47 | 22.38 | 24.03 | 23.02 |
| GRPO | 13.54 | 9.79 | 64.53 | 57.05 | 23.06 | 26.88 | 32.48 |
| FlowRL | 15.41 | 10.83 | 54.53 | 66.96 | 31.41 | 34.61 | 35.63 |
| **GFlowRL** | **17.29** | 9.79 | **67.66** | **76.89** | **33.62** | **40.25** | **40.92** |

GFlowRLはGRPOに**+8.44pp**、FlowRLに**+5.29pp**で全ベンチを圧倒。特にMATH500 +22.42ppの改善が目を引く。

### コーディング（DeepSeek-R1-Distill-Qwen-7B/14B）

| 手法 | LiveCodeBench | Pass@16 | Codeforces | HumanEval+ |
|------|--------------|---------|------------|------------|
| GRPO (7B) | 32.75 | 52.32 | 1314 | 80.13 |
| FlowRL (7B) | 37.43 | 56.27 | 1549 | 83.28 |
| **GFlowRL (7B)** | **38.62** | **58.06** | **1646** | **84.93** |
| **GFlowRL (14B)** | — | — | **2048** | — |

14Bスケールでの**Codeforces 2048**は衝撃的だ。o1（1891）を157 Elo上回り、o3-mini（2073）に**わずか25 Elo差**に迫る。FlowRL-14Bより+144 Elo、DeepCoder-14Bより+112 Elo。

### 敵対的レッドチーミング（AdvBench / HarmBench）

| 手法 | AdvBench Avg | HarmBench Avg |
|------|-------------|---------------|
| SEMA | 80.1 | 75.0 |
| **GFlowRL** | **82.5** | **79.5** |
| FlowRL | — | —（**発散**） |

**FlowRLはレッドチーミングタスクでも収束せず**、GFlowRLが両ベンチで最高ASR@1を達成。分布マッチングの多様性が敵対的プロンプト生成に有利に働く。

![数学・コーディングベンチマークにわたるGFlowRLの支配的結果](/images/gflowrl-distribution-matching-rl/fig3_main_results.png)

### MoEモデル：FlowRL全敗 vs GFlowRL安定動作

| モデル | GRPO | GFlowRL | FlowRL | GFlowRL vs GRPO |
|------|------|---------|--------|-----------------|
| Qwen3-30B-A3B (Math) | 75.78 | **78.32** | **発散** | +2.54 |
| Qwen3-235B-A22B (Math) | 82.40 | **83.35** | **発散** | +0.95 |

**235BパラメータではGFlowRLはわずか30ステップ**（GRPOは100ステップ）で訓練済み。効率性の差は明らかだ。

![応答多様性3.93（GRPOの3.2倍）とMoEモデルでの収束比較](/images/gflowrl-distribution-matching-rl/fig4_diversity_moe.png)

---

## 消融研究

### 分割関数の実質的寄与はゼロ

$\log Z_\phi$ をガウスノイズ $\mathcal{N}(0.5, 1)$ で置き換えても性能が**変わらない**（35.63% → 36.19%）。学習された分割関数は有益な情報を符号化しておらず、単なるノイズバイアスであることが実証された。

### フローギャップクリッピングの重要性

クリッピングを除去すると：
- 平均精度が **40.92 → 37.02**（-3.90）
- 勾配平均が **6.3倍**に増大（0.095 → 0.601）
- 勾配最大値が **3倍**に増大（6.18 → 16.57）

### 逆温度 $\beta$ の感度

$\beta \in [1, 10]$ の範囲で最適点（$\beta=8$: 38.2）の0.5点以内に収まる。低感度であり、本番でのチューニング負担が小さい。

### 応答多様性

| 手法 | 多様性スコア | 備考 |
|------|-------------|------|
| PPO | 1.15 | 最小多様性 |
| GRPO | 1.21 | 最小多様性 |
| FlowRL | 2.64 | 中程度 |
| **GFlowRL** | **3.93** | **高多様性**（GRPOの**3.2倍**） |

---

## 考察

### 「不要なものを特定する」アプローチの威力

GFlowRLの最大の教訓は、**「元の定式化のどの部分がこの設定で不要かを特定する」こと**が、新しい補助機構を追加することよりも拡張性の鍵になるという点だ。

Trajectory Balanceの分割関数 $Z_\phi$ は、理論上はマッチング目標に不可欠に見える。しかし、LLM後訓練の設定では、方策が事前学習済みモデルから初期化されているという事実が、$Z_\phi$ の役割を根底から覆す。バッチ内推定量がこの初期化の利点を直接活用するため、追加パラメータなしで同様の正規化効果を得られる。

### 分布マッチング vs 報酬最大化のトレードオフ

GFlowRLの圧倒的な多様性スコア（3.93）は、分布マッチングの理論的利点が実務でも顕現することを示している。ただし、AIME25の単一ベンチではFlowRL（10.83）> GFlowRL（9.79）と逆転するケースもあり、全てのタスクで分布マッチングが一貫して優れるわけではない。

### MoEでの実用性

235Bパラメータで30ステップ訓練という効率は、大規模展開で決定的な意味を持つ。FlowRLがMoEで完全に失敗するのに対してGFlowRLが安定動作する事実は、学習安定性の差がアーキテクチャの選択に直結することを示している。

---

## 関連研究

| 手法 | アプローチ | GFlowRLとの関係 |
|------|-----------|----------------|
| **FlowRL** (2024) | GFlowNet + 学習分割関数 $Z_\phi$ (MLP) | 直接の前身。GFlowRLは $Z_\phi$ を不要化 |
| **FOR** (2024) | 共有スカラー分割関数 | より簡略化された分割関数だが、同様の安定性問題 |
| **GRPO** (2024) | 報酬最大化、グループ正規化 | 本論文の主要ベースライン。GFlowRLは同じインフラで動作 |
| **GSPO** (2025) | シーケンス級重要度比でGRPO改善 | 最適化単位の変更。GFlowRLは目的関数自体を変更 |
| **UP** (2026) | 非対称最適化で探索-安定性両立 | 非対称性のアイデアが共通。UPはクリッピング範囲を操作 |
| **DAPO/POSO** | 報酬最大化RLの安定化 | 同じ問題領域だがパラダイムが異なる |

---

## まとめ

GFlowRLは、GFlowNet風RLを初めてDense・MoE双方の大規模LLMで安定動作させた手法だ。その核心は**分割関数ネットワークの完全削除**という「引き算のアーキテクチャ」にある：

1. **バッチ内MC推定量**が $Z_\phi$ を置き換え——勾配不安定性の根本原因を除去
2. **非対称フローギャップクリッピング**が外れ値を制御——正の補正に広い空間を確保
3. **重要度サンプリング補正**が方策ドリフトを修正

これらはGRPOとほぼ同じインフラで実現でき、追加コストはほぼゼロ。理論的にはTrajectory Balanceと同一の不動点を保持しながら、実用性では大幅に上回る。14BでCodeforces 2048、7B数学でGRPO +8.44pp、応答多様性3.2倍——分布マッチングRLが単なる理論上の美しさではなく、実務的な優位性を持つことが証明されたと言える。

---

## 参考文献

1. Liu, X., Xu, M., Stokes, J.W., Smolensky, P., Burger, D., & Gao, J. (2026). GFlowRL: Scaling Distribution-Matching RL to Large Language Models. arXiv:2607.13394.
2. Bengio, E., et al. (2021). Flow Networks as a framework for modeling distributions over trajectories. ICML.
3. Malkin, N., et al. (2024). FlowRL: Training LLMs with Recovery Learning and GFlowNets. NeurIPS.
4. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
5. Yuan, Z., et al. (2024). DeepSeek-Prover-V1.5: Training Large Language Models to Prove Theorems with Monte-Carlo Tree Search. arXiv:2502.03827.
