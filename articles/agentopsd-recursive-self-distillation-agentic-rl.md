---
title: "AgentOPSD：マルチターンAgent RLの信用割当てを再帰ベイズ信念更新で解決"
emoji: "🧠"
type: "tech"
topics: ["LLM", "AgentRL", "CreditAssignment", "SelfDistillation", "GRPO"]
published: true
published_at: "2026-08-09"
---

## TL;DR

マルチターンAgentタスクのRLでは、終端のスパース報酬をどのように各ターンに割り当てるか（credit assignment）が最大の課題だ。GRPOは単一の軌跡レベル優位度を全トークンに均等に广播するため、結果を決定づけた数回の「決定的ターン」を特定できない。

**AgentOPSD**（Wang et al., 清華大学, arXiv:2608.05987）は、Critic-freeの再帰的自己蒸留手法でこの問題を解く。Tokenレベルの教師-生徒対数確率差をターンレベル証拠に集約し、log-odds空間でベイズ信念状態を再帰更新する。信念修正量 $\Delta B_k$ が「そのターンが履歴の文脈でどれだけ成功確率の推定を変えたか」を測り、これを用いてGRPOの優位度を有界再形成する。

- **ALFWorld**: 7Bで **89.1%**（GRPO 81.2%, StepOPSD 88.4%）
- **WebShop**: 7Bで **90.2**（GRPO 80.9, SDAR 89.4）
- **Search-QA**: 7Bで **49.2%**（GRPO 42.0%）
- **ターン追加コスト**: GRPO比 **-0.54%/turn**（RLSD -3.59%, GRPO -2.91%）

![method overview](/images/agentopsd-recursive-self-distillation-agentic-rl/fig1.png)

## 背景：マルチターンAgent RLの信用割当て問題

LLMエージェントの強化学習（RLVR）では、環境が軌跡終端でスパースな二値報酬（成功/失敗）を返すのが一般的だ。標準的なGRPOは、この報酬から軌跡レベルの優位度 $A_{\mathrm{seq}}$ を計算し、軌跡内の全トークンに均等に割り当てる。

$$A_{\mathrm{seq}}^{(i)} = \frac{R^{(i)} - \bar{R}}{\widehat{\sigma}_R + \epsilon_0}$$

しかし30ターンのタスクでは、結果を決めたのはせいぜい2〜3回の決定的判断かもしれない。GRPOはこれを「すべてのターンが等しく貢献した」と仮定してしまう——無駄なだけではない。**間違った信用分配は学習を阻害する。**

既存のOPSD（On-Policy Self-Distillation）系手法は、教師-生徒のtokenレベル対数確率差から密な監督信号を得ようとしているが、2つのミスマッチがある：

1. **粒度ミスマッチ**: OPSDの信号はトークンレベルだが、環境のフィードバックはターン境界で発生する
2. **孤立性ミスマッチ**: 各ターンを独立に評価し、先行ターンの累積証拠を無視する

## 方法：再帰ベイズ信念状態によるターンレベル信用割当て

### コアアイデア

AgentOPSDの中心的な問いはこうだ：

> ターン $k$ の信用は、その局所信号だけで決まるべきではない。**「その信号が、最終成功の推定確率をどれだけ変えたか」**で決まるべきだ。

これは「マージナル信念修正（Marginal Belief Revision）」と呼ばれる。同じ局所証拠でも、結果がまだ不確実な初期ターンでは決定的だが、信念がすでに飽和した後期ターンでは冗長になる。

### Step 1: Tokenレベル証拠の計算

各ターン $k$ で、生徒（スキル条件なし）と教師（スキル条件あり）が同じアクションに対する対数確率を評価する。ここで $c^+$ は**訓練時のみ**使用する検索スキル（有用なサブゴールとアクションパターンの記述）。

$$\delta_{k,t} = \log \pi_\theta(y_{k,t} \mid h_{k,t}^+) - \log \pi_\theta(y_{k,t} \mid h_{k,t})$$

これをターン内の全トークンについて集約し、**ターンレベル証拠** $e_k$ を得る：

$$e_k = \sum_{t=1}^{L_k} \delta_{k,t}$$

$e_k$ は理想ベイズターン証拠（成功/失敗条件下でのアクションの対数尤度比）の代理として機能する。理論的には、成功条件下の挙動分布を近似する仮説（A1）の下で、$e_k$ は点相互情報となり、**符号がベイズ証拠の符号と一致する**ことが保証される。

### Step 2: 再帰信念状態更新

$e_k$ は各ターンの局所証拠だが、「前のターンとの関係」を無視している。そこで **減衰証拠累積器** を維持する：

$$c_k = \gamma \, c_{k-1} + e_k \quad (\gamma \in (0,1])$$

$$\ell_k = \mathrm{logit}(B_0) + c_k = \mathrm{logit}(B_0) + \sum_{j=1}^{k} \gamma^{k-j} e_j$$

$$B_k = \sigma(\ell_k)$$

ここで $B_0 = \mathrm{clip}(\bar{R}, \epsilon_0, 1 - \epsilon_0)$ はグループ成功率をclipした事前信念。log-odds空間で証拠を累積するため、$B_0$ は毎ターン保持され、減衰するのは $c_k$ のみ。

### Step 3: マージナル信念修正

各ターンの重要性は、**「履歴の文脈下でそのターンが信念をどれだけ修正したか」** で測る：

$$\Delta B_k = B_k - B_{k-1}$$

一階近似すると：

$$\Delta B_k \approx B_{k-1}(1 - B_{k-1}) \cdot (e_k - (1-\gamma)c_{k-1})$$

この式の美しさは、**不確実性が最大のときに証拠が最も効く**という性質にある。$B_{k-1} = 0.5$ のとき $B(1-B)$ は最大となり、$B \to 0$ または $B \to 1$ では抑制される。

### Step 4: 有界優位度再形成

信念修正を結果信号と整合させ、有界乗数で優位度を再形成する：

$$q_k = \mathrm{sign}(A_{\mathrm{seq}}) \cdot \Delta B_k$$
$$\tilde{A}_k^{(i)} = A_{\mathrm{seq}}^{(i)} \cdot [(1-\lambda) + \lambda \cdot w_k^{(i)}]$$

ここで $w_k \in [1-b, 1+b]$ は标准化された信用に基づく有界乗数。重要な理論的性質：

- **符号保存**: $\mathrm{sign}(\tilde{A}_k) = \mathrm{sign}(A_{\mathrm{seq}})$ — GRPOの更新方向を**決して反転しない**
- **有界性**: $\|\tilde{A}_k - A\| \leq \lambda b \|A\|$ — 変化量が制御可能
- **GRPO回復**: $\lambda = 0$ で厳密にGRPOに退化

最終的な損失関数は標準GRPOと全く同じ形で、ただ $A_{\mathrm{seq}}$ が $\tilde{A}_k$ に置き換わるだけ。追加の蒸留損失は**一切不要**。

![main results](/images/agentopsd-recursive-self-distillation-agentic-rl/fig2.png)

## 実験結果

### メイン実験

Qwen2.5-3B/7B-Instructで、ALFWorld（家事エージェント）、WebShop（オンラインショッピング）、Search-QA（検索QA）の3ベンチマークを評価。

**7Bモデルの主要結果**:

| ベンチマーク | GRPO | GRPO+OPSD | SDAR | StepOPSD | **AgentOPSD** |
|---|---|---|---|---|---|
| ALFWorld Avg | 81.2 | 80.4 | 85.9 | 88.4 | **89.1** |
| Search-QA Avg | 42.0 | 47.0 | 49.0 | 48.2 | **49.2** |
| WebShop Score | 80.9 | 86.8 | 89.4 | 87.2 | **90.2** |
| WebShop Acc | 72.6 | 76.5 | 82.8 | 78.1 | 79.7 |

全ベンチマークでGRPOを大幅に上回り、既存の最強ベースライン（SDAR, StepOPSD）も一貫して超える。3Bでも同様の傾向（ALFWorld 84.4%, WebShop 90.4）。

### 長期依存性の優位性

ALFWorld 7Bでのターン追加あたりの成功率低下を測定：

- **AgentOPSD**: -0.54%/turn（最も平坦）
- **GRPO**: -2.91%/turn
- **RLSD**: -3.59%/turn

ターン数が増えるほどAgentOPSDの優位性が大きくなる。これは再帰信念状態が長期依存を適切に捕捉していることを示している。

### 消融実験

![ablation](/images/agentopsd-recursive-self-distillation-agentic-rl/fig3.png)

| 構成 | ALFWorld 7B | 変化 |
|---|---|---|
| Full AgentOPSD | **89.1** | — |
| Per-token集約（粒度なし） | 85.9 | -3.2 |
| Raw $e_k$（再帰なし） | 82.8 | -6.3 |
| $\|\Delta B_k\|$ のみ（符号なし） | 80.5 | -8.6 |
| $B_0$ なし（事前なし） | 78.9 | -10.2 |

3つの役割が明確に分離されている：

1. **粒度**: トークン→ターンの集約が必須（-3.2）
2. **再帰**: 局所 $e_k$ ではなく $\Delta B_k$ で履歴文脈を反映（-6.3）
3. **結果整合**: 符号方向で検証器と整合させる（-8.6）
4. **事前錨定**: $B_0$ でタスク難易度を反映（-10.2）

### ハイパラ感度

- **$\lambda$（再形成重み）**: 最も感度が高い。0.5が最適。0でGRPOに退化
- **$\gamma$（証拠減衰）**: 0.8〜1.0で安定。0.95を使用
- **$\epsilon_{\mathrm{high}}$（クリップ範囲）**: ほぼ無感応

ターン数が少ないSearch-QAでは全ハイパラの差が縮まる——短期タスクでは信用割当ての差が反映されにくいことを示している。

![degradation and belief revision](/images/agentopsd-recursive-self-distillation-agentic-rl/fig4.png)

## 考察

### なぜ再帰信念更新が効くのか

AgentOPSDの成功の核心は、**「局所証拠の価値は履歴に依存する」**という認識だ。同じ $e_k = 0.5$ の証拠でも：

- ターン1（$B_0 = 0.4$、不確実性大）: 信念を大幅に修正する → 決定的ターン
- ターン7（$B_6 = 0.9$、信念ほぼ飽和）: ほとんど修正しない → 冗長ターン

この文脈依存性を $\Delta B_k$ が自然に捕捉する。$B(1-B)$ というゲートが、情報理論的に「最も学習価値の高いタイミング」を選択する。

### Critic-freeの利点

AgentOPSDは学習済みCriticを一切必要としない。オーバーヘッドはターンあたり教師の1回前向き伝播のみ。これは実用的な大きな利点で、長期ホライズンのエージェントRLでCriticを安定して訓練するのは難しいことが知られている。

### 制約と課題

- **スキル依存**: 証拠計算に検索スキル $c^+$ が必要。スキル品質が証拠品質に直結する
- **短期タスクでの差の縮小**: ターン数が少ないタスクでは再帰の利点が限定的
- **理論仮説**: スキル条件分岐が成功条件下の挙動を近似するという仮説（A1）が成立する範囲に依存

## 関連研究

| 手法 | 信号粒度 | 履歴依存 | 注入方式 |
|---|---|---|---|
| OPSD | Token | なし | 独立蒸留 |
| GRPO+OPSD | Token | なし | 補助監督 |
| SDAR | Turn | なし | ゲート/損失 |
| StepOPSD | Step | なし | 局部log-ratio |
| **AgentOPSD** | **Turn** | **再帰信念** | **有界優位度再形成** |

SEEDのon-policy蒸留、ABSeekerのanswer-backtracked密報酬など、エージェントRLでは「ロールアウトからより多くの情報を絞り出す」という方向が急速に収束している。AgentOPSDの貢献は、この流れに**理論的に動機付けられたターンレベル信用割当て**を持ち込んだ点にある。

## まとめ

AgentOPSDは、マルチターンAgent RLの信用割当て問題に対して3つの鍵となる設計を提案した：

1. **ターン粒度の集約**: 環境の転移境界に信号を整合させる
2. **再帰信念状態**: 履歴文脈下での証拠の価値を評価する
3. **有界再形成**: 符号保存・有界性を保証しつつGRPOの優位度をターン別に調整する

ALFWorld 89.1%、WebShop 90.2という結果は、7BモデルのエージェントRLにおける強力なベースラインを確立している。Critic-free・追加損失なし・GRPO完全互換という実用性も高く、エージェントRLの信用割当てにおける新しい設計軸としての位置づけが明確だ。

## 参考

- 論文: [AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2608.05987) (arXiv:2608.05987)
- コード: [github.com/ZethWang/AgentOPSD](https://github.com/ZethWang/AgentOPSD)
- GRPO: [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- OPSD: [On-Policy Self-Distillation](https://arxiv.org/abs/2405.21024)
- SDAR: [Self-Distillation Augmented Reinforcement Learning](https://arxiv.org/abs/2503.08746)
