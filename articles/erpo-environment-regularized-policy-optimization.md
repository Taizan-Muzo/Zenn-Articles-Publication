---
title: "ERPO: 安定性と探索のジレンマを入力側正則化で打破"
emoji: "🔄"
type: "tech"
topics: ["LLM", "強化学習", "ポリシー最適化", "GRPO"]
published: false
---

## TL;DR

LLMのポリシー最適化（GRPO/PPO等）は、Policy-KL正則化という「応答側」の制約で安定性と探索のトレードオフを制御してきた。残念ながら、残すと探索が阻害され、外すとドリフト制御がなくなるというジレンマに陥る。AlibabaのZhouらがEMNLP 2026に採択された論文「ERPO」は、正則化を**入力（クエリ）側**に移すことでこのジレンマを打破する。Query-KL（QKL）項でポリシーが誘導するクエリ分布のドリフトを拘束しつつ、応答分布への直接的な勾配圧力はゼロ――探索を犠牲にせずに安定性を獲得する。Qwen2.5-Math-7Bで6ベンチマーク平均+6.2%、32Bモデルでは高温サンプリング（T=1.5）で25.2%→80.8%と劇的な安定化を実現。Reward hackingの評価ギャップも51%削減。

## 背景

### Policy-KLのジレンマ

LLMのポリシー最適化では、学習済みモデル $\theta_0$ からのKL散度 $D_{KL}(\pi_\theta \| \pi_{\theta_0})$ を正則化項として加えるのが標準的な手法だ。GRPOでもPPOでも、このPolicy-KLは安定化の要として機能してきた。

しかし、固定されたPolicy-KL予算の下で観察される事実がある：**バッチ推定のQuery-KLは訓練中に一方向に上昇し続ける一方、Policy-KLは平坦なまま**である。つまり、応答側のKLを拘束しても、入力側のドリフトは止められない。

これが実務者を二重のジレンマに追い込む：

1. **Policy-KLを残す**：応答の多様性が制約され、探索予算が消費される
2. **Policy-KLを外す**：明示的なドリフト制御が失われ、長期訓練で不安定化する

![Main Results](/images/erpo-environment-regularized-policy-optimization/fig1_main_results.png)

## 方法

### 核心アイデア：正則化の入力側への移行

ERPOの着眼点はシンプルだが本質的である。LLMの訓練では、クエリ（プロンプト）は固定データセットからサンプリングされるため、これまで「環境」として扱われてこなかった。しかし、自己回帰モデルはクエリ自体の対数尤度 $\ell_\theta(q) = \log P_\theta(q)$ も計算している。ERPOはこれを「ポリシーが誘導するクエリ分布」 $\rho_\theta(q)$ と定義し、その事前RL参照分布 $\rho_{\theta_0}$ からのドリフトを拘束する。

### Query-KL (QKL)

正則化項を次のように定義する：

$$\mathcal{R}_{\text{query}}(\theta) = D_{KL}(\rho_\theta \| \rho_{\theta_0}) = \mathbb{E}_{q \sim \rho_\theta}\left[\ell_\theta(q) - \ell_{\theta_0}(q)\right]$$

**命題1（構造的解離）**によれば、この勾配は閉形式で：

$$\nabla_\theta \mathcal{R}_{\text{query}}(\theta) = \mathbb{E}_{q \sim \rho_\theta}\left[(\ell_\theta(q) - \ell_{\theta_0}(q)) \cdot \nabla_\theta \ell_\theta(q)\right]$$

 ここが決定的に重要なのだが、**この勾配には応答のスコア関数 $\nabla_\theta \log \pi_\theta(o|q)$ が一切現れない**ことだ。つまりQKLは応答分布に対して直接的な勾配圧力を及ぼさない。探索を完全に保ちながらクエリ側のドリフトだけを抑え込む。

### クエリ再重み付け (Query Reweighting)

QKLに加えて、各クエリの更新重みを参照モデルのクエリ尤度に基づいて調整する：

$$w(q_i) = \text{clip}\left(\frac{\bar{s}}{s_i},\, 0,\, 2\right), \quad s_i = -\log p_{\theta_0}(q_i)$$

参照分布 $\rho_{\theta_0}$ のもとで典型的なクエリに更新をバイアスし、低確率クエリ（高勾配分散の原因）の影響を緩和する。$\ell_{\theta_0}$ は訓練前にデータセット全体について**一度だけ計算してキャッシュ**されるため、追加のforward passは不要。

### ERPO-GRPOの損失関数

$$\mathcal{L}_{\text{ERPO-GRPO}}(\theta) = -\frac{1}{m}\sum_{q \in B}\frac{w_B(q)}{K}\sum_{o \in \mathcal{G}(q)} A_\theta^{\text{GRPO}}(q,o)\,\log\pi_\theta(o|q) + \alpha\,\widehat{\mathcal{R}}_{\text{query}}(\theta)$$

実装上は次の4ステップのドロップイン変更で済む：

1. $\ell_{\theta_0}$ を事前計算・キャッシュ
2. クエリ外側の重み $1/m$ を $w_B(q)/m$ に置換
3. 損失に $\alpha\,\widehat{\mathcal{R}}_{\text{query}}(\theta)$ を追加
4. 追加forward passなし・アーキテクチャ変更なし・内部PG推定器のclip/baselineロジック変更なし

## 実験結果

### メイン結果（Qwen2.5-Math-7B）

6つの数学推論ベンチマーク（AIME24/25, AMC, MATH500, Minerva, OlympiadBench）で評価。サンプリング温度0.1〜1.5の平均をとる。

| メトリクス | GRPO Avg. | ERPO Avg. | 差分 |
|-----------|----------|----------|------|
| Avg@32 | 57.5% | 61.1% | **+3.64pp** |
| Avg@1 | 27.5% | 33.2% | **+5.69pp** |

個別ベンチマークでは最大**+14.9pp**の改善。OlympiadBenchのPass@32で55.8%→59.3%、MATH500では85.0%→90.4%と着実に上昇。

### 温度安定性

![Temperature Stability](/images/erpo-environment-regularized-policy-optimization/fig2_temperature_stability.png)

ERPOの最大の強みは**高温サンプリングでの安定性**だ。GRPOはT=1.5でMATH500が0.4%に崩壊するが、ERPO（$\alpha=10^{-2}$）は8.6%を維持。$\alpha$ を $5 \times 10^{-2}$ に強めると15.0%にさらに向上する。

Qwen2.5-Math-32Bではこの差がより劇的になる：

| 温度 | GRPO | ERPO | 差分 |
|-----|------|------|------|
| T=1.0 | 81.2% | 83.6% | +2.4pp |
| T=1.5 | 25.2% | **80.8%** | **+55.6pp** |

### 機構の分解

![KL & Entropy Decomposition](/images/erpo-environment-regularized-policy-optimization/fig3_kl_entropy_decomposition.png)

消融実験から各コンポーネントの役割が明確に分かる：

- **QKL单独**が性能向上の主駆動力。Query-KLを0.9679から**0.0041**へ劇的に低減し、同時にPolicy-KLは0.1001に上昇（＝応答側の制約が緩み、探索が促進される）、エントロピーも0.5674に増加
- **クエリ再重み付け**は補完的な安定化役割。Query-KLを0.5933へと緩やかに低減し、Policy-KLを0.0113へ低下させることで勾配分散を低減
- **ERPO（両者の組み合わせ）**がバランス最適：Query-KL 0.0828、エントロピー 0.4244で安定性と探索の最適点を達成

### Reward Hackingの抑制

![Reward Hacking](/images/erpo-environment-regularized-policy-optimization/fig4_reward_hacking.png)

訓練精度と評価精度のギャップでreward hackingを測定：

| 方法 | 平均Train Acc | 平均Eval Acc | 平均Gap |
|-----|-------------|-------------|--------|
| GRPO | 77.3% | 70.1% | **6.47pp** |
| ERPO | 77.1% | 73.8% | **3.14pp** |

ギャップの**51%削減**。特にStep 240でGRPOの評価精度が58.4%へ急落（reward hacking発生）するのに対し、ERPOは78.4%で安定的に継続改善。

### 他アルゴリズムへの拡張

ERPOはGRPOだけでなく、DAPOやRLOOにもdrop-inで適用可能：

| ベース | 通常 | +ERPO | 差分 (T<=1.0) |
|-------|-----|-------|-------------|
| GRPO | 68.8% | 78.7% | **+9.94pp** |
| DAPO | 68.2% | 78.4% | **+10.24pp** |
| RLOO | 77.3% | 79.6% | **+2.28pp** |

DAPO+ERPOの組み合わせが最も大きな改善を示した。

## 考察

### なぜ入力側の正則化が有効なのか

ERPOの成功は「環境とポリシー空間の結合関係」の再認識に基づいている。自己回帰モデルはクエリの尤度も同時に計算しているため、クエリ分布のドリフトは暗黙のうちに進行する。Policy-KLは応答側のKLだけを見ているため、このドリフトを検知できない。

QKLが応答分布に勾配圧力をかけないという性質（命題1）は、安定性と探索のトレードオフを「空間的に分離」することを意味する。入力側でドリフトを抑え、出力側では自由に探索させる――この分離がジレンマ打破の鍵だ。

### クエリ再重み付けの相乗効果

プロンプトNLLとレスポンス確率には**95%の訓練サンプルでほぼ完全な正相関**（相関係数≈1）がある。低確率プロンプトは低確率レスポンスを生みやすく、これが勾配分散の主原因となる。再重み付けはこのノイズ源を効果的に抑制する。

### 実用上の魅力

- **追加forward passゼロ**：$\ell_{\theta_0}$ は事前キャッシュ、$\ell_\theta$ は既存のPG forward passから読み取り
- **アーキテクチャ非依存**：内部のclip/baselineロジックを変更する必要がない
- **マルチアルゴリズム対応**：GRPO/PPO/REINFORCE/DAPO/RLOOに共通して適用可能

## 関連研究

- **GRPO (Shao et al., 2024)**：ERPOの主な比較対象。グループ内帰一化アドバンテージによるcritic-free RLVR。
- **DAPO (Yu et al., 2025)**：Policy-KLを段階的に減衰させるアプローチ。ERPOとは逆に応答側の制約を緩める方向。
- **GSPO (Zheng et al., 2025)**：トークンレベルからシーケンスレベルへの重要度比の変更でGRPOの不安定性に対応。
- **RIPO (Cai et al., ICML 2026)**：リーマン多様体上の等角clippingで探索崩壊に対処。ERPOとは異なり応答側の幾何学的修正。
- **GFlowRL (Liu et al., 2025)**：GFlowNet的分布マッチングで探索多様性を確保。応答側の分布制御という点でERPOと対照的。

## まとめ

ERPOは「正則化を入力側に移す」というシンプルな転換で、LLMのポリシー最適化における安定性-探索のジレンマを打破した。QKLが応答分布への勾配圧力をゼロに保ちながらクエリ分布のドリフトを拘束するという構造的性質は理論的に美しく、実験的にも裏付けられている。

実用面でも追加forward passなしでGRPO/PPO/DAPO/RLOOにdrop-in適用可能という強力な互換性を持ち、特に高温サンプリングでの安定性（32BでT=1.5: 25.2%→80.8%）とreward hacking抑制（ギャップ51%削減）は即座に実務価値のある結果だ。

数学推論ベンチマークに限定されている点は今後の課題だが、コード生成やエージェントタスクへの拡張は自然な方向性であり、EMNLP 2026への採択もこのアプローチの妥当性を示している。

## 参考

- Zhou, X., Meng, X., He, Y., Qi, T., Guan, S., Zhang, X., Zhang, J., Li, X., Lin, Q., & Liu, J. (2026). Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization. arXiv:2608.23311v2. [link](https://arxiv.org/abs/2608.23311)
- コード: [https://github.com/AlibabaResearch/ERPO](https://github.com/AlibabaResearch/ERPO)
