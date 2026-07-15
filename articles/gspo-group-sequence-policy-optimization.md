---
title: "GSPO：シーケンス級重要性比でGRPOの崩壊を根本解決"
emoji: "⚙️"
type: "tech"
topics: ["LLM", "強化学習", "GRPO", "Qwen3", "MoE"]
published: false
---

# GSPO：シーケンス級重要性比でGRPOの崩壊を根本解決

## TL;DR

Qwenチーム（Alibaba）が提案する**GSPO（Group Sequence Policy Optimization）**は、GRPOの訓練不安定性を根本から解決するRLアルゴリズムだ。鍵となる発見はシンプルだが衝撃的——**GRPOのtoken級重要性比は統計的に無意味**であり、これが訓練崩壊の根源だった。

GSPOは重要度をtokenレベルではなく**シーケンスレベル**で評価する。具体的には、系列全体の対数尤度の平均をとったシーケンス級重要性比 $s_i(\theta) = (\pi_\theta(y_i|x)/\pi_{old}(y_i|x))^{1/|y_i|}$ を用い、clippingもrewardも最適化もすべて**系列単位**で行う。これにより：

- **訓練の安定性が劇的に向上**（GRPOでは不可逆的崩壊が頻発）
- **同計算量での訓練効率がGRPOを上回る**
- **MoEモデルのRL訓練がRouting Replayなしで安定化**（~10%のexpert切り替え問題を完全回避）
- **RLインフラの簡素化**（推論エンジンの尤度をそのまま使用可能）

このアルゴリズムは既に**最新Qwen3モデルのRL訓練に採用**されており、実戦で検証済みである。

![Token-level vs Sequence-level](/images/gspo-group-sequence-policy-optimization/fig1_token_vs_sequence_ratio.png)

## 背景：GRPOの隠された弱点

### GRPOの成功と限界

GRPOはcriticモデルを不要にし、グループ内相対優位度で報酬を正規化する画期的な手法として、LLMのRL後学習において広く採用されている。DeepSeek-R1をはじめ多くのモデルでSOTAを実現した。

しかし、大規模訓練においてGRPOは深刻な問題を抱えている。** catastrophicで不可逆的なモデル崩壊**だ。一度崩壊が起きると、checkpointを巻き戻してclipping rangeや生成長を調整しても復旧できないことが多い。

### 重要性比の誤用という根源

GSPOの論文は、この問題の根源を**重要性サンプリング（importance sampling）の原理違反**に特定する。

重要性サンプリングの基本原理を思い出そう：

$$\mathbb{E}_{z \sim \pi_{tar}}[f(z)] = \mathbb{E}_{z \sim \pi_{beh}}\left[\frac{\pi_{tar}(z)}{\pi_{beh}(z)} f(z)\right]$$

この等式が成立するには、**行動分布$\pi_{beh}$から多数のサンプル$N \gg 1$を生成**し、重み付き平均をとる必要がある。単一サンプルの重みは分散が大きすぎて分布補正に寄与しない。

GRPOは各token位置$t$で重要性比$w_{i,t}(\theta) = \pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{old}(y_{i,t}|x,y_{i,<t})$を計算するが、この重みは**単一サンプル$y_{i,t}$に基づいている**ため、統計的に妥当な分布補正にならない。論文はこの点を明確に指摘する：

> 「最適化目標の単位は報酬の単位と一致すべきである。報酬は系列に付与されるのに、token級でoff-policy補正を行うことは根本的に誤りである。」

さらに、token級の重みは**同一応答内で不均一**であり、$(0, 1+\varepsilon]$（正の優位度）や$[1-\varepsilon, +\infty)$（負の優位度）の範囲で変動する。この不均一性は訓練中に蓄積し、clipping機構によって増幅され、最終的に予測不可能な崩壊を引き起こす。

## 方法：GSPOのアルゴリズム設計

### シーケンス級重要性比

GSPOの核心は、token級の代わりに**シーケンス級の重要性比**を導入することにある：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$$

この設計には3つの重要な特徴がある：

1. **長さ正規化**（$1/|y_i|$）：長さの異なる応答間で重要性比を統一された範囲に制御し、分散を低減する
2. **単一のスカラー値**：系列全体の逸脱度を一つの数値で表現し、clipping範囲を統一的に設定可能にする
3. **各tokenの勾配重みが等価**：GRPOのような不均一な重み付けを排除し、安定した勾配推定を実現

### GSPO最適化目標

$$\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i(\theta)\hat{A}_i,\ \text{clip}(s_i(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right)\right]$$

優位度の推定はGRPOと同じグループ内相対方式：$\hat{A}_i = (r(x,y_i) - \mu_G)/\sigma_G$。

GRPOとの**決定的な違い**は、clippingがtoken単位ではなく**応答全体に対して一度だけ行われる**ことだ。これは「報酬は系列に与えられる」事実と最適化の単位を一致させる設計である。

### GSPO勾配の等権性

GSPOの勾配を展開すると（clippingを無視）：

$$\nabla_\theta\mathcal{J}_{\text{GSPO}} \propto \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})$$

応答内の**すべてのtokenが完全に等しい重みで勾配に寄与**する。対照的にGRPOでは：

$$\nabla_\theta\mathcal{J}_{\text{GRPO}} \propto \sum_{t=1}^{|y_i|}\underbrace{\frac{\pi_\theta(y_{i,t}|ctx)}{\pi_{old}(y_{i,t}|ctx)}}_{\text{不等重み}}\nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})$$

tokenごとに異なる重みが乗るため、特定のtoken位置が訓練を支配し、勾配の方向が不安定になる。

### GSPO-token：より細かい制御が可能な拡張

多ターンRL等でtoken級の優位度調整が必要な場合、GSPO-token変体を用いる：

$$s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t}|ctx)}{\text{sg}[\pi_\theta(y_{i,t}|ctx)]}$$

stop-gradient（sg）により、シーケンス級の安定性を保ちながらtoken級の優位度調整を可能にする。全tokenの優位度が等しい場合、GSPO-tokenとGSPOは**数値的に完全一致**する。

## 実験結果

### 訓練安定性と効率

Qwen3-30B-A3B-Baseをコールドスタート微調整した実験では、GSPOがGRPOに対して以下の優位性を示した：

- **全訓練期間で安定収束**（GRPOは不可逆的崩壊が頻発）
- **同計算量・同query消費量でより高い訓練精度とベンチマーク性能を達成**
- **訓練量の増加、queryセットの定期更新、生成長の延長による継続的改善が可能**

GSPOは既にQwen3シリーズのRL訓練に採用されており、最新モデルの性能向上に直結している。

### Clipping割合のパラドックス

![Clipping behavior comparison](/images/gspo-group-sequence-policy-optimization/fig2_clipping_fraction.png)

最も興味深い発見の一つが、clipping割合に関する**反直感的な結果**だ。

- **GRPOのclipped token割合は~1-8%**と極めて低い。大部分のtokenはclippingされずにそのまま勾配に反映される
- **GSPOのclipped token割合は~45-75%**と圧倒的に高い

直感では「clippingされすぎると学習信号が失われる」ように思えるが、実際は逆だ。GRPOのtoken級勾配推定は**本質的にノイズが多く、サンプル利用効率が低い**。GSPOはより少ないが信頼性の高い勾配信号で効率的に学習する。

clipping range自体も対照的——GRPOは$[0.2, 1.27]$、GSPOは$[3 \times 10^{-4}, 4 \times 10^{-4}]$と**2桁小さい**。これは長さ正規化されたシーケンス級比率が自然に1.0近傍に集まることを示している。

### MoE訓練の安定化

![MoE training stability](/images/gspo-group-sequence-policy-optimization/fig3_moe_stability.png)

MoE（Mixture-of-Experts）モデルのRL訓練は、**expert activation volatility**という独自の課題を抱えている。

論文の測定では、Qwen3-30B-A3B-Base（48層）において、同じrolloutサンプルに対して1回のRL勾配更新を行うだけで**約10%のexpertが異なるルーティング**になることがわかった。層が深くなるにつれてこの現象はさらに顕著になる。

GRPOでは、token $y_{i,t}$の尤度$\pi_\theta(y_{i,t}|ctx)$と$\pi_{old}(y_{i,t}|ctx)$が**異なるexpertセットを活性化**するため、重要性比が劇的に変動し、訓練が発散する。

#### GRPOの回避策：Routing Replay

GRPOでMoEを訓練するには、**Routing Replay**という追加テクニックが必須となる。これは$\pi_{old}$で活性化したexpertをキャッシュし、$\pi_\theta$の尤度計算時に同じルーティングパターンを「再生」することで、重要性比の計算を安定化する。

ただし、Routing Replayには重大な欠点がある——**キャッシュされたルーティングパターンにモデルの容量が制約され**、MoE本来の表現力を活かしきれない。

#### GSPOの解決：原理的回避

GSPOはこの問題を**原理レベルで回避**する。GSPOが重視するのはシーケンス尤度$\pi_\theta(y_i|x)$であり、個別のtoken尤度$\pi_\theta(y_{i,t}|ctx)$の変動には鈍感だ。MoEモデルは言語モデリング能力を維持し続けるため、シーケンス尤度は安定している。

結果として、GSPOは**Routing Replayを一切不要**にし、MoEモデルの全容量を活用したまま安定したRL訓練を実現する。

### RLインフラへの影響

![Algorithm comparison](/images/gspo-group-sequence-policy-optimization/fig4_algorithm_comparison.png)

GSPOの実用的メリットはアルゴリズム自体にとどまらない。RL訓練のインフラ設計にも大きな影響を与える。

大規模RL訓練では通常、**訓練エンジン**（例：Megatron）と**推論エンジン**（例：SGLang, vLLM）の間に精度の差異が存在する。このため、GRPOでは訓練エンジンで$\pi_{old}$の尤度を再計算し直す必要がある。

GSPOは**シーケンス級尤度のみ**を最適化に使用するため、精度の差異に対する許容度が高い。これにより、**推論エンジンの尤度をそのまま最適化に利用可能**になり、訓練エンジンでの再計算を省略できる。

partial rollout、多ターンRL、訓練-推論分離アーキテクチャなどでこの恩恵はとくに大きい。

## 考察

### なぜこれほどシンプルな修正がこれほど効果的だったのか

GSPOの核心は「最適化の単位を報酬の単位に合わせる」という原則への回帰にある。報酬が系列に与えられる以上、最適化も系列単位で行うべきというのは、重要度サンプリングの理論から見れば当然の帰結だ。

GRPOのtoken級重要性比は、理論的基盤を持たない**ヒューリスティック**だったと論文は指摘する。PPOのtoken級clippingをそのままGRPOに引き継いだことが、後年の不安定性の種だったと言える。

### クリッピング範囲の示唆

GSPOのclipping rangeが$[3 \times 10^{-4}, 4 \times 10^{-4}]$と極端に狭いことは、**正しく正規化された重要性比は本来1.0に極めて近い**ことを示している。GRPOの$[0.2, 1.27]$という広いclipping rangeは、token級比率の**本質的な不安定性**をマスクするためのバンドエイドに過ぎなかった。

### 実用上の影響

GSPOがQwen3のRL訓練に採用されたことは、このアルゴリズムが**大規模実運用で検証済み**であることを示している。Qwen3モデルの性能向上はGSPOの有効性の最も説得力のある証拠だ。

## 関連研究

- **PPO**（Schulman et al., 2017）：token級clippingの原型。GRPOはここからcriticを除去する方向で進化したが、token級重要性比の問題は継承した
- **GRPO**（Shao et al., 2024）：critic不要のグループ内相対優位度。実用性は高いが、安定性に課題
- **DAPO**（Yu et al., 2025）：KL正則化やclipping ratioの制御を導入したGRPO拡張。安定性の改善を図ったが、token級比率の問題は解決していない
- **UP**（Fan et al., 2025）：非対称最適化でGRPOの探索-安定性ジレンマを打破。GSPO-tokenとも相性が良い可能性がある

## まとめ

GSPOは「**報酬の単位に最適化の単位を合わせる**」という原則に立ち返り、GRPOのtoken級重要性比という根本的欠陥を修正した。シーケンス級重要性比の導入により：

1. **訓練安定性**：GRPOで頻発する不可逆的崩壊を解消
2. **訓練効率**：同計算量でGRPOを上回る性能向上
3. **MoE対応**：Routing Replayなしでの安定訓練を実現
4. **インフラ簡素化**：推論エンジンの尤度を直接利用可能

clipping rangeが2桁小さく、clipping割合が2桁大きいという反直感的な結果は、**正しい抽象化レベルの選択がいかに重要か**を示している。GSPOは既にQwen3の実運用で成果を出しており、LLMのRL訓練における新たなデファクトスタンダードとなる可能性が高い。

## 参考

- Zheng, C., Liu, S., Li, M., et al. (2025). Group Sequence Policy Optimization. arXiv:2507.18071
- Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
- Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
