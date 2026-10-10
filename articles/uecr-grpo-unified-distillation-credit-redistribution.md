---
title: "教師を信じる場所と信じない場所──UECR-GRPOがオンポリシー蒸留とGRPOをエントロピー校準信用再分配で統一する仕組み"
emoji: "⚖️"
type: "tech"
topics: ["RLVR", "On-Policy Distillation", "GRPO", "信用再分配", "数学推論"]
published: false
---

## TL;DR

強化学習による数学推論（RLVR）は正解だけを報酬として与えるが、個々のトークンに何の指針も与えない。一方、オンポリシー蒸留（OPD）は教師モデルから密なトークン級フィードバックを得られるものの、教師の好みが正しさを反映する保証はない。本稿が提案する **UECR-GRPO**（Unified Entropy-Calibrated Credit Redistribution for GRPO）は、この二つの信号を二層で統合する。第一層 **PUU**（Path-Utility Unification）は群正規化の前に検証器報酬と教師経路改良を合成し、教師に応答順序への発言権を与える。第二層 **ECR**（Entropy-Calibrated Redistribution）は教師のエントロピーで不確実な誘導を減衰させつつ、零和射影で検証器由来の信用予算を厳密に保存したままトークン級再分配を行う。Qwen3-1.7B で平均 17.21%、Qwen3-4B で 65.09%——それぞれ最強基線を +0.89 / +0.56 ポイント上回る。

![UECR-GRPOのベンチマーク比較](/images/uecr-grpo-unified-distillation-credit-redistribution/fig1.png)

## 背景

### RLVRの構造的限界

GRPO（Group Relative Policy Optimization）は、プロンプトあたり $G$ 個の応答をサンプリングし、群内統計から優位度を計算する：

$$A_i^{\text{GRPO}} = \frac{R_i^{\text{task}} - \text{Mean}(R_j^{\text{task}})}{\text{Std}(R_j^{\text{task}}) + \epsilon}$$

価値モデルを不要にする利点があるが、一つの応答級優位度を**全トークンに一括放送**するため、重要な代数ステップも無害な文体トークンも最初の局所的誤りも同じ優位度を受ける。さらに、群内の全応答が同じ二値報酬を得た場合、GRPOの更新は**完全に消滅**する。

### OPDの罠

オンポリシー蒸留は学生のon-policy応答上で教師の対数確率を追跡する。トークン級の密な信号が得られるが、教師の好みが**数学的正しさと無関係**なことがある。筆者らの分析（Figure 1）が明らかにした事実は重い：

- 教師スコアは退化群の 84.3–98.4% を判別できる（これは良い）
- だが、正解-不正解応対の 39.1–50.8% で教師の順序が検証器と**衝突**する
- 検証器平局組で教師の好みがOpus 4.8の過程品質判定と一致するのは 56.8% に過ぎない

**教師は応答間の相対的情報源として有用だが、端末検証の代わりにはならず、過程品質の信頼できるラベルでもない。**

![教師信号の信頼性分析](/images/uecr-grpo-unified-distillation-credit-redistribution/fig4.png)

### 既存ハイブリッドの限界

既存の手法はいずれも、教師信号が検証器の群正規化を通った**後に**入る。つまり検証器による応答順序が確定してから教師がトークン級微調整を加える形になり、教師が順序そのものに影響する余地がない。

| 手法 | 教師の順序参加 | 単一裁剪更新 | 負タスク信用 | エントロピー減衰 | 信用予算保存 |
|------|:-:|:-:|:-:|:-:|:-:|
| Naive GRPO+OPD | ✗ | ✗ | OPD独立 | ✗ | ✗ |
| ATOD | ✗ | ○ | 加性OPD | ✗ | ✗ |
| Distilled RL | ✗ | ○ | GRPOに戻す | ✗ | ✗ |
| PUU | ○ | ○ | ○（軌跡級） | ✗ | 放送 |
| **UECR-GRPO** | **○** | **○** | **○（符号保護）** | **○** | **○** |

## 方法详解

UECR-GRPOは二層構造を持つ。PUU が応答級の統一効用を構成し、ECR がトークン級の信用再分配を行う。

![二層設計の全体像](/images/uecr-grpo-unified-distillation-credit-redistribution/fig2.png)

### PUU: Path-Utility Unification

#### 経路対数比の恒等式

OPDのトークン級信号を足し上げると、**正確に**経路級対数密度比になる：

$$\sum_{t=1}^{L} \log \frac{\pi_T(y_t | x, y_{<t})}{\pi_0(y_t | x, y_{<t})} = \log \frac{P_T(y|x)}{P_0(y|x)}$$

この恒等式は、散在するトークン級の教師-旧ポリシー間隙が、実は応答全体の「教師が錨ポリシーよりこの経路をどれだけ好むか」を測る一つの量に集約されることを意味する。

#### 統一KL正則化目的

筆者らは検証器報酬と教師経路比を単一のKL正則化目的に統合する：

$$J(Q) = \mathbb{E}_{y \sim Q}\left[R_{\text{task}}(x,y) + \alpha \log \frac{P_T(y|x)}{P_0(y|x)}\right] - \beta\,\text{KL}(Q \| P_0)$$

Gibbs最適解は：

$$Q^*(y|x) \propto \exp\!\left(\frac{R_{\text{task}}(x,y)}{\beta}\right) P_T(y|x)^{\alpha/\beta}\,P_0(y|x)^{1-\alpha/\beta}$$

- $\alpha=0$ なら報酬正則化RLに退化
- $R_{\text{task}}=0, \alpha=\beta$ なら教師 $P_T$ に退化
- $R_{\text{task}}=0, \alpha>\beta$ なら教師超錨方向への外挿

#### On-policy群相対実装

応答長が教師寄与の尺度を暗黙に歪めるのを防ぐため、長さ正規化した教師スコアを用いる：

$$R_i^T = \frac{\sum_t m_{i,t}\,\delta_{i,t}}{\sum_t m_{i,t}}, \quad R_i^U = R_i^{\text{task}} + \alpha\,R_i^T$$

この**統一効用** $R_i^U$ を群正規化にかけ、優位度 $A_i^U$ を計算する。教師証拠が正規化の**前**に入るため、応答間の順序そのものを変えられる。これが PUU の核心である。

重要なことに、統一優位度は検証器成分と教師成分に**正確に分解**される：

$$A_i^U = A_i^{\text{task}|U} + \alpha\,A_i^{T|U}$$

分母を共有するため、係数 $\alpha$ の意味が群を跨いで一貫する。

### ECR: Entropy-Calibrated Redistribution

PUU は応答級の統一を果たしたが、トークン級では依然として放送（全トークンに同じ優位度）である。ECR は教師の不確実性を校準しつつ、検証器由来の信用を再分配する。

#### 方向と信頼度

教師-旧ポリシー間隙 $\delta_{i,t}$ とタスク優位度の符号 $s_i$ から有界方向を構成：

$$d_{i,t} = \tanh\!\left(\frac{s_i\,\delta_{i,t}}{2\tau_\delta}\right), \quad c_{i,t} = \exp\!\left(-\frac{H_{i,t}^T}{\tau_H}\right)$$

- $s_i$ は応答全体が正か誤かの符号。正応答では教師好みのトークンに正方向、負応答では**方向を反転**——負応答で教師好みトークンが無条件の模倣対象にならない
- $H_{i,t}^T$ は凍結教師の全語彙エントロピー。教師が自信のない箇所（高エントロピー）では信頼度 $c_{i,t}$ が減衰し、PUUの放送優位度に戻る

#### 零和射影

信頼度重み付き応答内平均を引く：

$$\mu_i^c = \frac{\sum_t m_{i,t}\,c_{i,t}\,d_{i,t}}{\sum_t m_{i,t}\,c_{i,t}}, \quad q_{i,t} = \frac{1}{2}\,m_{i,t}\,c_{i,t}\,(d_{i,t} - \mu_i^c)$$

零和射影により $\sum_t q_{i,t} = 0$ が成立する。重み $w_{i,t} = 1 + \rho\,q_{i,t}$ の算術平均は有効トークン上で正確に 1 になる。すなわち、**ECR は各応答の加法的タスク信用とトークン級符号を保存する**。

#### 最終優位度

$$A_{i,t}^{\text{final}} = A_i^{\text{task}|U}(1 + \rho\,q_{i,t}) + \alpha\,A_i^{T|U}$$

ECR は検証器由来成分のみを再分配し、PUU が導入した教師成分は手を触れない。$\rho=0$ で正確に PUU に退化。信頼度が無視できるか全方向が一致しても PUU に戻る。

## 実験結果

### 設定

- **1.7B設定**: Qwen3-1.7B-Base を初期化、Qwen3-4B-GRPO を凍結教師、DeepMath-103K 難易度5-7、515ステップ
- **4B設定**: Qwen3-4B 学生、Qwen3-8B-Math-GRPO 凍結教師、難易度6-8、160ステップ
- 評価は AIME 2024/2025、AMC 2023、HMMT 2025 Feb/Nov の5ベンチマーク Avg@12

### 主結果

| 手法 | 1.7B Avg@12 | 4B Avg@12 |
|------|:-:|:-:|
| Vanilla GRPO | 11.45 | 59.96 |
| Vanilla PG-OPD | 12.22 | 57.61 |
| Naive GRPO+OPD | 11.43 | 58.65 |
| Distilled RL | 16.12 | 64.53 |
| ATOD-aligned | 16.32 | 64.20 |
| **UECR-GRPO** | **17.21** | **65.09** |

1.7B で最強基線 ATOD-aligned を +0.89pt、4B で Distilled RL を +0.56pt 上回る。全ベンチマークで一様改善ではないが、平均では確実に先行する。

### 消融実験

![消融実験](/images/uecr-grpo-unified-distillation-credit-redistribution/fig3.png)

消融は設計の各段階が積み上げる改善を明示する：

1. **Task only → Separate norm.** (+0.97): 教師を別々に正規化して加えるだけで向上
2. **Separate norm. → PUU** (+2.09): 正規化前に統一するのが鍵——教師が順序に影響できることが大きい
3. **PUU → w/o entropy** (+0.83): トークン級再分配の追加、ただしエントロピー校準なし
4. **w/o entropy → w/o proj.** (+4.06): エントロピー校準の追加が大幅寄与
5. **w/o proj. → Full** (+0.45): 零和射影で信用予算を保存する最終仕上げ

ECR の二つの構成要素（エントロピー校準 + 零和射影）が合計 +4.49pt を担う。PUU の +2.09 と合わせて、二層設計がそれぞれ独立で不可欠な役割を果たす。

### PUU 信号分析

- 検証器報酬は 40–53% の擬似群で平局
- 56–90% の群が統一効用で非自明な分散を保持——教師が死んだ群を蘇らせる
- 混合群で教師成分は絶対成分振幅の 17–25% を占める

### α 感度分析

$\alpha=1$ で 17 チェックポイント・451 混合群・22,811 正誤対を検査した結果、**厳密な正誤順序反転は一つも観測されなかった**。最も制約の厳しい群の $\alpha_g^* \approx 3.88$。チェックポイント級 $\alpha_g^*$ の第5百分位は 5.3–9.3、中央値は 12.6–21.3 に分布。実用的な $\alpha=1$ は安全側に大きく余裕がある。

## 考察

### 教師の立ち位置の再定義

UECR-GRPO が描き出す教師像は、従来の「正解の案内人」でも「過程品質のラベル」でもない。教師は**応答間の相対的序列付けの情報源**として働く。検証器が「どの応答が正しいか」を告げ、教師が「同じ正しさの中でどの経路がより教師らしいか」を告げる——この二つが群正規化前に合成されることで、検証器だけでは平局に沈む群に分離が生まれる。

### 信用予算の保存が意味するもの

零和射影 $\sum_t q_{i,t} = 0$ は「応答に割り当てられた検証器信用の総量を、トークン間の再分配で増減させない」ことを保証する。ECR は信用の**配置**を変えるだけで**量**は変えない。この制約がないと、教師の偏った好みが検証器信用を体系的に増幅または減衰させ、過学習や学習消失の経路を開く。

### エントロピー校準の実用的意義

教師の全語彙エントロピー $H_{i,t}^T$ が高いトークン位置では、教師自身が「次に何が来るか」を自信を持って言えない。その箇所で教師の間隙 $\delta_{i,t}$ を信用するのは危険——ECR は $c_{i,t} = \exp(-H_{i,t}^T / \tau_H)$ で減衰させ、代わりに PUU の放送優位度（群相対の均一信号）に戻す。これは「自信のない教師は黙る」という設計原理の実装と言える。

## 関連研究

- **GRPO / DAPO / GSPO**: 群相対優位度に基づくRLVR手法群。UECR-GRPO は GRPO を PUU+ECR で拡張する位置づけ
- **G-OPD**: OPD を密KL正則化RLとして定式化。PUU の KL正則化視点はこれを一般化
- **Distilled RL**: 教師比率で正GRPO優位度を再重み付け。検証器平局で更新消失する点が UECR と異なる
- **ATOD**: 退火OPD優位度を事後正規化GRPOに加算。教師が順序に影響しない点が核心的差異
- **Entropy-Aware OPD**: 教師高エントロピー域で前向きKLに切り替え。ECR は全語彙エントロピーで門制する点が異なる

## まとめ

UECR-GRPO は、検証器と教師の信号を二層で統合する新しいRLVR手法である。第一層 PUU は、KL正則化理論に基づき検証器報酬と教師経路改良を群正規化前に合成——教師に応答順序への発言権を与える。第二層 ECR は、教師の全語彙エントロピーで不確実な誘導を減衰させつつ、零和射影で各応答の検証器信用予算を厳密に保存したままトークン級再分配を行う。

核心的な洞察は**「いつ・どこで教師を信じるか」を分離して扱う**ことにある。「いつ」は PUU が応答級で決め、「どこで」は ECR がトークン級で決める。教師は万能ではないが、検証器の盲点を補う情報源として——しかしその不確実性に応じて控えめに——活用される。

## 参考

- Zhang, J., Yang, J., Huang, Z., Liu, Y., & Huang, X. (2026). *When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment*. arXiv:2609.28385.
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300. (GRPO)
- Agarwal, R. et al. (2024). *On-Policy Distillation for RLVR*. (OPD)
- Xiong, W. et al. (2025). *ATOD: Adaptive Teacher-Student On-Policy Distillation*. (ATOD)
