---
title: "CrEST：RLの方向性は変えず、Teacherが「大きさ」だけを調整するマルチターンAgent信用割当て"
emoji: "🎯"
type: "tech"
topics: ["LLM", "ReinforcementLearning", "Agent", "CreditAssignment", "RLVR"]
published: true
published_at: "2026-08-15"
---

## TL;DR

マルチターンのツール利用Agentを強化学習（RL）で訓練する際、trajectory全体に一つの報酬を与えるだけでは、ターンごとの異質な結果が混ざり合ってしまい、信用割当て（Credit Assignment）が極めて困難になる。一方で、On-Policy Self-Distillation（OPSD）はtokenレベルで密な監督信号を提供できるものの、教師モデルの性能上限に縛られるか、あるいは勾配集中による崩壊（gradient concentration collapse）を起こすというジレンマがあった。

本稿が紹介する **CrEST**（**Cr**edit Assignment with **E**ntropy-gated **S**elf-**T**eacher）は、この二者択一を解消する階層型信用割当てフレームワークである。RLの「方向性」をそのまま残しつつ、特権付き自己教師（privileged self-teacher）からの密なtokenレベル信号を「大きさ」の調整にのみ利用するというシンプルな設計思想で、verifier boundedな性能天井を維持しながら密な信用割当てを実現する。

BFCL V3およびWildToolBenchでの実験において、CrESTはGRPO・OPSDの双方を一貫して上回り、特に**長 trajectory**（10ターン以上）および**厳密な session-level メトリクス**で最大の改善を示した。

## 背景

### ツール利用AgentのRLにおける信用割当て問題

LLMベースのツール利用Agentは、関数呼び出し、情報抽出、外部APIとの対話など、複数のターンにわたって環境と相互作用する。このようなマルチターンタスクをRLVR（Reinforcement Learning with Verifiable Rewards）で訓練する際、以下の2つのアプローチが主に使われてきた。

#### アプローチ1: Trajectory-Level RL（GRPOなど）

GRPOやDAPOなどのgroup-basedなRLアルゴリズムでは、trajectory全体の結果（成功/失敗）を報酬として使い、turn内の全tokenに同一の advantage を割り当てる。このアプローチの利点は **verifier bounded** であること——最終的な検証結果に基づく報酬を使うため、教師モデルの限界に縛られない。

しかし問題は **inter-turn dilution** である。例えば4ターンのtrajectoryでターン3が正しいツール呼び出しを行い成功した場合でも、失敗したターン1やターン2のtokenにも同じ正の advantage が伝播してしまう。異質なターン結果を単一の報酬信号に混ぜ込むことで、学習効率が著しく低下する。

#### アプローチ2: On-Policy Self-Distillation（OPSDなど）

OPSDは、モデル自身のlogitsを教師信号として使い、tokenレベルで密な監督を提供する。ターンごとの正確な信用割当てが可能になるため、短いCoT推理タスクでは高い効果を示す。

しかし、ツール利用Agentに拡張すると2つの壁にぶつかる：
1. **Teacher-bounded**: 教師（＝同じモデルの自己蒸留）の性能がpolicyの上限を決定してしまう
2. **Gradient concentration collapse**: 教師信号がdominantになると、学習が特定のtokenパターンに集中し、多様性を失う

### 本質的な問い：Teacherは「方向」か「大きさ」か

ここでCrESTは根本的な問いを立てる：

> **Teacherの役割は、policy updateの「方向」を決めることではなく、「大きさ」を調整することに還元できるのではないか？**

RLが提供するのはverifierに基づく「正しい方向」（成功したtrajectoryのtokenを強化し、失敗したものを弱める）。Teacherが提供すべきは、その方向に沿った各tokenの「調整幅」——つまり大きさである。この分離によって、両方の長所を活かすことができる。

## 手法详解

CrESTは2段階の階層型信用割当てを行う。

### Level 1: Turn-Segmented Verified Advantages

まずtrajectory全体をターンごとに分割し、各ターン $k$ に対して個別の verified advantage を計算する：

$$\hat{A}_k^{ver} = \frac{1}{|T_k|} \sum_{t \in T_k} \frac{r - \bar{r}}{\sigma_r}$$

ここで $T_k$ はターン $k$ に含まれるtoken集合、$r$ はtrajectory全体の報酬、$\bar{r}$, $\sigma_r$ はグループ内の平均と標準偏差である。

この設計により、**inter-turn dilution** が解決される。各ターンのtokenには、そのターンが属するtrajectoryのoutcomeに応じたadvantageが割り当てられる。正しいツール呼び出しを行ったターンには正の advantage が、失敗したターンには負の advantage が、より正確に伝播するようになる。

### Level 2: Entropy-Gated Self-Teacher Modulation

ターン内の各token $t$ について、自己教師からのKL divergence $\Delta_t^{KL}$ を計算し、**entropy gate** $g(H_t)$ で変調する：

$$\hat{A}_t^{mod} = g(H_t) \cdot \Delta_t^{KL} \cdot \hat{A}_k^{ver}$$

entropy gate $g(H_t)$ は、tokenの予測不確実性（entropy $H_t$）に基づいて、教師信号の影響度を制御するゲート関数である：

$$g(H_t) = \sigma(\alpha \cdot (H_t - \tau))$$

ここで $\sigma$ はsigmoid関数、$\alpha$ はスケーリングパラメータ、$\tau$ は閾値である。

このゲーティングの直感は以下の通り：

- **高エントロピーのtoken**（生成が多様で情報量が多い）→ ゲートが大きく開き、教師信号を強く反映
- **低エントロピーのtoken**（ほぼ確定的、定型表現など）→ ゲートが閉じ、教師信号を弱める

これにより、gradient concentration collapse を防ぎつつ、重要な意思決定tokenに密な監督信号を集中させることができる。

### 全体の目的関数

CrESTの最終的な目的関数は、turn-segmented advantageを基盤としつつ、entropy-gated teacher signalでtokenレベルの調整を行う形で定式化される：

$$\mathcal{L}_{CrEST} = -\mathbb{E}\left[\sum_t \min\left(\rho \hat{A}_t^{mod}, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t^{mod}\right)\right]$$

重要なのは、$\hat{A}_t^{mod}$ の **符号はRLのverified advantage $\hat{A}_k^{ver}$ に由来** すること。TeacherのKL divergence $\Delta_t^{KL}$ は常に非負であるため、Teacherはupdateの方向ではなく大きさにのみ寄与する。これが "Teach the Magnitude, Not the Direction" の核心である。

![CrEST framework overview](/images/crest-verifier-bounded-credit-assignment/fig2.png)

## 実験結果

### ベンチマークと設定

| ベンチマーク | タスク | 評価軸 |
|:---|:---|:---|
| **BFCL V3** | マルチステップ関数呼び出し | ast, multi-step, multi-hop, cross-ent. |
| **WildToolBench** | オープンツール利用 | Overall, Long-Trajectory (>10 turns), Session-Level (strict) |

モデルスケール: Qwen2.5-7B-Instruct および Qwen2.5-3B-Instruct

### 主要な結果

![Performance comparison](/images/crest-verifier-bounded-credit-assignment/fig3.png)

CrESTは両ベンチマークでGRPO・OPSDを一貫して上回った。特に注目すべきは：

1. **Long-Trajectoryでの顕著な改善**: WildToolBenchで10ターン以上の長いtrajectoryにおいて、CrESTはGRPOに対して最大+10.4ppの改善を示した。ターン数が増えるほどinter-turn dilutionの解消効果が大きくなる
2. **Strict Session-Level Metrics**: 最も厳密なsession-level評価でも+10.5ppの改善。これはturn-segmented advantageがsession全体の成功/失敗を正確に各ターンに反映できていることを示す
3. **モデルスケールの保存性**: 3Bと7Bの両方で一貫した改善を確認

### 消融実験の知見

- **Entropy Gateの有無**: ゲートなしでは精度が低下。特に低エントロピーtokenへの過剰な教師信号伝播がgradient concentration collapseを引き起こす
- **Turn Segmentationの有無**: ターン分割なし（trajectory-level advantageのみ）では、長いtrajectoryでの性能が著しく低下。inter-turn dilutionが確認された
- **Teacher Modulationの範囲**: 変調を方向にまで拡張（Teacherが符号も変更できるように）すると、verifier-bounded天井が失われ、OPSDと同等の性能に落ち込んだ——「大きさのみ」の制約が本質的であることが実証された

### 関連手法との比較

| 手法 | 信用割当て | 密な信号 | Verifier-Bounded | 長Trajectory |
|:---|:---|:---|:---|:---|
| GRPO | Trajectory-level | ✗ | ○ | 弱い |
| OPSD | Token-level | ○ | ✗ | 中程度 |
| AgentOPSD | Turn-level | ○ | △ | 中程度 |
| **CrEST** | **Hierarchical** | **○** | **○** | **強い** |

## 考察

### 「方向」と「大きさ」の分離の意義

CrESTの最も重要な洞察は、policy updateの「方向」と「大きさ」を明示的に分離したことである。RLのverified rewardが方向を決め、self-teacherが大きさを調整するという分業は、両者のジレンマを美しく解消する。

これは他の信用割当て手法とも共通する構造を持つ。CSCR（反実仮想感度信用再分配）もGRPOの符号を保持しつつtokenレベルの調整を行うが、CrESTはentropy gateというシンプルな機構で勾配崩壊を防ぐ点でより汎用的である。

### 階層型設計の普遍性

turn-levelとtoken-levelの2階層設計は、AgentOPSDのturn-level信用割当てを自然に拡張している。AgentOPSDがBayesian belief updateでターンレベルの重みを決定するのに対し、CrESTはverified advantageでターンレベルの信用を決定し、entropy gateでtokenレベルの細かさを制御する。

この階層性は、問題の粒度に応じた信用割当てという一般的な原理を体現している。マクロな粒度（turn/session）ではoutcome-basedな信号を、ミクロな粒度（token）ではteacher-basedな密な信号を——それぞれ適切なレベルで使う。

### 長Trajectoryへのスケーラビリティ

CrESTが長trajectoryで最も大きな改善を示したことは、マルチターンAgentの学習における本質的なボトルネックがinter-turn dilutionにあることを裏付けている。ターン数が増えるほど、trajectory-levelの報酬だけでは各ターンの寄与を区別できなくなる。turn-segmented advantageはこの問題に対する最も直接的な解法である。

## 関連研究

### マルチターンAgentの信用割当て

- **AgentOPSD** (Wang et al., 2026): 再帰的ベイズ信念更新によるターンレベル信用割当て。Critic-free。CrESTとは異なり、turn内のtokenレベルの調整を行わない
- **CSCR** (He et al., 2026): 反実仮想感度による長CoTのtoken信用再分配。GRPOの符号を保持しつつtokenの重みを調整。turn-levelの分割は行わない
- **TACO** (Lou et al., 2026): Tail-risk scoreによるpositive credit contaminationの修正。tokenレベルのcalibrationに焦点

### Self-Distillation系

- **OPSD** (Guo et al., 2025): On-policy self-distillation。密なtokenレベル教師信号だがteacher-bounded
- **DASH** (Hou et al., 2026): 発散適応型監視地平。ターン内のtoken-level信用割当てに対する異なるアプローチ
- **DemoPSD** (Li et al., 2026): Disagreement-modulated self-distillation。teacher-studentの不一致を利用した適応的漏洩制御

### GRPO拡張

- **GSPO** (Zheng et al., 2026): Group-level importance ratioでGRPOのtoken-levelサンプリング問題を修正
- **SoftmaxGRPO** (Hernandez et al., 2026): z-scoreの代わりにsoftmax advantageを使用して発散を防ぐ

## まとめ

CrESTは、マルチターンツール利用AgentのRL訓練において「方向」と「大きさ」の分離というシンプルだが強力な原理を提案した。

- **Turn-segmented verified advantages** が inter-turn dilution を解決し
- **Entropy-gated self-teacher modulation** が token-level の密な調整を勾配崩壊なしに実現する
- その結果、**verifier-bounded天井** を維持しながら、**密な信用割当て** の恩恵を受けることができる

実験結果は特に長trajectoryと厳密なsession-level評価で顕著であり、マルチターンAgentの実用的な訓練において重要な意味を持つ。今後の展望として、turn segmentationの粒度の自動決定や、より複雑なツール利用パターンへの適用が期待される。

## 参考

- Wang, Z., Lu, S., Zhang, H., Mo, L., Zhuang, C., & Gan, L. (2026). "Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents." arXiv:2608.13179.
- [論文PDF](https://arxiv.org/pdf/2608.13179)
