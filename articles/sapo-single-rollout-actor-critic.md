---
title: "SAPO: 1ロールアウト自己回帰Actor-CriticでAgent RLの壁を突破"
emoji: "⚡"
type: "tech"
topics: ["LLM", "ReinforcementLearning", "Agent", "RLHF"]
published: false
published_at: "2026-08-22"
---

## TL;DR

LLMエージェントの強化学習において、PPOは高精度だが別途Criticモデルのメモリが必要で、GRPOはメモリ効率が良いが長期タスクでadvantage collapseに陥る——この二律背反を**SAPO（Single-rollout Autoregressive Policy Optimization）**が一発で解消する。鍵は「自己回帰モデルの因果的境界を利用して、1つのバックボーンでPolicy・V・Qを同時に表現する」という設計。ALFWorld × Qwen2.5-1.5BでPPO +35.7pp、GRPO +17.3pp。1回のロールアウトのみでCriticなしの効率的学習を実現しつつ、明示的な temporal credit assignment を保持。実行時間もPPO比33.2%削減。

![SAPO architecture overview](/images/sapo-single-rollout-actor-critic/fig1.png)

## 背景

### エージェントRLの現状とジレンマ

LLMエージェントを環境フィードバックから学習させる「Agentic RL」は、数学推論のコーディング、Webナビゲーション、ツール活用など幅広いタスクで急速に普及している。現状の手法は大きく2つの系統に分かれる。

**PPO系（Actor-Critic）**: PPOは学習済みの価値関数（Critic）を用いてGAE（Generalized Advantage Estimation）によりtokenレベルまたはturnレベルの信用割当てを行う。長期タスクでの credit assignment に優れるが、ポリシーと同等規模のCriticモデルが別途必要で、メモリ・計算コストが大きい。

**GRPO系（Critic-free・Group-relative）**: DeepSeek-R1で採用されたGRPOは、同一プロンプトから複数ロールアウトをサンプリングし、グループ内の報酬統計でadvantageを正規化する。Criticが不要でメモリ効率が良いが、3つの構造的限界がある。

1. **価値の汎化不足**: 学習されたCriticが持つ状態間の価値一般化が得られない
2. **Advantage Collapse**: スパース報酬・長期タスクでは全ロールアウトが同じ報酬を受け、正規化advantageが消失（ゼロ勾配領域）
3. **サンプリングコスト**: 小グループはノイズが大きく、大グループはコストが増大

### 既存アプローチの限界

これらの限界に対し、複数のアプローチが提案されてきた。

- **Hydra-PPO**: 凍結バックボーンにPolicy/Valueアダプタを配置するが、アダプタ干渉のリスクあり
- **POISE**: Actorの中間表現から軽量プローブで価値を推定するが、クロスロールアウト推定に依存
- **SAO**: 非同期単一ロールアウト学習を実現するが、別途事前学習済みCriticが必要
- **VinePPO**: モンテカルロ継続で中間価値を推定するが、追加生成コストがかかる

**共通する課題**: 効率的な単一ロールアウト学習と明示的な temporal value estimation の両立が未解決だった。

## 方法論

### 核心アイデア: 自己回帰構造とActor-Criticの構造的対応

SAPOの設計は一つのシンプルな観察から出発する。

> Actor-Critic学習に必要な情報フローは、言語生成の因果的順序と自然に一致する。価値推定はアクション前に状態を要約すべきであり、アクション評価は生成された内容を追加的に条件付けできる。

この対応関係により、1つの自己回帰ストリーム内の適切な位置でPolicy生成とValue推定を表現でき、因果マスクがそれぞれの条件付けコンテキストを強制する。

### 因果的境界による2-token Value Basis

![Causal boundary value readout](/images/sapo-single-rollout-actor-critic/fig1.png)

ターン $t$ において、状態コンテキスト $c_t^s$ の後に応答 $a_t = (a_{t,1}, ..., a_{t,M_t})$ を生成する。SAPOは自己回帰モデルの語彙から2つのエントリ $w^+$, $w^-$ を**Value Basis**として予約する。

$$p_\theta(c) = \text{clip}\left(\frac{z_{\theta, w^+}(c) - z_{\theta, w^-}(c)}{\tau_v}, -1, 1\right)$$

ここで $\tau_v > 0$ は価値温度。報酬スケール $R_{\max}$ でスケーリングして：

- **状態価値** $V_\theta(s_t) = R_{\max} \cdot p_\theta(c_t^s)$ → 応答生成**前**の境界で読み取り
- **行動価値** $Q_\theta(s_t, a_t) = R_{\max} \cdot p_\theta(c_t^{sa})$ → 応答生成**後**の境界で読み取り

ポイントは、$w^+, w^-$ は生成トークンではなく**読み取り専用**であること。サンプリングもコンテキストへの追加も行われない。Policy語彙 $W_{\text{act}} = W \setminus \{w^+, w^-\}$ を定義し、アクション確率はこの制限語彙上でのみ正規化される。

### 単一ロールアウト軌跡Advantage推定

グループ相対手法が複数軌跡を比較するのに対し、SAPOは各タスク1つの軌跡をサンプリングし、学習済み状態価値で遅延フィードバックをターン間伝播する。

**Trajectory-level GAE**: 軌跡 $\tau = \{(s_t, a_t, r_t, d_t)\}_{t=1}^T$ について：

$$\delta_t = r_t + \gamma(1-d_t)V^{\text{old}}_{t+1} - V^{\text{old}}_t$$

$$A^{\text{GAE}}_t = \delta_t + \gamma\lambda(1-d_t)A^{\text{GAE}}_{t+1}$$

この再帰により、環境が終端報酬のみ提供する場合でも、各ターンに異なる学習シグナルが割り当てられる。

**2つの価値ターゲット**: 同一の後向き計算で、2つの因果的価値境界に対する補完的なターゲットを構成。

- 状態価値ターゲット: $y^V_t = V^{\text{old}}_t + A^{\text{GAE}}_t$ （λ-returnに向けて学習）
- 行動価値ターゲット: $y^Q_t = r_t + \gamma(1-d_t)Q^{\text{old}}_{t+1}$ （on-policy SARSAターゲット）

重要な設計選択: PolicyのadvantageはGAEから直接計算し、同時学習されるQとVの差からは計算しない。これにより、初期のQ値の誤較正がPolicyに直接伝播するのを防ぐ。

### バッチ正規化Turn Advantage

トークンにadvantageを割り当てる前、**現在のバッチ内の有効ターン全体**で正規化する。

$$\hat{A}^{b}_t = \frac{A^{e}_t - \mu_B}{\sigma_B + \epsilon_{\text{adv}}}$$

ここで $A^{e}_t$ は無効アクションインジケータによる補正済みadvantage。正規化**後**に全有効トークンにブロードキャストする。これにより、長い応答がadvantage統計で不均衡な重みを持つのを防ぐ。

### 統合目的関数

$$L_{\text{SAPO}}(\theta) = L_{\text{pol}} + c_V L_V + c_Q L_Q + \beta L_{\text{KL}} - c_H H(\pi_\theta)$$

- $L_{\text{pol}}$: クリップ付きPPOスタイルPolicy損失（turn-level advantageをトークンにブロードキャスト）
- $L_V, L_Q$: クリップ付き回帰損失（ターン単位で平均化、トークン単位ではない）
- $c_V, c_Q$: 生成と価値学習の干渉制御

Policy・状態価値・行動価値は別々のマスクを使用するが、**1回のバックワードパスで同じTransformerと言語モデルヘッドを更新**する。

## 実験結果

### ALFWorld

![ALFWorld results Qwen2.5-1.5B](/images/sapo-single-rollout-actor-critic/fig2.png)

**Qwen2.5-1.5B**でSAPOはALFWorld全体で90.1%の成功率を達成。PPO（54.4%）から+35.7pp、GRPO（72.8%）から+17.3ppの大幅改善。特にCleanとHeatカテゴリで**100%の完全制覇**を記録。

**Qwen2.5-7B**ではさらに向上し、全体94.0%。Clean・Heat・Lookの3カテゴリで100%を達成し、7BスケールでもSAPOの優位性が維持される。

### WebShop

![WebShop results comparison](/images/sapo-single-rollout-actor-critic/fig3.png)

WebShopでは約110万製品・12,000ユーザー指示の中から適切な商品を検索・購入するタスク。

- **1.5B**: Score 82.2（PPO 73.8から+8.4、GRPO 75.8から+6.4）
- **7B**: Score 88.6・成功率82.4%（PPO 81.4/68.8%から+7.2/+13.6pp）

7Bスケールでは全aggregate metricでPPO・GRPO・近接variantを上回る。

### 実行効率

![Runtime breakdown](/images/sapo-single-rollout-actor-critic/fig4.png)

ALFWorld・Qwen2.5-1.5Bでの1イテレーションあたりの実行時間を比較。

- **PPO合計**: 451.2秒
- **SAPO合計**: 301.4秒（**-33.2%削減**）

最大の節約は**Critic updateモジュールの完全除去**（61.4秒/iter）。生成時間も306.4秒→221.4秒に削減。Actor側のコスト（log prob計算・参照モデル評価・Actor更新）はほぼ同等で、価値推定の統合による追加オーバーヘッドは最小限。

## 考察

### 因果的境界の設計がなぜ機能するのか

SAPOの核心は「重み共有」自体ではなく、**自己回帰モデリングとActor-Critic学習の構造的対応**にある。自己回帰生成の因果的順序が、Actor-Criticで必要な情報の流れ（状態の要約 → アクション生成 → アクション評価）と完全に一致する。この対応により、別のエンコーダや手動でのコンテキスト分離なしに、同じバックボーンで3つの役割を兼務できる。

2-token Value Basisの設計も秀逸だ。logit差を取ることで共通シフトに対して不変になり、clipで学習範囲を報酬スケールに適合させる。予約トークンは生成されず読み取り専用であるため、Policy学習が価値basisを意図せず強化・抑制することはない。

### 単一ロールアウトでの長期信用割当て

GRPOのadvantage collapse問題（全ロールアウトが同一報酬 → 正規化advantage = 0）に対し、SAPOは学習済み価値関数によるGAEでターンレベルの信用割当てを実現。バッチ正規化も、グループ内ではなく**独立したターン・タスク間**で行うため、同一プロンプトの複数ロールアウトを必要としない。

### GAEを用いたPolicy Advantageの設計選択

QとVを同時学習しながら、Policy advantageはあえてGAEから直接計算する。初期のQ値較正誤差がPolicyに直接伝播するのを防ぐ工夫で、Q目的はあくまで行動条件付き価値表現の学習に専念させる。

### 残る課題

- 消融実験の詳細が論文内で限定的（各コンポーネントの寄与度を定量的に示すデータが不足）
- 言語のみの環境に限定され、マルチモダル（視覚・非同期ツール呼び出し）への適応は未検証
- Value温度 $\tau_v$ や係数 $c_V, c_Q$ の感受性分析が不十分

## 関連研究

| 手法 | 価値推定 | ロールアウト数 | 別Critic | 特徴 |
|------|----------|---------------|---------|------|
| PPO | 学習Critic + GAE | 1 | **必要** | 高精度だがメモリ重い |
| GRPO | グループ内統計 | 複数 | 不要 | Advantage collapseのリスク |
| RLOO | Leave-one-out基準 | 複数 | 不要 | 中間価値なし |
| DAPO | 動的サンプリング | 複数 | 不要 | 非対称クリッピング |
| SAO | 事前学習Critic | 1 | **必要** | 非同期単一ロールアウト |
| POISE | Actor隠れ状態プローブ | 複数 | 不要 | クロスロールアウト推定 |
| Hydra-PPO | 凍結BB + アダプタ | 1 | 不要* | アダプタ干渉リスク |
| **SAPO** | **因果境界V+Q** | **1** | **不要** | **単一ストリームActor-Critic** |

SAPOは「1ロールアウト」「Critic不要」「明示的temporal credit assignment」の3つを同時に達成した初の手法と言える。

## まとめ

SAPOは、自己回帰LLMの因果的構造をActor-Critic学習に構造的に対応させることで、これまでのエージェントRLのトレードオフを打破した。

- **因果的境界2-token Value Basis**: 1つのバックボーンでPolicy・V・Qを表現。予約トークンは読み取り専用でPolicyと干渉しない
- **Trajectory-level GAE + バッチ正規化**: 単一ロールアウトでターンレベルの信用割当てを実現。グループサンプリング不要
- **SARSA補助目的**: 行動価値のon-policy学習で即時の行動結果を学習
- **圧倒的効率**: Criticモデルを完全に除去し、実行時間33.2%削減
- **性能**: ALFWorld 94.0%（7B）、WebShop 88.6（7B）でPPO・GRPOを凌駕

明示的な価値学習は、モデルの重複や高コストなサンプリングを必ずしも必要としない——SAPOは効率的な長期エージェントRLへの実用的な道を示している。

## 参考

- Liang, D., Feng, L., An, B., & Liu, Y. (2026). SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning. arXiv:2608.19842.
- Shao, Z. et al. (2024). DeepSeekMath. (GRPO)
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. (PPO)
- Ahmadian, A. et al. (2024). Back to Basics: Revisiting REINFORCE-style Optimization. (RLOO)
- Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
- Hou, Z. et al. (2026). Single-Rollout Asynchronous Optimization for Agentic RL. (SAO)
- Santacroce, M. et al. (2023). Efficient RLHF: Reducing the Memory Usage of PPO. (Hydra-PPO)
- Choi, Y. et al. (2026). Your Language Model is Its Own Critic. (POISE)
