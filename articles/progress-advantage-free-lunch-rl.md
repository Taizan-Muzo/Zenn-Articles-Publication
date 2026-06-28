---
title: "Progress Advantage：RL後訓練が無償で与えるステップ評価シグナル"
emoji: "🎁"
type: "tech"
topics:
  - "LLM"
  - "強化学習"
  - "エージェント"
  - "推論"
published: true
---

## TL;DR

RL後訓練（PPO/GRPO/DAPO）を経たポリシーモデルと参照モデルのlog確率比が、追加訓練なしで最適アドバンテージ関数を厳密に復元する。この「Progress Advantage」は、テスト時拡張・不確実性定量化・障害帰属の3つの応用で、タスク固有の報酬モデルを上回る性能を示す。プロセス報酬モデルが不要になる、まさに"タダ飯"である。

---

## 1. 背景：なぜプロセス報酬モデルは難しいのか

LLMエージェントのステップレベル評価は長年の課題だ。人間によるフィードバックはスケールせず、モンテカルロ推定は長期間インタラクション・不可逆アクション・確率的環境フィードバックによって実行不可能になる。

これまで、専用のプロセス報酬モデル（Process Reward Model: PRM）を訓練することでこの問題に対処しようとしてきた。しかしPRMには以下の根本的困難がある：

1. **高コストなアノテーション**: 各中間ステップの正しさを人間が判断する必要がある
2. **ドメイン依存性**: 特定のタスク・環境で訓練したPRMは他ドメインに汎化しにくい
3. **分布シフト**: ポリシー更新に伴い、PRMの評価対象分布が変化する

本論文の核心的洞察はシンプルだ：**これらの困難に正面から取り組む必要はない**。RL後訓練そのものが、すでにステップレベル評価の材料を内包している。

---

## 2. 方法：Progress Advantageの理論的導出

### 2.1 トークンレベルMDPとしてのエージェント操作

まず、エージェントの操作を汎用的なトークンレベル確率MDP $(\mathcal{S}, \mathcal{A}, f, r, \rho)$ として定式化する：

- **状態** $s_t$：時刻 $t$ までに生成・観測された全トークン列
- **アクション** $a_t$：ポリシー $\pi(\cdot|s_t)$ が生成する次のトークン
- **確率的遷移** $f$：外部観測（ユーザーメッセージ、ツール出力）が新しい状態を形成
- **報酬** $r$：トークンごとのスカラー報酬（実際には軌跡終端でのみ与えられる）

RL後訓練の標準的目的関数はKL正則化付き：

$$
\max_{\pi_\theta} \mathbb{E}\left[\sum_t r(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]
$$

### 2.2 コア定理：Progress Advantage

**命題1（確率MDPにおけるProgress Advantage）**

$\tilde{\pi}^*$ をKL正則化RLの最適ポリシー、$\pi_{\text{ref}}$ を参照ポリシーとする。このとき、最適アドバンテージ関数はポリシー対の対数確率比で正確に復元される：

$$
\tilde{A}^*(s,a) = \beta \log \frac{\tilde{\pi}^*(a|s)}{\pi_{\text{ref}}(a|s)}
$$

計算は極めてシンプルだ：

$$
\text{ProgressAdvantage}(a_t|s_t) = \log \tilde{\pi}^*(a_t|s_t) - \log \pi_{\text{ref}}(a_t|s_t)
$$

![Progress Advantage概念図](/images/progress-advantage-free-lunch-rl/fig1.png)

### 2.3 なぜアドバンテージなのか：報酬との決定的差異

暗黙的報酬の閉形式解は以下で与えられる（備考1）：

$$
r(s_t, a_t) = \beta \log \frac{\pi^*(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + V^*(s_t) - \mathbb{E}_{s_{t+1}\sim f}[V^*(s_{t+1})]
$$

軌跡レベルで合計すると：

$$
\sum_t r_t = \sum_t \beta \log \frac{\pi^*(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \sum_t \delta_t
$$

ここで $\delta_t = V^*(s_t) - \mathbb{E}[V^*(s_{t+1})]$ は**残差項**である。決定論的MDPでは $\delta_t$ が望遠鏡和で相殺されるが、**確率的MDPでは $V^*$ がポリシー対から直接取得できない**。つまり、正確な報酬は復元不可能。

対して**アドバンテージ関数 $A(s,a) = Q(s,a) - V(s)$ は、定義によって確率性を自然に分離する**。対数確率比項が期待将来価値を吸収し、遷移モデルの知識なしに正確なアドバンテージを抽出できる。

さらに重要なのは、アドバンテージが**状態の難易度とアクションの質を分離**する点だ。簡単な状態での高報酬と難しい状態での中程度の報酬が、実際には同じアクション品質を反映している場合がある。アドバンテージはこの曖昧さを解消する。

### 2.4 ClippingベースRLへの一般化

**命題2** は、DAPOやDr. GRPOなど陽的KL正則化を用いないclippingベースのアルゴリズムでも、clipping制約 $R(s,a) \in [1-\varepsilon, 1+\varepsilon]$ が暗黙的に $D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \lesssim \varepsilon^2/2$ を課すことを証明する。これにより、PPOだけでなく**主要なRL後訓練アルゴリズム全般**がProgress Advantageの理論的枠組みに含まれる。

### 2.5 実装上の3つの設計選択

論文は理論から実践への橋渡しとして、以下の設計指針を提供する：

1. **参照ポリシーの指定**: $\pi_{\text{ref}}$ は $\tilde{\pi}^*$ から遠すぎず近すぎずが良い。RL-Zeroならプレトレイン基盤、標準RLならSFTチェックポイントが適切。TIES mergingで0.2〜0.7の混合比が安定した結果を示す（図4(a)）。

2. **アドバンテージの集約戦略**: 軌跡内のトークンごとのProgress Advantageをどう集約するかは応用に依存する（図4(b)）。単純和、トークン平均、位置重み付け、最小値、最大値など。

3. **トークン確率の表現**: 純粋なトークン確率はRL訓練中にノイズが多いため、top-k平均確率の使用も検討されている。

---

## 3. 実験結果

### 3.1 テスト時拡張（Best-of-N）

4つのベンチマーク（BFCLv4-MT, WebShop, AgentDojo, τ²-Airline）でBest-of-8選択を評価。

![Best-of-N結果](/images/progress-advantage-free-lunch-rl/fig2.png)

Progress Advantageは**Gemma4-4Bで平均15.5%、Qwen3.5-9Bで11.3%の成功率向上**を達成。特に高温探索が有効なWebShop（Gemma: 45.0% vs WildReward-8Bの41.0%）とτ²-Airline（Qwen: 72.0% vs Greedyの60.0%）で顕著な改善を示した。

注目すべきは、進歩アドバンテージが**タスク固有の訓練を受けたAgentPRM-7Bさえも上回る**点である（Table 12, Appendix C）。「訓練不要」でありながら「訓練済み専用モデル超え」という結果は驚異的だ。

### 3.2 不確実性定量化

τ²-Airlineとτ²-Retailで、各軌跡の成功/失敗をAUROCで予測するタスク。

![不確実性定量化結果](/images/progress-advantage-free-lunch-rl/fig3.png)

τ²-Airlineでは、Progress Advantageが**4モデル全てでClaude Sonnet-4.6（LLM-as-a-Judge）を凌駕**した。Gemma4-4BでAUROC 0.865（Claude 0.615）、Olmo3-7Bで0.799（Claude 0.715）。

より興味深いのは、Progress Advantageを**クロスポリシー評価**に使える点だ。Gemma4-4BのProgress AdvantageでQwen3.5-9Bの軌跡を評価するとAUROC 0.754を達成。これは、あるポリシー対から計算したProgress Advantageが**異なるポリシーの評価にも転用可能**であることを示す。

### 3.3 障害帰属

Who & Whenベンチマークで、多エージェントシステムの決定的エラーステップを特定するタスク。Progress Advantageは、このタスクのために専用訓練されたAgenTracerと同等の性能を示した。

---

## 4. 考察

### 本論文の最も重要なメッセージ

「無料の昼食などない」という格言がある。しかし本論文は、RL後訓練パイプラインに**すでに埋め込まれているが見過ごされてきた「無料の昼食」**を発見した。

$\log\tilde{\pi}^*(a|s) - \log\pi_{\text{ref}}(a|s)$ という、普段はKL正則化項として目的関数に現れるだけの量が、実は**最適アドバンテージ関数の厳密解**だったのである。

### 実践的意義

1. **プロセス報酬モデル不要**: ステップレベル評価のために別モデルを訓練する必要がなくなる
2. **即時利用可能**: 標準的なRL後訓練パイプラインの副産物として、追加計算なしで得られる
3. **ドメイン非依存**: タスク固有の訓練やアノテーションを必要としない
4. **ポリシー間転用可能**: あるポリシー対のProgress Advantageで別ポリシーの評価も可能

### 限界と注意点

- 参照ポリシーの選択が性能に影響する（遠すぎ/近すぎは良くない）
- 集約戦略は応用ごとにチューニングが必要
- 確率MDPの仮定が成り立たない環境では理論的保証が弱まる可能性がある

---

## 5. 関連研究

- **プロセス報酬モデル**: PRM（Lightman et al., 2023）、ThinkPRM、AgentPRMなど。いずれも専用訓練を必要とする。
- **暗黙的報酬**: DPOの暗黙的報酬 $r = \beta\log(\pi^*/\pi_{\text{ref}})$ はよく知られているが、確率MDPでの限界は見過ごされてきた。
- **確信度ベース評価**: Self-Certainty、DeepConfなど。追加訓練不要だが、Progress Advantageに一貫して劣る。
- **価値関数学習**: 別途価値ネットワークを訓練する手法。Progress Advantageは追加パラメータなしで同等以上を達成。

---

## 6. まとめ

Progress Advantageは、RL後訓練が自然に生み出す「無料」のステップレベル評価シグナルである。理論的に最適アドバンテージ関数と等価であり、3つの応用すべてで訓練済み報酬モデルを上回った。

実装は `log_prob(final_policy) - log_prob(reference_policy)` を計算するだけ。RL後訓練済みモデルを持っているなら、**今日から使える**技術である。プロセス報酬モデルの訓練に苦労している研究者・実務者にとって、これは見過ごせない発見だ。

---

## 参考

- [arXiv:2606.26080] Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents
- Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh, Tanwi Mallick, Sharon Li
- Submitted: June 24, 2026
