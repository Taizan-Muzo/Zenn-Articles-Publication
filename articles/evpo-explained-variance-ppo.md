---
title: "PPOかGRPOか？Explained Varianceが解くRLVRの最大のジレンマ"
emoji: "⚖️"
type: "tech"
topics: ["LLM", "強化学習", "PPO", "GRPO", "EVPO"]
published: false
---

# PPOかGRPOか？Explained Varianceが解くRLVRの最大のジレンマ

## TL;DR

- LLM後段訓練（RLVR）における最大の設計判断の一つは「Criticを使うかどうか」だが、PPO（Criticあり）とGRPO（Criticなし）のどちらが優れるかはタスク依存で、統一見解がない
- 北京大学・复旦大学が**EVPO**（Explained Variance Policy Optimization）を提案：Criticの予測品質を「Explained Variance」というメトリクスで監視し、**各ステップでCritic使用の可否を適応的に判定**する
- 基線選択問題を**カルマンフィルタ**として定式化し、PPOとGRPOがカルマンゲインの両極端であることを数学的に示した
- **定理1（性能崩壊境界）**: EV ≤ 0 ⇔ Criticノイズ ≥ 状態信号 ⇔ 最適カルマンゲイン ≥ 1/2、つまり「EV = 0が正確な切替境界」
- Sokoban 0.604（PPO比+92%）、FrozenLake 0.684（+12%）、WebShop 0.303（+20%）、MATH 0.416（+12%）で全4タスク一位

## はじめに：RLVRの設計ジレンマ

LLMの推論能力を引き出す強化学習（RLVR）は、現在のAI研究で最も活発な領域の一つだ。DeepSeek-R1に始まり、DAPO、R1-Zero、Kimi-k1.5など、実に多くの手法が提案されている。

その中で、研究者が直面する根本的な設計判断がある。

> **Critic（価値関数）を訓練して使うべきか、それとも使わずにバッチ統計だけで済ませるべきか？**

- **PPO**（OpenAIのChatGPTで採用）はCriticを訓練し、状態依存のベースラインで分散削減を行う
- **GRPO**（DeepSeek-R1で採用）はCriticを捨て、バッチ平均で代用する
- **DAPO**（ByteDanceのオープンソースRLシステム）もGRPOベースのCritic-free手法だ

どちらが良いか？——実はこれまで、明確な答えがなかった。

Sokobanのようなマルチターン計画タスクではCritic-based（PPO）が勝ち、MATHのような数学推論タスクではCritic-free（GRPO/DAPO）が健闘する。先行研究では「タスク依存」と片付けられてきた。

北京大学と复旦大学のチームがarXivに投稿した**EVPO**は、この「どちらを使うか」という二者択一自体を不要にするアプローチだ。

## 方法：Explained VarianceでCriticの「使える度」を測る

### Explained Varianceとは

EVPOの核となる指標は**Explained Variance（説明分散）**だ。強化学習の文脈では広く知られている概念だが、EVPOはそれを**アーキテクチャ選択の基準**として新たに位置づけた。

$$\mathrm{ev} = 1 - \frac{\mathrm{Var}(G - \hat{V}_\phi(s))}{\mathrm{Var}(G)}$$

直感的には「Criticが報酬の分散のうちどれだけを説明できているか」を表す。ev > 0ならCriticは意味のある信号を捉えている。ev ≤ 0なら——Criticのノイズが有益な情報を上回っている。

![fig1](/images/evpo-explained-variance-ppo/fig1_paradigm_comparison.png)

### カルマンフィルタの視点

ここからがEVPOの最も面白い部分だ。論文は基線選択問題を**カルマンフィルタの状態推定問題**として定式化する。

観測報酬 $G = V^\pi(s) + \epsilon$ を考えよう。ここで $V^\pi(s)$ は真の状態価値、$\epsilon$ はサンプリングノイズだ。Criticの予測は $\hat{V}_\phi(s) = V^\pi(s) + \delta(s)$ で、$\delta$ はCriticの推定誤差。

- $P_A = \mathrm{Var}(\delta)$：Criticのノイズ分散
- $P_B = \mathrm{Var}_s(V^\pi(s))$：状態価値の真の分散
- $R$：サンプリングノイズ

このときExplained Varianceは次のように表せる：

$$\mathrm{ev} = \frac{P_B - P_A}{P_B + R}$$

分母は常に正なので、**EVの符号は $P_B - P_A$ だけで決まる**。Criticのノイズが状態信号を上回ればEVは負になる。

PPOは $K = 0$（完全にCriticを信頼）、GRPOは $K = 1$（Criticを完全に無視）として、最適カルマンゲインの両極端に位置づけられる。

![fig2](/images/evpo-explained-variance-ppo/fig2_kalman_variance.png)

### 定理1：性能崩壊の正確な境界

論文の理論的貢献の中心は**定理1（Performance Collapse Boundary）**だ。

$$\mathrm{ev} \leq 0 \;\;\Longleftrightarrow\;\; P_A \geq P_B \;\;\Longleftrightarrow\;\; K \geq \tfrac{1}{2}$$

この三つの条件が同値であることが証明されている。つまり：

- EV > 0のとき → Criticは分散を減らす → **Criticを使うべき**
- EV ≤ 0のとき → Criticは逆に分散を増やす → **バッチ平均に戻すべき**
- 境界は**まさにEV = 0**であり、閾値のチューニングは不要

### EVPOアルゴリズム

これを実際のアルゴリズムに組み込むのはシンプルだ。各訓練ステップで：

1. $M$本のロールアウトからバッチレベルEVを計算
2. $\widehat{\mathrm{ev}}_\mathcal{B} > 0$ → Critic値をベースラインに使用
3. $\widehat{\mathrm{ev}}_\mathcal{B} \leq 0$ → バッチ平均 $\bar{G}_\mathcal{B}$ に切り替え

分散の理論的保証も美しい：

$$\mathrm{Var}(\hat{A}^{\mathrm{EVPO}}) = R + \min(P_A, P_B)$$

これはPPOの分散（$R + P_A$）とGRPOの分散（$R + P_B$）の**小さい方以下**であり、**各ステップで**成り立つ。つまり「最悪でも固定ベースラインのどちらかと同等」で、Criticの質が良ければそれを活かし、悪ければ安全に退避する。

## 実験結果

### 4タスクでの全面的優位

実験はQwen2.5-3B/7Bをベースに、4つの多様なタスクで評価された。

![fig3](/images/evpo-explained-variance-ppo/fig3_main_results.png)

| 方法 | Sokoban | FrozenLake | WebShop | MATH |
|------|---------|------------|---------|------|
| Base LLM | 0.139 | 0.133 | 0.025 | 0.332 |
| PPO | 0.314 | 0.611 | 0.252 | 0.373 |
| StarPO-S | 0.455 | 0.359 | 0.178 | 0.371 |
| GRPO | 0.156 | 0.234 | 0.143 | 0.379 |
| DAPO | 0.162 | 0.207 | 0.127 | 0.385 |
| **EVPO** | **0.604** | **0.684** | **0.303** | **0.416** |

注目すべき結果をいくつか挙げる。

**Sokoban**での圧倒的改善：EVPO 0.604はPPO（0.314）の**+92%**、GRPO（0.156）の**+287%**だ。マルチターン計画タスクでは状態依存ベースラインの価値が大きいことが分かるが、初期のノイズ多いCriticを回避することで、PPOの弱点を克服している。

**MATH**での一貫した改善：数学推論ではCritic-free手法（GRPO/DAPO）がPPOに近い成績を出していたが、EVPOはそれらをさらに**+8~10%**上回った。少ないながらもCriticが提供する信号を活かせることが示唆される。

**DAPOより大幅に優れる**：GRPOベースの現代的手法であるDAPOすら、Sokobanで0.162、WebShopで0.127と低い。バッチ統計だけではマルチターンタスクの信用割当が不十分で、EVPOの適応的切り替えの有効性が際立つ。

### ゲート挙動の分析

![fig4](/images/evpo-explained-variance-ppo/fig4_gating_analysis.png)

訓練中のCritic選択確率（P(EV > 0)）を追跡すると、タスクごとに異なるパターンが見られる。

- **Sokoban / MATH**：訓練初期はEV ≤ 0（バッチ平均モード）で、Criticが成熟するにつれて徐々にCriticモードへ移行。Criticの学習曲線に追従する適応的な振る舞い。
- **FrozenLake**：約80歩目でゲート頻度が逆転し、ちょうど訓練の停滞期と一致。ポリシーが困難な領域に入るとCritic品質が一時的に劣化するが、適応ゲートがこれを検知してバッチ平均モードに退避する。

この挙動は「Criticは常に信頼できるわけではない」という論文の主張を強く支持している。

### EV閾値の感度分析

EVの閾値 $\tau$ を -0.2 ~ 0.2 の範囲で変化させると、**$\tau = 0$ が経験的最適**であることが確認された。

- 正の閾値（Criticを保守的に使用）：訓練後半でCriticが必要なのにバッチ平均に留まり、性能が低下
- 負の閾値（Criticを積極的に使用）：訓練初期のノイズが訓練全体を引きずり、性能が低下

理論で導いた「ゼロ境界」がそのまま経験的最適操作点になったことは、理論の頑健性を示している。

## 考察：なぜEVPOはうまくいくのか

EVPOの成功はいくつかの重要な示唆を含んでいる。

### 1. 「Critic vs No-Critic」は偽のジレンマだった

RLVRコミュニティでは「Criticベース vs Critic-free」の議論が二項対立として語られてきた。しかしEVPOの結果は、**両者の長所を組み合わせる余地が十分にある**ことを示している。訓練の段階に応じて適切な方を選べば、どちらの固定戦略よりも良い結果が得られる。

### 2. Criticの「成熟過程」を尊重する

強化学習でCriticの品質が時間とともに変化することは経験的に知られていたが、これを体系的に活用した手法はなかった。EVPOはEVという軽量なメトリクス（追加の前向き伝播不要）でCriticの成熟度をモニタリングし、それに基づいてアーキテクチャ選択を行う。これはRLVRの実践的なプラクティスとして非常に有用だ。

### 3. カルマンフィルタの視点の説得力

基線選択をカルマンフィルタとして定式化する枠組みは、PPOとGRPOを統一的に理解する強力なレンズを提供する。両者が同一の原理の異なる極端であるという見方は直感的な理解を深める。

## 限界と今後の課題

論文自身もいくつかの限界を認めている。

- **有限サンプルの集中不等式**: 定理1は母集団EVに基づくが、実際は有限バッチで計算される。切り替えルールの有限サンプル収束は未証明。
- **メモリオーバーヘッド**: EVPOは訓練全体でCriticネットワークを維持する。完全なCritic-free手法にはメモリ面で劣る。
- **タスクカバー範囲**: 4タスクのみで評価。密な中間報酬や報酬モデルの学習など、異なる設定でのEVの挙動は不明。

個人的には、**GRPOの統計的バイアス問題**（難しいプロンプトを系統的に過小評価する点）へのEVPOの影響も興味深い。Criticモードでは状態依存ベースラインを使うため、このバイアスを緩和できる可能性がある。

## 関連研究の位置づけ

| 手法 | 年 | Critic | ベースライン | EVPOとの関係 |
|------|-----|--------|-------------|-------------|
| PPO | 2017 | あり | Critic値 | K=0の極端 |
| GRPO | 2024 | なし | バッチ平均 | K=1の極端 |
| DAPO | 2025 | なし | バッチ平均（改良） | Critic-free最強の基線 |
| StarPO-S | 2025 | あり | 分散フィルタ | Critic-basedの改良 |
| SPPO | 2026 | なし | CB定式化 | 別アプローチ（Sequence-Level） |
| **EVPO** | **2026** | **あり** | **適応切替** | **両極端の統合** |

EVPOの位置づけは明確だ。PPO/GRPOの「どちらかを選ぶ」設計から、「両方を状況に応じて切り替える」設計へのパラダイムシフト。

## まとめ

EVPOはRLVRの後段訓練におけるCritic活用問題に、理論的に根拠のある実用的な解答を提示した。

- **Explained Variance**がCriticの有用性を判定する十分な指標である
- **カルマンフィルタの定式化**がPPOとGRPOの統一的理解を可能にした
- **定理1**がEV = 0を正確な切替境界として特定した
- 4タスクで**全ての固定ベースライン手法を一貫して凌駕**した

「EVPOが全てを解決する」と言うつもりはないが、RLVRの実装においてEVを監視し、Criticの品質に基づいてベースライン戦略を切り替えるというアイデアは、即座に採用可能で、既存のRLパイプラインに最小限の変更で組み込める。実務的なインパクトは大きいと評価したい。

## 参考

- 論文: [EVPO: Explained Variance Policy Optimization for Adaptive Critic Utilization in LLM Post-Training](https://arxiv.org/abs/2604.19485) (arXiv:2604.19485)
- 著者: Chengjun Pan, Shichun Liu, Jiahang Lin, et al. (北京大学, 復旦大学, 上海人工智能ラボ)
- 日付: 2026-04-21
