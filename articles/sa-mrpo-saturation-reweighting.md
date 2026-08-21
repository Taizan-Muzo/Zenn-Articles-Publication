---
title: "SA-MRPO：飽和認識の報酬再配分で多目的方策最適化を刷新"
emoji: "⚖️"
type: "tech"
topics:
  - "LLM"
  - "強化学習"
  - "RLHF"
  - "GRPO"
  - "多目的最適化"
published: true
---

## TL;DR

複数報酬を扱うLLM方策最適化では、従来GRPOはスカラー和でグループ標準化を行うため**異なる報酬プロファイルが同一 advantage になってしまう**、また**飽和済み目的にも勾配を配り続けてしまう**という2つの構造的問題を抱えていた。本稿で取り上げる **SA-MRPO（Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization）** は、各報酬目的を独立に標準化したうえでバッチ平均から推定した**飽和比率** $s^{(k)}$ に応じて $(1-s^{(k)})^\gamma$ で重みを減衰させ、最適化努力をまだ伸びしろのある目的へ動的に再配分する。Qwen2.5-7Bの数学推論3目的設定でAIME24を +5.0pt、適応推論でAMC23を +9.2pt押し上げるなど、単一ハイパーパラメータ $\gamma=0.5$ のみでGDPOを一貫して上回る結果を報告している。

---

## 背景

### RLVR と GRPO の隆盛

2025年以降、LLMの推論能力を引き上げるデファクトの手法が **Reinforcement Learning with Verifiable Rewards (RLVR)** である。中でも **GRPO（Group Relative Policy Optimization）** は、同じプロンプトに対して $G$ 本の方策 rollouts をサンプリングし、グループ内で報酬を標準化して advantage とするcritic-freeな手法として爆発的に普及した。

しかし現実の推論モデルには複数の最適化したい目的が存在する。

- **正しさ**（解答が真の答えに一致するか）
- **長さ**（出力トークン数が制限内か）
- **フォーマット**（XMLタグなどを正しく閉じているか）
- **安全性・実行可能性**（コードがコンパイル・実行可能か）

これらを**一つのスカラーにまとめてからグループ標準化**する——これが既存の選択肢だった。

### 既存手法の2つの限界

Wangらは、このありがちな実装に潜む**2つの構造的欠陥**を明示的に指摘する。

**限界1：スカラー化で報酬解像度が失われる**

重み $\{w_k\}$ で線形結合したスカラー報酬は、異なる $(r_1, r_2, \ldots, r_n)$ プロファイルに対して同一の値を返し得る。たとえば重みが等しいとき、$(1, 0)$ と $(0, 1)$ は区別できない。**グループの standardization はこのロスを回復できない。**

**限界2：固定重みは飽和度を無視する**

どの目的も訓練を通じて常に同じ相対的重みで寄与し続ける。たとえば2番目の目的の**達成率**が既に90%なのに、その目的が勾配予算の半分を食い続ける。**より伸ばせる余地のある目的へ予算を回す仕組みがGRPO系列にはない。**

![SA-MRPO が解決する2つの構造的問題](/images/sa-mrpo-saturation-reweighting/fig1_pipeline.png)

---

## 方法詳解

### 飽和認識のグループ相対 advantage

SA-MRPOの枠組みは次の3ステップで成る。

**Step 1：目的ごとの独立標準化**

各報酬目的 $k\in\{1,\ldots,n\}$ に対し、クエリ $q_i$ 内の $G$ 本のrolloutsで

$$
\mu_k^{(i)} = \mathrm{mean}\{r_k^{(i,1)},\ldots,r_k^{(i,G)}\},\quad
\sigma_k^{(i)} = \mathrm{std}\{r_k^{(i,1)},\ldots,r_k^{(i,G)}\}
$$

を取り、

$$
A_k^{(i,j)} = \frac{r_k^{(i,j)} - \mu_k^{(i)}}{\sigma_k^{(i)}}
$$

を目的ごとの advantage とする。**スカラー化を経ない**ので、まずスカラー化由来の報酬ロス問題は消える（これが GDPO と等価）。

**Step 2：バッチレベルでの飽和度推定**

目的 $k$ の**現在のバッチ平均** $\bar r^{(k)}$ を

$$
\bar r^{(k)} = \mathrm{mean}\{r_k^{(i,j)} : i\in[B], j\in[G]\}
$$

で計算し、既知の上下界 $(r_{\min}^{(k)}, r_{\max}^{(k)})$ から

$$
s^{(k)} = \frac{\bar r^{(k)} - r_{\min}^{(k)}}{r_{\max}^{(k)} - r_{\min}^{(k)}} \in [0,1]
$$

という**飽和比率**を定義する。$s^{(k)}$ が大きいほどその目的は既に達成領域に入りつつある。

**Step 3：飽和認識の重み付けと最終 advantage**

$$
\tilde w_k = w_k (1 - s^{(k)})^\gamma,\qquad
\tilde A^{(i,j)} = \sum_{k=1}^n \tilde w_k A_k^{(i,j)}
$$

これをバッチ全体 $\mathcal A$ で再度標準化して

$$
\hat A_{\mathrm{SA}}^{(i,j)} = \frac{\tilde A^{(i,j)} - \mathrm{mean}(\mathcal A)}{\mathrm{std}(\mathcal A)}
$$

とし、通常どおり clipping を施した PPO 様目的

$$
\mathcal J_{\mathrm{SA\text{-}MRPO}}(\theta) = \mathbb E\Big[\frac{1}{G}\sum_j\frac{1}{|o_{i,j}|}\sum_t \min(\rho_{i,j,t}\hat A_{\mathrm{SA}}, \mathrm{clip}(\rho,1\!-\!\epsilon,1\!+\!\epsilon)\hat A_{\mathrm{SA}})\Big]
$$

で $\theta$ を更新する。**飽和認識の介入は advantage の構成にのみ及び、方策更新の本体は GRPO のまま**を保つ。

![GRPO → GDPO → SA-MRPO の進化と、各段階での重み処理の違い](/images/sa-mrpo-saturation-reweighting/fig1_pipeline.png)

### $\gamma=0$ で GDPO に退化し、単一目的で GRPO に戻る

$\gamma=0$ のとき $\tilde w_k = w_k$ となり、SA-MRPO は GDPO の独立標準化部分と完全に一致する。さらに目的が1つならば GRPO そのものに戻る。**SA-MRPO は GDPO および GRPO の上位集合**であり、本質的な追加設計は「各目的のバッチ飽和度から再配分係数を作る」一点に絞られる。

![$(1-s)^\gamma$ の幾何と、相対重み比の絶対的なずれ](/images/sa-mrpo-saturation-reweighting/fig2_saturation_weight.png)

### 飽和認識は magnitude だけでなく sign も反転し得る

単なる weight rescaling と侮ってはいけない。各目的勾配 $g_k(\theta) = \nabla_\theta J_k(\theta)$ に対する一次近似で、$\tilde w_a\|g_a\|^2 + \sum_{k\neq a} \tilde w_k g_a^\top g_k$ が**負に転じれば、その目的 $a$ の advantage 符号は更新後にさえ反転し得る**。これは「予算配分ルール」であって「制約付き多目的最適化」ではないという論文の注意書きと整合する。

---

## 実験結果

論文は3ドメインで評価している。

### 5.1 数学推論

Qwen2.5-3B / Qwen2.5-7B を DeepScaleR-Preview で訓練し、AIME24 / Minerva / AMC23 / MATH500 / OlympiadBench を平均 pass@1 で評価した。

| 設定 | ベンチ | GDPO | SA-MRPO | Δ |
|------|--------|------|---------|---|
| 7B / 3-obj | AIME24 | 11.5% | **16.5%** | **+5.0pt** |
| 7B / 3-obj | MATH500 | 64.2% | **67.7%** | +3.5pt |
| 7B / 3-obj | Olympiad | 25.3% | 26.1% | +0.8pt |
| 3B / 3-obj | AMC23 | 31.5% | **35.2%** | +3.7pt |
| 3B / 2-obj | AIME24 | 5.0% | **8.5%** | +3.5pt |

15組中 **12組で SA-MRPO が GDPO を上回り**、主評価（AIME24 / MATH500）で明確な伸びを記録している。

### 5.2 適応推論（DeepSeek-R1-Distill-7B）

長さに明示的な飽和領域（$B_{\min}{=}1024$, $B_{\max}{=}2048$）を持たせた**段階的長さ報酬**下で訓練。

| ベンチ | GDPO Acc | SA-MRPO Acc | Δ |
|--------|----------|-------------|---|
| AIME24 | 5.2 | 7.3 | +2.1 |
| Minerva | 15.4 | 15.9 | +0.5 |
| AMC23 | 28.3 | **37.5** | **+9.2** |
| MATH500 | 47.1 | 51.5 | +4.4 |
| Olympiad | 18.1 | 20.9 | +2.8 |
| **平均** | 22.8 | 26.6 | **+3.8** |

5ベンチ全てで改善し、平均 +3.8pt。**飽和構造を持つ報酬設計下でこそ SA-MRPO の価値が最も際立つ**ことがわかる。

### 5.3 コード生成

Qwen2.5-7B を Eurus-2-RL で学習し、テスト通過率と実行可能性の2目的で最適化。

| ベンチ | GDPO Pass | SA-MRPO Pass | Δ |
|--------|-----------|--------------|---|
| APPS | 53.2 | 53.8 | +0.6 |
| CodeContests | 19.2 | **20.6** | +1.4 |
| Codeforces | 10.6 | **12.9** | **+2.3** |
| TACO | 36.0 | 35.6 | -0.4 |

実行可能性は比較的早期に飽和する一方、テスト通過率は伸びが続く——SA-MRPO は前者を保護しつつ後者に予算を集中している。

![3ドメインでの主要結果](/images/sa-mrpo-saturation-reweighting/fig3_main_results.png)

### 5.4 $\gamma$ 単独の感度

Qwen2.5-3B / 2-obj における数学5ベンチ平均の AIME24 スコアは、

| $\gamma$ | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|----------|-----|------|-----|------|-----|
| AIME24 | 5.0% | 8.5% | **9.0%** | 8.7% | 7.4% |
| 5ベンチ平均 | 26.4% | 27.5% | **28.0%** | 27.8% | 27.6% |

と**U字カーブ**を描き、$\gamma=0.5$ が安定してスイートスポットになる。$\gamma$ 自体を**転移学習と同様にハイパーパラメータ探索で決定できる**点は実用上有利。

![$\gamma$ の感度と最適点](/images/sa-mrpo-saturation-reweighting/fig4_gamma_ablation.png)

---

## 考察

### 「既に解けたもの」ではなく「まだ伸びるもの」に予算を回す

タイトル **"Learn What's Left, Not What's Mastered"** が示す哲学は、bitter lesson 的であると同時に極めて実用的な帰結を持つ。**どんな多目的最適化でも、上限に近い目的へ勾配を注ぎ続けるのは無駄であり、しかも他目的の進捗を阻害する**——RLVR の文脈では尚更である。

### 制約付き多目的最適化ではない

論文は繰り返し強調する：SA-MRPO は**適応的予算配分ルール**であり、**$\tilde w_k$ 自体に上限下限の制約や KL ペナルティを持たせた制約多目的最適化ではない**。ある目的の advantage 符号が反転し得ることも明示しており、ユーザー側の解釈余地を残している。補助目的を**「ほぼ保証された範囲」**に保ちたい場合は、$\gamma$ を下げるか、$(r_{\min}^{(k)}, r_{\max}^{(k)})$ を保守的に見積もるのが無難。

### $r_{\min}/r_{\max}$ の設計と前提

飽和比率 $s^{(k)}$ は $(r_{\min}^{(k)}, r_{\max}^{(k)})$ を既知とするが、これは**目的設計者が与えなければならない**。たとえば正しさ報酬なら $[0,1]$、長さ報酬なら $[0, B_{\max}]$ と素直に決まる。一方「スタイル整合」のように上限が曖昧な目的を足した場合、見積もりを誤ると飽和判定が破綻する。**実運用では目的を足した瞬間にこの境界を保守側に置く運用ルールが有効**だろう。

### 何が斬新で、何が当然だったか

斬新な点：

- 飽和率を **batch平均** から推定する（追加 forward 不要、追加報酬モデル不要）
- magnitude だけでなく **sign も反転し得る** 配置を理論的に明示
- $\gamma$ 単独で挙動を制御し、勾配クリップなどの既存機構を温存

当然な点：

- 「伸びる余地へ予算を寄せる」自体は多目的最適化の教科書の知識
- 独立標準化は GDPO / DVAO / GD2PO など近傍に既にあり、飽和スケーリングが本質的な差分

つまり SA-MRPO の価値は、**古典的な多目的最適化の知恵を、GRPO 系の標準 clipping と両立する形で実装した**ところにある。

---

## 関連研究

### 多報酬グループ相対最適化

- **GDPO**（Wang et al., 2024）：各次元を独立標準化してから集約。SA-MRPO の基礎
- **DVAO**：報酬分散に応じて重みを動かす。**飽和度そのものは推定しない**
- **GD2PO**：目的間 advantage の衝突を検出するが、伸びしろではなく不一致に基づく

### 動的報酬重み付け

- **Focal-RL**：ルーブリック基準の達成度に基づくが**人手設計を要す**
- **Dynamic Reward Weighting** / **SAW**：分散駆動で、**目標の上限に対する達成比率は無視**
- SA-MRPO は**有界報酬に対するバッチ平均 + 既知上限**という最小情報だけで動く

### On-Policy 系蒸留と critic-free RL

- **GRPO** 系一連（DeepSeek R1, DAPO, REINFORCE++）を前提としつつ、**critic を学ばず advantage だけ拡張**する立場

### Self-Reflection・適応テスト時計算

- 飽和認識という発想は、推論長制御（CAT, DASH）や自己検証にも通じる。**「やっても無駄な努力」を削る**視点の近傍に位置づけられる

---

## まとめ

SA-MRPO は、**複数報酬を扱う GRPO 系訓練に「目的ごとの飽和度から再配分する」という1行の介入**を持ち込むだけで、Qwen2.5 系と R1-Distill 系の双方で GDPO を安定的に上回ることを示した。

- **実装**：各目的 $k$ の独立標準化 + バッチ平均からの $s^{(k)}$ 推定 + $(1-s^{(k)})^\gamma$ 重み付け
- **結果**：数学推論15組中12組で GDPO 超え、AIME24 最大 +5.0pt、適応推論で AMC23 +9.2pt
- **実用**：追加 forward なし、追加モデルなし、$\gamma$ 単独で感度調整可能
- **制約**：制約多目的最適化ではなく適応配分ルール、目的の上下界 $(r_{\min}, r_{\max})$ が正しく与えられる前提

「**伸びる余地へ予算を回す**」——LLM 強化学習が多目的化していく今後の方向性と極めて整合的な、シンプルかつ強力な拡張である。

---

## 参考

- Wang, Y., Chen, Y., Zhang, H., Luo, H., Wu, X., Ni, J., Fu, Y., Vasconcelos, N., & Li, Y. *Learn What's Left, Not What's Mastered: Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization.* arXiv:2608.16072, 2026.
- Shao, Z., Wang, P., Zhu, Q., et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024. — GRPO原典
- Wang, Y., et al. *GDPO: Group Reward-Decoupled Normalization for Multi-Objective RL.* 2024. — SA-MRPO の退化先
- Kazdan, J., et al. *DAPO: An Open-Source LLM RL Recipe at Scale.* 2025. — clipping ベースラインの一つ
- Lai, X., et al. *CAT: Confidence-Adaptive Thinking for Efficient Reasoning of Large Reasoning Models.* arXiv:2607.00862, ACL 2026 Industry Track. — 計算飽和の文脈で関連
- Wei, H., et al. *DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation.* arXiv:2608.06243, 2026. — 時間方向の飽和制御で関連
