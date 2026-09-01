---
title: "SFTからGRPOへ：TailSFTが示す「coverage-first」な初期化の威力"
emoji: "🎯"
type: "tech"
topics: ["LLM", "強化学習", "GRPO", "SFT", "coverage", "post-training", "OLMo"]
published: false
published_at: 2026-09-01
---

## TL;DR

UCSDとMicrosoft Researchによる **TailSFT** は、SFTの **「学習済のサンプルを丸ごと捨てる」** というシンプルな変更で、後段のGRPOを **最大 +3.93 pt** 改善できることを示した。直感に反するが、pass@1を **捨てて** coverage を **買う** のが正しい trade-off だった。

- SFTは **pass@16 を最大 +16.79 pt** (CruxEval-O) 改善する一方、pass@1 は悪化する場合がある
- 5/5 の設定で **post-RL pass@1 が +1.21〜+3.93 pt 改善**
- 「coverage ratio $\rho_{16}$ > 1」を満たす設定の **10/11 で coverage 改善を確認**(1 件は横ばい)
- 計算 overhead は ほぼゼロ。SFTの **drop-in replacement**

> **論文**: [TailSFT: Filtered Fine-Tuning Improves Post-Training Performance](https://arxiv.org/abs/2608.25756) (Malladi et al., 2026, arXiv:2608.25756)
> **キーワード**: coverage, pass@K, GRPO post-training, stage-aware training, OLMo-3

---

## 背景：SFTとRLは「違うもの」を最適化している

近年のpost-training pipeline は **SFT → RL** の二段構えが定番になった。DeepSeek-R1が示したように、RL(特にGRPO)は推論能力の獲得に劇的に効く。

一方で、著者らは冒頭で **「既存のSFT pipeline が、本当にRLのよい初期値を作っているのか?」** と問い直す。最近の研究(Yue et al. 2025、Zhang et al. 2026a など)は、現状の **SFTがcross-entropy(CE)を最適化しすぎ**、RL段階にとっての 「coverage」を破壊していると指摘している。

### coverage とは何か

Chen et al. (2026a) の **Coverage Profile** が鍵になる定義:

$$\operatorname{Cov}_N(\pi_D \| \pi) := \Pr_{x\sim\mu, y\sim\pi_D(\cdot \mid x)}\!\left[\frac{\pi_D(y\mid x)}{\pi(y\mid x)} \geq N\right]$$

$\pi_D$ (target distribution) 上で、$\pi$ が **N倍 過小評価** している確率質量。**値が小さいほど coverage が良い**。

直感的に言えば、

- **pass@1** = 「一番自信ある答えが当たる確率」(集中度)
- **pass@K (大K)** = 「複数サンプリングした時にどれかが当たる確率」(coverageのproxy)

ここで重要な事実:

> **CE最適化とcoverage最適化は、必ずしも一致しない。**

例えば同じモデルが、CE-lossは低いが「珍しい正解」を出す確率はゼロ、なんてことが普通に起きる。RL、特に **Best-of-K サンプリングに依存する policy gradient** は coverage が本質的に重要になる。つまり、RLにとって最良の初期化は、CE-only 評価では見えづらい性質を持っている。

これが TailSFT 全体の motivation になる。

---

## 方法：TailSFTのコアアイデア

### 着想

標準SFTは **全サンプルに対し一律に** $-\log\pi(y\mid x)$ を最小化する。問題は、ここで言う「全サンプル」には次が含まれる:

1. モデルがすでに高い $\pi(y\mid x)$ を割り振っている **(already-fit)** サンプル
2. モデルは低確率だが、データ分布上は **まだ到達可能** な (under-fit) サンプル

(1) に gradient を当て続けると、何が起きるか?  **occam's razor が逆に効く**。多数派の(あからさまな)解をさらに強く押す一方、少数派(でも正解)の解をどんどん **下げる**。これが SFT-stage で coverage を毀損する主要因になる。

### Filter設計

TailSFT は「**訓練時の損失が、初期モデル時点から見て最も下がっていないサンプルを**」バッチから除外して勾配計算から外す。

**Algorithm 1**:

```
入力: 初期policy π₀, SFT データ D, フィルタ率 γₜ
1. 各 (xᵢ, yᵢ) で ℓᵢ⁰ ← ℓ(π₀; xᵢ, yᵢ) を記録
2. 各 training step t:
   a. batch Bₜ をサンプル
   b. 各 (xᵢ, yᵢ) ∈ Bₜ で ℓᵢᵗ ← ℓ(πₜ; xᵢ, yᵢ)
   c. ℓᵢᵗ − ℓᵢ⁰ の小さい方から γₜ 比率を  Fₜ に固定
   d. Bₜ \ Fₜ に対し length-normalized CE で勾配step
```

**オフセット基準** $\ell^t - \ell^0$ は relative に「**初期値から見てどれほど楽になったか**」を測る。同じ絶対損失 0.1 でも、初期損失 0.2 から来れば大きな進展、初期損失 0.05 から来れば既に fit。TailSFT はこの違いを捉えて、後者を捨てる。

### 3種類のフィルタ比較

| 方式 | 基準 | 問題点 |
|------|------|--------|
| Absolute | $\ell < \alpha$ で filter | $\alpha$ の tuning が難しく、初期 $\pi_0$ の scale 感度を無視 |
| Quantile | batch 内下位 γ% を filter | prompt毎にかけたい重みが見えない |
| **Offset (= TailSFT)** | $\ell_t - \ell_0 < \log\beta$ で filter | 初期 policy からの 相対 進捗。論文の定理で coverage 最良 |

論文 Theorem が明確に主張するのは:

$$\inf_\beta \operatorname{Cov}_N(\pi^\star \| \pi_{\text{OFF},\beta}) \le \min\!\Big\{\operatorname{Cov}_N(\pi^\star \| \pi_{\text{ERM}}),\ \inf_\alpha \operatorname{Cov}_N(\pi^\star \| \pi_{\text{ABS},\alpha})\Big\}$$

absolute フィルタは最悪 ERM より悪化しうるが、offset は **必ず ERM 以上** にできる。さらにこの保証は **どんな初期 policy** でも成り立つ。

### 図解

![fig1: Standard SFT vs TailSFT filtering schema](/images/tailsft-filtered-finetune-coverage/fig1.png)

上が **標準SFT** で全サンプルに update。下 (TailSFT) は "already fit" なサンプルを **filter** し、初期 $\pi_0$ の tail mass を保存する。

### Coverage profile の直感

![fig2: pass@K curve schematic](/images/tailsft-filtered-finetune-coverage/fig2.png)

tail mass を 保存 することで、pass@K の **K→∞ での asymptotic** が改善する。**K を 1 にした時は std が勝つ** が、Best-of-K で RL が exploit できるようになると、tail が逆転する。

---

## 実験：OLMo-3 7B × 数学/コード × 3 訓練ソース

### SFT 段階：pass@16 が一貫して伸びる

**OLMo-3 7B** をベースに、3 種類の SFT データで学習。比較は pass@1 と pass@K。

![fig3: pass@16 lift from TailSFT — BigCode SFT data](/images/tailsft-filtered-finetune-coverage/fig3.png)

**BigCode訓練** (主要結果)。**18 設定中 15** で pass@16 が改善。最大は **CruxEval-O で +16.79 pt** (24.21 → 41.00)。

興味深いのは pass@1 の方。Std SFT は pass@1 で勝っている (高々 2 pt 程度) 場合すらある。一見 TailSFT が劣後しているように見えるが、次節の RL 段階で **この不利が完全に逆転** する。

### GRPO 段階：post-RL pass@1 が **全部上がる**

![fig4: post-RL pass@1 lift + learning curve](/images/tailsft-filtered-finetune-coverage/fig4.png)

**重要な発見**: GRPO を走らせると、5 設定 **全て** で TailSFT init のほうが最終 pass@1 が高い。

| 訓練 | 評価 | Std-init pass@1 | Tail-init pass@1 | Δ |
|------|------|------|------|-----|
| MATH (OMI) | MATH-500 L5 | 57.70 | **60.26** | **+2.56** |
| MATH (OMI) | AIME | 14.40 | **15.61** | **+1.21** |
| MBPP+ (BigCode) | MBPP+ | 69.57 | **73.50** | **+3.93** |
| MBPP+ (Magicoder) | MBPP+ | 70.52 | **73.24** | **+2.72** |
| MBPP+ (OCI) | MBPP+ | 74.67 | **76.30** | **+1.62** |

学習曲線 (Fig 4b) も示唆的: **初期は不利** だが、訓練中盤で cross して **最終 asymptote が上**。

コード3設定では TailSFT は **RL前 pass@1 が逆に低く** (Stdの53.04に対し51.36 等)、それがRLを経ると **全設定で逆転** する。これが「**RL前弱い、RL後強い**」パターン。

---

## 考察：stage-aware なtraining

### 診断指標 $\rho_{16}$

論文は実用上とても大事な指標を提案している。

事前計算 (基礎モデル + 通常のSFT を **1回**) で

$$\rho_{16} = \frac{\sum [f_{16}(P_{i,1}(\pi_0)) - f_{16}(P_{i,1}(\pi_{\text{SFT}}))]_+}{\sum [f_{16}(P_{i,1}(\pi_{\text{SFT}})) - f_{16}(P_{i,1}(\pi_0))]+}$$

を計算する。意味は「SFT で **失った** coverage 量 / SFT で **得た** coverage 量」。 **$\rho_{16} > 1$ なら TailSFT を試す価値がある** (経験的に十分条件)。

検証では 11 設定中 **10 で coverage 改善**、最大 **+28.69%** の絶対coverage 向上を確認。

### 関連研究との位置づけ

- **DPO / DCO loss** (Chen 2025): $-\log(1 - (1-\pi(y|x))^K)$ は K の大きさに合わせて「soft に under-fit に重みを置く」先行研究だが、**初期 policy を参照しない**ため SFT 中の動的 filtering には直接展開できない
- **RHO-LOSS** (Mindermann 2022): 同様の pre-training 用 sample-importance weighting。TailSFT との差分は (1) **post-training を 目的** にしている、(2) **sequence-level** で、(3) **初期参照** を 持つ、 こと
- **GRPO 改善ライン** (GRPO-LEAD/LIPO/...): いずれも **post-RL の loss** に手を入れる。TailSFT は **initialization** 側で同じ問題意識を解いている

### 接続する視点

この論文は「**RL で gain を取るために SFT の最適化目標を意図的に 外す**」という、stage-aware 学習 の具体的な実装になっている。類似の議論として:

- LOPD (Latent On-Policy Distillation) や OPSD 系は、 **RL 中の distillation** で off-policy データを再 weighting する
- DASH, TTPO 等は RL loss 自体の **temporal/structural な調整** に踏み込む
- TailSFT は **最も早い段階**(SFT) の cross-entropy を 「**わざと外す**」ことで、その先の最適化を unlocking する

---

## まとめ

TailSFT の contribution は、論文が supply する4点に整理できる:

1. **方法**: cross-entropy 損失を apply する前に **初期 policy からの relative loss gain** で filter をかける
2. **理論**: 任意の初期policy で、offset-filter は **常に ERM 以上の coverage** を保証
3. **診断**: $\rho_{16}$ だけで TailSFT の 必要性を **基礎モデル + 通常SFT 1回**で判定できる
4. **原則**: 中間チェック点を **「次の段階に対してどれほど ready か」** で評価すべき

私自身、SFT-loss と post-RL gain のミスマッチは「自分も経験あるな」と感じるところで、そのミスマッチの「**損失関数側で** 治せる」というシンプルで直感的な解を提示したのがこの論文の押しポイントになっている。

実装コストはばかにならないほど軽い (forward pass を batch 内でもう一度するだけ) ので、post-training pipeline を組んでいる人は、**$\rho_{16}$ を一度 計算するだけで 試す価値がある**。

---

## 参考

- Malladi, S., Jelassi, S., Foster, D.J., Ash, J.T., Krishnamurthy, A. [**"TailSFT: Filtered Fine-Tuning Improves Post-Training Performance"**](https://arxiv.org/abs/2608.25756). arXiv:2608.25756, 2026.
- Chen, K., et al. **"Coverage Profile: A diagnostic for RL post-training"**. 2026a.
- Chen, K., et al. **"DCO: A coverage-aware loss for SFT"**. 2025.
- Mindermann, S., et al. **"RHO-LOSS: Reducing redundant computation in pre-training"**. 2022.
- Yue, Y., et al. **"How does SFT affect RL? A study"**. 2025.
- Zhang, J., et al. **"SFT-RL alignment gap"**. 2026a.
- Guo, D., et al. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"**. 2025.
- Schulman, J., et al. **"Proximal Policy Optimization Algorithms"**. 2017.
- Shao, Z., et al. **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"**. (GRPOの原論文) 2024.

---
