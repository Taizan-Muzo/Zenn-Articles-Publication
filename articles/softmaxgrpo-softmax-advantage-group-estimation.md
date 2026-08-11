---
title: "SoftmaxGRPO：Softmaxアドバンテージ推定でGRPOのz-score発散を克服・COLM2026"
emoji: "🌡️"
type: "tech"
topics: ["RLHF", "GRPO", "SoftmaxGRPO", "強化学習", "LLM", "ポリシー最適化", "COLM2026"]
published: true
---

## TL;DR

GRPO の z-score 正規化は、二値報酬下で「ほぼ解けている簡単なプロンプト」に勾配予算を発散的に集中させる。**SoftmaxGRPO** は z-score を温度スケール softmax アドバンテージに置き換える **一行レベルの変更** でこの問題を解決。大規模理論解析により、SoftmaxGRPO が REINFORCE から MaxRL、さらに ML へと温度で連続的に補間する統一的目的族であることを示す。Qwen2.5-1.5B で DeepMath **51.8%**、Poetry を **35.0%→68.0%**（+13.4pt）に引き上げ、全タスクで GRPO を上回る。COLM 2026 採録。

---

## 背景：GRPOの見過ごされてきた「プロンプト難易度バイアス」

強化学習による LLM 後訓練（RLVR）のデファクトスタンダードとなった GRPO（Group Relative Policy Optimization）は、各プロンプトに対して $M$ 個のロールアウトをサンプリングし、グループ内で報酬を z-score 正規化したアドバンテージでポリシーを更新する。

この「グループ内正規化」は単なる実装上の細部ではない。**二値報酬下では、プロンプトの通過率 $p$ に応じて勾配の重みが $1/\sqrt{p(1-p)}$ に比例する**。つまり、$p\to 1$（ほぼ常に正解）や $p\to 0$（ほぼ常に不正解）の極端なプロンプトで重みが発散する。

数値で見ると衝撃的だ。GSM8K での実験では、GRPO は勾配予算の **36.4%** を $p\geq0.9$ の「ほぼ解けている」プロンプトに費やしている。これらのプロンプトからは新しい学習シグナルがほぼ得られないにもかかわらず、だ。

DAPO、Dr.GRPO、CISPO、DPPO などの後続手法は、この問題をクリッピングやフィルタリングで対症療法的に扱ってきたが、根本的な目的関数の幾何学に手を付けてはいなかった。

### 問題の定式化

GRPO のグループ内 z-score アドバンテージは：

$$A_i^{\text{GRPO}} = \frac{R_i - \bar{R}}{\sigma_R}$$

二値報酬 $R_i\in\{0,1\}$ を仮定すると、正解率 $p$ のプロンプトに対する実効的なプロンプト重みは $\omega(p) \propto 1/\sqrt{p(1-p)}$ となり、$p\to 0$ と $p\to 1$ の両端で発散する。これは極端な difficulty のプロンプトに不当な重みがかかることを意味する。

---

## 方法详解：SoftmaxGRPOの理論と実装

### コアアイデア：一行の置き換え

SoftmaxGRPO の中心的な変更は驚くほど単純だ。GRPO の z-score アドバンテージ計算を、**温度パラメータ $\tau$ でスケールされた softmax に置き換える**：

$$w_i = \frac{\exp(R_i/\tau)}{\sum_{j=1}^{M} \exp(R_j/\tau)}, \quad A_i = M w_i - 1$$

これだけである。$\sum_i A_i = 0$ が自動的に満たされるため、グループ内加算シフト不変性は保持される。全ロールアウトが同一報酬の場合は更新ゼロとなる。

![図1: プロンプト重み関数の比較](/images/softmaxgrpo-softmax-advantage-group-estimation/fig1.png)

**図1** に示すように、GRPO の重みが $p\to 1$ で発散するのに対し、SoftmaxGRPO の重みは全領域で有界に留まる。温度 $\tau$ が小さいほど難しいプロンプト（低 $p$）に重みがシフトする。

### 理論的基盤（1）：二値報酬での厳密な有限グループ母集団目的関数

著者らは二値報酬 $R_i\in\{0,1\}$ に対して、SoftmaxGRPO の **正確な有限-$M$ 母集団勾配** を導出している。これがこの論文の理論的なハイライトだ。

$S \sim \mathrm{Binomial}(M-1, p)$（サンプル $i$ 以外の成功ロールアウト数）を条件付けると、成功・失敗間の中心アドバンテージギャップは：

$$\Delta_s^{(\tau)} = M\left(\frac{c}{M+(s+1)(c-1)} - \frac{1}{M+s(c-1)}\right),\quad c=e^{1/\tau}$$

これにより、厳密なプロンプト重み関数が Bernstein 多項式として閉形式で得られる：

$$\omega_{M,\tau}(p) = \mathbb{E}_{S\sim\mathrm{Binomial}(M-1,p)}[\Delta_S^{(\tau)}]$$

$$= \sum_{s=0}^{M-1} \Delta_s^{(\tau)} \binom{M-1}{s} p^s (1-p)^{M-1-s}$$

この結果の重要性は二つある：
1. **有界性の証明**：$p\to 1$ で $\omega_{M,\tau}(1) = 1 - \frac{M}{1+(M-1)e^{1/\tau}} < \infty$
2. **低温極限が MaxRL に一致**：$\tau\to 0$ で $\omega_{M,0}(p) = \frac{1-(1-p)^{M-1}}{p}$、これは $T=M-1$ の MaxRL 重みと完全一致

### 理論的基盤（2）：大グループ極限と Log-MGF 目的関数

一般の有界スカラー報酬に対して、$M\to\infty$ の大グループ極限では、SoftmaxGRPO の更新は **報酬の log 積率母関数（log-MGF）** の勾配に収束する：

$$\lim_{M\to\infty} \mathbb{E}_{\mathcal{G}(x)}\left[\nabla_\theta \mathcal{J}_{\text{SoftmaxGRPO}}^{\text{uc}}\right] = \nabla_\theta \log Z_\tau(\theta; x)$$

ここで $Z_\tau(\theta; x) = \mathbb{E}_{z\sim\pi_\theta}[e^{R(x,z)/\tau}]$ である。

これは極めてエレガントな結果だ。SoftmaxGRPO は大グループにおいて「報酬分布のテール特性を最適化する」と解釈できる。$\tau$ が小さいほど高報酬テールに感度が集中し、大きいほど報酬分布全体を均等に考慮する。

### 「有限グループスカラー目的関数の非存在」という重要な否定的結果

著者らはさらに、**報酬が 3 値以上の場合、有限 $M$ では一般的なスカラーポテンシャル関数が存在しない**ことを示している。具体的には、$M=2$ で 3 レベルの報酬があるとき、クロス偏微分の非対称性：

$$\frac{\partial(m_1-m_3)}{\partial p_2} = -\frac{1}{15} \neq \frac{1}{15} = \frac{\partial(m_2-m_3)}{\partial p_1}$$

により、更新が非保存的（経路依存）になることが証明される。

これは「理論的にこれ以上は望めない」という一種の完全性定理であり、二値報酬で厳密解が得られることの特別さを際立たせている。

### 実用的な PPO 形式

実際の訓練では、標準的な PPO クリッピングと KL 正則化を組み合わせる：

$$\mathcal{L}_{\text{SoftmaxGRPO}}^{\text{clip}}(\theta) = -\mathbb{E}\left[\frac{1}{\sum_i T_i}\sum_{i=1}^{M}\sum_{t=1}^{T_i}\min\left(\rho_{i,t}A_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)A_i\right)\right]$$

$$\mathcal{L}_{\text{SoftmaxGRPO}} = \mathcal{L}^{\text{clip}} + \beta \cdot \mathbb{E}\left[\text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

### 適応的温度：ESS に基づく自動調整（Appendix）

付録では、有効サンプルサイズ（ESS）を固定値に保つ適応的温度スケジューリングも提案されている：

$$c = \frac{k(M-k) + \sqrt{k(M-k)\nu(M-\nu)}}{k(\nu-k)}, \quad \tau = \frac{1}{\log c}$$

ここで $k$ は成功数、$\nu$ は目標 ESS。これにより、訓練の進行に伴って自動的に温度が調整される。

---

## 実験結果

### 検証可能タスク（数学推論）

Qwen2.5-1.5B をベースに、GSM8K（多段算数）、Countdown（組合せ最適化）、DeepMath（汎用数学推論）の 3 タスクで評価。**SoftmaxGRPO は全タスクで GRPO を上回った**：

| 手法 | GSM8K | Countdown | DeepMath |
|------|-------|-----------|----------|
| Base | 23.0 | 2.0 | 30.0 |
| GRPO-Exact | 73.5 | 57.7 | 50.9 |
| **SoftmaxGRPO-Exact** | **75.8** | **58.1** | **51.8** |

さらに、文字列類似度（ROUGE/BLEU）のみの **弱報酬（SoftmaxGRPO-Sim）でも GRPO-Sim を一貫して上回る**。これは特に、検証器が用意できない非検証タスクへの展開において重要な意味を持つ。

![図3: 主要結果](/images/softmaxgrpo-softmax-advantage-group-estimation/fig3.png)

### 非検証タスク：Poetry で +13.4pt の衝撃

非検証タスクでは SoftmaxGRPO の真価が発揮される。Poetry（詩の創作）タスクでは、文字列類似度のみの弱い報酬信号にもかかわらず、**35.0% → 68.0%（+13.4pt）と GRPO-Sim の 54.6% を大きく引き離した**。

| 手法 | Poetry | MeetingBank | AlpacaEval 2.0 | MMLU | GPQA |
|------|--------|-------------|----------------|------|------|
| GRPO-Sim | 54.6 | 62 | 2.24 | 62.2 | 23.8 |
| OPD | 42.6 | 42 | 2.41 | 64.1 | 25.3 |
| **SoftmaxGRPO** | **68.0** | **70** | **2.50** | **65.2** | **27.1** |

AlpacaEval 2.0（2.50）、MMLU（65.2）、GPQA（27.1）のすべてで先行手法を上回る。

### 勾配配分の可視化：本当に「簡単な問題」から予算が移動している

![図2: 勾配予算の配分](/images/softmaxgrpo-softmax-advantage-group-estimation/fig2.png)

**図2** が示すように、GSM8K において GRPO は勾配予算の **36.4%** を $p\geq0.9$ のプロンプトに割り当てているのに対し、SoftmaxGRPO はわずか **10.0%**。代わりに $p\in[0.2,0.9)$ の中難度プロンプトに **82.7%** を割り当てている（GRPO は 58.9%）。

これは SoftmaxGRPO の「勾配予算の再配分」仮説を直接的に裏付ける証拠である。

### 温度 $\tau$ の消融実験

![図4: 温度消融と理論フレームワーク](/images/softmaxgrpo-softmax-advantage-group-estimation/fig4.png)

GSM8K での温度スイープの結果、$\tau \leq 0.5$ の低温領域で Pass@1 が 74.4–75.8% と安定する一方、$\tau \geq 1.0$ では急激に性能が劣化する（例：$\tau=10.0$ で $M=4$ が 31.7%）。Countdown でも同様の傾向が確認された。

推奨設定：報酬を $[0,1]$ に正規化した上で $\tau \in [0.1, 0.3]$ が出発点として信頼できる。

### モデルサイズ間の汎化

Qwen2.5-3B でも評価され、SoftmaxGRPO の優位性は維持された：

| モデル | GSM8K (GRPO→SoftmaxGRPO) | Countdown (GRPO→SoftmaxGRPO) |
|--------|--------------------------|------------------------------|
| 1.5B | 73.5 → **75.8** (+2.3) | 57.7 → **58.1** (+0.4) |
| 3B | 80.2 → **82.3** (+2.1) | 50.9 → **60.4** (+9.5) |

3B の Countdown で +9.5pt と顕著な改善。モデルサイズが大きくなっても SoftmaxGRPO の優位性は一貫している。

---

## 考察：なぜSoftmaxGRPOは「ただの正規化変更」以上の価値があるのか

### 1. 統一的目的族としての理論的意義

SoftmaxGRPO は単なるヒューリスティックな改善ではない。$\tau$ を変化させることで、REINFORCE（$\tau\to\infty$）から MaxRL（$\tau\to 0$）、さらに $M\to\infty$ で ML（$1/p$ 重み）へと連続的に補間する **統一的目的族** を提供する。これは、既存の RLVR 目的関数を一つの枠組みで理解し直す理論的道具としても価値が高い。

### 2. 「存在しない」ことの証明の価値

有限 $M$ での一般スカラー報酬に対するポテンシャル関数の非存在証明は、理論的限界を明確にしたという点で重要だ。この結果は「二値報酬の特殊性」を浮き彫りにし、なぜ検証可能タスクでの RL がここまで成功しているのかについての間接的な説明にもなっている。

### 3. 実用性：ゼロコストの置き換え

SoftmaxGRPO は既存の GRPO 訓練パイプラインに **コード一行の変更で導入できる**。追加の推論コスト、モデルパラメータ、外部報酬モデルは一切不要。これは産業応用の観点から極めて重要な特性だ。

### 4. 制限事項と今後の課題

- 有限 $M$ の厳密理論は二値報酬に限定される
- 主な評価が 1.5B モデルに留まっている（3B の追加評価はあるが、7B+ の実験は未実施）
- 非検証タスクの評価は重複報酬と LLM-as-judge に依存
- PPO クリッピングと KL 正則化下での理論的保証は未解決

---

## 関連研究

**GRPO 改良系統**：DAPO（動的クリッピング）、Dr.GRPO（報酬正規化の改善）、CISPO（クリッピング範囲の適応）、DPPO（Dual-clip PPO）など、いずれも z-score 正規化の症状に対処するもので、根本的な目的関数の置き換えではない。

**指数報酬重み付け**：RAML（Reward Augmented Maximum Likelihood）、MPO（Maximum a Posteriori Policy Optimization）、OCD（Optimal Completion Distillation）などの先行研究があるが、グループベースの RLVR 設定での系統的解析は行われていなかった。

**プロンプト難易度を考慮したカリキュラム学習**：難易度ベースのサンプリング戦略とは異なり、SoftmaxGRPO は目的関数自体に難易度適応を埋め込む。

---

## まとめ

SoftmaxGRPO は GRPO の長年の問題——z-score 正規化によるプロンプト難易度バイアス——を、**温度スケール softmax アドバンテージという単純かつ理論的に裏付けられた一行変更**で解決する。

理論面では、二値報酬での厳密な有限グループ目的関数、MaxRL 極限との接続、log-MGF 最適化としての大グループ解釈、そして一般スカラー報酬での有限グループ目的関数の非存在証明——と、驚くほど豊かな理論体系を構築している。

実験面では、検証可能タスク・非検証タスクの両方で GRPO を一貫して上回り、勾配配分の可視化によって理論的予測の正しさを実証した。Poetry で +13.4pt という結果は、弱報酬下での RLVR の可能性を大きく広げる。

COLM 2026 に採録された本手法は、既存の GRPO パイプラインに即座に導入可能な「明日から使える改善」でありながら、RLVR の目的関数設計に新しい理論的視点を提供する重要な仕事である。

---

## 参考

- Jefferson Hernandez, Jaywon Koo, Zilin Xiao, Chen Wei, Vicente Ordonez. "SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation." arXiv:2608.09271, 2026. (COLM 2026)
- Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024. (GRPO 原論文)
- Guo et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. (DPO)
- Norouzi et al. "Reward Augmented Maximum Likelihood for Neural Structured Prediction." NeurIPS 2016. (RAML)
- Abdolmaleki et al. "Maximum a Posteriori Policy Optimisation." ICLR 2018. (MPO)
