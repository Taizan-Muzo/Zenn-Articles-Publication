---
title: "DemoPSD: 逆KL重心でOPSDの特権情報漏洩を解消・SDPO超え"
emoji: "🔬"
type: "tech"
topics: ["LLM", "強化学習", "自己蒸留", "推論", "Post-Training"]
published: true
---

# DemoPSD: 教師-学生分岐に基づく選択的自己蒸留による特権情報漏洩の解消

## TL;DR

On-Policy Self-Distillation (OPSD) はLLMの推論訓練で広く採用されているが、教師が「正解情報 (特権情報)」を条件付けとして利用するため、学習後に学生がテスト時には使えないショートカットを暗記してしまう**特権情報漏洩 (Privileged Information Leakage)** が致命的な弱点だった。

**DemoPSD** (Li et al., 2026) は、教師と学生の予測分布間の**Jensen-Shannon 分岐 (JSD)** をトークンごとに測定し、分岐が大きい位置ほど教師の指導を弱める **reverse-KL 重心ターゲット** を構築する手法だ。分岐が小さい位置では従来のSDPOと同等の密監督を維持しつつ、分岐が大きい位置では特権情報に過度に依存した指導を選択的に減衰させる。

SciKnowEval 4分野で SDPO を mean@16 +1.68、GRPO を +5.21 で凌駕。訓練エントロピーは 33-98% 高く維持され、GPQA での分布外汎化も SDPO を平均 +7.91 で上回る。

## 背景：RLVRとOPSDのジレンマ

### RLVRの信用割当ボトルネック

強化学習と検証可能報酬 (RLVR) は、LLMの推論訓練における標準パラダイムだ。GRPO などは各問題に対して複数のロールアウトをサンプリングし、結果の正解性を報酬信号とする。しかしこのアプローチには根本的な**信用割当ボトルネック**がある——ロールアウトレベルの報酬を全トークンに均等に分配するため、個々のトークンの貢献度を区別できない。

### On-Policy Self-Distillation (SDPO)

SDPO はこの問題に対する有力な解決策として注目を集めている。同じモデルが「特権情報（検証済み推論軌跡や標準答案 $y^*$）」で条件付けられた教師と、問題だけを受け取る学生を兼ねる。Qwen3 や DeepSeek-V4 など、産業界でも広く採用されている。

$$\mathcal{L}_{\text{SDPO}}(\theta) = \mathbb{E}_{x} \mathbb{E}_{\hat{y} \sim \pi_\theta} \left[\sum_{t=1}^{|\hat{y}|} \text{KL}\big(\pi_\theta(\cdot|x,\hat{y}_{<t}) \| \text{stopgrad}(\pi_\theta(\cdot|x, y^*, \hat{y}_{<t}))\big)\right]$$

![OPSDパイプラインと漏洩の現れ方](/images/demopsd-disagreement-policy-self-distillation/fig1.png)
*図1: 左) OPSDの基本パイプライン。教師は特権情報 $y^*$ を条件として受け、学生は問題のみで推論する。右) 特権情報漏洩の典型的な現れ方——初期に有益な指導が性能を押し上げるが、後期にはショートカットの暗記が汎化を損なう。*

### 特権情報漏洩：取り返しのつかない情報ギャップ

問題は、教師が条件付けする特権情報 $y^*$ が、テスト時には決して利用できないという点だ。条件付きエントロピーの不等式から、不可避の相互情報ギャップが存在する：

$$I(y_t; y^* \mid x, y_{<t}) > 0$$

このギャップは、問題と生成済み接頭辞を条件付けしても、$y^*$ が次トークンの予測に**追加の情報を提供し続ける**ことを意味する。結果として、学生モデルは「問題→正解」の直接的相関をエンコードし、本質的な推論能力の代わりに暗記に依存するようになる。

実験的に見ると、SDPO の性能は訓練初期にピークに達した後、残りの訓練フェーズで**単調に劣化**していくという致命的なパターンが確認されている。

## DemoPSD：分岐に基づく選択的教師採用

### コアアイデア：教師の指導を「賢く」使う

DemoPSD の出発点はシンプルだ：**すべての教師の指導が等しく信頼できるわけではない**。

- 教師と学生の予測が**合理的に一致**している位置 → 特権情報の影響が限定的。教師の指導は安全に採用できる
- 教師と学生の予測が**大きく分岐**している位置 → 教師の出力は特権情報に過度に影響されている。この位置での蒸留はリスクが高い

DemoPSD は各トークン位置でこの分岐を直接測定し、それに基づいて教師指導の採用度を**適応的に制御**する。

![DemoPSDのreverse-KL重心メカニズム](/images/demopsd-disagreement-policy-self-distillation/fig2.png)
*図2: DemoPSDの核心メカニズム。(a) 低JSD分岐の場合、αは小さく、ターゲットは教師に近い。(b) 高JSD分岐ではαが大きくなり、ターゲットは学生分布に寄る。(c) αはJSDの単調増加関数で、α_maxで飽和する。*

### 分岐の測定：Jensen-Shannon Divergence

各トークン位置 $t$ での分岐を、対称で有界な Jensen-Shannon Divergence (JSD) で測定する：

$$d_t = \text{JSD}(\pi_S^t \| \pi_T^t) = \frac{1}{2}\text{KL}(\pi_S^t \| m_t) + \frac{1}{2}\text{KL}(\pi_T^t \| m_t)$$

この分岐値から、漏洩減衰係数 $\alpha_t$ を計算する：

$$\alpha_t = \big(\sigma(\beta \cdot d_t) - 0.5\big) \cdot 2 \cdot \alpha_{\max}$$

ここで $\beta$ は分岐感度、$\alpha_{\max}$ は上限値。$\alpha_t = 0$ なら完全に教師に従い、$\alpha_t = \alpha_{\max}$ なら最大限教師の影響を減衰させる。

### Reverse-KL Barycenter Target

DemoPSD の真の革新はターゲット分布の設計にある。算術平均ではなく、**幾何混合（reverse-KL 重心）**を用いる：

$$\pi_{\text{target}}^{\alpha_t}(v) \propto \big(\pi_T^t(v)\big)^{1-\alpha_t} \cdot \big(\pi_S^t(v)\big)^{\alpha_t}$$

これは以下の最適化問題の解として定義される：

$$\pi_{\text{target}}^{\alpha_t} = \operatorname*{arg\,min}_{q} \left\{(1-\alpha_t)\text{KL}(q \| \pi_T^t) + \alpha_t \text{KL}(q \| \pi_S^t)\right\}$$

**幾何混合の利点**：確率の掛け算により、教師と学生の**両方が支持するトークンだけが高いターゲット品質を獲得**する。教師だけが支持し学生の確率が極めて低いトークンは自然に抑制される——まさに漏洩を減衰させる効果だ。

最終的な損失関数は：

$$\mathcal{L}_{\text{DemoPSD}} = \mathbb{E}_{x} \mathbb{E}_{\hat{y}} \left[\sum_t \text{KL}\big(\pi_\theta(\cdot|x,\hat{y}_{<t}) \| \text{stopgrad}(\pi_{\text{target}}^{\alpha_t})\big)\right]$$

勾配を見ると、DemoPSD の勾配は従来の SDPO と同じ reverse-KL score-function 形式を保ちつつ、分岐に基づく因子 $(1-\alpha_t)$ で教師誘導の対数比信号をスケーリングする：

$$\nabla_\theta \mathcal{L}_{\text{DemoPSD}} \propto (1-\alpha_t) \cdot \log \frac{\pi_\theta(\hat{y}_t|x,\hat{y}_{<t})}{\pi_\theta(\hat{y}_t|x, y^*, \hat{y}_{<t})}$$

## 理論的保証

DemoPSD は二つの重要な理論的性質を満たすことが証明されている。

### 定理1：漏洩減衰 (Leakage Attenuation)

DemoPSD の有効漏洩率は、標準 OPSD の漏洩率に対して**厳密に小さい**：

$$\mathcal{R}_{\text{leak}}^{\text{DemoPSD}} = \mathbb{E}_t\left[(1-\alpha_t)^2 \|\Delta_t\|^2\right] < \mathbb{E}_t\left[\|\Delta_t\|^2\right] = \mathcal{R}_{\text{leak}}$$

ここで $\Delta_t(v) = \log \pi_T^t(v, y^*) - \log \pi_S^t(v)$ は特権情報による対数確率偏移だ。$\alpha_t > 0$ となる位置（すなわち分岐が存在する位置）では、$(1-\alpha_t)^2 < 1$ となり、漏洩が確実に減衰する。

### 定理2：探索保持 (Exploration Preservation)

DemoPSD のターゲットは完全教師分布より**厳密により高いエントロピー**を維持する：

$$\mathcal{H}(\pi_S^t) \geq \mathcal{H}(\pi_{\text{target}}^{\alpha_t}) \geq \mathcal{H}(\pi_T^t)$$

証明の鍵は、幾何混合 $q_\gamma^t \propto (\pi_T^t)^\gamma (\pi_S^t)^{1-\gamma}$ が $\gamma$ をパラメータとする**指数族**であること。エントロピーの微分：

$$\frac{d\,\mathcal{H}(q_\gamma^t)}{d\gamma} = -\gamma\,\text{Var}_{q_\gamma^t}[\Delta_t] - \text{Cov}_{q_\gamma^t}(\Delta_t, \log \pi_S^t)$$

両項が非正であるため、$\mathcal{H}(q_\gamma^t)$ は $\gamma$ について単調減少。DemoPSD は $\gamma = 1-\alpha_t < 1$ で止まるため、完全教師 ($\gamma=1$) より高いエントロピーを維持する。

## 実験結果

### 設定

- 基礎モデル: Qwen3-4B-Instruct
- 訓練データ: SciKnowEval（4択科学問題、4分野）
- 評価: SciKnowEval (in-domain) + GPQA Extended (out-of-distribution)
- ベースライン: GRPO, SDPO
- ハードウェア: 8× H20 GPUs, FSDP

### SciKnowEval での性能

![SciKnowEval主要結果とエントロピー比較](/images/demopsd-disagreement-policy-self-distillation/fig3.png)
*図3: (a) SciKnowEval 4分野での mean@16。DemoPSD は全分野で SDPO と GRPO を上回る。(b) 訓練エントロピーの保持。DemoPSD は SDPO に対して 33-98% 高いエントロピーを維持し、エントロピーエントロピークラッシュを回避する。*

| Domain | GRPO | SDPO | DemoPSD | Δ vs SDPO |
|--------|------|------|---------|-----------|
| Biology | 33.51 | 36.88 | **39.25** | +2.37 |
| Chemistry | 65.83 | 71.70 | **72.98** | +1.28 |
| Material | 76.32 | 76.13 | **76.53** | +0.40 |
| Physics | 66.31 | 68.98 | **71.64** | +2.66 |
| **Avg** | **60.49** | **63.42** | **65.10** | **+1.68** |

DemoPSD は mean@16 で SDPO を +1.68、GRPO を +5.21 で上回る。best@16 の改善が最も大きく（+2.82 vs SDPO）、DemoPSD が維持した高い探索エントロピーがより質の高い推論パスをサンプリングで発見していることを示唆する。

### 分布外汎化：GPQA Extended

![GPQA分布外汎化とJSD分布](/images/demopsd-disagreement-policy-self-distillation/fig4.png)
*図4: (a) GPQA Chemistry での OOD 汎化。SDPO は早期に最適 OOD 性能に達した後、特権ショートカットの暗記により劣化する。DemoPSD は訓練全域で安定した汎化を維持。(b) トークン単位 JSD 分布。全分野で強い右偏り——大部分のトークンの分岐は無視できる程度で、2-5% のみが 0.25 を超える。*

| Method | Biology | Chemistry | Physics | Avg |
|--------|---------|-----------|---------|-----|
| SDPO | 57.81 | 28.62 | 52.99 | 46.47 |
| **DemoPSD** | **61.42** | **41.75** | **59.98** | **54.38** |

SDPO の Chemistry OOD 性能は 40.45 から 28.62 へと **12ポイント以上劣化**。一方 DemoPSD は訓練全域で安定し、平均 +7.91 のアドバンテージを確保。これは特権情報ショートカットがドメイン特有であるため、ドメインを越えるとショートカットが破綻する現象を DemoPSD が回避していることを示している。

### 訓練ダイナミクス

- **エントロピー**: DemoPSD は全分野で SDPO より 33-98% 高い最終エントロピーを維持。特に Material Science では SDPO のエントロピーが 0.150 にまで低下（エントロピークラッシュの兆候）する中、DemoPSD は 0.297 を維持
- **分岐の希薄性**: 平均 $\bar{\alpha_t}$ は 0.033-0.055 と小さく、大部分のトークンでは教師分布に近いターゲットが使用される。高 JSD トークンは全体のわずか 2-5%
- **β 感度**: 分岐の小さい領域（Physics, $\bar{d_t}=0.026$）では高い β が有効。分岐の大きい領域（Biology, $\bar{d_t}=0.046$）では低い β が良い。β ∈ [25, 100] の範囲で DemoPSD は常に SDPO を上回るか同等

## 考察

### 既存アプローチとの差異化

DemoPSD の最大の特徴は、**分岐そのものを直接利用**して漏洩を制御する点にある。既存の RLSD, HDPO, DASD, SRPO などはいずれも教師エントロピーやサンプル正解性といった**間接プロキシ**に依存していた。

DemoPSD のアプローチは原理的に美しい——特権情報が教師の予測をどれだけ歪めたかを、学生自身の予測との比較で**直接測定**する。プロキシではなく本質的な信号を使うことで、より正確で適応的な漏洩制御が可能になる。

### 分岐の希薄性という発見

JSD 分布の強い右偏りは重要な実験的発見だ。大部分のトークン（95%以上）では教師と学生の予測が実質的に一致しており、これらの位置では DemoPSD は SDPO とほぼ同等の密監督を提供する。漏洩リスクが高いのはわずかな位置（数値答案や解法を明らかにするステップなど）だけであり、DemoPSD は**必要最小限の介入**で最大の効果を発揮する。

### 幾何混合 vs 算術混合

ターゲットに幾何混合（対数空間補間）を選んだ点も重要だ。算術混合は分布のモードを平均化し、エントロピーの膨張を引き起こす。幾何混合はトークンの確率を掛け合わせるため、両分布が支持するトークンだけが高いターゲット品質を獲得し、モードの崩壊を回避する。

### 実用性

EMA レート 0.05 の追加計算と top-k=100 での分岐計算という軽量なオーバーヘッドで済み、既存の SDPO 実装への統合が容易。特に β が広い範囲でロバストに機能する点は、実運用面で大きな魅力だ。

## 関連研究

- **SDPO** (Hübotter et al., 2026): 自己蒸留による RL の形式化。特権情報漏洩の問題を明示的に扱わない
- **DASD** (Zhang et al., 2026): 自己蒸留信号の方向に基づく監督調整。間接プロキシを使用
- **SRPO** (Li et al., 2026): GRPO と自己蒸馏をサンプルルーティングで統合
- **AMiD** (Shin et al., 2026): α-混合補助分布の導入。概念上 reverse-KL 重心と関連
- **HDPO** (Ding, 2026): クリフプロンプトに限定した特権自己蒸留
- **OPSD** (2026-05-12記事): Post-RL 自己蒸留の圧縮効率化

## まとめ

DemoPSD は OPSD の核心的な弱点である特権情報漏洩に対し、**「教師の指導を全面的に信じるのではなく、トークンごとに分岐を測定して信頼度を調整する」** という直感的で理論的に厳密な解決策を提示した。

reverse-KL 重心ターゲットの設計は、漏洩減衰と探索保持という二つの理論的性質を同時に満たし、SciKnowEval と GPQA の両方で一貫した改善を示す。分岐の希薄性という発見は、OPSD の問題が「全体的な」ものではなく「位置的な」ものであることを明らかにし、狙いを絞った介入が十分に有効であることを証明している。

## 参考文献

1. Li, Y. et al. (2026). "DemoPSD: Disagreement-Modulated Policy Self-Distillation." arXiv:2607.02502.
2. Hübotter, J. et al. (2026). "Reinforcement Learning via Self-Distillation." (SDPO)
3. Zhang, Y. et al. (2026). "DASD: Direction-Adaptive Self-Distillation."
4. Li, Y. et al. (2026). "SRPO: Unifying GRPO and Self-Distillation."
5. Ding, Z. (2026). "HDPO: Hybrid Distillation Policy Optimization."
6. Shin, J. et al. (2026). "AMiD: Alpha-Mixture Assistant Distribution."
7. Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in LLMs." (GRPO)
8. Feng, G. et al. (2024). "SciKnowEval: A Scientific Knowledge Evaluation Benchmark."
9. Rein, D. et al. (2023). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark."
