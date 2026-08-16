---
title: "LOPD：特権コンテキストをエンドツーエンド学習する自己蒸留"
emoji: "🎯"
type: "tech"
topics: ["LLM", "強化学習", "自己蒸留", "エージェント"]
published: false
---

## TL;DR

On-Policy Self-Distillation（OPSD）は、エージェントが自身のrollout上で教師から密なtokenレベル監督を得られる強力な枠組みだが、既存手法はすべて設計者が手作業で指定した「特権コンテキスト」（正解、フィードバック、スキル等）に依存している。

**LOPD**はこの前提を覆す。教師の特権コンテキスト自体を**エンドツーエンドで学習可能**にし、経験ベースの検索→QFormer圧縮→連続潜在token生成というパイプラインで、コンテキストの内容を学習プロセスに委ねる。特権マージン制約とアンカリング項で崩壊を防ぎ、3モデル・7ベンチマーク全設定でRLVR・OPSD系手法を上回る。しかもGRPO/Skill-SDの**最終性能をrollout予算30%未満で超過**する。

論文: [arXiv:2608.13040](https://arxiv.org/abs/2608.13040) (2026/08/13)
コード: [github.com/bingreeky/LOPD](https://github.com/bingreeky/LOPD)

## 背景：OPSDの「特権コンテキスト問題」

OPSDの核心はシンプルだ。学生がrollout τを生成し、固定された教師が同じprefix上でtoken分布を出力し、そのKL乖離で学生に密な勾配を与える。問題は**教師に何を見せるか**。

既存手法の特権コンテキストはいずれも**設計者が先験的に決めた形式**に固定されている：

| 手法 | 特権コンテキスト |
|------|----------------|
| OPSD | 成功軌跡全文（システムメッセージ） |
| SDPO | 同rollout内の成功自己対 |
| Skill-SD | 構造化スキル要約 |
| CSCR | 反事実感度に基づく重み |
| DASH | 適応的伝播ゲート |

これらは「どの情報が学生にとって有用か」を先験的に決めており、学習可能性と拡張性の上限を人間の設計判断に依存してしまう。まさに論文が指摘するように：

> *「どの教師が十分強いか」から「自教師にどの特権コンテキストを与えるべきか」へと問題は移行したが、その答えを依然として設計者が決めている」*

## 方法：LOPDの仕組み

### 全体アーキテクチャ

![LOPD Architecture](/images/lopd-latent-on-policy-self-distillation/fig1_architecture.png)

LOPDのパイプラインは3段構成：

1. **検索段階**: 経験DB（成功rolloutのみ）から、Qwen3-Embedding-8Bでタスクに類似するtop-J件を検索
2. **Composer（潜在コンテキスト生成）**: 凍結バックボーン+LoRAで各経験をエンコードし、QFormerスタイルの交差注意で学習クエリによりK個の連続潜在tokenに圧縮。合計J×K個（デフォルト96個）の潜在tokenを生成
3. **蒸留段階**: 学生がrolloutを生成し、潜在tokenで条件付けられた教師がreverse KL（Top-M + Tail）で密な監督を提供

**推理時は学生のみをデプロイ**。経験DB、検索器、Composerは一切不要になる。

### Composerの具体設計

- **エンコーダ**: 凍結バックボーンにLoRA（rank=8, α=16）を7つの線形投影に付与
- **圧縮器**: QFormer風パーセプタ。8層共有パラメータの交差注意、FFN倍率4
- **学習クエリ**: $\mathcal{N}(0, 1/\sqrt{d})$から初期化。トレーニング前に成功軌跡上でバックボーン凍結のnext-token NLLでコールドスタート

### 特権マージン制約：崩壊防止メカニズム

ここがLOPDで最も重要な設計判断だ。単に$\phi$を蒸馏損失で最適化するだけでは、Composerは「教師の分布を学生に近づける」ことで損失をゼロにできてしまう（つまり$\pi^T \to \pi^S$に崩壊し、特権コンテキストは無情報化する）。

![Privilege Margin](/images/lopd-latent-on-policy-self-distillation/fig2_privilege_margin.png)

LOPDは3つの仕組みでこれを防ぐ：

**① 特権マージン $\Delta(\phi) \geq m$**

各tokenの教師-学生対数確率差（特権）を定義：

$$\delta_{t,n}(\phi) = \log\pi^T(a_{t,n} | s_t, c_\phi, a_{t,<n}) - \text{sg}[\log\pi^S(a_{t,n} | s_t, a_{t,<n})]$$

これを軌跡レベルの検証結果 $A(\tau) = 2r(\tau) - 1 \in [-1, 1]$ で加重し、双対変数$\beta$でラグランジュ制約として強制：

$$\Delta(\phi) = \mathbb{E}\left[\frac{\sum \omega_{t,n} \cdot A(\tau) \cdot \delta_{t,n}(\phi)}{\sum \omega_{t,n}}\right] \geq m$$

$\beta$は $\beta \leftarrow [\beta + \eta_\beta(m - \Delta)]_+$ で更新。Composerが崩壊して$\pi^T \approx \pi^S$になれば$\delta \to 0$となり$\Delta < m$でペナルティが発動する。

**② アンカリング項**

潜在空間での初期点$\phi_0$からのドリフトを罰する：

$$\lambda \|c_\phi - \text{sg}[c_{\phi_0}]\|_2^2$$

追加のforward passなしで計算可能。

**③ コールドスタート初期化**

初期$\phi_0$を成功軌跡上のnext-token NLLで学習済みにする。これによりcomposerは最初から有情報なコンテキストを生成でき、崩壊からの出発を防ぐ。

### 完全目的関数

$$\min_{\theta, \phi} \max_{\beta \geq 0} \; \mathcal{L}_{\text{distill}}(\theta, \phi) + \beta(m - \Delta(\phi)) + \lambda\|c_\phi - \text{sg}[c_{\phi_0}]\|_2^2$$

- $\theta$: 学生パラメータ（全パラメータ訓練可能）
- $\phi$: Composerパラメータ（LoRA + 圧縮器のみ訓練）
- $\bar{\theta}$: 教師バックボーン（固定。勾配は$\phi$のみ流れる）

## 実験結果

### ツール使用タスク

![Performance Comparison](/images/lopd-latent-on-policy-self-distillation/fig3_performance.png)

Qwen3-4B/8BでEnvScaler（200 held-out）、BFCL-v3、ACEBenchを評価。LOPDは**全10の骨格-ベンチマーク組み合わせで最高スコア**を記録。

ハイライト：
- **EnvScaler 8B**: 66.4（Skill-SDの60.2から+6.2、GRPOの57.3から+9.1）
- **ACEBench 8B**: 62.7（GRPOの58.0から+4.7）
- **BFCL-v3 Avg 8B**: 29.88（全手法最高）

注目すべきは**ベースラインの不稳定性**だ。SDPOはQwen3-4BのBFCL-v3でVanillaの22.88から**15.75に悪化**（-7.13）。OPSDはLiveCodeBenchでVanilla**未満**に低下。同じコンテキスト形式でも骨格・タスクによって効果が反転する現象が明確に確認された。LOPDは**全設定でVanillaを一貫して上回る**。

### コード生成タスク

Qwen3-4BとOlmo3-7BでLiveCodeBench v5/v6、HumanEval+、MBPP+を評価。

- **Olmo3 LiveCodeBench Avg**: 50.98（GRPOの48.29から+2.69）
- **Qwen3-4B EvalPlus Avg**: 81.36（SDFTの80.07から+1.29、GRPOの79.34から+2.02）

### サンプル効率

![Efficiency & Sensitivity](/images/lopd-latent-on-policy-self-distillation/fig4_efficiency_sensitivity.png)

同じ1,600世代の生成予算でEnvScalerの平均報酬を追跡。LOPDは約320世代（**予算の30%未満**）でGRPOの最終性能0.611を超過し、以降0.63-0.64の狭い帯で推移。GRPOとSkill-SDは1,600世代経っても0.61/0.59に到達しない。

### 消融実験：学習可能コンテキストは必要条件

| 設定 | EnvScaler報酬 |
|------|--------------|
| Composer凍結（$\phi_0$） | 0.573 |
| **マージンなし（m=0）** | **0.551（悪化！）** |
| 弱マージン（m≤0.01） | 改善せず |
| m=0.02 | 凍結基線を超過 |
| **m=0.05（デフォルト）** | **0.637** |

マージンなしでは**凍結composerより性能が悪化**する。つまり、無制約な蒸馏勾配はcomposerを**退化させる**。特権マージン制約は性能向上の**必要条件**。

### 感度分析

- **潜在token容量K**: K=8/16では~0.56に留まるが、K=32で**0.637に跳躍**（容量閾値の存在）。K=64/128でも一貫した改善なし
- **検索数n_ret**: 1→3で0.605→0.637と改善。5以上でも単調改善せず

### 行動内化の証拠

| 指標 | Vanilla | Base+Composer | LOPD |
|------|---------|---------------|------|
| 首ステップ長 | 9,937 | 6,695 | **6,210**（-37.5%） |
| ステップあたりツール呼出 | 3.50 | 1.21 | **1.11** |
| 環境ステップ数 | 11.12 | 16.31 | **17.04** |

LOPD学生は「一度に大量の投機的呼出しを出す」から「より段階的な計画を実行する」へと行動を変化させた。潜在コンテキスト条件付けされたベースモデルが最終学生と**同じ定性的パターン**を示し、LOPDが教師の手続き的指導を**内化**していることを示している。

## 考察

### 設計原理の転換

LOPDの最も深い貢献は、新しいアルゴリズムを提案したことではなく、**設計原理を転換した**ことだ。論文は最後にこう述べる：

> *「拡張可能な自己進化は、ますます精密に設計された人間記述の経験フォーマットに依存すべきではない。経験表現は、それが改善しようとする戦略のためにエンドツーエンド最適化されるべきである」*

これは「特権コンテキスト」という概念自体の再定義だ。固定されたテキスト表現ではなく、戦略の改善に最適化された**連続潜在表現**へ。

### 特権マージン制約の優雅さ

崩壊防止のアプローチが非常に優雅だ。CSCRは反事実感度でtoken重みを再計算し、DASHは適応的伝播ゲートを導入する。LOPDはシンプルに「教師が学生より有利でなければならない」という**1つの制約**で、過度な設計なしに崩壊を構造的に排除する。

ただし、$m=0.05$という閾値の設定がタスク非依存で良いかは未検証。コード生成のように報酬が密なタスクでは異なる最適値がある可能性がある。

### 冷启动依存の限界

Composerの初期化はベースモデル自身の成功rolloutに依存する。コールドスタート時点でベースモデルが有意な成功軌跡を生成できなければ、初期$\phi_0$は無情報になる可能性がある。高難度タスクや初期性能の低い小規模モデルへの適用は追加の検討が必要だろう。

## 関連研究

- **OPSD / AgentOPSD / CSCR / DASH**: 既存OPSD系手法。特権コンテキストは固定形式。LOPDはこれらの「コンテキスト内容」を学習に委ねる点で一線を画す
- **Skill-SD**: スキル要約を特権コンテキストとする手法。LOPDはスキルという構造化概念を陽に設計せず、潜在空間に任せる
- **QFormer (VLP)**: LOPDのComposerにQFormer風交差注意を採用。元はVision-Language Pre-trainingの技術
- **GRPO**: RLVRベースライン。結果報酬のみを使用し、教師なし。LOPDは30%未満の予算でGRPOを超過
- **World Model RL (WMRL)**: [arXiv:2608.12564](https://arxiv.org/abs/2608.12564)。環境実行をワールドモデルで代替し、3-4倍の加速。LOPDの潜在コンテキストは環境モデリングの側面も持つが、その目的は教師の条件付けに限定される

## まとめ

LOPDはOPSDの「特権コンテキスト問題」に根本的かつ実用的な解答を提示した。

1. **特権コンテキストをエンドツーエンド学習可能に**: 手作業の経験フォーマット設計から解放
2. **特権マージン制約で崩壊防止**: シンプルなラグランジュ制約が構造的に無情報解を排除
3. **圧倒的サンプル効率**: GRPO/Skill-SDの最終性能を30%未満のrolloutで到達
4. **一貫した性能**: 3モデル・7ベンチマーク全設定でベースラインを上回る（既存手法では不可能だった）

推理時のオーバーヘッドはゼロ（学生のみデプロイ）。Composerの追加計算はトレーニング時のみ。これはエージェント自己進化の**実用的なステップ**だ。

今後の展開として、多段階進化（composerの自己改善サイクル）、より大規模モデルへの適用、コールドスタッチ問題の解消が挙げられる。

## 参考

- Zhang, G., Lyu, J., Sun, R., Yu, X., Zhao, H., Ren, Q., & Yan, S. "Latent On-Policy Self-Distillation." arXiv:2608.13040, 2026.
- コード: [github.com/bingreeky/LOPD](https://github.com/bingreeky/LOPD)
- モデル: [Qwen3-8B-LOPD](https://huggingface.co/liunanfu1992/Qwen3-8B-LOPD), [OLMo-3-7B-Think-LOPD](https://huggingface.co/OLMo-3-7B-Think-LOPD)
