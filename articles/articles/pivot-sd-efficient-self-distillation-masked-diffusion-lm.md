---
title: "少数の決定が全体を決める──Pivot-SDがDiffusion LMの高影響Commitmentを見つけ出す仕組み"
emoji: "🎯"
type: "tech"
topics: ["機械学習", "NLP", "拡散モデル", "LLM", "自己蒸留", "EMNLP2026"]
published: false
---

## TL;DR

Masked Diffusion Language Model（dLM）は並列生成の利点を持つが、従来の後訓練手法は「どのtokenが回答を決定づけたか」という拡散モデル特有の情報を無視していた。Pivot-SDは、denoising過程で残りのマスク位置の不確実性を最も削減するcommitment（pivot）を情報利得で選び出し、成功軌道のpivotをcross-entropyで強化、失敗軌道のpivotをtargeted unlikelihoodで抑制する。200問・軌道4本という極小データで、LLaDA-8B-Instructをfull-sequence SFTや5,000ステップのonline RLを上回る性能に引き上げた。監視tokenは256中たった10個（3.9%）で、wall-clock時間もRLの1/8以下。

**論文**: [Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2610.03665) (arXiv:2610.03665, EMNLP 2026 Main Oral)
**著者**: Seo Hyun Kim, Sunwoo Hong, Younwoo Choi, Chen-Hao Chao, Se-Young Yun, Rahul G. Krishnan (KAIST AI / University of Toronto & Vector Institute)

---

## 背景

### 拡散言語モデルの台頭と独自の課題

Transformer以降、自然言語生成の主流は左から右へとtokenを並べるautoregressive（AR）モデルだった。しかし近年、拡散モデルの離散版であるMasked Diffusion Language Model（dLM）が、並列生成と複雑な推論タスクへの適用可能性で注目を集めている。LLaDA（8B）やDream（7B）といったインストラクションモデルが登場し、ARモデルに対する実用的な代替案としての地位を固めつつある。

dLMの生成は「完全にマスクされた状態から始まり、徐々にtokenを開いていく」というプロセスを経る。confidence-based samplerのもとでは、モデルが最も確信を持てる位置から順に埋めていくため、生成順序は固定されていない。この特性がdLMに独自のcredit-assignment問題をもたらす。

### なぜ「どのtokenを訓練するか」が重要か

大規模言語モデルの推論能力向上には、SFTやRLによる後訓練が不可欠だ。ここで重要なのは、**すべての出力tokenが訓練にとって同等に重要というわけではない**という事実である。ARモデルでも、一部の「決定的なtoken」が後続の大半を拘束するという研究がある（Wang et al., 2025b; Lin et al., 2025）。

dLMではこの現象がさらに顕著になる。denoisingの初期段階でcommitされた数個のtokenが、残りのマスク位置の予測分布全体を大きく変え、最終回答の骨格を決定してしまう。対照的に、多くのtokenは周辺の文脈が既に確定した後に埋められる「自明な埋め合わせ」に過ぎない。

### 既存手法の盲点

既存のdLM後訓練は大きく二つに分かれるが、いずれもこのcommitmentの非対称性を無視している。

一つ目は、ARモデルから借用したアプローチで、最終的な完成テキストに対してSFTを行うものだ（Zhao et al., 2025; Wang et al., 2025a）。この場合、denoisingの最初期に「大局を決めた」tokenと、最後に埋められた周辺tokenが同じ重みで訓練される。失敗軌道では、正しい中間過程までが誤りとともに一括ペナルティを受ける。

二つ目は、denoisingの中間ステップに介入するonline RLである（Chen et al., 2025; Tang et al., 2026）。しかしこれらは通常、一ステップ全体に報酬を割り当てるか、policy gradientをサンプリングされたステップに適用するに留まり、**個別のcommitmentが回答に与えた影響**までは追っていない。

実際、論文の分析（Fig. 1）によれば、dLM軌道のマスク位置予測エントロピーは、わずかなステップで急激に低下し、その他のステップではほとんど変化しない。このエントロピー急降下を引き起こすcommitmentこそが、回答を形作る「pivot」なのである。

![dLMのDenoising過程とエントロピー変化](/images/pivot-sd-efficient-self-distillation-masked-diffusion-lm/fig1.png)
*図1: (a) dLMは完全マスク状態から開始し、confidence-based samplerで順次tokenをcommitしていく。Pivotステップでは大局が決まる。(b) マスク位置の平均予測エントロピーは、Pivotステップで急激に低下する。*

---

## 方法詳解

### Pivot-SDの設計思想

Pivot-SD（Pivot Self-Distillation）は、凍結されたベースモデルから一度だけ軌道をサンプリングし、その中から高影響のcommitment（pivot）を選び出して局所的に蒸留するオフライン手法である。成功軌道からは「このcommitmentをもっとやれ」、失敗軌道からは「このcommitmentを減らせ」という方向性で更新する。

核心的な洞察は三つある。

第一に、**dLM軌道はcommitmentの履歴を記録している**。ARモデルと異なり、dLMでは「どのtokenがどのステップ・どの位置でcommitされたか」が軌道に直接残る。これはAR生成には存在しない情報だ。

第二に、**不確実性削減量でcommitmentの影響力を測れる**。freezeモデルの予測分布から、commit前後の残りマスク位置のエントロピー変化を計算すれば、そのcommitmentが「残りをどれだけ確定させたか」を定量化できる。

第三に、**失敗軌道からも局所的な負の学習信号を抽出できる**。pivot tokenだけをunlikelihoodで抑制すれば、失敗軌道中の正しい部分まで連座でペナルティを受けることを防げる。

### Pivot選択：Information Gain

Pivot-SDでは、各denoisingステップ$t$における情報利得スコア$g(t)$を定義する。

ステップ$t$でまだマスクされた位置の集合を$\mathcal{A}_t$、本ステップでcommitされる位置の集合を$\mathcal{U}_t$とする。commit前後で、**まだマスクされた残り位置**における予測エントロピーの変化を測る。

$$
H_{\mathrm{pre}}(t) = \sum_{i \in \mathcal{A}_t \setminus \mathcal{U}_t} h_{\theta_0}(i \mid M_t)
$$

$$
H_{\mathrm{post}}(t) = \sum_{i \in \mathcal{A}_t \setminus \mathcal{U}_t} h_{\theta_0}(i \mid M_t^{[\mathcal{U}_t]})
$$

ここで$M_t^{[\mathcal{U}_t]}$は、ステップ$t$のcommit後の状態を表す。重要なのは、求和から本ステップでcommitされた位置$\mathcal{U}_t$を**除外**している点だ。これにより、純粋に「残りへの影響」だけが測られる。

ステップが進むにつれて残り位置が減るため、生のエントロピー差は早期ステップに偏る。そこで残り位置数で正規化する。

$$
g(t) = \frac{H_{\mathrm{pre}}(t) - H_{\mathrm{post}}(t)}{|\mathcal{A}_t \setminus \mathcal{U}_t|}
$$

各軌道で$g(t)$が最も高い$K$ステップ（本論文では$K=10$）を選び、それらのステップでcommitされたtokenをpivotとする。このスコア計算に追加のforward passは不要で、samplerが既に実行した推論結果から直接算出できる。

![Pivot選択のメカニズム](/images/pivot-sd-efficient-self-distillation-masked-diffusion-lm/fig2.png)
*図2: Information Gainの計算プロセス。Pre-commitment状態とPost-commitment状態で、残りマスク位置のエントロピーを比較し、平均的な不確実性削減量としてスコア化する。*

### 訓練目標

各軌道はverifier（数学はexact-answer matching、コードはunit-test execution）によって成功・失敗に分類される。軌道の結果スコアを$s(\tau) \in \{+1, -1\}$とする。

各pivotは訓練tuple $z = (M_t, p, y_p, s(\tau))$に変換される。$M_t$はcommitmentが発生した部分マスク状態、$p$は位置、$y_p$はcommitされたtokenである。

**成功軌道（$s(\tau) = +1$）**では、モデルに対してpivot token $y_p$を状態$M_t$下でより高く予測させるよう、cross-entropy lossを適用する。

$$
\ell^{+}_{p}(\theta; M_t, y_p) = -\log P_{\theta}(y_p \mid M_t)
$$

**失敗軌道（$s(\tau) = -1$）**では、pivot token $y_p$の予測確率を下げるよう、token-level unlikelihoodを適用する。

$$
\ell^{-}_{p}(\theta; M_t, y_p) = -\log(1 - P_{\theta}(y_p \mid M_t))
$$

ここで$P_{\theta}$は数値安定性のためclippingされる。

最終的なpivot lossは以下の通り。

$$
L_{\mathrm{pivot}}(z) = \mathbf{1}[s(\tau)=+1] \, \ell^{+}_{p} + \lambda_{\mathrm{neg}} \, \mathbf{1}[s(\tau)=-1] \, \ell^{-}_{p}
$$

負のスケーリング係数$\lambda_{\mathrm{neg}}$は、オフラインデータセット内の正負軌道比に基づいて自動設定される。例えばGSM8Kでは正負比が約5:1なので$\lambda_{\mathrm{neg}} = 5$とし、両ブランチの寄与を釣り合わせる。コードタスクのみ分布シフトが大きいため、小規模なheld-outセットで対数スケールで探索し$\lambda_{\mathrm{neg}} = 0.02$とした（これが本文唯一のハイパーパラメータ探索）。

### 極めて疎な監督

Pivot-SDの監督の疎さが際立つ。最大256tokenの生成バジェットに対し、各軌道で监督されるtokenは$K=10$個だけ、すなわち**3.9%**に過ぎない。残り96%以上のtokenには一切lossを与えない。これはfull-sequence SFTがすべてのマスク位置に均等にCEをかけるのと対照的である。

---

## 実験結果

### 主実験：4ベンチマークでの優位性

LLaDA-8B-Instructをバックボーンに、数学（MATH500, GSM8K）とコード（HumanEval+, MBPP+）の4ベンチマークで評価した。訓練データは各タスクから200問、各問4本の軌道（計800軌道）のみ。オフライン手法はすべて1,000 optimizer steps、バッチサイズ8で訓練した。

| Method | MATH | GSM8K | HumanEval+ | MBPP+ |
|:---|---:|---:|---:|---:|
| Base | 31.40 | 75.13 | 32.32 | 43.12 |
| SFT-GT | 35.47 | 76.94 | 33.94 | 46.32 |
| diffu-GRPO (extended, 5k steps) | 35.20 | 77.63 | 34.15 | 42.59 |
| wd1++ (extended, 5k steps) | 35.40 | 76.65 | 34.15 | 44.97 |
| **Pivot-SD (ours)** | **37.47** | **79.51** | **40.43** | **47.12** |

*表1: LLaDA-8B-Instructの推論性能比較（256-token generation）。‡印の方法は3回の完全再ランダム化の平均。*

Pivot-SDは**すべてのベンチマークで最高平均精度**を達成した。特筆すべきは、5,000ステップのextended RL実行をも上回った点である。MATHではSFT-GTを2.0ポイント、GSM8Kではextended diffu-GRPOを1.9ポイント、HumanEval+では最も近いSFT-SDを4.7ポイント上回った。MBPP+ではSFT-GTと0.8ポイント差でトップタイ。

![実験結果の比較](/images/pivot-sd-efficient-self-distillation-masked-diffusion-lm/fig3.png)
*図3: LLaDA-8B-Instructにおける各手法の推論性能比較。Pivot-SDは4ベンチマークすべてで最高または同等の性能を達成。*

### 計算効率

wall-clock時間の比較では、Pivot-SDの優位性がさらに際立つ。

| Method | Steps | Total Time (h) |
|:---|---:|---:|
| diffu-GRPO (extended) | 5,000 | 22.9 |
| wd1++ (extended) | 5,000 | 17.3 |
| diffu-GRPO (budget-matched) | 1,000 | 5.4 |
| wd1++ (budget-matched) | 1,000 | 3.6 |
| **Pivot-SD (ours)** | **1,000** | **2.8** |

*表2: 2x NVIDIA L40でのwall-clock訓練時間。Pivot-SDはextended RLの約1/8、budget-matched RLよりも1.3〜1.9倍高速。*

Pivot-SDの総算力（PFLOPs）はbudget-matched RLの約1/1.65に抑えられた。訓練自体はわずか0.3時間で、軌道生成の2.5時間が支配的だ。軌道生成はオフラインかつ一度きりなので、複数デバイスやプロンプト間で並列化できる。

![計算効率の比較](/images/pivot-sd-efficient-self-distillation-masked-diffusion-lm/fig4.png)
*図4: (a) wall-clock訓練時間の比較。Pivot-SDは最も高速。(b) 1軌道あたりの監督token数は256中10個（3.9%）に過ぎない。*

### バックボーン汎化

第二のdLMバックボーンであるDream-7B-Instructでも同様の傾向が確認された。Pivot-SDは4ベンチマークすべてでBaseおよび両SFT基線を上回り、特にMATH（42.20）とGSM8K（81.88）で大きな改善を示した。

| Method | MATH | GSM8K | HumanEval+ | MBPP+ |
|:---|---:|---:|---:|---:|
| Base | 37.97 | 79.55 | 54.27 | 60.58 |
| SFT-GT | 35.40 | 61.49 | 54.27 | 59.26 |
| **Pivot-SD** | **42.20** | **81.88** | **58.54** | **61.64** |

*表3: Dream-7B-Instructにおける汎化実験。*

### ドメイン外（OOD）性能

訓練ドメインと評価ドメインを入れ替えたOOD実験では、Pivot-SDが**全ドメインのマイクロ平均で最高**（48.61）を記録した。特にGSM8Kで訓練した場合のOOD平均（39.82）が、どの基線よりも高かった。

### 消融実験：何がPivot-SDを機能させるか

様々な消融変体との比較で、Pivot-SDの各設計選択の重要性が検証された。

| Method | State選択 | Target | Pos. | Neg. | MATH | GSM8K |
|:---|:---|:---|:---|:---|---:|---:|
| Pos+Neg All-token | IG | All | CE | State-UL | 35.00 | 74.44 |
| Random-Step Pivot | Random | Pivot | CE | Pivot-UL | 29.93 | 76.51 |
| Random-Token Pivot | IG | Random | CE | Pivot-UL | 34.87 | 76.47 |
| Entropy pivot | Entropy | Pivot | CE | - | 32.27 | 74.75 |
| **Pivot-SD** | **IG** | **Pivot** | **CE** | **Pivot-UL** | **37.47** | **79.51** |

*表4: Pivot選択とcredit assignmentの消融。IG=Information Gain、All=状態内すべてのマスクtoken、Pivot=選ばれたpivot tokenのみ。*

**Pivot-localなcredit assignmentが重要**：失敗軌道にULを適用する場合、lossをpivot tokenだけに絞る（Pivot-SD）と、選択された状態内のすべてのマスクtokenに適用する（Pos+Neg All-token）より、MATHで2.5ポイント、HumanEval+で3.6ポイント高い。全状態ULでは、失敗軌道中の本来正しいtokenまで抑制されてしまうためである。

**Pivot選択基準が鍵**：Random-Step Pivot（ランダムに選んだステップのpivotを監督）はMATHで29.93と、未訓練Base（31.40）を下回った。Random-Token Pivot（IGで選んだステップからランダムにtokenを選ぶ）もPivot-SDに劣る。つまり、**監督の疎さそのものではなく、どのcommitmentを選ぶかが性能を分ける**。

**UL導入で選択基準の差が拡大**：正の更新のみ（CE）の場合、Entropy pivotとIG pivotの差は2ポイント以内に留まる。しかし負の更新（UL）を加えると、Random-StepではMATHが7.5ポイント、Random-Tokenでも2.6ポイント低下する。ULは影響力の高いtokenにのみ適用すべきであり、誤った位置にULを当てると有害となる。

---

## 考察

### なぜ「失敗から学ぶ」がここで機能するのか

失敗軌道からの学習はRLでは一般的だが、SFTベースの手法では難しい。Pivot-SDがこれを実現できたのは、**失敗の「元凶」を局所的に特定**できたためである。

従来のrejection samplingやfull-sequence ULでは、失敗軌道全体が廃棄または一括ペナルティを受ける。しかし複雑な推論タスクでは、失敗軌道の多くの部分は構文的・論理的に正しい。Pivot-SDは「この軌道が失敗したのは、この特定のpivot tokenが悪かったからだ」と特定し、それ以外の正しい部分はそのまま残す。

この設計は、過程報酬モデル（PRM）の思想と通じるものがあるが、違いは監督単位にある。PRMはテキスト上の推論ステップにスコアをつけるが、Pivot-SDはdLM軌道内のcommitmentイベント$(t, p, y_p)$に対して、それが「残りの不確実性をどれだけ削減したか」という拡散モデル特有の信号で重みをつける。

### データ効率の驚異

200問・軌道4本というデータ規模は、現代のLLM後訓練の文脈で考えると極めて小さい。それでもPivot-SDは十倍のデータ・五倍のステップ数を使うRLを上回る。これは、pivot選択が「どの部分を学ぶべきか」という情報を追加していると解釈できる。

1,000ステップの訓練時間が0.3時間（18分）というのも注目に値する。研究開発サイクルの短縮に大きく寄与する。

### 限界と今後の方向

論文自身でもいくつかの限界を認めている。

第一に、$K$や$\lambda_{\mathrm{neg}}$といったハイパーパラメータは手動設定のまま残っている。ステップごとのエントロピー崩壊パターンに基づく適応的な規則が望ましい。

第二に、現状の公式は1つのdenoising block内で完結している。長文脈推論では、block境界を跨いだstep-conditioned credit assignmentが必要になる。

第三に、実験は8B〜7B規模のdLMに限定されている。より大規模なdLMへの適用は自然な次のステップだ。

---

## 関連研究

### Diffusion Language Modelの後訓練

dLMの推論能力向上を目指した最近の研究には、d1（masked SFT + diffu-GRPO）、DCoLT（中間逆拡散步を潜在思考アクションとしてRLで最適化）、d2（SFTなしのpolicy-gradient）、GDPO（variance-reduced ELBO-based最適化）などがある。これらはすべてonline RLに依存する。Pivot-SDは凍結モデルからのオフライン軌道で同等以上の性能を達成し、RL特有の不安定性（zero-reward問題など）を回避する。

### Process SupervisionとCredit Assignment

数学推論の改善では、中間ステップへの監督が有効であることが知られている（Lightman et al., 2024; Wang et al., 2024）。Pivot-SDも同じく細粒度のcredit assignmentを目指すが、監督単位が異なる。これらの手法はテキスト上の「推論ステップ」にスコアをつけるのに対し、Pivot-SDはdLM軌道内のcommitmentイベント$(t, p, y_p)$に対して、部分マスク状態$M_t$下での影響力で重みづける。これはdLMがtokenを非順次的にcommitする特性に合わせた設計である。

### Token Importance

AR方向では、policy-gradient更新を高エントロピーtokenに限定する研究（Wang et al., 2025b）や、誤り軌道のcritical tokenをペナルティする研究（Lin et al., 2025）がある。dLM方向ではGIFT（Xu et al., 2025）がエントロピーに基づく重要性重みでSFTを改善した。これらは最終系列の位置にスコアをつける。Pivot-SDは「commitmentが発生したステップ」でスコアをつけ、それが「まだマスクされた位置への影響」に基づく。これは最終系列には記録されていない情報を利用している。

---

## まとめ

Pivot-SDは、Masked Diffusion Language Modelの後訓練において、「どのtokenを学ぶか」という問いに対して、dLM特有の構造を活かした回答を提示した。

**核心の貢献**は三つ。

1. **dLM固有の信号の同定**：denoisingステップが残りマスク位置の不確実性をどれだけ削減したかを定量化するinformation gainスコアを定義し、これをpivot selectionに応用した。

2. **局所的なオフライン自己蒸留**：凍結モデルからサンプリングした軌道のpivotを、commitment発生時の部分マスク状態でリプレイして訓練。成功軌道ではCEで強化、失敗軌道ではtoken-level ULで抑制し、周囲の正しい部分は保護する。

3. **極めて高いデータ・計算効率**：200問・4軌道・1,000ステップという最小限のリソースで、full-sequence SFTや5,000ステップのonline RLを上回る。監督tokenはわずか3.9%、wall-clock時間はRLの1/8以下。

Masked Diffusion LMはARモデルとは根本的に異なる生成プロセスを持つ。Pivot-SDは、その違いを「制約」ではなく「追加の教師信号」として活用した好例である。拡散モデルの並列性という利点を、推論性能向上にも結びつける道を開いたと言えるだろう。

---

## 参考

- **Pivot-SD** (arXiv:2610.03665) — Seo Hyun Kim et al., EMNLP 2026 Main Oral
- **LLaDA** — Nie et al., 2025. Large Language Diffusion with Masking
- **Dream** — Ye et al., 2025. Diffusion Reasoning Model
- **diffu-GRPO** — Zhao et al., 2025. d1: Scaling Diffusion Language Models
- **DCoLT** — Huang et al., 2025. Diffusion Chain-of-Thought
- **d2** — Wang et al., 2025a. Diffusion for Math Reasoning
- **GDPO** — Rojas et al., 2026. Guidance-Diffusion Policy Optimization
- **GIFT** — Xu et al., 2025. Guided Importance Fine-Tuning
- **Process Reward Model** — Lightman et al., 2024. Let's Verify Step by Step
- **Math-Shepherd** — Wang et al., 2024. Verification and Reinforcement
