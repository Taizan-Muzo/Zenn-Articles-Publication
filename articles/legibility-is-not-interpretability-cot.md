---
title: "可読性は解釈可能性にあらず──Chain-of-Thought推論における判定重要性と実際の重要性の比較"
emoji: "🔍"
type: "tech"
topics: ["Chain-of-Thought", "解釈可能性", "Process Reward Model", "LLM", "推論の忠実性"]
published: false
---

## TL;DR

Chain-of-Thought（CoT）推論モデルの出力する推論軌跡は、一見するとモデルがどのように答えに至ったかを窥える「可読な窓」に見える。しかし本論文は、**推論ステップのテキストが実際にどのステップが重要かを符号化しているか**という根本的な問いを提起する。著者らは推論ステップの重要性を RL の「アドバンテージ」関数として定義し、Monte Carlo rollout で推定する手法を提案。その結果、LLM 裁定者や微調整批評者は高アドバンテージ・ステップの特定において、普遍性ベースラインを上回るもののノイズ天井には遠く及ばないことを示した。とくに**正解レスポンスでは重要性の復元がほぼ不可能**であり、「可読性 ≠ 解釈可能性」という強い警告を発している。COLM 2026 採録。

## 背景

### Chain-of-Thought の「可読性」という約束

CoT 推論モデルが段階的に思考を記述する仕組みは、単に正答率を向上させるだけでなく、モデルの推論過程を人間が読み取れるという約束を含んでいる。この約束のもと、以下のような実践が広まっている。

- **LLM 裁定者**による推論エラーの診断
- **忠実性（faithfulness）** の評価
- **Process Reward Model（PRM）** によるステップレベルの報酬付与
- **生成型批評者（generative critic）** による推論品質の評価

これらの実践はすべて、推論ステップのテキストがそのステップの機能的役割に関する情報を担っているという暗黙の前提に依存している。

### 前提への疑問

しかし、この前提は本当に成り立つのか？ 著者らは競技幾何の問題における次の例を挙げる。

1. *"Alternatively, maybe I made an error in the problem setup. Let me check again."* — 正答率に**ほぼ影響なし**
2. *"Alternatively, maybe my error is in the coordinate assignment for point A."* — 記号エラーを発見し、正答率を ≈52% → ≈94% に**急上昇**

テキストのみからはこの二つのステップを意味的に区別することはほぼ不可能だが、実際の重要性は天と地ほど違う。この例は、可読性が解釈可能性を保証しないことの直観的な動機となる。

## 方法詳解

### アドバンテージによる重要性の定義

著者らは推論ステップの重要性を、強化学習の枠組みにおける**アドバンテージ関数**として定義する。

- 語彙表 $\Sigma$ 上の言語モデル $\pi$
- プロンプト $x$ に対する推論軌跡 $a_1, a_2, \ldots, a_t$
- 状態 $s_t = x \circ a_1 \circ \ldots \circ a_{t-1}$

報酬関数として二つの自然な選択を考える。

1. **正確性報酬**：$r = \mathbf{1}[\text{最終答えが正しい}]$
2. **自己整合報酬**：$r = \mathbf{1}[\text{最終答えが元軌跡の答えと一致}]$

これらのもとで、**アドバンテージ**を次のように定義する。

$$A^\pi(s_t, a_t) \triangleq Q^\pi(s_t, a_t) - V^\pi(s_t)$$

ここで $Q^\pi$ はステップ $a_t$ を取った後の期待報酬、$V^\pi$ は現状からの期待報酬である。正のアドバンテージはそのステップが正答の可能性を高めることを、負のアドバンテージは有害であることを意味する。

![アドバンテージの概念図と、類似テキストでも重要性が大きく異なる例](/images/legibility-is-not-interpretability-cot/fig1.png)

### Monte Carlo 推定

$V^\pi, Q^\pi, A^\pi$ はモデルの確率的方策のもとでの期待値であり、直接計算できない。そこで Monte Carlo rollout により推定する。

- $\hat{V}(s_t)$：$\pi(\cdot|s_t)$ から $N$ 個の補完をサンプリングし、報酬を得た割合
- $\hat{Q}(s_t, a_t)$：$\pi(\cdot|s_t \circ a_t)$ から $M$ 個の補完をサンプリングし、報酬を得た割合
- **アドバンテージ推定**：$\hat{A}(s_t, a_t) = \hat{Q}(s_t, a_t) - \hat{V}(s_t)$

本実験では $N = 50$ を使用。

### Consequential ステップの同定

ステップ $a_t$ が **consequential**（影響的）であるとは、$|A^\pi(s_t, a_t)| > \delta$（効果量 $\delta$ を超えて期待報酬を変化させる）を満たすこと。そうでない場合は **uninformative**（非情報的）と呼ぶ。

単純な逐步骤仮説検定は多重比較問題を引き起こすため、著者らは**変点検出（changepoint detection）**アプローチを採用する。

1. **PELT 分割**：価値軌跡を区分定数モデルとみなし、Pruned Exact Linear Time アルゴリズムで変点を検出
2. **ペナルティ校正**：平坦な零系列上で 1 系列あたり 2% の偽陽性率に校正
3. **Beta 事後効果量フィルタ**：検出された変点について、隣接区間平均に Beta 事後を置き、$P(|A^\pi| > \delta) \geq 0.95$（$\delta = 0.1$）を検定
4. **単一ステップ位置要件**：変点の輪郭尤度信頼区間が単一ステップに収まることを要求（約 25% の境界を破棄）

この手法の統計的検出力は、逐步骤比較の最悪ケース最小検出可能効果 ≈0.28 に対し、区間集約により ≈0.1 の跳びを検出可能。

## 実験結果

### 実験設定

- **生成モデル**：Qwen3-1.7B, 4B, 8B（thinking off）、Qwen3-1.7B（thinking on）
- **データセット**：AIME 24/25/26, AMC 23, MATH500, GSM8K（各 30 問、計 180 問）
- **Monte Carlo rollout**：$N = 50$
- **評価指標**：PR-AUC、Precision@$k$%、ノイズ天井

### 推論パターンの分析

価値軌跡を六つのパターンに分類した結果、**thinking モードとスケールの性能向上は「最初から価値が高い」レスポンスの増加にほぼ完全に吸収される**ことが判明した。「Always High」パターンが非 thinking の 24% から thinking の 61% へと増加。つまり、推論の途中で答えを見つけるのではなく、**推論開始前にすでに答えが実質的に確定しているケースが増えている**。

![Thinking vs 非 Thinking モデルの推論パターン分布](/images/legibility-is-not-interpretability-cot/fig3.png)

### Consequential ステップの性質

- 非 thinking モデルでは、正解レスポンスで「不確実性管理」「自己チェック」ステップが最も consequential
- Thinking モデルでは「能動的計算」「不確実性管理」が上位
- **Consequential ステップは希少**：ID でわずか約 1.8%
- より難しい AIME データセットほど consequential ステップが多く、易しい GSM8K では推論の冗長性が高い

![Consequential ステップのタイプ分布と忠実性テスト結果](/images/legibility-is-not-interpretability-cot/fig4.png)

### LLM 裁定者と微調整批評者の性能（核心結果）

#### 開箱即用裁定者（OOB Judges）

性能はスケール向上に伴い普遍性ベースラインを上回るが、**ノイズ天井には遠く及ばない**。最良の裁定者（Qwen3.6-27B）でも ID で天井の約 1/9、OOD で約 1/6。

#### 微調整批評者（Fine-tuned Critics）

**誤りレスポンス**では比較的良性能：PR-AUC がランダムの 10–15 倍、Precision@0.5% で保守天井に到達。ただし 2% 予算で急速に低下。

**正解レスポンス**ではほぼ失敗：PR-AUC はランダムの 3.5–5 倍にすぎず、ノイズ天井のわずか 10–20%。Precision@0.5% も天井の 10–50% に留まる。

![裁定者・批評者の性能比較と Precision@k% 予算曲線](/images/legibility-is-not-interpretability-cot/fig2.png)

### 忠実性テスト

Scruples データセットを用いた実験では、手がかりなしでは 58% のレスポンスに consequential ステップが含まれるが、**手がかりありではわずか 15%** に減少。手がかりがあると価値がほぼ定数 1.0 となり、すべての推論ステップが新情報を付加しない。

## 考察

### 可読性 ≠ 解釈可能性

本論文の最も重要なメッセージは、推論軌跡のテキスト可読性がその解釈可能性を意味しないという点にある。とくに正解レスポンスにおける consequential ステップの重要性はテキストからほぼ復元不可能である。これは「aha! moment」とも呼ばれる、推論の転換点こそが最も解釈価値が高いにもかかわらず、それをテキストから読み取れないという深刻な結果である。

### 正解 / 誤りレスポンスの非対称性

誤りレスポンスでは、明白な誤りがテキスト上も識別しやすいため、ステップ重要性が部分復元可能。しかし正解レスポンスでは、推論が「正しく機能している」こと自体がテキスト上の特徴として表れにくく、復元が困難になる。

### 性能向上の真の源泉

Thinking モードとスケールの性能向上は、推論過程で答えを発見する能力の向上ではなく、**より強い事前分布**（推論開始前に答えがすでに実質確定）によるものと示唆される。これは「推論とは何か」という問いを根底から揺るがす発見である。

### Process Reward Model への示唆

ステップレベルのテキスト可解釈性を前提とする PRM には根本的な限界が存在する可能性がある。ステップの重要性がテキストから部分復元可能な誤りレスポンスでは PRM は有効だが、正解レスポンスでは実質的に無意味な報酬を分配する危険がある。

### RL と解釈可能性の橋渡し

アドバンテージの概念は、「どのステップが重要か」という解釈可能性の問いと、RL における「信用帰属（credit assignment）」の問いを接続する。この架け橋は、解釈可能性研究に形式的な道具を提供する。

## 関連研究

### CoT 忠実性

Turpin et al. (2023) や Chen et al. (2025) は摂動テストにより CoT の不忠実性を実証。Lanham et al. (2023) はより大きいモデルほど不忠実になりうることを報告。Emmons et al. (2025) は逆に、モデルが推論を必要とする状況では忠実性が高まることを示唆。Boppana et al. (2026) は CoT が「上演的（performative）」であり、モデルが答えを確定後もトークンを生成し続けることを指摘。

### アドバンテージ vs 反事実的必要性

Pearl (2009) の反事実的必要性（「そのステップがなければどうなるか」）はアドバンテージと一致するとは限らない。論理的に必要な計算ステップであっても、アドバンテージはゼロになりうる。これは当該ステップが方策に新たな予測価値を加えないことを正確に信号する。

### KL-based 重要性との比較

Bogdan et al. (2025) は隣接ステップ間の KL ダイバージェンスとして重要性を定義。KL は分布の全体シフトを測るが方向性を示さない。アドバンテージは忠実性や正確性に関連する結果に直接ターゲットする。

### Math-Shepherd との関係

Wang et al. (2024) は中間ステップの「ポテンシャル」を再サンプリング rollout で評価。数学的にはアドバンテージと等価だが、$N=8$ と少なく、補完モデル間の価値移行を仮定する点が異なる。

## まとめ

本論文は、Chain-of-Thought 推論における「可読性は解釈可能性にあらず」というテーゼを、RL のアドバンテージ関数という形式的な枠組みで実証した画期的な研究である。主な貢献は以下の通り。

1. **重要性の定義**：推論ステップの重要性をアドバンテージとして定義し、Monte Carlo rollout で推定する手法を提案
2. **変点検出手法**：PELT + Beta 事後フィルタにより、統計的に検出力の高い consequential ステップ同定を実現
3. **裁定者の限界の実証**：LLM 裁定者・微調整批評者は普遍性ベースラインを上回るがノイズ天井に遠く及ばず、とくに正解レスポンスでは重要性復元がほぼ不可能
4. **推論パターンの解明**：thinking モードの性能向上は「最初から答えが高い」パターンの増加によるものであり、推論途中での発見によるものではない
5. **忠実性との補完性**：自己アドバンテージは手がかりの有無を二値判定ではなく粒度良く検出

本研究は、PRM、LLM 裁定者、人間評価など、推論ステップのテキストを解釈可能な信号として扱うすべての実践に対して根本的な再考を促すものである。

## 参考

- Du, K., Hoyle, A., Ruis, L., & Locatelli, A. (2026). Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning. *COLM 2026*. [arXiv:2609.04194](https://arxiv.org/abs/2609.04194)
- コード: [github.com/kdu4108/importance-advantage](https://github.com/kdu4108/importance-advantage)
- データ: [hf.co/datasets/kducohere/MC-Math-Rollouts](https://huggingface.co/datasets/kducohere/MC-Math-Rollouts)
- 可視化: [hf.co/spaces/kducohere/mc-math-rollouts-viewer](https://huggingface.co/spaces/kducohere/mc-math-rollouts-viewer)
- Turpin, et al. (2023). Language Models Don't Always Say What They Think. *ICML 2024*.
- Lanham, et al. (2023). Measuring and Inducing Formal Thought in Language Models.
- Wang, et al. (2024). Math-Shepherd: A Verify-and-Reinforce LLM for Math.
- Bogdan, et al. (2025). Resampling-based importance via KL divergence.
- Boppana, et al. (2026). CoT as performative generation.
