---
title: "思考が正解を殺す——MARGOで混合モードAdvantage正則化による事実性 hallucination 抑制"
emoji: "🛡️"
type: "tech"
topics: ["LLM", "hallucination", "RLVR", "GRPO", "事実性QA"]
published: true
---

## TL;DR

大型推論モデル（LRM）において、明示的な思考（Chain-of-Thought）は事実性QAで正しい直答を覆して hallucination を引き起こす「思考誘発 hallucination」を生む。MARGOは **非思考 rollout を同一モデル参照として GRPO advantage に組み込み**、思考が事実性に正の残差値を追加するか否かを評価し、有害な思考を抑制・有益な思考を保持する。Qwen3-4B/8Bで平均精度 +2.68/+1.98%、6ベンチマーク全勝、数学推論能力も完全に維持。

## 背景：思考が正解を殺す現象

DeepSeek-R1やQwen3などに代表される大型推論モデルは、`<think>...</think>`ブロックで明示的な思考経路を生成し、最終回答を導く。数学やコーディングでは推論の分解・検証が有効だが、**事実性QAでは事情が異なる**。

TriviaQAでQwen3-8Bを評価したinstance-level分析が衝撃的：

| 移行パターン | 比率 | 含意 |
|---|---|---|
| $r_{0,1}$：非思考誤→思考正 | 11.74% | 思考が知識を回復 |
| $r_{1,0}$：非思考正→思考誤 | **7.50%** | 思考が正答を覆す |
| $r_{1,1}$：両モード正 | 44.68% | 思考が正答を維持 |

7.50%——**同じモデルが非思考モードでは正解できた問いを、思考経路の生成後に誤答に変えてしまう**。これを本稿は **思考誘発 hallucination** と定義する。

![思考誘発 hallucination：3ベンチマークで r0,1（思考が救う）と r1,0（思考が殺す）の推移](/images/margo-mixed-mode-hallucination/fig1.png)

NQ_Openでは8B/14B/32Bで $r_{1,0}$ がそれぞれ8.12%/7.89%/7.62%に達し、$r_{0,1}$ に匹敵する。**推論モデルが大きくなっても、思考が正解を殺す比率は低下しない**。対照的にGSM8Kでは $r_{1,0}<2\%$——数学推論では思考が正答を覆すことは稀。事実性QAの特殊性が明確だ。

思考経路の内容を調べると、「But wait」「Hmm, not sure」「Wait, no, maybe that's not right」といった**不安定・修正マーカー**が頻出。正しい知識を持っているのに、長い思考が不確実性を増幅・実体混同・推測的関連付けを導入し、事実性 drift を引き起こす。

## 方法详解：MARGOの三層構造

### 1. 思考残差値の定式化

非思考応答はモデルの**直接的知識傾向**（direct-answer tendency）を反映する。思考応答はその上に「思考残差」を追加する。残差値を次式で定義：

$$\Delta_{\text{res}}(x) = \mu_T(x) - \mu_N(x)$$

$\Delta_{\text{res}} > 0$ なら思考が事実性に正の価値を追加。$\Delta_{\text{res}} < 0$ なら思考が有害な残差汚染を導入。

### 2. 混合モード rollout 群の構築

標準GRPOは思考モードのみの rollout 群 $\{y_1^T, \ldots, y_K^T\}$ で advantage を計算する。これでは**思考内での相対品質**は評価できるが、思考自体が直答より事実性に貢献しているか否かは見えない。

MARGOは各問いで思考・非思考 rollout を混合：

$$\mathcal{G}(x) = \{y_1^T, \ldots, y_{K_T}^T\} \cup \{y_1^N, \ldots, y_{K_N}^N\}$$

実装では $K_T=6, K_N=2$（$\alpha=0.75$）。**同一モデル・同一パラメータ・プロンプトのみ異なる**2モードを同一群に置く。

![MARGOの混合モード advantage 分解：非思考 rollout を同一モデル参照として組み込む](/images/margo-mixed-mode-hallucination/fig2.png)

### 3. Advantage 分解と正則化効果

混合群の baseline は：

$$\mu_{\text{mix}}(x) = \alpha\mu_T(x) + (1-\alpha)\mu_N(x)$$

**命題1（残差値分解）**：思考 trajectory の混合モード advantage は

$$A_{\text{mix}}^T(x, y^T) \propto \underbrace{(R(x,y^T) - \mu_T(x))}_{\text{within-mode advantage}} + \underbrace{(1-\alpha)\Delta_{\text{res}}(x)}_{\text{残差値調整}}$$

- $\Delta_{\text{res}} > 0$：思考が有益→advantage 増加→思考 trajectory が強化
- $\Delta_{\text{res}} < 0$：思考が有害→advantage 減少→思考 trajectory が抑制

非思考 trajectory は逆の調整を受け、思考が有害な場合に相対的に好まれる。**報酬関数自体はモードを区別しない**——正則化効果は群内 advantage normalization から**自然に浮上**する。

### 報酬設計

事実性QAは数学のような確定的 verifier が存在しない。同義語・略語・言い換え等で exact match が不可。Qwen3-32Bを **judge** として使用——ただし**独自知識で事実性を判定するわけではなく**、question・ground-truth・prediction を与えて意味的整合性を判定する3-way（CORRECT/INCORRECT/NOT_ATTEMPTED）評価。

$$r = r_{\text{corr}} + r_{\text{fmt}}$$

$r_{\text{corr}}$: 正解なら+1、不然なら-1。$r_{\text{fmt}}$: `<think>`/`</think>` ブロックが正しく閉じられていれば+0.05。

### 訓練データ構築

TriviaQA訓練分割から、思考・非思考モードで各6サンプルを生成し、**2モード間の事実性 gap が明確な例**を選択：
- 非思考好適例：$\Delta(x) \leq -0.7$ かつ $S_N(x) \geq 0.7$
- 思考好適例：$\Delta(x) \geq 1.0$ かつ $S_T(x) \geq 1.0$

Qwen3-4Bで5,660例、Qwen3-8Bで5,721例。ランダム選択の ablation では改善がほぼ消滅——**明確なモード間 gap を持つ例**が正則化信号の源泉。

## 実験結果

### 事実性QA：6ベンチマーク全勝

| Method | SimpleQA | SimpleQA-V | TriviaQA | NQ_Open | PopQA | HotpotQA | **Avg** |
|---|---|---|---|---|---|---|---|
| FixedNoThink | 3.74 | 3.80 | 39.79 | 29.50 | 17.07 | 27.86 | 20.29 |
| FixedThink | 3.98 | 4.10 | 46.12 | 32.94 | 19.05 | 30.64 | **22.81** |
| AdaptiveSFT | 3.35 | 4.30 | 43.91 | 30.28 | 18.36 | 28.08 | 20.96 |
| FixedThink+RL | 4.48 | 4.10 | 40.14 | 30.61 | 18.40 | 27.70 | 21.11 |
| **MARGO** | **6.73** | **10.00** | **49.12** | **34.76** | **20.48** | **31.82** | **25.49** |

Qwen3-4Bで FixedThink +2.68%、**全6ベンチマーク最高**。SimpleQA/SimpleQA-Vでは特に顕著（+2.75/+5.90）。

Qwen3-8Bでも同様：

| Method | Avg |
|---|---|
| FixedThink | 27.59 |
| FixedThink+RL | 27.92 |
| **MARGO** | **29.57 (+1.98)** |

![6ベンチマークでの全手法比較（4B/8B）](/images/margo-mixed-mode-hallucination/fig3.png)

### 遷移分析：r1,0 削減 + r1,1 増加

FixedThink+RL は $r_{1,0}$・$r_{1,1}$ の変化が微少（-0.14~-0.16/+0.11~0.52）。MARGO は：
- TriviaQA $r_{1,0}$：**-0.45**（有害遷移を大幅削減）
- NQ_Open $r_{1,0}$：**-0.50**
- PopQA $r_{1,0}$：**-1.40**
- $r_{1,1}$：全ベンチマークで増加（+0.0~+1.67）

改善は RL 訓練やデータ自体ではなく、**混合モード advantage 正則化**由来。MARGOはモデルを非思考に偏らせるわけではなく、思考が事実性に貢献するか否かで最適化信号を調整する。

### 数学推論能力の維持

AMC23/AIME24/AIME25で Mean@16 を評価：

| | FixedThink Avg | MARGO Avg | 差分 |
|---|---|---|---|
| Qwen3-4B | 78.0% | **78.5%** | +0.5 |
| Qwen3-8B | 79.2% | **80.7%** | +1.5 |

**数学推論能力は完全に維持**、しかも微増。事実性正則化が推論一般化を損なわない。

![遷移比率変化と数学推論維持](/images/margo-mixed-mode-hallucination/fig4.png)

## 考察

### 思考の二面性とモード依存性

「思考は常に有益」という前提は事実性QAで成立しない。推論モデルでは思考が正答を殺す比率がモデルサイズに依存しない（1.7B~32Bで一貫して $r_{1,0}$ が存在）。これは**知識検索型タスクにおける思考の根本的な不安定性**を示唆する。

### 単純なモード選択では不十分

AdaptiveCLS（DeBERTa分類器）とAdaptiveSFTは FixedThink を一貫して上回れない。問いレベルの「思考するか否か」の硬い決定は、同一問い内でもサンプル依存の思考有無性を捉えられない。MARGOは**硬決定を回避**し、advantage 計算内でモード参照を柔軟に利用する。

### 同一モデル参照の意義

非思考 rollout は外部モデルではなく**同一パラメータの別プロンプト出力**。これにより：
- モデル固有の知識傾向を正確に参照
- モデル更新に参照も追従
- 外部参照モデルの前提分布ミスマッチを回避

### 生成長への影響

Qwen3-4Bでは MARGO の生成長が FixedThink 比 -3.4~-5.2%短縮、Qwen3-8Bでは +0.1~+1.5%延長。**一貫した短縮傾向はない**——明示的な長さペナルティではなく、advantage 正則化からの自然な結果。

## 関連研究

- **AdaptThink** [Zhang et al., EMNLP 2025]：問いレベルの思考/非思考選択をRLで学習。MARGOは硬決定ではなく soft 正則化。
- **DASH** [Lee et al., 2607.00482]：セグメント別 advantage で過思考抑制。事実性 drift ではなく回答 drift を対象。
- **CAT** [Jiang et al., ACL 2026]：Self-Certainty で推論長適応制御。モード選択ではなく長さ制御。
- **DemoPSD** [Li et al., 2607.02502]：JSD分岐で教師指導減衰。OPSD特権漏洩を解決。MARGOも「参照で正則化」の思想を共有。
- **RLCR** [ICLR 2026]：校正報酬で RLVR の確信度改善。事実性 hallucination ではなく自信度校正。
- **Chen et al. [2025b]**：LRM の長文 hallucination で online RL + 事実性報酬。思考誘発 hallucination のinstance-level分析は未踏。

## 总结

MARGOの核心は3点：

1. **思考誘発 hallucination の発見**：事実性QAで、思考が正しい直答を覆して hallucination を引き起こす比率がモデルサイズに依存せず7~8%に達する
2. **残差値定式化**：思考を「直答傾向への残差」と見なし、$\Delta_{\text{res}}$ で有無性を評価
3. **混合モード advantage 正則化**：非思考 rollout を同一モデル参照として GRPO advantage に組み込み、有害思考を抑制・有益思考を強化——**報酬関数にモード差は不要**、群内 normalization から自然に浮上

結果：4B/8Bで6ベンチマーク全勝、数学推論完全維持、$r_{1,0}$ 大幅削減。LRMの改善には「思考を促す」だけでなく、「思考が直答に事実性価値を追加するか否かを学習する」視点が不可欠だ。

## 参考

- Wang, K. et al. "Mitigating Factual Hallucination in Large Reasoning Models via Mixed-Mode Advantage Regularization." arXiv:2607.05861, 2026.
- Shao, Z. et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.
- Zhang, J. et al. "AdaptThink: Reasoning Models Can Learn When to Think." EMNLP 2025.
- Lee et al. "Know When to Stop: Segment-Level Credit Assignment for Reducing Overthinking." arXiv:2607.00482, 2026.
- Li et al. "DemoPSD: Disagreement-Modulated Policy Self-Distillation." arXiv:2607.02502, 2026.
