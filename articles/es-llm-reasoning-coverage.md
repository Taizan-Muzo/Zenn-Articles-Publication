---
title: "ESはGRPOを超えるか：推論カバレッジの理論と実験"
emoji: "🧬"
type: "tech"
topics: ["LLM", "ReinforcementLearning", "EvolutionStrategies", "PostTraining"]
published: true
---

# TL;DR

LLMの推論ポストトレーニングといえばGRPO一強だが、本稿は**Evolution Strategies（ES）**がGRPOとは異なる強みを持つことを理論と実験の両面から示す。ESはPass@1こそGRPOに譲るが、**Pass@Kで一貫して上回り**、エントロピーコラプスを起こさない。GRPOの18箇所中15箇所でPass@16/32がBase Modelを下回るのに対し、ESは全モデルで上回る。さらに、パラメータ空間での大きなドリフトにもかかわらず、**機能的スパース性**により性能寄与は少数の大幅更新に集中し、カタストロフィックフォゲッティングは起きない。**ESはGRPOのメモリ効率版ではなく、独立した推論ポストトレーニングパラダイム**である。

# 背景

LLMの推論能力を引き出すポストトレーニング手法として、Group Relative Policy Optimization（GRPO）がデファクトスタンダードになりつつある。GRPOは各プロンプトからG個の応答をサンプリングし、グループ内で標準化したアドバンテージで方策を更新する。

一方、**Evolution Strategies（ES）**は勾配を使わない最適化手法で、パラメータ空間に直接摂動を加え、その結果の報酬から探索方向を決定する。GPUの逆伝播メモリを必要としないため、大規模モデルでのメモリ効率が優れるとされてきたが、GRPOとの直接的な性能比較や、ESの最適化挙動の理解は不十分だった。

本稿（Ba et al., 2026, arXiv:2608.27351）は、ESの推論ポストトレーニングにおける特性を3つの研究問いに沿って体系的に解明する。

![fig1](/images/es-llm-reasoning-coverage/fig1_pass_k_comparison.png)

# RQ1：ESの推論カバレッジ優位性

## 理論的解析

ESの核心的な違いは、パラメータ空間の摂動が**ポリシー多様性**を生み出す点にある。

**Lemma 1（摂動誘発多様性）**: 固定プロンプト$x$に対し、$N$個の摂動で生じるポリシー間のJS多様性は、摂動スケール$σ → 0$のとき

$$\mathbb{E}[JS_N^{pol}(x)] = \frac{σ^2}{2}\left(1 - \frac{1}{N}\right) \cdot \mathrm{tr}\, \mathcal{I}_x(\theta) + O(\sigma^4)$$

となる。ここで$\mathcal{I}_x$はプロンプト条件付きFisher情報行列で、モデルの表現力が高いほど多様性が大きい。

**Lemma 2（多様性→カバレッジ）**: 種群の成功確率JS多様性$JS_N^{succ}$はポリシーJS多様性$JS_N^{pol}$で上から抑えられ、$N$個の異なるポリシーで各1回ずつサンプリングする成功率は同一ポリシーの$N$回サンプリングを上回る。

**Lemma 3（報酬加重改善）**: 報酬とメンバーの成功率が正相関する場合、報酬加重混合ポリシーの成功確率は均一加重平均を上回る。

**Proposition 1**: 中心への遷移誤差が十分小さく、比較器のマージンが$K\sqrt{\varepsilon/2}$を超えれば、ES更新後のPass@Kは更新前を上回る。

## 実験結果

### Easy Setting（GSM8K学習）

3つのモデル（Qwen2.5-1.5B/7B-Instruct、Llama-3.2-3B-Instruct）でGSM8Kを2 epoch学習し、6タスクで評価。

![fig2](/images/es-llm-reasoning-coverage/fig2_entropy_dynamics.png)

**GRPOのエントロピーコラプス**: GRPOはPass@1を大幅に改善するが、held-outのGPQA tokenエントロピーが急激に低下。その結果、**18箇所のPass@16/32比較のうち15箇所でBase Modelを下回る**。

**ESの多様性維持**: ESはエントロピーの低下が緩やかで、全モデルの平均Pass@16/32でBaseとGRPOの両方を上回る。例えばQwen2.5-1.5BではES平均Pass@16が76.0%（Base 75.4%、GRPO 75.1%）。

### Hard Setting（DeepScaleR学習）

DeepSeek-R1-Distill-Qwen-1.5BでDeepScaleRを1 epoch学習し、AIME24/25、AMC23、MATH-500で評価。

| Method | Avg Pass@1 | Avg Pass@16 | Avg Pass@32 |
|--------|-----------|-------------|-------------|
| Base | 47.7 | 73.5 | 77.4 |
| GRPO | **52.9** | 74.7 | 78.0 |
| ES | 49.9 | 75.0 | **78.9** |
| ES→GRPO | 52.3 | **75.8** | **79.2** |

ES単独ではPass@1でGRPOに劣るが、Pass@32では逆転する。

### 順次混合訓練

GRPOのPass@1強みとESのPass@K強みを組み合わせるため、同一予算を2段階に分割。

- **ES→GRPO**: Hard Settingで最高Pass@32（79.2%）を達成しつつ、GRPOのPass@1利得の大部分を保持
- **GRPO→ES**: 両者のPareto前沿上に非支配トレードオフ点を追加

![fig4](/images/es-llm-reasoning-coverage/fig4_pareto_scaling.png)

# RQ2：機能的スパース性

## パラメータドリフトの大きさ

ESのパラメータ移動量はGRPOの**約40〜44倍**に達する。

| Model | GRPO D_rel | ES D_rel | Ratio |
|--------|-----------|----------|-------|
| Qwen2.5-1.5B | 0.048 | 1.933 | **40.7x** |
| Llama-3.2-3B | 0.095 | 4.185 | **43.9x** |
| Qwen2.5-7B | 0.083 | 3.664 | **44.1x** |

これほどのドリフトにもかかわらず、held-out評価ではESの平均Pass@32が**GRPOを上回る**。

## 更新のスパース性

鍵は**機能的スパース性**だ。更新の幅閾値$τ$以下の小さな更新の割合を測ると：

| Threshold | Qwen2.5-1.5B | Llama-3.2-3B | Qwen2.5-7B |
|-----------|-------------|-------------|-----------|
| $\tau$=1.0e-3 | 79.1% | 78.1% | 78.4% |
| $\tau$=1.5e-3 | 92.5% | 92.6% | 93.0% |
| $\tau$=2.0e-3 | 97.6% | 97.9% | 98.1% |

$\tau$=1.5e-3で**約93%の非ゼロ更新が小幅**に収まる。性能寄与は残り7〜22%の大幅更新に集中しており、彩票仮説やNeural Thickets（Gan & Isola, 2026）と整合する。

![fig3](/images/es-llm-reasoning-coverage/fig3_functional_sparsity.png)

## 大幅更新の局在化

ESの大幅更新は**LayerNorm重みとattention投影**に集中する。Llama-3.2-3BではTop 100の最大更新のうち72個が正規化パラメータ。一方GRPOはtoken embeddingsと言語モデル出力頭に集中。ESが正規化層を通じて隠れ状態のスケーリングや情報ルーティングを調整している可能性が高い。

## 幅閾値化実験

小幅更新を段階的にゼロに置換しても、高スパース性まではPass@1がほぼ維持される。Base Model性能を下回るのは95%以上をゼロにしたとき。つまり、大半の更新は機能的に冗長で、少数の鍵パラメータが性能を支えている。

# RQ3：ハイパーパラメータ設計

## 報酬正規化

z-score正規化（GRPOのグループ正規化と同様）はES訓練の**必須成分**。正規化なしでは早期に報酬過学習に陥る。

## 摂動スケール$\sigma$

$σ$が小さすぎると狭い近傍の探索に閉じこもり局所最適に陥り、大きすぎると不安定になる。論文では初期の報酬過学習を防ぎつつ安定した報酬ガイド付き進行を維持するバランスが推奨される。

## 種群サイズのスケーリング則

Qwen2.5-{0.5B, 1.5B, 3B}-InstructでN$\in$\{8, 16, 32, 64\}を系統的に評価。

- **0.5B**: N=32でN=64に匹敵（N=16では不十分）
- **1.5B**: N=16でもN=64の0.5%以内
- **3B**: N=16でN=64の0.3%以内

**大きなモデルほど小さい種群で十分に機能する**。これは大規模モデルがより多くの有効部分構造を含み、近傍に性能改善摂動が密集しているためと解釈される。

## 1点推定器 vs 2点推定器

2点ES推定器（対称摂動の差分）はSST-2などの監督タスクで分散削減効果があるが、GSM8Kのような自回帰再生成タスクでは**相関が弱まり効果がない**。推論タスクでは1点推定器が推奨される。

# 考察

本稿の最も重要なメッセージは、**ESはGRPOの劣化版ではなく、異なるトレードオフ空間を最適化する独立パラダイム**だということだ。

GRPOが「ベストアンサーを絞り込む」アプローチなら、ESは「正解に至る複数ルートを維持・拡張する」アプローチと言える。これはtest-time computeの文脈で決定的に重要になる——Pass@Kは$K$回の独立サンプリングで少なくとも1つが正解の確率を測る指標であり、モデルの実用的な推論能力をより正確に反映する。

機能的スパース性の発見も示唆に富む。パラメータ空間での巨大な移動が必ずしも能力の喪失を意味しないという事実は、LLMのパラメータ空間に大域的な冗長性が存在することを示唆している。Lottery Ticket Hypothesisの延長線上にあり、今後のパラメータ効率改善に繋がる可能性がある。

# 関連研究

- **Agentic ESOpt**（Zheng et al., 2026）: エージェントタスクにESを適用し、推論レベルのGPUメモリでfull-parameter微調整を実現
- **MeZO**（Malladi et al., 2023）: LLMのzero-order最適化の先駆的研究
- **Abdi et al. (2026)**: ESの大規模パラメータドリフトによるカタストロフィックフォゲッティングを報告（本稿はより広い条件でこれを反駁）
- **SoftmaxGRPO**（arXiv:2608.09271）: GRPOのz-score発散問題を温度softmaxアドバンテージで解決
- **Neural Thickets**（Gan & Isola, 2026）: 大規模モデルの有効部分構造の存在を理論的に示唆

# まとめ

| 発見 | 内容 |
|------|------|
| 推論カバレッジ | ESはPass@KでGRPOを一貫して上回る。GRPOは18/15でPass@16/32がBase以下に |
| 順次訓練 | ES→GRPOでHard Setting最高Pass@32（79.2%）達成 |
| 機能的スパース性 | パラメータドリフト40xでも性能維持。93%の更新は小幅 |
| 種群スケーリング | 大モデルほど小種群で十分。3BではN=16でN=64に匹敵 |
| 1点推定器 | 自回帰推論タスクでは2点推定に優る |

ESはPass@1でGRPOに譲るものの、Pass@Kという実用指標では明確に優位性を持つ。順次混合訓練で両者の利点を統合できることから、**推論モデルのポストトレーニング設計においてESは無視できない選択肢**になったと言える。

# 参考

- Ba, Y. et al. "Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO." arXiv:2608.27351, 2026.
- Frankle, J. & Carbin, M. "The Lottery Ticket Hypothesis." ICLR 2019.
- Gan, C. & Isola, P. "Neural Thickets." 2026.
- Malladi, S. et al. "Fine-Tuning Language Models with Just Forward Passes." arXiv:2305.17333, 2023.
