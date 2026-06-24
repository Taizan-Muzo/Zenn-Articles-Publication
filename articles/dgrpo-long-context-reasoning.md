---
title: "dGRPO: 長文脈推論のための蒸留付き方策最適化"
emoji: "🔗"
type: "tech"
topics: ["LLM", "強化学習", "知識蒸留", "LongContext", "GRPO"]
published: false
---

# dGRPO: 長文脈推論のための蒸留付き方策最適化

LLMを数万トークンの長文脈タスクに適応させるには、学習信号と露光バイアスのバランスが鍵になる。Off-policy手法（SFT/KD）は安定しているがモデル自身の生成分布を反映できず、on-policy RL（GRPO）は文脈が長いほど稀疏報酬で不安定になる。

Ramos et al. (IST Lisbon) はこのジレンマを解消すべく、**dGRPO（Distilled GRPO）**を提案した。GRPOのスパース報酬最適化と、教師モデルからのdenseトークンレベル蒸留（OPD）を単一目的関数に統合する手法だ。これを2段階パイプライン（SFTコールドスタート → dGRPO）で適用し、Qwen3-1.7Bを128K文脈に拡張した結果、**RULERで平均64.5→74.8**、128K長では**11.0→44.5**と大幅改善しつつ、短文脈性能は完全に保持された。

## TL;DR

- **dGRPO**: GRPO + OPDを単一目的に統合。スパース報酬最適化にdense教師正則化を付加
- **LongBlocks**: 多言語合成长文脈データセット（193K QAペア、30+自然言語、15+プログラミング言語）
- **実績**: Qwen3-1.7BベースでRULER 128Kが+33.5pp、∞Bench・LongBenchでも一貫改善
- **短文脈保持**: 数学・コード生成でbase modelと同等以上を維持
- 教師モデル（Qwen3-32B）の質が重要。自己蒸留でも改善するが外部教師が圧倒的に有利

## 背景：長文脈post-trainingのジレンマ

LLMの長文脈推論は実応用で不可欠になりつつある——コードベース全体の理解、複数文書の横断検索、数十万トークンにまたがるマルチセッション対話など。しかしpost-training段階では3つの障壁がある。

| 手法 | 学習軌跡 | 信号密度 | 報酬最適化 | 露光バイアス |
|------|---------|---------|-----------|------------|
| SFT / KD | Off-policy | Dense | 不可 | 高い |
| OPD | On-policy | Dense | 不可 | 低減 |
| GRPO | On-policy | Sparse | 可能 | 低い |
| **dGRPO** | **On-policy** | **Dense** | **可能** | **低い** |

![fig1](/images/dgrpo-long-context-reasoning/fig1.png)

Off-policy手法（SFT・KD）は安定でデータ効率が良いが、固定の専門家データに依存するため**露光バイアス（exposure bias）**が生じる。学習中は教師のトークン接頭辞だけを見るが、推論時はモデル自身の生成が接頭辞になる。文脈が長いほど、このミスマッチが累積して性能劣化を引き起こす。

一方、GRPOはon-policyで露光バイアスを回避できるが、長文脈では**軌跡レベルのスパース報酬**が遅延し、安定性とサンプル効率が著しく低下する。reward hackingのリスクも高い。

OPD（On-Policy Distillation）はdenseなトークンレベル信号を提供し安定しているが、報酬信号を直接最適化できない。

dGRPOはこの3つの利点を統合する。

## dGRPO：手法详解

### 目的関数

dGRPOの核心は、GRPO目的に教師モデルからの逆KLダイバージェンス正則化項を追加すること：

$$J_{dGRPO}(\theta) = J_{GRPO}(\theta) - \beta \cdot \mathbb{E}_{p, \{o_i\}} \left[ \sum_{t=1}^{|o|} D_{KL}\left( \pi_\theta(\cdot|p, o_{<t}) \| \pi_{teacher}(\cdot|p, o_{<t}) \right) \right]$$

- **第1項**: GRPO。グループ内相対アドバンテージによるスパース報酬最適化
- **第2項**: OPD。逆KLで生徒を教師に接近。生徒が実際に訪れる軌跡上でのみ監督
- **β = 0.5**: 正則化強度。0なら純GRPOに退化

![fig2](/images/dgrpo-long-context-reasoning/fig2.png)

### 2段階パイプライン

**Stage 1 — SFTコールドスタート**: LongBlocks + Nemotron（10%短文脈/90%長文脈）で2エポック。長文脈roLLoutの品質安定化を図る。

**Stage 2 — dGRPO**: Qwen3-32Bを教師に、同一データミックスで1エポックのon-policy RL。β=0.5、8生成サンプル、クリッピング閾値0.2。

![fig2](/images/dgrpo-long-context-reasoning/fig2.png)

### LongBlocksデータセット

多言語合成长文脈QAデータセットで、以下のソースから193,219ペアを生成：

- ArXiv, Wikipedia, Project Gutenberg, StackExchange, FineWeb2-HQ, Institutional Books
- 30以上の自然言語、15以上のプログラミング言語
- Qwen3-Next-80B-A3B-Thinkingで合成 → LLM-as-a-Judgeで品質検証

## 実験結果

### RULER: 文脈長別性能

RULERベンチマークで文脈長ごとの平均精度を比較した。

![fig3](/images/dgrpo-long-context-reasoning/fig3.png)

| 文脈長 | Base | dGRPO | 差分 |
|--------|------|-------|------|
| 4K | 87.4 | 88.3 | +0.9 |
| 8K | 84.2 | 85.4 | +1.2 |
| 16K | 81.0 | 84.7 | +3.7 |
| 32K | 72.0 | 80.4 | +8.4 |
| 64K | 51.5 | 65.6 | +14.1 |
| **128K** | **11.0** | **44.5** | **+33.5** |

128K文脈で33.5ポイントの改善。文脈が長いほどdGRPOの優位性が顕著にになる。

### 短文脈保持

| ベンチマーク | Base | Ours |
|-------------|------|------|
| IFEval | 88.0 | 88.3 |
| MMLU-Pro | 70.1 | 70.4 |
| GSM8K | 67.7 | 69.5 |
| MATH-500 | 60.4 | 54.0 |
| HumanEval | 54.0 | 54.1 |
| GPQA ♦ | 34.4 | 33.8 |

短文脈ではほぼ維持。コード生成（HumanEval）で僅かに改善。数学はやや低下するものの、長文脈の大幅ゲインを考慮すれば許容範囲。

### 学習ダイナミクス

![fig4](/images/dgrpo-long-context-reasoning/fig4.png)

GRPO単独は報酬が増えるものの変動が大きく、長文脈性能の改善が限定的。OPDは安定しているが最終報酬が低い。dGRPOは**最も滑らかな学習曲線**と**最高の最終報酬**を達成し、長文脈性能が最も高い。

### 教師モデルの影響

- **自己蒸留**: ベースラインより改善するが、dGRPOのポテンシャルを十分に引き出せない
- **外部教師（Qwen3-32B）**: 自己蒸馏を大幅に上回る。教師信号の質が直接dGRPOの効果に反映

### βの感受性

β=0（純GRPO）が最悪。βの増大とともに性能が向上し、**β=0.5でピーク**。それ以上では正則化が強すぎて探索が阻害される。

## 考察

### なぜdGRPOは長文脈で有効か

長文脈推論の難しさは、数千〜数万トークンにわたって推論構造を維持しつつ、文脈全体から証拠を統合しなければならない点にある。GRPOはこの問題を「文脈全体の結果が正かどうか」というスパース信号だけで解こうとするため、中間ステップでの逸脱が見逃され、reward hackingや推論崩壊が起きやすい。

OPDの教師正則化は**各トークンステップで「正しい」次トークンの分布を教える**ため、GRPOのスパース信号をdenseに補完する。しかもon-policyで動くため、モデルが実際に生成する（間違えやすい）接頭辞に対して教師が介入できる——これが露光バイアスの根本的な解消だ。

### 2段階設計の必然性

RL-Zero（SFTなしで直接dGRPO）は短文脈/長文脈ともに悪い結果になった。これは長文脈の初期rollout品質が低すぎて、アドバンテージ推定が脆くなるため。SFTコールドスタートで基盤を安定させた上で、dGRPOでon-policy微調整する2段階設計が最適パターンとして浮かび上がった。

### 10/90ミックスの意義

短文脈10%・長文脈90%の固定ミックスが、壊滅的忘却（catastrophic forgetting）を防ぐ鍵。RL中にも短文脈タスクを維持することで、長文脈獲得による一般的推論能力の劣化を抑制している。

## 関連研究

- **GRPO (Shao et al., 2024)**: グループ相対アドバンテージでクリティック不要のPPO簡易化。DeepSeek-V3/R1で採用
- **OPD (Agarwal et al., 2024)**: on-policy蒸留のオリジナル提案。自己蒸留にも拡張可能
- **DAPO (Yu et al., 2025)**: GRPOの改良版。アドバンテージ正規化やトークンレベルペナルティ
- **LongAlign (Bai et al., 2024)**: 長文脈向けoff-policyアライメント手法
- **EffOPD (Prior work in this series)**: OPDの効率化。指数チェックポイント外挿で3倍加速
- **CIPO**: 失敗軌跡修正RLVR。dGRPOとは対照的にnegative sampleを活用
- **RiM**: 作業記憶ベース潜在推論。推論構造をlatent spaceに符号化

## まとめ

dGRPOは「スパース報酬のon-policy最適化」と「dense教師信号のon-policy蒸留」という、一見相反する2つの学習パラダイムを単一目的関数に統合することで、長文脈post-trainingの安定性と効果性を同時に達成した。

- **128K文脈で33.5pp改善**（RULER 11.0→44.5）
- 短文脈性能は完全保持
- 教師モデルの質が性能を直接左右
- β=0.5がバランスの最適点

2段階パイプライン（SFT → dGRPO）とデータミックス設計（10%短/90%長）により、長文脈の獲得と短文脈の保持を両立する実用的レシピが確立された。

実応用の観点からは、dGRPOがOpenRLHFやHugging Face TRLのような既存のGRPO実装に**教師正則化項を数行追加するだけで統合可能**である点も実装上の利点だ。βのチューニングはタスク依存だが、論文の知見（β=0.5が最適）は強力な初期値を提供する。

## 参考

- Ramos, M. M., Alves, D. M., & Martins, A. F. T. (2026). *Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models*. arXiv:2605.12227
- [LongBlocks Dataset](https://huggingface.co/datasets/utter-project/LongBlocks)
- [Paper PDF](https://arxiv.org/pdf/2605.12227v1)
