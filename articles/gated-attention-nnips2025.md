---
title: "Gated Attention：NeurIPS 2025最優秀論文"
emoji: "🏆"
type: "tech"
topics: ["LLM", "Transformer", "NeurIPS", "Attention", "DeepLearning"]
published: false
---

## TL;DR

- Softmax Attentionの出力に**ヘッドごとのSigmoidゲート**を挟むだけ。それがNeurIPS 2025最優秀論文のアイデア
- Attention Sink（最初のトークンに集中する現象）を本質的に緩和し、長文脈処理が安定する
- 15B MoE / 1.7B Dense で30種以上の変体を比較。すでにQwen3-Nextに組み込まれている
- 追加パラメータは0.1〜0.5%。計算オーバーヘッドもほぼ無視できる

## 1. なぜこの論文を選んだか

2025年11月、NeurIPSの最優秀論文が発表された。全4本のうち、LLMのアーキテクチャそのものに切り込んだのがQwenチームの「Gated Attention for Large Language Models」だ。

TransformerのAttentionは、2017年の「Attention Is All You Need」以来ずっとSoftmax + Scaled Dot-Productが基本で、ほとんど手が入っていない。それが**Sigmoidゲート一つ**で訓練安定性も長文脈性能も改善するというなら、NLPエンジニアとしては読まない手はない。

## 2. 問題設定：Softmax Attentionの「罠」

### 2-1. Attention Sinkとは

LLMのAttentionを可視化すると、奇妙な現象が起きている。**どのレイヤーでも、最初のトークン（`<bos>`など）に異常なほどAttentionが集中している**のだ。

論文のデータが物語っている。1.7B Denseモデルの第21層では、**最初のトークンが全体の83%のAttentionを占有**していた。つまり、83%の計算リソースが最初のトークンに吸い取られ、意味のあるトークン群にはほとんど振り向けられていない。

これが「Attention Sink」と呼ばれる現象で、Shiら（2023）が初めて体系的に報告した。

![fig2](/images/gated-attention-nlp/fig2_attention_heatmaps.png)
*図2：Baseline（上段）では最初のトークンが赤く光る（Attention Sink）。Gated（下段）ではAttentionが全体に分散している。*

### 2-2. なぜSinkが起きるのか

Softmaxの性質が原因だ。

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

Softmaxはすべてのスコアの合計を必ず1に正規化する。クエリに対して「どのキーも重要でない」という場合でも、どこかにAttentionを割かなければならない。そこで**一番手っ取り早い逃げ道が最初のトークン**になる。

加えて、FFNの初期段階で異常に大きい値が出力され、それが残差接続を通じて蓄積する。最初のトークンは全シーケンスの情報を「溜め込み」やすく、学習初期から安定したアンカーとして機能してしまう。一度このパターンが定着すると、後から矯正するのはかなり難しい。

### 2-3. 実際に困ること

- **長文脈の性能劣化**：コンテキストが長くなるとSinkの影響が増幅し、後半のトークンの情報が希薄になる
- **訓練の不安定性**：Loss spike（突発的な損失の跳ね上がり）が頻発し、学習率を低く抑えざるを得ない
- **表現力の低下**：重要な情報がSinkに圧迫されて、モデルの性能上限を下げている

## 3. 提案手法：Gated Attention

### 3-1. アイデアの骨子

提案は驚くほどシンプルだ。

> **SDPA（Scaled Dot-Product Attention）の出力に対して、クエリ依存のSigmoidゲートを掛ける**

図で見ると一目瞭然だ。

![fig1](/images/gated-attention-nlp/fig1_attention_comparison.png)
*図1：左が標準のSoftmax Attention。右が提案するGated Attention。赤い「Sigmoid Gate」ブロックが追加されているだけ。*

数式で書くとこうなる。

$$O_{\text{gated}} = \sigma(X W_\theta) \odot \text{Attention}(Q, K, V)$$

- $X$：クエリの隠れ状態
- $W_\theta$：ヘッドごとの学習可能なパラメータ
- $\sigma$：Sigmoid関数（出力は0〜1の範囲）
- $\odot$：要素ごとの積（アダマール積）

要するに、**Attentionの計算結果に「このヘッドの出力をどれくらい通すか」のバイパス制御を挟む**わけだ。

### 3-2. なぜ出力の「後」なのか

ここが肝だ。論文では30種以上の変体を比較している。

- Q / K / V投影の**前**にゲート → 効果薄
- Softmaxの**前**にゲート → 効果薄
- Valueと掛け合わせた**後**、出力投影の**前**にゲート → **これが最強**

結論として、**G1（SDPA出力直後）**が最も性能が良いことがわかった。理由は2つある。

1. **非線形性の導入**：Softmax Attentionは本質的に低ランクな線形写像だ。Value投影と出力投影の間にSigmoidゲートを入れることで、この線形ボトルネックを打破できる
2. **クエリ依存のスパース性**：Sigmoidの出力は0〜1のスカラーで、大部分の値が0に近い。つまり、**各クエリに対して不要なヘッド出力を動的にフィルタリング**できる

### 3-3. スパース性こそが鍵

論文のデータを見ると、Gating Scoreの平均値は**わずか0.116**。大部分のスコアがほぼ0で、一部だけが1に近い。

![fig3](/images/gated-attention-nlp/fig3_gating_analysis.png)
*図3左：Gating Scoreの分布。圧倒的に0付近に集中している（スパース）。図3右：各レイヤーにおける最初のトークンのAttention占有率。Baselineの46.7%が、Gatedでは4.8%まで低下。*

もしこのスパース性を意図的に潰す（Sigmoidの代わりに均一なゲートを使う）と、性能向上はほぼ消失する。**スパースであること自体が重要**で、それは「ほとんどの情報はノイズなので捨てていい」というモデルの判断を反映している。

## 4. 実験結果

### 4-1. 訓練の安定性

1.7B Denseモデルで学習させたところ、Gated Attentionを導入したモデルは**Loss spikeがほぼ完全に消失**した。これまでは学習率を慎重に下げてLoss spikeを回避する必要があったが、Gatedならもっと高い学習率でも安定して収束できる。

![fig4](/images/gated-attention-nlp/fig4_training_context.png)
*図4左：学習曲線の比較。BaselineはLoss spikeが散見されるが、Gatedは滑らかに収束している。図4右：RULERベンチマークでの長文脈性能。コンテキスト長を4K→128Kに伸ばしても、Gatedは性能劣化が小さい。*

### 4-2. 長文脈の外挿性能

RULERベンチマークで評価。コンテキスト長を訓練時の4Kから128Kまで延ばした結果。

| コンテキスト長 | Baseline | Gated (G1) |
|:---:|:---:|:---:|
| 4K | 92 | 93 |
| 32K | 55 | 76 |
| 128K | 22 | 58 |

128K時点で**36ポイントもの差**がついた。Sinkが緩和されたことで、後半のトークンの情報もちゃんと使えるようになっているのだろう。

### 4-3. 評価ベンチマーク

1.7B Denseモデルでの主要ベンチマーク結果。

| 指標 | Baseline | Gated (G1) | 差分 |
|:---:|:---:|:---:|:---:|
| PPL | 基準 | **-0.265** | 改善 |
| MMLU | 基準 | **+2.03** | 改善 |
| GSM8K | 基準 | **有意に改善** | — |

パラメータ増加はわずか0.1〜0.5%。このインパクトに比べれば、コストはほぼゼロに等しい。

## 5. 消融実験から見える設計の要諦

論文の消融実験は非常に勉強になる。要点を整理する。

**ゲートの位置**：G1（SDPA出力後）が最強。G2（Value投影後）もPPLは改善するが、総合性能ではG1に劣る

**活性化関数**：Sigmoid > Tanh > ReLU。0〜1に制限されるSigmoidが最もスパースになりやすい

**スパース性の必要性**：非スパースなゲート（常に0.5付近）だと性能向上が消失。スパース性は不可欠

**クエリ依存性**：固定ゲート（入力によらず一定）より、クエリ依存ゲートのほうが圧倒的に良い。各トークンに応じた動的なフィルタリングが重要

**ヘッドごと vs 要素ごと**：ヘッドごと（Headwise）と要素ごと（Elementwise）の2つの実装があり、どちらも有効。ヘッドごとは軽量で、要素ごとは表現力が高い

## 6. 実装の実際

コードはGitHubで公開されている。

```python
# configuration_qwen3.py
self.headwise_attn_output_gate = True       # ヘッドごとのゲート
self.elementwise_attn_output_gate = False   # 要素ごとのゲート
```

たったこれだけの設定で有効になる。`modeling_qwen3.py`の`Qwen3Attention`クラス内で、SDPA出力後にSigmoidゲートが適用される。

Qwen3-Next-80B-A3B-Instructですでに採用されており、**1Mトークンのコンテキスト長**をサポートしている。実戦投入済みのテクノロジーだ。

## 7. 考察：なぜこれほどシンプルな変更が効くのか

個人的に最も興味深いのは、**TransformerのAttentionは長年「Softmax + Dot Product」で固定されてきたが、実はそこにボトルネックがあった**という事実だ。

Softmax Attentionを数式的に振り返ると：

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

これって結局、QとKの類似度行列でVを加重平均しているだけだ。加重平均は**アフィン変換（線形写像）**の一種で、出力は入力の凸結合に過ぎない。つまり、表現力的にはかなり制約がきつい。

そこにSigmoidゲートを挟むと、$O_{\text{gated}} = g \cdot O$ となり、$g \in (0, 1)$の要素ごとの乗算が加わる。これが**非線形性**をもたらし、単なる加重平均では表現できない情報の取捨選択を可能にする。

加えて、この非線形性は**スパース**に働く。大部分のゲートが0に近いということは、「このヘッドの出力は今のクエリにとってノイズだ」とモデルが判断している。これはAttentionの重みそのものを変えるのではなく、**「計算結果として使うか捨てるか」を事後的に決定する**メカニズムで、Softmaxの確率分配の枠組みとは独立に機能する。

## 8. 関連研究との位置づけ

**Attention Sinkの先行研究**：
- Shi et al. (2023) 「Efficient Streaming Language Models with Attention Sinks」— Sink現象の発見とStreamingLLMの提案
- Xiao et al. (2023) 「DynaLLM」— Sink対策の動的削除

**Attentionの改善**：
- FlashAttention (Dao et al., 2022) — 計算効率の改善（本論文とはアプローチが異なる）
- Multi-Query Attention / Grouped Query Attention — K/Vの共有化による効率化
- 本論文は「表現力」の軸でAttentionを改善する

**ゲートメカニズムの系譜**：
- LSTM / GRU (1997, 2014) — リカレント構造におけるゲート
- Highway Networks (2015) — 残差接続へのゲート
- 本論文は「Attention出力へのゲート」という新しい適用箇所

## 9. 限界と今後の課題

いくつか気になる点も残る。

**スパース性のトレードオフ**：大部分のゲートが0に近いということは、計算結果の多くを捨てている。ハードウェア的には0を掛ける計算も実行されるので、計算量の削減には直結しない。専用のスパース演算器との組み合わせで初めて、メモリ帯域や電力消費の削減が見込めるかもしれない。

**他のアーキテクチャへの適用**：論文ではQwen系のDecoder-only Transformerでのみ検証されている。Encoder-Decoderモデルや、State Space Model（Mambaなど）、Linear Attentionへの応用は今後の課題だ。

**推論時のオーバーヘッド**：訓練時には無視できるレベルでも、推論時にSigmoidの計算がボトルネックになるケースがあるかもしれない。ただし、推論のボトルネックは通常メモリアクセスにあるので、実用上は問題ないだろう。

## 10. まとめ

NeurIPS 2025最優秀論文「Gated Attention」は、Transformerの最も基本的な構成要素に**一本のシンプルな道**を開いた。

- Sigmoidゲート一つで、Attention Sink、訓練不安定性、長文脈性能劣化の3つの問題を同時に緩和
- 30種の変体を体系的に比較し、設計選択の根拠を明確に示した
- すでにQwen3-Nextで実戦投入済み

LLMのアーキテクチャは、FlashAttention、RoPE、GQAといった積み重ねで進化してきた。Gated Attentionはその系譜に加わる、小さいが確実な一歩だ。

何より印象的なのは、**「新しい損失関数」でも「新しい正則化」でもなく、「一本のSigmoidゲート」でここまで変わる**という事実。Transformerのまだ見ぬポテンシャルを感じさせる論文だった。

---

## 参考

- 論文: [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) (NeurIPS 2025 Best Paper)
- コード: [github.com/qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention)
- モデル: [huggingface.co/QwQZh/gated_attention](https://huggingface.co/QwQZh/gated_attention)
- NeurIPS 2025 Blog: [blog.neurips.cc](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
