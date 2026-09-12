---
title: "捨ててはいけない──Layer DropoutがLLMの訓練効率と推論弾性を同時に開く仕組み"
emoji: "🧱"
type: "tech"
topics: ["LLM", "Layer Dropout", "推論高速化", "訓練効率化", "自己投機デコード"]
published: false
---

## TL;DR

大規模言語モデル（LLM）の事前学習からdropoutが消えつつある。激活dropoutは大規模単パス訓練で精度を劣化させるという観測が広まったためだ。だが本論文は、**layer dropout**（層単位のstochastic depth）は別物であることを2400本超の訓練実験で証明した。増加層分布（ILD）＋漸減時刻スケジュール（DTS）の組み合わせにより、訓練FLOPsを最大25%削減しつつ検証Lossを稠密基線と同等かそれ以下に抑え、さらに訓練時に獲得した深さ方向の頑健性が零样本の早期退出・層スキップ・自己投機デコードを可能にし、最大**1.55×**の推論高速化を実現する。ICML 2026採録。

> **Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference**
> Mostafa Elhoushi, Alex Pretko, Nolan Dey, Bin Claire Zhang, Gavia Gray, Gurpreet Gosal, Abdulrahman Mahmoud, Shane Bergsma, Joel Hestness
> arXiv:2609.05275 / ICML 2026

![増加層分布の概念図](/images/dont-drop-dropout-layer-sparsity-efficient-llm/fig1.png)
*図1: 増加層分布（ILD）では、浅い層ほどスキップ確率が低く、深い層ほど高くなる。各層の色の濃さがドロップアウト確率を表す。*

## 背景

### DropoutはなぜLLMから消えたのか

Transformer登場当初、dropoutは標準的な正則化手法として組み込まれていた。しかしGPT-3やOPTを除けば、PaLMは微調整時にのみdropoutを使い、LLaMA系ではdropoutが明示的に除外されている。背景には二つの観測がある。

1. **大規模単パス訓練では過学習が起こりにくい**——万亿tokenを1回しか見ない設定では、古典的な正則化の必要性が薄い。
2. **激活dropoutは精度を劣化させる**——Liu et al. (2025)が、大規模事前学習において激活dropoutがLossを悪化させることを報告。

これらから「大規模LLMにdropoutは不要」という認識が定着した。だが本論文は、**layer dropoutと激活dropoutを同一視することに問題がある**と指摘する。Layer dropoutは層全体をスキップする構造化スパース性であり、激活dropoutとは異なる正則化効果と計算削減をもたらす。

### Layer Dropoutの独自の位置づけ

Layer dropoutは残差接続内でTransformer block全体を確率的にバイパスする。スキップされた層の計算を丸ごと省略できるため、FLOPs削減がほぼ線形に効く。さらに、訓練時に「層が欠けても推論できる」体験をモデルに強いるため、推論時の深さ方向最適化（早期退出や層スキップ）に対する頑健性が自然に備わる。これは、量化認識訓練（QAT）が推論時の低精度にモデルを適応させるのと同じ発想だ。

## 方法詳解

### 定式化

標準的な残差層は次式で表される。

$$\mathbf{H}^{\ell+1,t} = \mathbf{H}^{\ell,t} + f^{\ell}(\mathbf{H}^{\ell,t})$$

Layer dropoutを導入すると、訓練時は次のようになる。

$$\mathbf{H}^{\ell+1,t} = \mathbf{H}^{\ell,t} + r^{\ell,t}_{\text{train}} \mathbf{M}^{\ell,t} f^{\ell}(\mathbf{H}^{\ell,t})$$

ここで $\mathbf{M}^{\ell,t} \sim \text{Bernoulli}(1-p^{\ell,t})$ はマスク、$r_{\text{train}}$ は訓練時スケーリング因子。マスクが0のシーケンスではその層の計算を丸ごとスキップできるため、訓練FLOPsが平均ドロップアウト率に比例して削減される。

### スケーリング因子の選択——ここが従来の落とし穴

各フレームワークで $r_{\text{train}}$ と $r_{\text{eval}}$ の取り方がバラバラだった。PyTorchとTensorFlowは $r_{\text{train}}=1/\rho$、Stochastic Depthの原論文は $r_{\text{train}}=1$、fairseqのLayerDropも $r_{\text{train}}=1$ である。

本論文の重要な理論的貢献は、**最大残差流更新の期待**（Expectation 1）を導入したことだ。各残差ブロックの重みが $O(1/L)$ の寄与をするという仮定のもと、$r_{\text{train}}=1/\rho$ のみがこの期待を満たし、ドロップアウト率が変わっても最適超パラメータをそのまま転用できることを示した。一方、$r_{\text{train}}=1$ では率ごとに超パラメータを再調整しなければならない。実験でも $1/\rho$ の優位性が確認されている。

### ドロップアウト粒度

**モデル粒度**として、SubLayer Dropout（AttentionとFFNに独立マスク）とLayer Dropout（層全体でマスク共有）を比較。結果、Layer Dropoutが一貫して優位だった。AttentionとFFNの協調的スキップが、独立スキップより精度面で有利という解釈。

**テンソル粒度**としては、per-sequence（各シーケンスで独立サンプリング）がper-batchを上回る。より細かい粒度のスパース性が精度に寄与する。

### 層分布——どこを多くスキップするか

三つの分布を比較した。

| 分布 | 定義 | 平均ドロップアウト率 |
|------|------|-------------------|
| Uniform | $p^\ell = p_{\max}$ | $p_{\max}$ |
| ILD (Increasing Layer Dist.) | $p^\ell = \frac{\ell}{L-1} \cdot p_{\max}$ | $0.5 \cdot p_{\max}$ |
| ALD (Alternating Layer Dist.) | $p^\ell = p_{\max}$ if $\ell$ is odd | $\approx 0.5 \cdot p_{\max}$ |

ILDは浅い層を温存し、深い層ほど高確率でスキップする。ALDは奇数層のみスキップする。同一FLOPs削減率で比較すると、**非一様分布（ILD・ALD）が一様分布を一貫して上回る**。小規模モデルではALDがやや有利だが、規模が大きくなるとILDの優位性が顕著になる。

### 時刻スケジュール——いつドロップアウトを強めるか

三つのスケジュールを検討した。

- **Constant**: $p^{\ell,t} = p^\ell$（全ステップで一定）
- **ITS (Increasing Time Schedule)**: $p^{\ell,t} = p^\ell \cdot \frac{t}{T-1}$（徐々に強める）
- **DTS (Decreasing Time Schedule)**: $p^{\ell,t} = p^\ell \cdot (1 - \frac{t}{T-1})$（徐々に弱める）

![時刻スケジュールとFLOPs削減の関係](/images/dont-drop-dropout-layer-sparsity-efficient-llm/fig2.png)
*図2: 左——3つの時刻スケジュールの比較。DTSは訓練初期に高く後期に低下する。右——FLOPs削減率と検証Loss変化率の関係。ILD+DTSの5%削減設定で、Lossを稠密基線以下に改善できる（緑色バー）。*

**DTSが三つの中で最優**という結果は直感に反するかもしれない。筆者の仮説は次の通り。訓練初期の高いドロップアウトは重み空間の広い探索を促し（バイアス削減）、後期の低下で最適解へ安定的に収束させる（バリアンス削減）。これは**確率的モデル成長**とも解釈できる——有効容量（実質的な深さ）が訓練とともに滑らかに増加し、明示的な再初期化なしでカリキュラム学習的な効果を得られる。

## 実験結果

### 同FLOPsでのLoss比較——dropoutありの方が良いことすらある

ILD+DTSの組み合わせで5%のFLOPs削減を行うと、906Mモデルでは検証Lossが稠密基線より**0.065%低い**値を達成。つまり計算を削減しながら精度まで向上する設定が存在する。10%削減でも906MモデルのLoss劣化はわずか0.10%。

### 大規模実験——億パラメータ級での検証

1.8B〜8.2Bパラメータのモデルで、$p_{\max}$ を0.6〜0.99まで漸増させた激しい設定を試行。

| モデル | $p_{\max}$ | FLOPs削減 | 検証Loss (dropout) | 検証Loss (dense) |
|--------|-----------|----------|-------------------|-----------------|
| 1.8B | 0.6 | 15% | **1.836** | 1.849 |
| 3.9B | 0.8 | 20% | 1.745 | **1.732** |
| 8.2B | 0.99 | 25% | **1.663** | — |

1.8Bと8.2Bでは、dropout訓練の方がLossが低い。3.9Bではやや劣るが、差は0.013点と微小。重要なのは、3.9Bモデルで訓練開始時の有効深さが $0.6L$ にまで縮小し、最終層は80%の時間スキップされるという極端な設定でも、稠密基線と遜色ない結果が得られたことだ。**大規模モデルほど高いドロップアウト率に耐性がある**。

![大規模実験の結果](/images/dont-drop-dropout-layer-sparsity-efficient-llm/fig3.png)
*図3: 左——検証Lossの比較。中——自己投機デコードの高速化倍率。右——代替層スキップ時のLossで頑健性を評価。全指標でdropout訓練モデルが稠密基線を大きく上回る。*

### 推論最適化——訓練で蒔いた種が推論で花開く

#### 零样本の早期退出と層スキップ

稠密基線では1層でも早期退出するとLossが急激に悪化する。一方、layer dropoutで訓練したモデルは、複数層の退出やスキップ後もLossが安定。3.9Bモデルで代替層（奇数層）をスキップしたとき、稠密基線のLossは6.446に跳ね上がるが、dropout訓練モデルは2.129に抑える。これは3倍の差だ。

#### 自己投機デコード

訓練時に高いドロップアウト率を使ったモデルほど、自己投機デコードの高速化倍率が大きい。8.2Bモデルで**1.55×**の推論高速化を達成。稠密基線では自己投機デコードの恩恵がほとんどない（1.02〜1.10×）ことと対照的だ。層スキップに対する頑健性が、投機草稿モデルの受け入れ率を底上げする仕組みである。

## 考察

### ILD vs ALD——目的で使い分ける

ILDは基礎精度と早期退出の頑健性に優れるが、非連続スキップでは劣る。ALDは代替層スキップで最も優雅に劣化するが、早期退出では崩壊する。実践者はデプロイ要件に応じて選ぶべきだ。著者らの推奨はILD+DTS。

### DTSはなぜ効くのか——二つの読み方

一つ目は**確率的モデル成長**。有効深さが訓練の進行とともに増加し、小さなモデルから始まって大きなモデルへ成長するカリキュラム効果。二つ目は**探索・利用の切り替え**。初期の高ドロップアウトは重み空間の探索を広げ、後期の低下で収束品質を高める。

### スケーリング因子の重要性——「超パラメータくじ」を避ける

$r_{\text{train}}=1$ を使うと、ドロップアウト率ごとに学習率・バッチサイズ・重み減衰を再調整しなければならない。これは探索空間を率の数だけ増やすことで、実験設計を著しく困難にする。$r_{\text{train}}=1/\rho$ はこの問題を回避し、一つの超パラメータ設定を全率に適用できる。これまでのlayer dropout研究が混合結果を報告してきた背景には、このスケーリング因子の不統一があった可能性が高い。

### 大規模ほどドロップアウトに強い

$p_{\max}$ の許容値がモデル規模とともに増加する（1.8B→0.6, 3.9B→0.8, 8.2B→0.99）。これは、パラメータの冗長性が規模とともに増し、個別層の寄与が相対的に小さくなるためと考えられる。

## 関連研究

- **Stochastic Depth** (Huang et al., 2016): CNN向けに提案。本論文はLLMへの本格的なスケーリング分析を初めて行った。
- **LayerDrop** (Fan et al., 2020): BERT向け。ALDを採用し、訓練後剪定を報告。LLMでの大規模評価はなかった。
- **Progressive Layer Dropout** (Shen et al., 2022): ITSスケジュールを提案。本論文はDTSがITSを上回ることを示した。
- **LayerSkip** (torchtune): $r_{\text{train}}=1$ とper-sequenceを採用。本論文は $1/\rho$ の優位性を理論・実験の両面で示した。
- **Mixture-of-Depths** (Raposo et al., 2024): 学習可能な深さルータ。Layer dropoutは追加パラメータなしで同等の効果を達成する。
- **Once-for-All / MatFormer / Nemotron-Elastic**: 明示的な弾性アーキテクチャ。Layer dropoutはアーキテクチャ変更なしで弾性を誘導できる点が異なる。

## まとめ

Layer dropoutは、LLMの事前学習において訓練効率と推論弾性を同時に開く、見過ごされてきた強力なレバーである。増加層分布（ILD）＋漸減時刻スケジュール（DTS）という推奨設定は、既存のアーキテクチャや訓練スタックに非侵入的で、量化や蒸留など他の最適化とも直交する。2400本超の訓練実験が裏付けた要点は次の三点。

1. **訓練FLOPs最大25%削減**で、検証Lossを稠密基線と同等以下に維持。
2. **推論最大1.55×高速化**——自己投機デコードの受け入れ率が、訓練時のドロップアウトが誘導した層スキップ頑健性によって押し上げられる。
3. **モデル規模が大きいほど高いドロップアウト率に耐性**があり、$p_{\max}=0.99$ の極端な設定でも8.2Bモデルは安定して収束する。

![推奨設定の全体像](/images/dont-drop-dropout-layer-sparsity-efficient-llm/fig4.png)
*図4: Layer Dropoutの推奨設定サマリー。スケーリング因子・粒度・層分布・時刻スケジュールの四要素が組み合わさって、訓練効率化と推論高速化を同時にもたらす。*

より広い文脈では、本論文の成果は「訓練中に有効容量を漸増させる」というモデル成長パラダイムの一实例でもある。幅・量子化幅・非構造化スパースなど他の次元にもこの発想は拡張可能だ。Layer dropoutを再評価することは、LLMの訓練レシピに柔軟性を取り戻す第一歩となるだろう。

## 参考

- Elhoushi, M. et al. "Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference." ICML 2026. [arXiv:2609.05275](https://arxiv.org/abs/2609.05275)
- Huang, G. et al. "Deep Networks with Stochastic Depth." ECCV 2016.
- Fan, A. et al. "Reducing Transformer Depth on Demand with Structured Dropout." ICLR 2020.
- Liu, Z. et al. "How to Train Your LLM: A Practitioner's Guide." 2025.
- Shen, Z. et al. "Progressive Layer Dropout: Unifying Uncertainty Regularization and Computation Acceleration." 2022.
- Raposo, R. et al. "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-based Language Models." 2024.
