---
title: "CRPO: 対比学習が解くAgent RLにおける自蒸留の露出バイアス問題"
emoji: "🔀"
type: "tech"
topics: ["LLM", "Reinforcement Learning", "Agent", "Self-Distillation", "Contrastive Learning"]
published: false
published_at: "2026-08-01 10:00"
---

## TL;DR

- **CRPO（Contrastive Reinforced Policy Optimization）** は、On-Policy Self-Distillation（OPSD）に潜む**露出バイアス（exposure bias）**を対比学習の視点から解決する新しいAgent RL手法
- 自教師モデルが持つ特権情報のせいで、生徒は推論経路を模倣に偏らせてしまう問題を、**予測エントロピー差**で正例（反省的探索）と負例（露出バイアス）に分類し、InfoNCEで位置ごとの蒸留強度を制御
- 13の推論・深度検索ベンチマークで一貫した改善。Qwen3-8B + CRPO*はGAIAで**47.6%**を達成し、32Bモデルの直接推論（QwQ 15.5%、DeepSeek-R1-32B 14.2%）を**32.1ポイント**も上回る
- CRPO* = GRPO + \lambda・CRPO という構成。結果報酬の骨格はそのままに、正則化項を静的な参照モデルから「位置認識の対比的自蒸留」に置き換える

## 背景：Agent RLはなぜ自蒸留に頼るのか

LLMのポストトレーニングでは、近年、**RLVR（Reinforcement Learning with Verifiable Rewards）**と**OPSD（On-Policy Self-Distillation）**の二軸が支配的になっている。RLVRは最終結果の正誤を報酬にして強化学習する。シンプルで強力だが、報酬は軌道全体に対するスカラー1つだけなので、中間のツール使用や検索動作への信用割り当ては粗い。

一方、OPSDは同一モデルを生徒・教師の両方に使い、教師に特権情報（正解軌道や環境フィードバック）を与えることで、トークン単位の密な教師信号をゼロ追加コストで得ようとする。DeepSeek-R1系列の成功以降、この流れはますます加速している。

しかしOPSDには見落とされがちな欠点がある。本論文はそれを**2つの現象**として鮮明に描き出す。

**現象1：自教師の推論経路が収束する**
深度検索タスクでは、ツール呼び出し後に外部情報が流入すると生徒のエントロピーが急上昇する。不確実性が高い状況で、特権を持つ自教師は「自分の自信を高めるために参考軌道に近い推論路を選びやすくなる」。その結果、教師のエントロピーは不自然に下がり、生徒は「安全な模倣」に引きずられる。

**現象2：多ターン蒸留で最適化方向が曖昧になる**
単一ターンの数学問題ならKL Divergenceで教師を追従するのでも問題ないが、Agentのような長い対話では、外部情報の注入によって教師信号が不安定化する。OPSDの報酬曲線は揺らぎ、KLは急増する。生徒は「何を学べばいいのか」見失う。

要するに、OPSDの教師信号は「どの位置が信頼できるのか」という選別なしに使われている。これが露出バイアスの核心だ。

## 方法：OPSDを対比学習に再定式化する

CRPOの出発点はシンプルだ。「OPSDとは実は対比学習の一形態ではないか」という問いである。

### 問題設定

プロンプト$x$に対して、方策$\pi_\theta$がツールと対話しながら応答$y_i$を生成する。各位置$t$で、生徒と教師は異なる文脈を持つ。

- 生徒の文脈：$S_{i,t} = (x, y_{i,<t})$
- 教師の文脈：$T_{i,t} = (x, f, y_{i,<t})$

ここで$f$は、同じグループ内の参照成功軌道や、Pythonインタプリタの実行結果などの環境フィードバックを含む特権情報。

古典的なOPSDの損失は次の通り。

$$
\mathcal{L}_{\text{OPSD}}(\theta) = \frac{1}{G}\sum_{i,t}\text{KL}(\pi_\theta(\cdot|S_{i,t}) \| \text{sg}(\pi_\theta(\cdot|T_{i,t})))
$$

sgはstop-gradient。すべての位置で等しく教師を模倣するため、露出バイアスが乗り移る。

### 予測エントロピーによる正負分類

CRPOは、生徒と教師の予測分布のエントロピー差に注目する。

$$
\Delta E_{i,t} = E^{S}_{i,t} - E^{T}_{i,t}
$$

| 状況 | 解釈 |
|------|------|
| $E^S \ll E^T$ | 生徒の方が自信を持てない＝教師が「反省的な探索」を示している位置 |
| $E^S \approx E^T$ | 境界ケース |
| $E^S \gg E^T$ | 生徒の方が迷っているのに教師が収束している＝露出バイアスの疑い |

グループ内の全位置の$\Delta E_{i,t}$をソートし、下位$p$%を**正例集合$\mathcal{P}$**（教師の行動を信頼して蒸留すべき位置）、残りを**負例集合$\mathcal{P}^c$**（教師の影響を減らすべき位置）とする。

### InfoNCEによる位置認識の対比損失

相似度関数を負のKLと定義する。

$$
\text{sim}(i,t) = -\text{KL}(\pi_\theta(\cdot|S_{i,t}) \| \text{sg}(\pi_\theta(\cdot|T_{i,t})))
$$

そしてInfoNCE型の損失を構成する。

$$
\mathcal{L}_{\text{CRPO}}(\theta) = -\log \frac{\sum_{(i,t)\in\mathcal{P}}\exp(\text{sim}(i,t)/\tau)}{\sum_{(i,t)\in\mathcal{P}\cup\mathcal{P}^c}\exp(\text{sim}(i,t)/\tau)}
$$

この損失を最小化すると、正例位置では生徒-教師間のKLが縮まり、負例位置では広がる。つまり、**教師の「よい」部分だけを取り込み、「よくない」部分は意識的に遠ざける**。

### 理論的解釈

本論文の命題1は、$\nabla_\theta\mathcal{L}_{\text{CRPO}}$が以下の形に分解されることを示している。

$$
\nabla_\theta\mathcal{L}_{\text{CRPO}}(\theta) = \mathbb{E}_{y_i\sim\pi_\theta}\bigg[-\frac{1}{\tau}\sum_{i,t}c_{i,t} \cdot \mathbb{E}_{\hat{y}_t\sim\pi_\theta(\cdot|S_{i,t})}[\hat{A}_{i,t}\nabla_\theta\log\pi_\theta(\hat{y}_t|S_{i,t})]\bigg]
$$

ここで、$\hat{A}_{i,t} = \log \frac{\text{sg}(\pi_\theta(\hat{y}_t|T_{i,t}))}{\pi_\theta(\hat{y}_t|S_{i,t})}$はトークン単位の優位性、$c_{i,t}$はソフトマックス正規化された対比重みで、正例では$c_{i,t} \leq 0$、負例では$c_{i,t} \geq 0$となる。これはまさに「**対比重み付きのトークン単位方策勾配、付ソフトゲート**」である。

### CRPO*：GRPOとの統合

CRPOは単独でも動くが、結果報酬と組み合わせてより強力になる。

$$
\mathcal{L}_{\text{CRPO*}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) + \lambda\mathcal{L}_{\text{CRPO}}(\theta)
$$

GRPOのKL正則化項を、凍結参照モデルへの引き戻しから「自教師との位置認識対比」に置き換える構造だ。これにより、結果報酬による正誤の学習と、自蒸留による行動の学習が、お互いの弱点を補完し合う。

![CRPO framework](/images/crpo-contrastive-reinforced-policy-optimization/fig2.png)

## 実験：13ベンチマークでの一貫した優位

### 実験設定

- **モデル**：Qwen2.5-3B/7B-Instruct、Llama3.1-8B-Instruct、Qwen3-8B/14B
- **タスク**：
  - 数学・知識推論：AIME24/25、MATH500、GSM8K、MATH、HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle、WebWalker
  - 深度検索：GAIA、WebWalkerQA、Humanity's Last Exam、xbench-DeepSearch
- **ベースライン**：GRPO、ARPO、OPSD、SDPO、RLSD
- **訓練**：冷始動SFT（Tool-Star 54K + STILL 0.8K）→ RL
- **評価**：QAタスクはtoken-level F1、それ以外はQwen2.5-72BによるLLM-as-Judge

### 主要結果

#### 数学・知識推論（10タスク）

3つのバックボーンでCRPO*が平均値トップとなった。Qwen2.5-3BではCRPO*が54.9%で、GRPO（50.4%）を**+4.5ポイント**上回った。Llama3.1-8BではCRPO* 58.5%で、ARPO（55.3%）を**+3.2ポイント**上回った。Qwen2.5-7BではCRPO* 60.7%で、ARPO（58.3%）を**+2.4ポイント**上回った。

目を引くのはOPSDの脆弱さだ。Qwen2.5-3BではOPSDが26.1%で、ベースモデルと同水準。Qwen2.5-7Bでも34.3%に留まり、GRPOやARPOに大きく劣る。これは、OPSDの露出バイアスが数値として明確に現れていることを示す。

#### 深度検索（4タスク）

深度検索ではCRPO*の優位がさらに大きくなる。

| モデル | GAIA Avg | WWQA Avg | HLE | xBench Avg |
|--------|----------|----------|-----|------------|
| Qwen3-8B + GRPO | 32.0 | 29.0 | 7.9 | 20.0 |
| Qwen3-8B + ARPO | 38.8 | 30.5 | 7.3 | 25.0 |
| Qwen3-8B + CRPO* | **47.6** | **38.0** | **13.0** | **27.0** |
| Qwen3-14B + CRPO* | **50.5** | **45.5** | **13.9** | **36.0** |

Qwen3-8B + CRPO*はARPOに対してGAIAで**+8.8ポイント**、WWQAで**+7.5ポイント**の差をつけた。14BではGAIAで**+6.8ポイント**、WWQAで**+9.5ポイント**だ。報酬が最も希薄な長期タスクで、CRPOの位置認識信号が最も効いていると解釈できる。

![Deep Search results](/images/crpo-contrastive-reinforced-policy-optimization/fig3.png)

### 大規模モデルとの比較

CRPO*は、小規模モデルがはるかに大きなモデルを追い抜くという現象も示した。GAIA Averageで比較すると：

| モデル | GAIA Avg |
|--------|----------|
| Qwen3-32B (thinking) | 15.5% |
| DeepSeek-R1-32B | 14.2% |
| QwQ-32B | 15.5% |
| **Qwen3-8B + CRPO*** | **47.6%** |

8Bモデルが32Bの既存推論モデルを**32.1ポイント**も上回った。これは単なる「強化学習でよくなる」というレベルを超えて、Agentタスクにおける「訓練効率」と「推論時コスト」の両面で意味の大きい結果だ。

![Scaling comparison](/images/crpo-contrastive-reinforced-policy-optimization/fig4.png)

### サンプリング拡張分析

Pass@1からPass@5への拡張も調べた。Qwen3-14B + CRPO*では、GAIAでPass@1 50.5%からPass@5 **70.7%**（+20.2ポイント）、HLEで13.9%から**35.0%**（+20.4ポイント）、WWQAで45.5%から**68.9%**（+23.4ポイント）、xbenchで36.0%から**60.7%**（+24.7ポイント）へと拡張した。

これはCRPO*が多様で情報量の多いツール使用を促進し、有効なサンプリング空間を広げている証拠となる。

### 訓練ダイナミクス

Qwen3-8Bで100ステップの訓練を追跡したところ：

- **正解率**：CRPOとCRPO*が一貫して最上位
- **エントロピー損失**：CRPO/CRPO*はARPOに近い狭い帯を保ち、GRPOより低い
- **ツール呼び出し数**：CRPO/CRPO*はARPOよりわずかに多いが、GRPOやOPSDより大幅に少ない
- **KL損失**：OPSDは60〜100ステップで急激なエントロピー崩壊を示し、生徒が自教師の偏りに引きずられていることを示唆

### ハイパーパラメータ感度

正例比率$p$と対比重み$\lambda$を変えたときの挙動は、CRPOの設計が本質的であることを裏付ける。

- **$p=30\%$**でピーク。$p \to 100\%$にすると、すべての位置が正例になりCRPOはOPSDに退化する。実際に各ベンチマークで大きく劣化した。
- **$\lambda=5$**でピーク。$\lambda \to 0$にすると対比正則化が消え、CRPO*はGRPOに退化する。これも各ベンチマークで劣化した。

つまり、**正例-負例の区別**も、**自教師との対比カップリング**も、どちらも不可欠であることが実験的に確認された。

## 考察

CRPOの面白さは、単に「対比学習をOPSDに混ぜた」という技術的寄与にとどまらない。それは、Agent RLにおける「教師信号の質」の問題を、**情報量（エントロピー）の観点から再定義**した点にある。

### なぜエントロピー差が効くのか

生徒の分布が一様に近く、教師の分布が尖っている位置では、教師は「特権情報に基づいて自信を持って選んだ1つの行動」を示している。これが正しい推論の手がかりなら模倣すべき（正例）。しかし同じ状況で、もし教師が単に参考軌道に引きずられているだけなら、それは「露出バイアス」に過ぎない（負例）。

CRPOはこの二者を区別するために、グループ内相対の$\Delta E_{i,t}$を使う。絶対的なエントロピー値ではなく、同じ問題に対する複数軌道の中での相対位置に着目することで、タスクやモデルに依存しにくい頑健な分類が可能になっている。

### 結果報酬との関係性

CRPO*がGRPOと正交であることも重要だ。GRPOは軌道レベルで「正解か不正解か」を学ぶ。CRPOはトークンレベルで「どう推論すべきか」を学ぶ。前者はクレジット割り当ての粗さを補いきれないが、後者は結果の正誤を直接扱わない。CRPO*はこの2つを組み合わせることで、お互いの得意領域を活かしている。

### 限界と今後の方向

本論文の分析は主に数学・知識QA・深度検索に集中している。ツールの種類や環境の複雑さがさらに増すオープンエンドなタスク、例えばOS上でのコンピュータ利用やマルチエージェント協調では、$f$の設計やエントロピー推定の安定性が新たな課題になる可能性がある。

また、$p=30\%$や$\lambda=5$が最適だったが、この値がモデルサイズやタスク複雑さにどう依存するかは完全には解明されていない。例えば、非常に長い軌道では「負例」の割合を増やすべきか、それとも減らすべきかは議論の余地がある。

## 関連研究

**自蒸留・蒸留系列**
- **OPSD / OPD**：特権付き教師によるlogitレベル蒸留の基盤
- **SDPO**：教師-学生KLを密な報酬としてRL目標に組み込む
- **RLSD**：GRPOの方策勾配を自蒸留信号で変調
- **DemoPSD**：不一致度に基づくPolicy Self-Distillation
- **BIRD**：冷始動問題をSFT+逆KLで解決する自己推論蒸留

**Agent RL系列**
- **GRPO / DAPO / GSPO**：軌道レベル・シーケンスレベルの重要性サンプリングを議論
- **ARPO**：ツール呼び出し後のエントロピー上昇を利用した適応的探索
- **BPO**：分岐木ロールアウトによる分散削減
- **Harness-G**：検索Agent向けのグラフ構造化harness

**対比学習系列**
- **InfoNCE**：表現学習の対比損失。CRPOは出力logitを「表現」、負のKLを「類似度」として適用
- **RLCSD**：推論タスクで成功・失敗軌道のスタイル差を対比信号にする先行研究。CRPOは位置レベルのエントロピー差を使う点で異なる

## まとめ

CRPOは、Agent RLにおけるOPSDの露出バイアス問題を、対比学習の言葉で整理し、予測エントロピー差という自然な指標で解決した。理論的には、それがソフトゲート付きのトークン単位方策勾配と等価であることが示されている。実験的には、13のベンチマークで一貫した改善を達成し、8Bモデルが32B推論モデルを大きく凌駕する結果も示した。

個人的に印象に残ったのは、**「正例だけを蒸留し、負例は遠ざける」という直感的なアイデアを、InfoNCEという既存の枠組みで自然に実装している点**だ。これにより、OPSDの弱点を過剰に複雑な仕組みで補うことなく、位置認識の密な教師信号を獲得できている。

特に長期の深度検索タスクでの伸びが大きいことは、Agent RLの次のフロンティアが「結果報酬をどう密集化するか」だけでなく、「どの信号を信じてどの信号を無視するか」という**選択的蒸留**の領域にあることを示唆している。

## 参考

- Wu, X., Liu, J., Liu, X., Zhu, X., Wang, J., Guo, L., Li, X., Cao, X., & Cai, X. **"Contrastive Reinforced Policy Optimization via Privileged Self-Distillation."** arXiv:2607.28026 [cs.LG], July 2026.
- DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.
- Guo, S., Zhang, B., Liu, J., et al. "R1-Zero's "Aha Moment" in Visual Reasoning on a 3B Budget." arXiv:2503.05132.
- Zhao, Y., Pang, T., Du, C., et al. "Bayesian Self-Distillation for Sampling Efficient Zero-Shot Reasoning." ICLR 2026.
- Hübotter, J., Saunshi, N., Zhao, Y., et al. "Self-Distillation Improves Reasoning in Language Models." ICML 2026.
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. "Visual Instruction Tuning." NeurIPS 2023.
- Shao, Z., Wang, P., Zhu, Q., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
