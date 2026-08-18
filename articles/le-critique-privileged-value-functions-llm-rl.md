---
title: "Le Critique: LLM強化学習における価値関数の復権 — PVFとTETHER"
emoji: "🎭"
type: "tech"
topics: ["LLM", "強化学習", "価値関数", "GRPO", "信用割当て"]
published: true
---

## TL;DR

LLMの強化学習において、GRPOを代表とするcritic-free手法が主流となって久しい。理由は明確で、価値関数（critic）の学習は不安定であり、インフラコストもかさむ。しかしcritic-free手法には本質的な限界がある——トークンレベルの信用割当てができないことと、グループサンプリングに伴うstraggler問題だ。

本論文は「価値関数をRLパイプラインに戻す」ための二つの相補的戦略を提案する。**Privileged Value Functions（PVF）**は、criticにポリシーが直接使えない特権情報（正解や他のロールアウトの報酬など）を条件付け、分散を削減しつつ勾配の不偏性を保つ。**TETHER**は、学習初期のcriticの不正確さをLOO group baselineとの適応的補間で補い、タスク固有のハイパラ調整なしに安全にcriticを活用する。Qwen3-4Bを用いた4タスク（Reasoning Gym、CodeIO、Sudoku、MiniF2F）の実験で、PVFは全タスクで最高性能を記録し、TETHERは一貫して標準VFを上回った。

![4つのbaseline戦略の比較](/images/le-critique-privileged-value-functions-llm-rl/fig1.png)
*図1: GRPOのgroup-mean baseline、標準価値関数、PVF、TETHERの比較。PVFは特権情報を追加し、TETHERはgroupとvalueを適応的に混合する。*

---

## 背景: GRPOの台頭と価値関数の衰退

### Critic-free手法が勝った理由

DeepSeek-R1以降、LLMのRLVR（Reinforcement Learning with Verifiable Rewards）においてGRPOが事実上の標準となった。GRPOの核心は単純だ——各プロンプトに対してK個のロールアウトを生成し、その平均報酬をbaselineとして各ロールアウトのadvantageを計算する：

$$A_i^{\text{GRPO}} = R_i - \bar{R}, \quad \bar{R} = \frac{1}{K}\sum_{j=1}^{K} R_j$$

この方式が好まれた理由は三つある。第一に、criticを訓練する必要がない——インフラが単純で、バグの入り込む余地が少ない。第二に、GRPOのbaselineは常に「そのプロンプトの他のロールアウト」との相対評価なので、タスクの難易度に自動的に適応する。第三に、実証的に強力だった——特に数学推論タスクで顕著な性能を示した。

### しかし、限界がある

GRPOのadvantageは**シーケンスレベル**であり、同一プロンプトの全トークンに同じ値が割り当てられる。長い推論チェーン（数千〜数万トークン）において、どのトークンが最終的な成功に寄与したかを区別できない。これは「multi-armed bandit」的な最適化であり、きめ細かい信用割当てを放棄している。

第二に、K個のロールアウトが必要であるため、**straggler問題**が発生する——ある1つのロールアウトが極端に長い場合、グループ全体の処理がそれを待たねばならず、スループットが低下し、off-policynessが増大する。

価値関数は理論的にこれらの問題を両方解決する。トークンレベルのadvantageを提供し、K=1でも動作する。ではなぜ使われないのか。criticの学習が難しいからだ。初期のcriticは不正確で、GRPOのgroup baselineよりも悪いbaselineを生成してしまう。これが「価値関数は使えない」という実践的判断を導いた。

本論文は、この判断を覆す二つのアプローチを提示する。

---

## 方法詳解

### 1. Privileged Value Functions (PVF)

#### 核心アイデア

PVFの発想は直感的だ。価値関数が不正確なのは、使える情報が足りないからだ。ポリシーが直接アクセスできない情報——正解、他のロールアウトの結果、検証器の詳細——をcriticに「特権情報」として与えれば、より正確な価値推定が可能になる。しかも、この特権情報はポリシーの最適化目標を歪めない。

形式的には、PVFは次のように定義される：

$$V^{\pi}(h_{i,t}, z_{i,t}) := \mathbb{E}_{\tau_i \sim \pi}[R_i \mid h_{i,t}, z_{i,t}]$$

ここで $h_{i,t} = (x_i, y_{i,<t})$ はポリシーが観測できる状態、$z_{i,t}$ は特権情報である。advantageは：

$$\hat{A}^{\text{PVF}}_{i,t} = R_i - V_{\phi}(h_{i,t}, z_{i,t})$$

#### 不偏性の保証

PVFがポリシー勾配を偏らせないための条件は、特権情報 $z_{i,t}$ が現在のトークン $y_{i,t}$ と**条件独立**であること：

$$z_{i,t} \perp\!\!\!\perp y_{i,t} \mid h_{i,t}$$

これは「特権情報は、ポリシーがすでに知っている履歴 $h_{i,t}$ を通じてしか現在のトークンに影響しない」という意味だ。

**許容される特権情報：**
- 正解・参考解答（数学の正解、証明の骨格、コードの修正パッチ）
- Leave-One-Out group（同じプロンプトの他K-1個のロールアウトとその報酬）
- 検証器の採点ルーブリック

**許容されない特権情報：**
- 現在のロールアウトの未来トークン
- 実現済みの報酬
- その後の環境フィードバック

![PVFの情報フローと条件独立性](/images/le-critique-privileged-value-functions-llm-rl/fig2.png)
*図2: PVFの仕組み。特権情報zがcriticに追加条件として与えられ、条件独立性を満たす限り勾配の不偏性が保たれる。分散はより豊かな条件付けにより常に減少する。*

#### 分散削減の理論保証

条件付けが豊かになるほど、baselineの分散は単調に減少する：

$$\mathbb{E}\left[\left(R_i - \mathbb{E}[R_i \mid h_{i,t}, z_{i,t}]\right)^2\right] \leq \mathbb{E}\left[\left(R_i - \mathbb{E}[R_i \mid h_{i,t}]\right)^2\right]$$

これは条件期待値の性質から直接導かれる。つまり、PVFの特権情報は理論上、baselineを改善するか同等にするしかない——悪化することはない。

#### Self-Distillationとの違い

PVFは自己蒸留（SDPOやOPSDなど）と一見似ているが、本質的に異なる。自己蒸留は新しいKLマッチング目標を導入し、ポリシーの最適解を変える。PVFはRLの元目標を変えず、単にbaselineの品質を向上させる。$\lambda_{\text{GAE}}=1.0$（MC目標）の設定では完全に不偏であり、ハイパラ感受性も低い。

### 2. TETHER: 適応的Group-Value基線

#### 動機

PVFでcriticの品質は向上するが、学習初期のcriticは依然として不正確だ。純粋な価値関数baselineを使うと、初期段階でポリシーが誤った方向に更新され、連鎖的な性能劣化を引き起こす。

TETHERは、LOO group baselineとトークンレベル価値関数を**適応的に線形補間**する：

$$b^{\text{TETHER}}_{i,t} = (1-\rho) b^{\text{LOO}}_i + \rho V_{i,t}$$

- $\rho = 0$：純粋なLOO group baseline（GRPOと等価）
- $\rho = 1$：純粋な価値関数baseline
- 中間値：両者の利点を組み合わせる

#### 二段階更新プロトコル

TETHERの肝は、$\rho$を**現在のバッチの報酬で自分自身に適合させない**ことだ。これをやると不偏性が壊れる。代わりに二段階プロトコルを採用する：

1. **Phase 1（ポリシー訓練）**: 前のバッチで得た $\rho_{k-1}$ を使って現在のバッチ $\mathcal{B}_k$ のadvantageを計算し、ポリシーを更新する
2. **Phase 2（$\rho$ 適合）**: $\mathcal{B}_k$ の報酬を使って新しい $\hat{\rho}_k$ を適合させる：

$$\hat{\rho}_k = \arg\min_\rho \sum_{(i,t) \in \mathcal{B}_k} \left(R_i - b^{\text{TETHER}}_{i,t}(\rho)\right)^2$$

3. **EMA平滑化**: ノイズを減らすため、指数移動平均で平滑化する：

$$\rho_k = d \cdot \rho_{k-1} + (1-d) \cdot \hat{\rho}_k, \quad d = 0.95$$

更新された $\rho_k$ は**次の**バッチ $\mathcal{B}_{k+1}$ で使われる。現在のバッチには使わない。この一貫した「時間差」が不偏性を保証する。

![TETHERの二段階更新とrhoダイナミクス](/images/le-critique-privileged-value-functions-llm-rl/fig3.png)
*図3: TETHERの二段階プロトコル（左）と、タスクごとのrho収束ダイナミクス（右）。全タスクがrho=0（LOO group）から出発し、訓練の進行に伴いcriticが改善するとrhoが上昇する。Sudokuは最も高いrhoに収束する。*

#### TETHERはPVFの特例

TETHERは実はPVFの簡単な形として理解できる。LOO groupの報酬 $\{R_j\}_{j \neq i}$ が軌跡 $i$ の特権情報として働き、それを平均でスカラー化して $b^{\text{LOO}}_i$ とし、トークン価値と線形結合する。学習するパラメータは $\rho$ だけだ。

#### EVPOとの比較

EVPOは「criticのexplained varianceが正なら使う、負ならgroup meanに戻す」というハードスイッチを行う。TETHERは連続的な平滑補間を行うため、理論的に優れた分散-バイアストレードオフを持つ（Appendix Cで証明）。

---

## 実験結果

### 設定

| 要素 | 詳細 |
|------|------|
| モデル | Qwen3-4B-Instruct-2507（MiniF2FのみQwen3.5-4B） |
| タスク | Reasoning Gym (K=1, K=8)、CodeIO (K=4)、Sudoku (K=4)、MiniF2F |
| Baseline | Mean (GRPO)、VF (標準価値関数)、Pvf、Tether |
| 特権情報 | RG: 正解、CodeIO: LOO group、Sudoku: 完全解盤 |
| 価値関数 | base policyのコピー + ランダム初期化value head、warmup 20 steps |
| $\lambda$ | $\lambda_{\text{target}} = \lambda_{\text{GAE}} = 1.0$（MC目標、不偏advantage） |
| インフラ | PRIME-RLベースの非同期訓練（prime-values） |

### PVFの結果

**PVFは全4タスク設定で最高性能を記録した。**

![実験結果の比較](/images/le-critique-privileged-value-functions-llm-rl/fig4.png)
*図4: PVF（左）は全タスクで最高。TETHER（右）は全タスクでVFを上回り、多くのタスクでMeanを上回る。*

各タスクの特徴的な結果：

- **Reasoning Gym (K=1)**: group baselineが使えない（K=1）状況で、PVFはVFを上回る。特権情報（正解）が価値推定を大幅に改善。
- **Reasoning Gym (K=8)**: group baselineが強力なため改善幅は小さいが、PVFは依然としてMeanとVFを上回る。
- **CodeIO (K=4)**: PVFはLOO group（タスク固有情報なし）のみを使うが、Meanを超越し、VFとの差は訓練の進行とともに拡大する。
- **Sudoku (K=4)**: 改善が最も大きい。多ターン長horizonタスクでは、特権情報（完全解盤）の価値が最大化される。

#### Explained Variance（EV）との相関

PVFは全環境でVFより多くの報酬分散を説明する。そしてEVの改善幅と最終的な報酬改善幅には**正の相関**がある——SudokuでEV差が最大で報酬改善も最大、Reasoning Gym K=8でEV差が最小で報酬改善も最小だ。

### TETHERの結果

**TETHERは全4タスクでVF baselineを一貫して上回った。**

- Reasoning Gym、MiniF2F：TETHERはMean（GRPO）をも上回る
- CodeIO：TETHERはMeanと同等
- Sudoku：TETHERはVFの大幅な劣化を防ぐが、Mean水準までは回復しない

#### $\rho$ の適応ダイナミクス

全実行が $\rho=0$（純粋なLOO baseline）から出発する。これは設計通り——学習初期のcriticは信用できない。訓練の進行に伴い $\rho$ は上昇し、criticの改善を反映する。

注目すべきは、$\rho$ の収束値が**タスクに強く依存**することだ。SudokuはVF baselineの性能が最も弱いにもかかわらず、最も高い $\rho$ に収束する。これは一見矛盾するが、次のように説明できる：訓練初期に不正確な価値関数に頼ると初期ポリシー学習が阻害され、連鎖的な性能劣化が起きる。TETHERは初期にgroup baselineに依存することでこの失敗モードを回避し、結果としてcriticが改善した後に高い $\rho$ に到達できる。

### λチューニングの発見

本論文のもう一つの重要な発見は、$\lambda_{\text{GAE}}$ の微調整だ。$\lambda=1.0$（完全MC目標）の近傍に極めて重要だが見逃されやすい区間がある：

$$\lambda = 0.999888$$

この値は、8192長の応答の最初のトークンが約40%の不偏終端信号を保持するように選ばれる——$\lambda = 0.4^{1/8192} \approx 0.999888$。このわずかな減衰が、VFとPVFの報酬を顕著に向上させる。

先行研究（DeepSeek-R1など）は $\lambda=0.95$ と $\lambda=1.0$ の比較にとどまっていたが、この狭い区間を粗い探索で見逃していた可能性が高い。

---

## 考察

### 価値関数「が」悪かったのではなく、使い方が悪かった

本論文の最も重要な含意は、「価値関数自体が問題だったのではなく、criticの初期の不正確さと特権情報の欠如が問題だった」という再解釈だ。PVFで情報を補強し、TETHERで初期の不安定性を緩和すれば、価値関数はGRPOに匹敵するかそれ以上の性能を発揮する。

### PVFとTETHERの補完性

二つの手法は異なる問題を解決する。PVFはcriticの**品質**を向上させ、TETHERは**不完全なcriticの安全な利用**を実現する。両者は直交しており、組み合わせて使うことができる。

### LOO baselineは「訓練不要のPVF」

LOO group baseline $b^{\text{LOO}}_i = \frac{1}{K-1}\sum_{j \neq i} R_j$ は、特権情報（他のロールアウトの報酬）を使うPVFの特例と見なせる。TETHERはこれをtoken価値と組み合わせることで、訓練不要の特権情報と訓練済みcriticの最適なブレンドを実現する。

### GRPOとRLOOの等価性

論文は付録でGRPOのbaselineが持つ「バイアス」が実際には定数スケーリングに過ぎないことを示す：

$$A_i^{\text{GRPO}} = \frac{K-1}{K} A_i^{\text{LOO}}$$

つまりGRPOのbaselineはRLOOを定数倍したものに過ぎず、最適化に本質的な影響を与えない。「GRPOのバイアス」は長らく懸念されてきたが、実は問題にならない。

### 制限事項

- **スケール**: 実験は4Bモデルのみ。長horizon agentic訓練へのスケーリングは未検証。
- **インフラコスト**: 価値関数の推論と訓練のGPUオーバーヘッドは無視できず、GRPOとの厳密な計算量マッチングは行われていない。
- **PVFの限界**: 完全な応答からの回顧的フィードバックは使えない（条件独立性違反）。自己蒸留はこの制約を持たない。

---

## 関連研究

- **GRPO / RLOO** (DeepSeek-R1, Yu et al. 2025): critic-free手法の標準化。本論文はその限界（シーケンスレベル信用割当て、straggler問題）を指摘。
- **EVPO** (He et al. 2025): explained varianceに基づくgroup/valueのハードスイッチ。TETHERは連続的補間で理論的に優位。
- **Self-Distillation** (SDPO, OPSD系): 教師の特権情報をtokenレベルで注入するが、ポリシー目標を変える。PVFは目標を変えずbaselineのみ改善。
- **CrEST** (Wang et al. 2026): 階層型信用割当てでverifier-bounded ceilingを保持しつつtokenレベル信号を統合。PVFは異なる角度（特権情報注入）から同じ問題に取り組む。
- **PRIME-RL** (Cui et al. 2025): 非同期RLインフラストラクチャ。本論文の実装基盤。

---

## まとめ

Le Critiqueは、「LLM RLから価値関数が消えた」状況に対する明確な処方箋を提示する。PVFは特権情報でcriticを強化し、TETHERはgroup baselineとの適応的補間で安全性を担保する。両者は補完的で、PVFがcriticの質を上げ、TETHERが不完全なcriticを安全に使う。

4Bモデルでの実験とはいえ、結果は明確だ——PVFは全タスクで最高、TETHERは全タスクでVFを上回る。$\lambda \approx 0.9999$ という極狭帯域の重要性も、今後のLLM RL研究に実用的な知見を提供する。

価値関数の「復権」という文脈で、本論文は一つの明確なメッセージを送っている——criticを使わない選択は、必ずしもcriticが使えないからではなく、正しく使う方法が見つかっていなかったからだ。PVFとTETHERがその方法を提示した。

---

## 参考

- Venkatraman, S., Dinot, M., & Aitchison, L. "Le Critique: Privileged Value Functions for LLM Reinforcement Learning." arXiv:2608.16739, 2026.
- [arXiv:2608.16739](https://arxiv.org/abs/2608.16739)
- [HTML version](https://arxiv.org/html/2608.16739v1)
- コード: [prime-values (GitHub)](https://github.com/HyperPotatoNeo/prime-values)
