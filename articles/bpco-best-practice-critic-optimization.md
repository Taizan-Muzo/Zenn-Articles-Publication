---
title: "Criticの復権: BPCOがGRPOを16分の1サンプルで凌駕する5つの設計選択"
emoji: "🎭"
type: "tech"
topics: ["LLM", "強化学習", "Critic", "PPO", "GRPO"]
published: false
published_at: 2026-08-27
---

## TL;DR

GRPOがLLMのRL post-trainingのデファクトスタンダードになり、Criticは「使えない」と見捨てられて久しい。しかしQi, Zhou, Leeによる**BPCO (Best Practice Critic Optimization)** は、5つの設計選択を積み重ねるだけでCriticベースのRLを安定化し、**1ロールアウトでGRPOの16ロールアウトに匹敵・超過**する性能を実現した。追加の定理や新アルゴリズムではない――地道だが決定的なエンジニアリングの勝利だ。

[/images/bpco-best-practice-critic-optimization/fig1_pipeline.png](/images/bpco-best-practice-critic-optimization/fig1_pipeline.png)

## 背景: なぜCriticを見捨てたのか

LLMのpost-trainingにおけるRLは、ここ1年で劇的な進化を遂げた。その中心にあるのが**GRPO (Group Relative Policy Optimization)** だ。GRPOは各プロンプトに対して複数（通常8〜16個）の応答をサンプリングし、グループ内の相対順位をアドバンテージの代わりに使うことで、**Criticネットワークを完全に排除**した。

理由は明確だった。標準的なPPO + Criticの組み合わせは:

- **訓練が頻繁に崩壊する** — 報酬が急激に発散し、Criticが意味のない値を出力し始める
- **解釈分散（Explained Variance）が嘘をつく** — bootstrapターゲットを使うと、自己言及的な目標によってEVが人工的に1に近づき、実際の予測品質を隠蔽する
- **長いChain-of-Thoughtに対応できない** — 固定 discount factor では、数千tokenの応答において初期tokenが末端報酬を受け取れなくなる

これらの問題は「Criticというアプローチ自体が悪い」のではなく、「Criticの使い方が雑だった」ことに起因していた。BPCOはまさにこの点を突く。

## 方法: 5つの設計選択

BPCOは新しいアルゴリズムではなく、**既存のコンポーネントを正しく組み合わせるレシピ**だ。だが、各選択の影響は制御実験で厳密に検証されている。

### 1. DPPO: 絶対確率クリッピング

標準PPOのクリッピングは確率比率 $\rho_t = \pi_\theta / \mu$ に対して行う。大語彙では低確率tokenは比率が大きくなりやすく、高確率tokenは小さな変化しか許容されない——つまり**語彙全体で一様でない制約**になっている。

DPPOはこれを修正し、サンプリングされた各tokenの**絶対確率変化**を $\varepsilon$ で制約する:

$$|\pi_\theta(y_t|s_t) - \mu(y_t|s_t)| \leq \varepsilon$$

消融実験でこれが**最も重要なコンポーネント**であることが示された。PPOでは訓練報酬が初期上昇後に崩壊するが、DPPOは安定した上昇を維持する。他の4つの改善をすべて盛り込んでも、PPOに戻すと崩壊する。

### 2. 有界値予測

線形値ヘッドは**無界**だが、報酬が既知の範囲 $[R_{\min}, R_{\max}]$ に収まることが多い（二値報酬なら $[0,1]$）。有界確率変数の期待値は同じ区間に収まるはずなのに、無界の出力は範囲外の極端な値を生成し、訓練を不安定にする。

BPCOは**スケーリング済みarctan関数**で値を $[R_{\min}, R_{\max}]$ の開区間に写像する:

$$V_\phi(s_t) = R_{\min} + (R_{\max} - R_{\min})\left(\frac{1}{2} + \frac{1}{\pi}\arctan(z_\phi(s_t))\right)$$

これにより、Criticは常に報酬範囲内の妥当な予測を出力するようになる。

### 3. 無偏モンテカルロ値目標

GAE ($\lambda < 1$) を使った bootstrap ターゲットは、**自己言及的**だ — 目標値自体に古いCriticの予測が含まれる。結果として、解釈分散が訓練初期から急速に1に近づき、Criticが実際には何も学べていないにもかかわらず「完璧に予測している」ように見える。

BPCOは方策のGAEとCriticの目標で**異なる $\lambda$ を使用**する:

- **方策**: $\lambda_\pi = 0.99$ （分散削減の恩恵を維持）
- **Critic**: $\lambda_V = 1$ （bootstrap バイアスを完全に排除）

$\gamma=1$ で結果報酬のみの場合、$\lambda_V=1$ の目標は**純粋なモンテカルロサンプル** $R(x,y)$ に退化する。これは $V^\mu(s_t)$ の**無偏推定量**だ。

[/images/bpco-best-practice-critic-optimization/fig2_ablation_stability.png](/images/bpco-best-practice-critic-optimization/fig2_ablation_stability.png)

### 4. 未正規化アドバンテージ

標準的なRL実装ではアドバンテージをバッチ正規化する: $\tilde{A}_t = (\hat{A}_t - \bar{A})/\sigma_A$。しかしBPCOはこれを**行わない**。

理由は2つある:

1. **最適での消失を阻害する** — 方策が最適に近づくと、真のアドバンテージとその分散は共にゼロに近づくべきだが、$\sigma_A$ で割ると小さな推定ノイズが大きな訓練シグナルに増幅され、最適点で更新が消えなくなる
2. **符号反転が探索を妨げる** — バッチ平均を引くことで、正だが平均以下のサンプルの符号が反転し、その行動が罰される。

BPCOでは正規化を除去した結果、アドバンテージの範囲は訓練を通じて小さく安定に留まる。一方、正規化を維持すると範囲が訓練と共に**増大**していく。

### 5. 長さ適応型GAE (LA-GAE)

固定 $\lambda_\pi < 1$ では、長い応答の初期tokenほど末端報酬の寄与が指数関数的に減衰する（$\lambda^{T-t}$）。数千tokenのCoTでは、最初の方の推論ステップが最終結果からほとんどシグナルを受け取れない。

LA-GAEは応答長さ $L$ に応じて $\lambda$ を動的に調整する:

$$\lambda_\pi(L) = 1 - \frac{1}{\alpha L}$$

ここで $\alpha > 0$ はバイアス-バリアンスのトレードオフを制御する（最適値 $\alpha=0.4$）。鍵となる性質は、最初のtokenのアドバンテージに含まれる末端残差の係数が:

$$(1-1/\alpha L)^L \approx \exp(-1/\alpha) \approx 0.082$$

となり、**応答長さにほぼ依存しない**ことだ。$\lambda=0.99$ では $L=24{,}000$ でこの係数は $\approx 10^{-105}$ にまで壊滅的に減衰するが、LA-GAEでは常に一定の情報伝達が保証される。

[/images/bpco-best-practice-critic-optimization/fig3_la_gae.png](/images/bpco-best-practice-critic-optimization/fig3_la_gae.png)

### 特権情報 conditioning

Criticは**訓練時のみ**存在する。推論時には削除される。この非対称性を利用し、Criticの入力に**方策には見せない情報**を条件付けできる:

- **BPCO+Ans**: 参考答案をCriticにのみ入力
- **BPCO+Sol**: 公式解法をCriticにのみ入力
- **BPCO+Rubrics**: 評価ルーブリックをCriticにのみ入力

これは多エージェントRLの**集中学習分散実行 (CTDE)** パラダイムに直接的に対応する。Rollout時の方策はプロンプト $x$ と生成プレフィックスのみを観測し、Criticだけが $q(x)$ を参照する。

## 実験結果

### 設定

| 実験 | ベースモデル | データセット | 最大生成長 | 基線 |
|------|-------------|-------------|-----------|------|
| Sanity Test | DeepSeek-R1-Distill-Qwen-1.5B | 1,460問 | - | - |
| 大規模 | 同上 | DeepScaleR (~40.3K問) | 24,000 tokens | Dr.GRPO (16 samples) / Critic baseline |
| 大モデル | Qwen3-30B-A3B | DAPO-Math-17K | 12,000 tokens | 同上 |
| Rubric | Qwen3-4B-Base | OpenRubrics | - | 同上 |

全設定で**各イテレーションの総トラジェクトリ数を固定**し、BPCO（1 sample/prompt）とGRPO（16 samples/prompt）を公平に比較している。

### Sanity Test (1.5B, 1,460問)

制御実験により各コンポーネントの寄与を段階的に検証:

1. **PPO + Critic**: 訓練報酬が初期上昇後に**崩壊**
2. **DPPO + Critic**: 安定した上昇を維持（**DPPOが不可欠**）
3. **+有界値予測**: 報酬範囲外の極端値を排除
4. **+MC目標**: 偽りの高解釈分散を解消、収束加速
5. **+正規化除去**: 過学習リスクを緩和
6. **+LA-GAE ($\alpha=0.4$)**: $\lambda=1$ の遅さと $\lambda=0.99$ の過学習を両回避

### DeepScaleR (1.5B, ~40.3K問)

BPCOは全変体でcritic baselineとgroup baselineを**一貫して上回った**。訓練・検証性能ともに改善し、解釈分散も全訓練過程で持続的に高い水準を維持した。

消融実験では:
- **有界値予測の除去**: 訓練報酬の改善が鈍化、AIME 2025 avg@32 が低下
- **正規化の再導入**: アドバンテージの振幅が訓練中に増大（小データセットほど顕著）

特権情報の効果:
- **BPCO+Ans**: より速い訓練、より高い解釈分散、より良いAIME 2025性能
- **BPCO+Sol**: 控えめな改善（7.3K/40.3K問のみ利用可能）

### Qwen3-30B-A3B (DAPO-Math-17K)

最も注目すべき結果だ:

- **Critic baseline**: Qwen3-30B-A3B（Instruct）上で訓練開始100歩以降**AIME 2025が完全に停滞**（不安定な最適化）
- **BPCO**: 同モデルで**大幅に高い**AIME 2025精度を達成
- **vs Group baseline**: Qwen3-30B-A3Bで**優越**、Baseモデルで**匹敵**

スケールアップするほどCriticの不安定性が顕著になり、BPCOの各設計選択の価値が増すことが示された。

### Rubric-Based報酬 (4B)

OpenRubricsデータセット + 凍結Qwen3-4B-Instructによるルーブリック評価:

- BPCOはgroup/critic baselineより**速く学習**
- Group baselineは最終的に**同等の性能**に到達
- Critic baselineは最終報酬が**やや低い**
- **特権情報は性能を改善せず**（解釈分散は高いが、タスクが比較的簡単なためCriticが十分学習可能）
- BPCOの核心（有界値予測 + 未正規化アドバンテージ）はルーブリック評価下でも**有効**

[/images/bpco-best-practice-critic-optimization/fig4_efficiency.png](/images/bpco-best-practice-critic-optimization/fig4_efficiency.png)

## 考察

### Critic debtという概念

BPCOが示した最も重要なメッセージは、Criticを捨てたのは「数学が壊れていた」からではなく「使い方が雑だった（skill issue）」からだということだ。論文はこれを**Critic debt**と呼ぶ — 基礎的な値推定を修正する代わりに、GRPOという回避策に走った結果、コミュニティが蓄積した技術的負債である。

### 16倍のプロンプト多様性

同じ計算予算で、BPCOはGRPOの**16倍の異なるプロンプト**をカバーできる。これは特に:

- 大規模データセットでの**汎化性**に直結する
- 長いCoT生成ではrolloutコストが支配的になるため、**1サンプル化の恩恵は極めて大きい**

### CTDEパラダイムの自然な適用

特権情報の conditioning は、理論的に新しくない（多エージェントRLでのCTDEは古くから知られている）が、LLM RLへの応用は実用的で強力だ。参考答案やルーブリックは多くのタスクで利用可能であり、Criticが訓練のみに使われるという事実はこのアプローチを**追加コストゼロ**で実現する。

### 限界

- 報酬範囲 $[R_{\min}, R_{\max}]$ が既知である必要がある
- 数学的推論とルーブリック評価に限定（コーディングやエージェントタスクは未検証）
- Critic自体の計算・メモリオーバーヘッドはトータルコスト比較に含まれていない
- 特権情報は全タスクで利用可能ではない

## 関連研究

**Critic復権の系譜**: Le Critique（PVF + TETHERによる特権価値関数）は同時期にCriticの有効性を示したが、BPCOは異なるアプローチ — レシピの最適化による安定化 — を取る。SAPOは単一ロールアウトでActor-Criticを統合したが、因果境界によるValue Basisを導入した。

**Group-based最適化**: GRPO/Dr.GRPOが現在の主流だが、SPO++は同期ボトルネックを除去し、GSPOはシーケンスレベル重要度比で問題を緩和した。BPCOは「groupの計算コストを払わずにgroupと同等以上の性能を出す」という別の切り口だ。

**オーバーサンプリングの課題**: UP (Asymmetric Optimization) は確率キャパシティの観点からクリッピングの限界を指摘し、GSPOは長さ正規化で対応した。BPCOのDPPOも同じ問題意識から出発しているが、解決策が異なる。

## まとめ

BPCOは派手な新しいアルゴリズムではなく、**5つの正しい設計選択**の積み重ねだ:

1. **DPPO** — 絶対確率制約で語彙全体で一様な更新
2. **有界値予測** — arctan写像で報酬範囲内に拘束
3. **MC目標** — 自己言及bootstrapの嘘を排除
4. **未正規化アドバンテージ** — 最適点での更新消失を保証
5. **LA-GAE** — 応答長さに適応したシグナル伝達

結果として、**1ロールアウトで16ロールアウトのGRPOに匹敵・超過**し、Criticの特権情報 conditioning というボーナスも手に入る。Criticを見捨ててから2年、ようやく「技術的負債」の返済が始まった。

## 参考

- Qi, P., Zhou, X., & Lee, W. S. (2026). Best Practice Critic Optimization. arXiv:2608.23566v2.
- コード: https://github.com/QPHutu/golden_critic
- フレームワーク: verl (commit 86e8123, 2026-06-16)
