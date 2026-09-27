---
title: "外挿が教師を作る──RISEがRLVRの訓練軌跡から逐次改善の教師を自前で構築する仕組み"
emoji: "🚀"
type: "tech"
topics: ["LLM", "強化学習", "蒸留", "RLVR", "推論"]
published: true
---

## TL;DR

On-policy distillation (OPD) は逐 token の密な教師信号を与える強力な枠組みだが、その効果は**教師の品質**で頭打ちになる——外部教師は分布不整合を起こし、特権条件付きの自蒸留 (OPSD) は in-context learning の容量限界に阻まれる。RISE (Recursive Improvement via Self-Extrapolating Policy Distillation) は、この袋小路を**自前の教師構築**で突破する：RLVR の訓練軌跡に沿ってパラメータ空間または logit 空間で外挿し、モデル自身の「到達すべき未来」を合成教師として構築する。結果報酬が外挿方向の正当性を保証し、外挿教師の逐 token 分布が細粒度の信用割当てを提供する——二段が補完し合う再帰的改善ループがここで回る。数学・STEM・コード・エージェントの四系統で一貫して RLVR-only と OPSD 基線を上回り、競技ベンチマークでは +16.7 pt (OLMo3-7B, AIME'24) という大幅改善を達成。追加サンプリングコストはゼロ、実行時間は 1.3–1.6× だけ。

## 背景

### RLVR の信用割当てボトルネック

GRPO や DAPO などの RLVR 手法は、結果報酬を逐 token の利得に変換して方策勾配更新を行う。しかし結果報酬は**系列レベル**のスカラーであり、同一応答内の全 token に同一の利得が割り当てられる——どの推論ステップが正解に貢献し、どのステップが無関係または有害だったかを区別できない。この粗い信用割当てが RLVR の根本的な壁である。

### OPD は解になり得る——教師が良ければ

On-policy distillation はこの問題を根本から解決するアプローチだ：教師モデルが各位置で完全な次 token 分布を指定すれば、結果報酬では不可能な逐 token の細粒度指導が可能になる。だが**信頼できる教師をどこから調達するか**が問われる。

| 教師の選択 | 問題点 |
|-----------|--------|
| 外部強教師 | 学生が探索した新経路上で分布不整合——教師が見たことのない前缀では分布が当てにならない |
| 特権条件付き自蒸留 (OPSD) | in-context learning の容量不足で特権情報を活用しきれない；教師と学生で入力が異なり分布不整合 |

啓発的な補正——token-level gating、divergence mixing、DAgger 式 rollout 混合、軌跡精製——はいずれも「欠陥のある教師をなんとか許容する」方向の対症療法であり、「そもそも良い教師を作れないか」という問いには答えていない。

### 訓練軌跡が教えること

近年の分析が示すように、LLM の後段訓練におけるパラメータ更新は**低ランク部分空間**に支配され、**近似的に線形**に推移する——たかだか 3 方向で約 87% の分散を説明できる。Model soups はこの線形性を $\beta \in [0,1]$ の**内挿**で活用して頑健性を得ている。RISE は同じ構造を $\beta > 1$ の**外挿**に転用し、能力向上を狙う。

## 方法詳解

### 外挿で「未来の方策」を合成する

RLVR によって方策 $\pi_{\theta_n}$ が $\pi_{\theta'_{n+1}}$ に更新されたとする。RISE の**自外挿教師**は次で定義される：

$$\varphi(\pi_{\mathrm{future}}) = \varphi(\pi_{\theta_n}) + \beta \cdot (\varphi(\pi_{\theta'_{n+1}}) - \varphi(\pi_{\theta_n})), \quad \beta > 1$$

$\varphi$ は方策を線形演算が意味を持つベクトル空間へ写すマッピング。$\theta_n$ は anchor、$\theta'_{n+1}$ は RLVR 後の checkpoint、$\beta > 1$ が訓練軌跡に沿って現在の先へ踏み出す外挿係数である。

![図1: RISEの外挿幾何と訓練ループ](/images/rise-recursive-self-extrapolating-distillation/fig1.png)
*図1: (a) RLVR で更新した方向を $\beta > 1$ 倍に外挿して教師を構築し、OPD で逐 token 蒸留する幾何。(b) 二段を交互に繰り返す再帰的改善ループ。*

### 二つの実装化

**重み空間外挿** ($\varphi = \theta$)：

$$\theta_{\mathrm{future}} = \theta_n + \beta \cdot (\theta'_{n+1} - \theta_n)$$

Task arithmetic に外挿係数を掛けたもので、$\theta_{\mathrm{future}}$ をパラメータとして持つモデルが教師になる。位置をまたいで予測が整合する完全なモデルが得られるが、パラメータを実体化し OPD の各 step で教師の forward pass が必要。

**Logit 空間外挿** ($\varphi = \log \pi$)：

$$\log \pi_{\mathrm{future}}(\cdot|s_t) = \log \pi_{\theta_n}(\cdot|s_t) + \beta \cdot (\log \pi_{\theta'_{n+1}}(\cdot|s_t) - \log \pi_{\theta_n}(\cdot|s_t)) + \mathrm{const}$$

これは $\pi_{\mathrm{future}} \propto \pi_{\theta_n}^{1-\beta} \cdot \pi_{\theta'_{n+1}}^{\beta}$——確率比 $\pi_{\theta'_{n+1}} / \pi_{\theta_n}$ を増幅する**幾何混合**。パラメータ操作が不要で、OPD 開始前に Top-K logit を二回 forward して教師分布をキャッシュすれば、以後はキャッシュから読み出すだけ。

重み空間外挿は logit 空間外挿の**一階 Taylor 近似**に相当する。$f$ が線形なら両者は完全に一致するが、ニューラルネットでは異なる——実験でも一貫した優劣はつかず、**外挿の原理そのもの**が効いていることが示される。

### KL 正則化の分解が見せる二つの力

蒸留損失を分解すると：

$$D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{future}}) = -(\beta-1) D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta_n}) + \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta'_{n+1}}) + \log Z$$

$\beta > 1$ のとき：

- **第一項**（負係数）は**斥力項**——現在の方策を anchor から遠ざけ、RLVR の改善方向を延長する
- **第二項**（正係数）は**正則項**——方策を RLVR 後の $\pi_{\theta'_{n+1}}$ の近傍に引き戻し、OPD 中の行き過ぎを防ぐ

両項が**逐 token レベル**で働く——結果報酬では不可能な細粒度の信用割当てがここで実現する。

### 外挿係数の減衰スケジュール

固定 $\beta$ は全訓練軌跡に適さない。方策が最適に近づくにつれ安全な $\beta$ の範囲は狭まる——訓練初期は大きく外挿してよくても、後期には $\beta = 1.5$ でも崩壊する。RISE は単調減衰スケジュール $\beta_n = 1 + (\beta_0 - 1)(1 - n/N)$ を採用し、初期は攻撃的に外挿し後期は慎重に近づく。

### Anchor の動学

デフォルトの anchor は前 checkpoint ($\eta = 1$)。EMA anchor $\theta_{\mathrm{anchor}} \leftarrow (1-\eta)\theta_{\mathrm{anchor}} + \eta \cdot \theta_{n+1}$ は複数 iteration にわたり変位方向を平滑化し、ノイズの多い RLVR 更新に対して安定化に寄与する。ただし OLMo のように GRPO が既に大きく安定した変位を生む場合は $\eta = 1$ が優位——EMA がかえって遅れを生む。

### OPD 段階の役割──なぜ $\theta_{\mathrm{future}}$ を直接採用しないのか

外挿点は方策の信頼領域の外にある：大きな $\beta$ で直接採用すれば退化挙動や RLVR ノイズの増幅を招く。OPD は**信頼領域への射影**として働く——外挿教師の逐 token 分布の方へ方策を移動させつつ、発散項が $\pi_{\theta'_{n+1}}$ の近傍に錨を下ろす。

### 訓練手続き

1. **Phase 1 — RLVR**: $\pi_{\theta_n}$ から rollout をサンプリング → 結果報酬 → 方策勾配で $\theta'_{n+1}$ に更新
2. **Phase 2 — OPD**: 同一 rollout から $\pi_{\theta'_{n+1}}$ と $\pi_{\theta_n}$ を用いて外挿教師 $\pi_{\mathrm{future}}$ を構築 → $D_{\mathrm{JSD}}$ で $\theta_{n+1}$ に蒸留

**同一 rollout を再利用**するため、追加サンプリングコストはゼロ。Jensen-Shannon 発散は $\log 2$ で上界付き、数値安定性に優れ、唯一の最小化器は同じ。

## 実験結果

### 数学推論

![図2: 数学推論ベンチマーク主結果](/images/rise-recursive-self-extrapolating-distillation/fig2.png)
*図2: 三つのモデル規模で RISE が一貫して全基線を上回る。競技ベンチマーク (AIME) での差が最も顕著。*

**Qwen3-8B**: Math Avg 60.0 (GRPO) → 62.7 (RISE-weight), +2.7 pt。OOD Avg も 70.6 → 72.0 と改善。

**Qwen3-1.7B**: Math Avg 45.4 (GRPO) → 50.2 (RISE-logit), +4.8 pt。小規模での改善がより大きい。

**OLMo3-7B**: Math Avg 47.6 (GRPO) → 56.4 (RISE-logit), +8.8 pt。AIME'24 は 30.2 → 46.9 で +16.7 pt と最大の改善。

特権条件基線 (GRPO+SDPO) は Qwen3-8B で 55.9 vs GRPO の 60.0 と**下回る**——OPSD の分布不整合が有害に働いた例である。RISE は外部モデルも特権情報も使わずにこれを上回る。

### 多域 STEM (Qwen3-4B-Base)

Math Avg 40.2 (GRPO) → 44.8 (RISE-weight), STEM Avg 45.5 → 47.5。AIME'24 で +5.0 pt、TheoremQA で +3.7 pt——多域軌跡の外挿が単域の改善を希釈しない。サンプル効率でも RISE は全行程で GRPO を上回り、AMC'23 で RISE(weight) が 70.0% に達する頃、GRPO は訓練中に一旦上がってから 59.4% に下落している。

### コード生成 (Qwen3-8B-Base)

RISE は GRPO より早期収束し、HumanEval+ で RISE(logit) が step 50 で GRPO の最終精度に到達——GRPO は step 90 を要する。改善幅は数学ほど大きくないが、サンプル効率の面で優位。

### Agentic タスク (Qwen2.5-3B-Instruct)

![図4: Agenticタスク結果](/images/rise-recursive-self-extrapolating-distillation/fig4.png)
*図4: ALFWorld と WebShop で RISE (weight) が GRPO を大きく上回る。報酬が疎かで遅延する系列決定タスクでも外挿原理が通用する。*

ALFWorld 75.0 → 84.4 (+9.4 pt)、WebShop Acc 63.3 → 74.2 (+10.9 pt)。エージェントの行動系列では結果報酬が一層疎かになるため、逐 token の密な指導の寄与が際立つ。

## 考察

### 外挿が効くのは RLVR が方向を保証するから

RLVR 段階を完全に除去し、自蒸留による変位だけを外挿した実験では、MATH-500 精度がベースラインから 2.4% に崩落し、応答長が 2K から 8K (上限) に爆増した。**結果報酬の錨がないと、変位は純粋に自己言及的**——毎 iteration が前回の移動方向を反復増幅し、方策は退化した非終了生成に陥る。

### OPD がないと改善は消える

$\theta_{\mathrm{future}}$ を直接次の方策として採用した実験では、Qwen3-8B で Math Avg 60.0 → 60.3 (+0.3 ptのみ)、1.7B で 45.4 → 45.6 (+0.2 pt)。**重み空間で長い歩みを取るだけでは方策は動くが改善しない**——OPD 段階が外挿方向を逐 token 分布の射影を通じて方策の真の改善に変換する。

### 安全な外挿範囲は訓練とともに狭まる

![図3: βの安全範囲の縮小](/images/rise-recursive-self-extrapolating-distillation/fig3.png)
*図3: 訓練初期は β=2.0 でも安全だが、中盤以降は急激に危険になる。RISE の減衰スケジュールはこの経験的事実に対応する。*

Step 50 では $\beta = 2.0$ で +7.3 pt だが、step 100 では同 $\beta$ で -15 pt の崩壊、step 200 では $\beta = 1.5$ でも -7.7 pt。RISE の $\beta_0 = 1.2$ から 1.0 への線形減衰は、安全領域の縮小に歩調を合わせる設計である。

### RISE は解の覆盖を広げる

RISE は既存の解を鋭くするだけでなく、解ける問題の集合を拡張する。1.7B で最も顕著：AIME'24 pass@16 が +9.6 pt (avg@16 は +6.3 pt)。外挿教師が方策の探索能力を広げている証拠である。

### 計算コスト

| モデル | RISE / GRPO 実行時間比 |
|--------|------------------------|
| Qwen3-8B | ~1.6× |
| Qwen3-1.7B | ~1.3× |

小規模では rollout 生成が実行時間の大部分を占める (GRPO の 73%) ため、OPD の相対オーバーヘッドが小さい。GRPO-2× (同一 rollout で二度目の勾配更新) と同一予算で比較しても、RISE は +2.2–2.9 pt 上回る——**改善は追加の最適化 step ではなく外挿教師の品質**に由来する。

## 関連研究

**RLVR と信用割当て**: GRPO と DAPO は結果報酬から逐 token 利得を推定するが、系列レベルの粗さは不可避。RLSD は利得の重み付けで細粒度化を図り、SDAR は教師-学生確率差で GRPO 利得をゲートする——いずれも結果報酬の情報量の壁を越えない。RISE は外挿教師という**追加の情報源**でこの壁を突破する。

**On-policy distillation**: 外部教師 OPD は分布不整合、OPSD は ICL 容量限界に直面する。ExOPD は構造上 RISE の logit 空間公式と最適方策が一致するが、**教師動学**が決定的に異なる——ExOPD の教師は固定静態方策 (ハード上限あり)、RISE の教師は学生の改善に伴い毎 iteration 刷新 (上限が漸進的に上昇)。この非平穏性が再帰改善をもたらす。

**Task arithmetic と線形軌跡**: Model soups は $\beta \in [0,1]$ の内挿で頑健性を得る。RISE は同じ線形構造を $\beta > 1$ の外挿に転用する——内挿が「既知の良い点の間を埋める」のに対し、外挿は「既知の改善方向の先にあるまだ見ぬ良い点を推測する」。

## まとめ

RISE が示した核心は、**良い教師は外部から借りるものではなく、自分の訓練軌跡から構築できる**ということだ。RLVR の結果報酬が外挿方向の正当性を検証し、外挿教師の逐 token 分布が細粒度の信用割当てを提供する——二段の補完関係が再帰的な改善ループを生む。外部モデルも特権情報も不要で、追加サンプリングも不要、実行時間 1.3–1.6× だけという計量の軽さも実用上重要だ。

一方で限界も明確だ。外挿の妥当性は訓練軌跡の低次元性に依存しており、$\beta$ がいつ危険になるかを原理的に検知する方法は未解決。また RLVR の報酬がハッカブルなら外挿は偽の方向を増幅する——報酬モデルの不確実性を組み込んだ外挿の制限は今後の重要な方向である。

## 参考

- Yang Li, Semih Yavuz, Shafiq Joty. "RISE: Recursive Improvement via Self-Extrapolating Policy Distillation." arXiv:2609.05295, 2026.
- GRPO: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." 2024.
- DAPO: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System." 2025.
- Task Arithmetic: Ilharco et al., "Editing Models with Task Arithmetic." ICML 2023.
- Model Soups: Wortsman et al., "Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time." ICML 2022.
