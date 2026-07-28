---
title: "ESTR：非同期RLの重要性比をtoken熵でスケーリングし2.6倍加速"
emoji: "⚖️"
type: "tech"
topics: ["reinforcement-learning", "LLM", "asynchronous-RL", "GRPO", "importance-sampling", "trust-region"]
published: true
---

## TL;DR

- 非同期RLはrollout生成と戦略更新を並列化してスループットを上げるが、古い（stale）行動方策由来のoff-policyデータが最適化を不安定化し、最悪の場合**不可逆な方策崩壊**を引き起こす
- 従来は重要性比 $r_t=\pi_\theta(y_t\mid s_t)/\mu(y_t\mid s_t)$ の**絶対値**だけでtokenを棄却/保持していた
- 本論文の核心：重要性比の自然な尺度は**tokenの局所エントロピー $H_t$** によって決まる、すなわち $\mathbb{E}[\delta_t^2]\propto H_t$
- これを無視すると、低エントロピー領域では**増幅されたサンプリングノイズ**を誤って保持し、高エントロピー領域では**in-flight更新による正当な探索**を誤って切り捨てる
- 提案手法 **ESTR（Entropy-Scaled Trust Region）** は、各tokenのoff-policy偏差を $\sqrt{H_t+\epsilon}$ で標準化し、信頼領域の境界をエントロピーに応じて膨らませたり縮めたりする
- BrowseComp-Plus（Qwen3-30B-A3B）でavg@1 37.3%、avg@4 **95.7%**を達成。同期GRPOと同等の精度を保ちながら**2.6倍**のスループット向上を実現
- 軌道内バージョン切り替えや軌道間の遅延を意図的に高めたストレステストでも**ゼロ崩壊**。非同期RLの実運用化に大きく近づいた一作

## 背景

### 非同期RLの吸引力と落とし穴

最近のLLM推論強化では、RL with Verifiable Rewards（RLVR）とその代表的実装であるGRPOが支配的になっている。大量の長い思考列を生成する必要があるため、**rollout生成と戦略更新を並列化する非同期設計**が注目されている。すると、GPUはrollout中も更新中も遊ばずに動かせるはずだ。

しかし、魅力的な反面、非同期化は fundamentally な問題を抱え込む：

1. **軌道が複数の重みバージョンを横断する**。長いマルチターン軌道の途中で方策が更新されると、同じ軌道の前半と後半が異なる重みで生成されたことになる。これを**軌道内陳旧性（intra-trajectory staleness）**と呼ぶ。
2. **ロールアウトと最適化のバージョンギャップが大きくなる**。最後に生成されたtokenの重みと、今更新しようとしている目標方策の重みとの差が開く。これを**軌道間陳旧性（inter-trajectory staleness）**と呼ぶ。

この状態で重要性サンプリングをそのまま使うと、比 $r_t$ が暴走し、勾配方差が爆発して学習が崩壊する。既存の対策は主に3系統あるが、本論文はそれらが2つの前提を誤っていると指摘する：

- tokenの信頼性は重要性比の**大きさ**で読める
- 各軌道は**単一の明確な行動方策**から生成される

実際には、軌道内で重みが切り替われば行動方策も切り替わるので、復元可能な「一つの行動方策」など存在しない。そして、重要性比の大きさはtokenごとのエントロピーに比例している。

## 方法詳解

### エントロピーと重要性比のスケーリング律

まず重要なのは、経験的事実として

$$
\mathbb{E}[\delta_t^2] \propto H_t
$$

が成立する点である。ここで $\delta_t=\log r_t=\log\frac{\pi_\theta(y_t\mid s_t)}{\mu(y_t\mid s_t)}$、$H_t=-\sum_w\mu(w\mid s_t)\log\mu(w\mid s_t)$ である。

論文のAppendix Aでは、目標方策を行動方策のlogitに摂動 $\eta_w$ を加えたものと近似すると、一次展開により

$$
\delta(y)\approx\sum_w\left(\mathbf{1}[w=y]-\rho_w\right)\eta_w
$$

が得られ、さらに $\rho_w(1-\rho_w)\approx-\rho_w\log\rho_w$（集中分布で良好な近似）を使うと

$$
\mathbb{E}[\delta^2]\approx\sigma_t^2\sum_w\rho_w(1-\rho_w)\approx\sigma_t^2 H_t
$$

が示される。つまり、重要性比の自然な分散尺度はエントロピーそのものなのだ。

### 二重の現象

このスケーリング律を無視すると、以下の二重の過ちを犯す。

**低エントロピー領域：増幅ノイズを誤って保持する**

モデルが自信を持っている位置では分布が2値的になる：主tokenの確率が $1-q$、外れたtokenの確率が $q\approx0$。エントロピーは $H_{\mathrm{bin}}(q)$ に近づく。もし $q$ が小さい外れたtokenがサンプリングされると、訓練-推論の不一致 $\xi$ が分母 $q$ で増幅される：

$$
\mathbb{E}[\delta_t^2\mid q]\approx\frac{\mathrm{Var}(\xi)}{q^2}\propto\frac{1-q}{q}\xrightarrow{q\to0}\infty
$$

これは方策の本質的な変化ではなく、ただのサンプリングノイズである。固定閾値を使うとこの異常値を受け入れてしまう。

**高エントロピー領域：正当な探索を誤って切り捨てる**

逆に、軌道内でin-flight更新が発生すると、新しい重みが古い重みの生成したprefixを評価する。その切り替え点では、エントロピーと重要性比が同時に急騰する。これを

$$
\|\delta_t\|=\sigma_t\sqrt{H_t},\qquad z_t\triangleq\frac{\delta_t}{\sqrt{H_t}},\qquad \mathbb{E}[z_t^2]\approx\sigma_t^2
$$

と分解すると、標準化後の偏差 $z_t$ は有界で、実は高エントロピーの包絡線の中の正当な探索なのだ。固定閾値はこれを漂移とみなして捨ててしまう。

### ESTR: Entropy-Scaled Trust Region

提案手法は驚くほどシンプルだ。各tokenについて

$$
S_t\triangleq\frac{\delta_t^2}{H_t+\epsilon},\qquad M_{i,t}=\mathbf{1}[S_{i,t}\leq\tau]
$$

という**エントロピースコア**を計算し、$S_t\leq\tau$ のtokenだけを損失に残す。$\epsilon$ は低エントロピーでの退化を防ぐための小さな定数（論文では0.01）。

これは以下の信任領域と同値である：

$$
\|\delta_t\|\leq\sqrt{\tau(H_t+\epsilon)}
$$

低エントロピーでは境界が $\sqrt{\tau\epsilon}$ へと縮小し、増幅ノイズを厳しく弾く。高エントロピーでは $\sqrt{\tau H_t}$ に比例して境界が広がり、in-flight更新による探索を保持する。

完全な損失関数は以下の通り：

$$
\mathcal{L}_{\mathrm{ESTR}}(\theta)=-\mathbb{E}_{x\sim\mathcal{D},\{o_i\}\sim\mu}\left[\frac{1}{\sum_i\|o_i\|}\sum_{i=1}^{G}\sum_{t=1}^{\|o_i\|}M_{i,t}\cdot\min\!\big(r_{i,t}A_{i,t},\,\mathrm{clip}(r_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})A_{i,t}\big)\right]
$$

ここで注目すべきは、**マスク $M_{i,t}$ とPPOクリップ $C_{i,t}$ は役割が異なる**ことだ。

- $M_{i,t}=0$：そもそも損失から除外（勾配ゼロ）
- $C_{i,t}=0$：損失には入るが勾配がゼロ（クリップ）

ESTRのマスクは「信頼領域の内外」を判定し、PPOクリップは「信頼領域内の最大歩幅」を制御する。両者は直交している。

実装コストはゼロに近い。エントロピー $H_t$ はrollout生成時のlogitsからそのまま取得でき、追加のforward passもバージョン切り替え検出も不要だ。

### 陳旧性の分解

本論文は陳旧性を明快に分解する：

$$
\Delta^{\mathrm{intra}}\triangleq v_{\mathrm{last}}-v_{\mathrm{first}},\qquad
\Delta^{\mathrm{inter}}\triangleq v_{\mathrm{tgt}}-v_{\mathrm{last}}
$$

- $\Delta^{\mathrm{intra}}$：1本の軌道内で最初から最後までに経た重みバージョン数
- $\Delta^{\mathrm{inter}}$：軌道の最後の重みと目標方策の重みの差

どちらも固定閾値では扱えないが、エントロピーで標準化すれば両方を同じルールで吸収できる。

## 実験結果

### 評価設定

論文は3種類のタスクで評価している。

| タスク | モデル | 報酬 | 性質 |
|--------|--------|------|------|
| BrowseComp-Plus | Qwen3-30B-A3B (MoE) | LLM judge | 長期マルチターンagent |
| Multi-turn GSM8K | Qwen2.5-7B | ルールベース | マルチターン+ツール |
| DAPO-Math | Qwen2.5-7B | — | 単輪数学推理 |

基線は同期GRPO、非同期GRPO（無修正）、IcePop（固定閾値硬マスク）、KPop（双方向KL閾値）の4つ。

### 主結果

**BrowseComp-Plus（Qwen3-30B-A3B）**

| Method | avg@1 | avg@4 |
|--------|-------|-------|
| GRPO (Sync) | 38.5 | 59.6 |
| GRPO (Async) | 28.9 | 60.7 |
| IcePop | 32.5 | 65.0 |
| KPop | 34.9 | 70.5 |
| **ESTR (Ours)** | **37.3** | **95.7** |

ESTRは同期GRPOのavg@1（38.5）にほぼ並び、一方でavg@4では**95.7%**と他を圧倒する。これは、ESTRが高エントロピー領域の探索を保持することで、解の多様性（pass@4）を大きく向上させたことを示唆する。

**Multi-turn GSM8K（Qwen2.5-7B）**

| Method | avg@4 |
|--------|-------|
| GRPO (Sync) | **96.0** |
| GRPO (Async) | 60.7 |
| IcePop | 71.4 |
| KPop | 65.0 |
| **ESTR (Ours)** | **95.7** |

非同期GRPOやIcePop、KPopは数百stepで不可逆に崩壊するが、ESTRは同期GRPO（96.0）をほぼ再現する95.7%を達成。

**DAPO-Math → AIME（Qwen2.5-7B）**

| Method | AIME Avg. avg@4 | AIME Avg. pass@4 |
|--------|----------------|------------------|
| GRPO (Sync) | 17.5 | 27.7 |
| GRPO (Async) | 13.6 | 23.0 |
| IcePop | 15.8 | 25.0 |
| KPop | 16.2 | 25.5 |
| **ESTR (Ours)** | **17.0** | **28.4** |

avg@4では同期GRPOに肉薄し、**pass@4では同期GRPOを上回る**。ここでも探索保持の効果が現れている。

### 訓練効率

| Method | Throughput (tokens/s/GPU) | sec/step | Speedup |
|--------|---------------------------|----------|---------|
| GRPO (Sync) | 82.56 | 1356.47 | 1.0× |
| **ESTR** | **214.39** | **514.84** | **2.6×** |

ESTRは同期GRPOと比べて**2.6倍**のスループットを達成し、かつ精度を維持する。非同期化による遅延を補うどころか、信任領域の設計がより良い探索を生んでいる。

### 陳旧性ストレステスト

著者は $\Delta^{\mathrm{intra}}\in\{1,5,7,9\}$ および $\Delta^{\mathrm{inter}}\in\{1,5,15,20,30\}$ まで意図的に陳旧性を高めた。結果：

- **どの設定でも崩壊はゼロ**
- 報酬は陳旧性の増大に伴って**単調かつ優雅に減少**
- ESTRは軌道内バージョン切り替えと軌道間遅延の両方を、陳旧性専用のチューニングなしで吸収する

![/images/estr-entropy-scaled-trust-regions/fig4_robustness_speedup.png]

*図4：陳旧性ストレステスト（左）と精度vsスループットのトレードオフ（右）*

### 訓練動態の可視化

BrowseComp-Plusでの訓練動態では、ESTRは他の非同期手法よりも高い精度を維持するだけでなく、方策エントロピーも安定的に増加し、rollout-目標KLが最も低く抑えられる。つまり、**最も訓練-推論一貫性が高い**状態で探索を維持している。

Multi-turn GSM8Kでは、vanilla非同期GRPOやIcePopのエントロピーが爆発し、マスク率も暴走するのに対し、ESTRのマスク率は低く安定する。固定閾値では「ノイズも探索も同じ基準で断ち切る」ため、安定を保とうとすると過剰にマスクし、性能が頭打ちになる。

![/images/estr-entropy-scaled-trust-regions/fig3_training_dynamics.png]

*図3：3タスクでの訓練動態。ESTRは非同期化による崩壊を回避しつつ、同期GRPOに匹敵する精度を達成する。*

## 考察

### なぜ固定閾値は失敗するのか

固定閾値の根本的な問題は、**異方性（heteroscedasticity）を無視している**点にある。重要性比の二乗はtokenエントロピーに比例して変動する。したがって、どの定数 $c$ を選んでも、低エントロピーでは緩しすぎてノイズを通し、高エントロピーでは厳しすぎて探索を潰す。

KLダイバージェンスの二階展開 $D_{\mathrm{KL}}(\mu_t\|\pi_\theta)\approx\frac{1}{2}\mathbb{E}[\delta_t^2]$ は、信任領域が実質的にlog-ratioの二乗モーメントを制約していることを示す。二乗モーメントの自然な尺度がエントロピーなら、信任領域もエントロピーに応じてスケーリングするのが自然だ。

### ESTRは「マスクを減らす」ことで強くなる

興味深いのは、ESTRのマスク率が固定閾値基線より**約1桁低い**ことだ。固定閾値は不安定な勾配を抑えるために多くのtokenを落とす。ESTRは低エントロピーノイズだけを的確に落とし、高エントロピー探索は残すので、無駄なマスクが減り、結果としてより多くの学習信号を使える。

### 非同期化の性能がむしろ上回るケース

DAPO-Mathのpass@4やBrowseComp-Plusのavg@4では、同期GRPOを非同期ESTRが上回る結果が出た。これは「非同期化は単なる高速化の代償ではなく、適切に正規化すればより多様な探索を生むメカニズムにもなりうる」ことを示唆する。in-flight更新が引き起こす高エントロピーは、同期設定では存在しない「自然な探索ブースター」として機能しうるのだ。

### 実運用への示唆

ESTRの実装は非常に軽量である。エントロピーはrolloutエンジンから既に得られる情報であり、追加forward不要、バージョン検出不要、行動方策再構築不要。したがって、既存の非同期RLシステムへの差し込みコストは低い。一方で、これまで「非同期は不安定で実用が難しい」とされていた長期マルチターンagentタスクでも、安定して同期並みの性能を出せるようになった点は大きい。

## 関連研究

本論文は既存の非同期RL対策を3群に分類して整理する。

**重要性サンプリング修正系**

比率クリッピングや硬マスクが多い。Çağatanら（2026）、Fuら（2026b）、Shenら（2026）、Liら（2026c）は区間クリッピング。Yaoら（2025）、IcePop、Guoら（2026）/KPop、Zhengら（2025b）は閾値を超えたtokenをマスク。いずれも「比率の大きさ＝信頼性」と仮定しており、本論文が根本的に否定した前提に立っている。

**より細粒度な安定信号系**

ESPO（Shengら2026）はエントロピーで系列を再構成して修正するが、バッチ内のエントロピー構造が安定していることを仮定する。AEPO（Dongら2025）は高エントロピー位置やターン切り替え点を明示的に定位しようとするが、追加のオーバーヘッドがある。VCPO（Huangら2026）は有効サンプルサイズを用いる。これらはエントロピー或いは分散を「外部からの重み付け/グループ化」に使うが、ESTRは信任領域自体をエントロピーに応じて変形させる点で異なる。

**陳旧性-不整合の分離系**

AReaL（Fuら2026a）、Guanら（2026）は実際に軌道を生成した行動方策を近似して、漂移と良性の陳旧性を分離しようとする。しかし、非同期agent設定では軌道が混合重みバージョンから生成されるため、単一の行動方策を復元すること自体が不可能。ESTRは行動方策を再構築せず、tokenごとの局所統計で信任領域をスケーリングする。

**Off-policy目標設計系**

Ritterら（2026）、VESPO、Luoら（2026）はクリッピングを分布の分散ペナルティに置き換える。Yuanら（2025）は重要性比を使わずに参照方策への回帰を行う。前者はバッチレベルの位置不変予算をかけているため、低エントロピー高分散ノイズと高エントロピー正当探索の区別がつかない。後者は非同期性の利点を捨てている。

## まとめ

ESTRは、非同期RLの不安定性を「重要性比の絶対値」ではなく「tokenエントロピーで標準化した偏差」に置き換えることで解決する。核心は以下の3点に集約される：

1. **エントロピーと重要性比の基本関係**：$\mathbb{E}[\delta_t^2]\propto H_t$。これは集中分布の自然な帰結である。
2. **二重現象の解明**：低エントロピーではサンプリングノイズが増幅され、高エントロピーではin-flight更新による正当な探索が発生する。固定閾値は両方とも誤って扱う。
3. **シンプルで実装コストゼロの解決策**：$S_t=\delta_t^2/(H_t+\epsilon)\leq\tau$ という熵スケーリング信任領域。

実験では、同期GRPOと同等以上の精度を保ちながら2.6倍の加速を達成し、陳旧性ストレステストでもゼロ崩壊。長期マルチターンagentタスクを含む非同期RLの実用化に向けて、信頼性のある一歩を踏み出した。

## 参考

- Zhao, G., Xie, Z., Zheng, B., Gong, E., Lu, J., Yang, Y., Hu, A., & Chen, Z. (2026). *Deconstructing Off-Policy Ratios: Entropy-Scaled Trust Regions for Asynchronous Reinforcement Learning*. arXiv:2607.22186.
- Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
- Sheng, G., et al. (2026). ESPO: Entropy-Accelerated Sequential Preference Optimization.
- Fu, Y., et al. (2026a). AReaL: Asynchronous Reinforcement Learning for Large Language Models.
- Guo, X., et al. (2026). KPop: KL-guided Policy Optimization.
- Yuan, Z., et al. (2025). Advancing Offline Reinforcement Learning via Preference Modeling.

---

*本記事は arXiv:2607.22186 を読んで書いた個人的な読書メモです。数式や実験設定の詳細は原著をご参照ください。*
