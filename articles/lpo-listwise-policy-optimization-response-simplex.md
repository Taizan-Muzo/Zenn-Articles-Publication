---
title: "LPO：Response Simplex上の明示的射影でGRPOを刷新する"
emoji: "🎯"
type: "tech"
topics: ["LLM", "強化学習", "RLVR", "GRPO", "応答最適化"]
published: false
---

# LPO：Response Simplex上の明示的射影でGRPOを刷新する

## TL;DR

- **問題提起**: GRPOをはじめとするgroup-based policy gradientは、実はresponse simplex（サンプル応答が張る確率単体）上の「暗黙のターゲットへの近似射影」に過ぎなかった。1次近似のたびに誤差が積もる。
- **LPOの提案**: ターゲット計算と射影を分離し、ターゲットを明示的に求める**Listwise Gibbs Target** + 厳密なKL最小化による射影、という二段階フレームワークに再構築。
- **二つの変種**: **LPO_fwd**（forward KL、mode-covering、log-barrierで多様性確保）と**LPO_rev**（reverse KL、mode-seeking、on-policy点で標準PGと等価）。
- **実験結果**: Pass@1で3基線×5シナリオ = **15設定中13勝**、Pass@kでLPO_fwdは**15/15全勝**。応答エントロピー・勾配ノルム・応答長さのすべての側面で安定性が向上。
- **実装コスト**: 温度τは基線の統計量（$\sigma_G$ / $\mu_G$ / 1）から自動決定、計算オーバーヘッドはゼロ。

---

## 1. 背景：「GRPOって何してるんだろう」

DeepSeek-R1以降、**RLVR（Reinforcement Learning with Verifiable Rewards）**は大規模推論モデルの後学習における定番になった。GRPO（Group Relative Policy Optimization）はその中核で、1プロンプトからK本の応答をサンプリングし、verifierで0/1報酬を付けた上で、グループ内の相対的なadvantageで方策を更新する。criticが要らない軽量さが受けて、DAPO、Dr.GRPO、RLOO、MaxRLなど多数の派生が生まれた。

ただ、2026年8月に清華大学自動化系の季向陽教授チームとTencent Hunyuan LLM部門が共同で公開した**LPO（Listwise Policy Optimization）**[^1]は、これらの手法をまったく別の角度から眺め直している。著者らの主張は明快だ。

> **Group-based policy gradientはみんな、response simplex上のターゲット分布に「1次近似」で射影していた。LPOはこれを「exact projection」に置き換える。**

### 1.1 Response Simplexとは

1プロンプト$x$に対して$K$本の応答$\{y_k\}_{k=1}^K$をサンプリングしたとする。current policy $\pi_\theta$は各応答に確率を割り振っているが、論文ではこれを以下のように**logit化した上でsoftmaxを掛けたlistwise分布**として扱う。

$$
P_{\theta,k} = \mathrm{softmax}(s_\theta)_k, \quad s_{\theta,k} = \log\frac{\pi_\theta(y_k|x)}{\pi_b(y_k|x)}
$$

この$P_\theta$は**K次元確率単体**$\Delta^{K-1} = \{p \in \mathbb{R}^K: p_k \ge 0, \sum_k p_k = 1\}$の上にあり、論文ではこれを**response simplex**と呼んでいる。on-policy点では$s_\theta = 0$、つまり$P_\theta = (1/K, \ldots, 1/K)$の**一様分布**に退化する。

### 1.2 既存手法の「暗黙のターゲット」

GRPOのadvantageは$A_k = (R_k - \mu_G)/\sigma_G$。これを$\mathrm{softmax}$に通すと$\mathrm{softmax}(R/\sigma_G)$になる。実はDr.GRPO、MaxRL、RLOOなど、advantageの正規化方式が違うように見えて、すべて$\mathrm{softmax}(R/\tau)$の形の**同じ構造のターゲット**を誘導している。$\tau$（温度）の選び方が違うだけ。

論文ではこれを以下の表で整理している。

| 手法 | advantage $A_k$ | 暗黙のターゲット $w^*$ | 温度 $\tau$ |
|------|----------------|-------------------------|--------------|
| Dr.GRPO / RLOO | $R_k - \mu_G$ | $\mathrm{softmax}(R)$ | $1$ |
| GRPO / DAPO | $(R_k - \mu_G)/\sigma_G$ | $\mathrm{softmax}(R/\sigma_G)$ | $\sigma_G$ |
| MaxRL | $(R_k - \mu_G)/\mu_G$ | $\mathrm{softmax}(R/\mu_G)$ | $\mu_G$ |

問題はここから先だ。group-based PGは**このターゲットに向かって1次近似で「ふわっと」射影する**。on-policy点では正確だが、方策が更新されて$\pi_\theta \ne \pi_b$になるたびに、応答ごとの係数に$\mathcal{O}(\bar\delta \cdot (1+\|A\|_\infty)/K)$の近似誤差が乗る。**1回1回の更新誤差は小さく見えても、蓄積すれば無視できなくなる**。

![Response Simplex: Implicit vs Explicit Projection](/images/lpo-listwise-policy-optimization-response-simplex/fig1.png)

---

## 2. 提案手法：LPO

### 2.1 二段階フレームワーク

LPOの核となる考え方は、ターゲット計算と最適化を**分離**すること。

$$
\underbrace{w^* = \arg\max_{w \in \Delta^{K-1}} \hat{J}(w)}_{\text{(i) Target: 何に射影するか}} \qquad \underbrace{\theta' = \arg\min_\theta D(w^* \| P_\theta)}_{\text{(ii) Projection: どう射影するか}}
$$

「**何を目指すか**」と「**どうそこへ近づけるか**」を別々に設計できる。これがLPOの柔軟さの源泉だ。

### 2.2 ターゲット計算：Listwise Gibbs Target

response simplex上に近接目的を定義する。

$$
\max_{w \in \Delta^{K-1}} \hat{J}(w) = \sum_{k=1}^K w_k R_k - \tau\, D_{\mathrm{KL}}(w \| P_t)
$$

**定理1（Listwise Gibbs Target）**: 上記の唯一の最大化解は

$$
w_k^* = \mathrm{softmax}(\phi)_k, \quad \phi_k = \frac{R_k}{\tau} + s_{t,k}
$$

特にon-policy条件（$P_t$が一様分布）では$w^* = \mathrm{softmax}(R/\tau)$となり、既存手法の暗黙ターゲットを**厳密に再現**する。さらに$K \to \infty$では$K$個限定だった和が全応答空間に拡張され、古典的なKL正則化RL目的$\pi^*(y) \propto \pi_t(y)\exp(R(y)/\tau)$に収束する。つまりLPOターゲットは**「有限サンプルの近似」ではなく、有限サンプルの世界でこそ閉形式で解ける**定式化になっている。

### 2.3 性能改善保証

**定理2（Performance Improvement Bound）**: 完全射影（$\epsilon_{\mathrm{proj}}=0$）下では、$P_t \ne w^*$である限り報酬が**厳密に改善**する。

$$
\hat{R}(P_{t+1}) \ge \hat{R}(P_t) + \underbrace{\tau\bigl[D_{\mathrm{KL}}(w^*\|P_t) + D_{\mathrm{KL}}(P_t\|w^*)\bigr]}_{\text{Jeffreys ダイバージェンス} \ge 0} - 2R_{\max}\epsilon_{\mathrm{proj}}
$$

これは**TRPOのsurrogate gap**に相当する保証を、有限単体上で閉形式で与えている。蓄積する1次近似誤差は**理論上ゼロ**になる。

### 2.4 射影の二つの選択肢

**Forward KL版（LPO_fwd）**：

$$
\min \mathcal{L}_{\mathrm{LPO_{fwd}}} = D_{\mathrm{KL}}(w^* \| P_\theta) \;\Rightarrow\; \nabla_\theta \mathcal{L}_{\mathrm{LPO_{fwd}}} = \sum_k \underbrace{(P_{\theta,k} - w_k^*)}_{c_k^{\mathrm{fwd}}} \nabla_\theta \log \pi_\theta(y_k|x)
$$

この勾配係数$c_k^{\mathrm{fwd}}$は以下の3つを満たす（**推論1**）。

- **有界性**: $|c_k^{\mathrm{fwd}}| \le 1$
- **零和性**: $\sum_k c_k^{\mathrm{fwd}} = 0$
- **自己補正性**: $P_\theta \to w^*$で$c_k^{\mathrm{fwd}} \to 0$

さらにforward KLは**log-barrierでmode-covering**を保証する（**推論2**）。$w_k^* \ge \alpha$で$D_{\mathrm{KL}}(w^*\|P_\theta) \le D$のとき、$P_{\theta,k} > \alpha\exp(-D/\alpha - 1)$。**どれか1本の高報酬応答が確率質量ゼロに潰されることが原理的に防げる**。

**Reverse KL版（LPO_rev）**：

$$
\min \mathcal{L}_{\mathrm{LPO_{rev}}} = D_{\mathrm{KL}}(P_\theta \| w^*) \;\Rightarrow\; \nabla_\theta \mathcal{L}_{\mathrm{LPO_{rev}}} = \sum_k \underbrace{P_{\theta,k}(d_k - \bar{d})}_{c_k^{\mathrm{rev}}} \nabla_\theta \log \pi_\theta(y_k|x)
$$

ここで$d_k = s_{\theta,k} - \phi_k$、$\bar{d}$は$P_\theta$重み付き平均。$c_k^{\mathrm{rev}}$も零和で自己補正。**on-policy点でLPO_revは標準戦略勾配と厳密に等価**になる。すなわちLPO_revは「今やっているPGの計算を、exact射影で再解釈し直したもの」だ。

### 2.5 全体像

![LPO Framework: Target-Projection](/images/lpo-listwise-policy-optimization-response-simplex/fig2.png)

フレームワーク全体としては、**ターゲット計算（青）→ fwd/rev分岐 → 勾配更新**の3ステップで、既存group-based PGと同じトレーニングパイプライン上に乗る。温度$\tau$は**既存基線の統計量から自動決定**されるので、新たなチューニング負担は発生しない。

---

## 3. 実験結果

### 3.1 設定

- **タスク**: 論理推論（Countdown-34）、数学（MATH/AIME24/25/AMC23/OlympiadBench）、コード（PRIME code）、マルチモーダル幾何（Geometry3k）
- **モデル**: Qwen3-1.7B/4B/8B-Base、Qwen2.5-VL-3B-Instruct、Qwen/Llama/DeepSeek/Mistralの4ファミリー
- **基線**: GRPO（$\tau=\sigma_G$）、Dr.GRPO（$\tau=1$）、MaxRL（$\tau=\mu_G$）の3種
- **LPO変種**: 各基線に対し**完全に同じ$\tau$でLPO_fwd / LPO_revを実装**。純粋に「近似vs厳密射影」の差分を切り分けている
- **実装**: verlフレームワーク

### 3.2 主要結果

| 指標 | LPO_fwd | LPO_rev |
|------|---------|---------|
| Pass@1（vs 3基線×5タスク） | **13/15 勝** | **13/15 勝** |
| Pass@k | **15/15 全勝** | 11/15 勝 |

![LPO vs Group-based PG: Head-to-head Comparison](/images/lpo-listwise-policy-optimization-response-simplex/fig3.png)

LPO_fwdが**Pass@kで15/15全勝**しているのが目を引く。これはforward KLの**log-barrierによるmode-covering**の賜物だ。複数の正解経路を保持できる性質が、k本サンプリングしてどれか1本が当たるPass@k評価で効いてくる。

### 3.3 訓練動態

LPOの効果を理解するうえで、Pass@1やPass@kの数字以上に面白いのが**訓練プロセスの可視化**だ。

![Training Dynamics: LPO maintains diversity, stability, and depth](/images/lpo-listwise-policy-optimization-response-simplex/fig4.png)

3つのメトリクスで一貫した傾向が見える。

- **応答エントロピー（左）**: LPO_fwd / LPO_revはGRPOより**明らかに高いエントロピーを維持**。モード崩壊が起きていない。
- **勾配ノルム（中央）**: LPOは**小さく安定**。推論1の有界性保証通り、$|c_k| \le 1$が効いている。
- **応答長（右）**: LPOは**より長いCoT**を生成。LPO_fwdが最長で、mode-covering的な振る舞いと整合的。

### 3.4 重要なAblation

#### 3.4.1 Listwise vs Pointwise射影

LPOの性能向上が**ターゲット設計ではなくlistwise射影の構造**に由来することを切り分けるため、ターゲット$w^*$を固定したままlistwise分布$P_\theta$をpointwise処理に置き換えたablationを実施。

$$
\mathcal{L}_{\mathrm{point}} = -\sum_k w_k^* \log \pi_\theta(y_k|x)
$$

結果：**pointwise版は深刻な性能劣化**を起こした。応答間の競合・共依存が消えると最適化が破綻する。これはLPOの効果が「ターゲット設計」ではなく「listwise射影が持つ構造的分散削減」にあることを示している。

#### 3.4.2 グループサイズ$K$の影響

$K \in \{2, 4, 8, 16, 32\}$で検証した結果：

- LPOは**全$K$で競争力**を維持
- 特に**小グループ（$K=2, 4$）でLPOの優位が顕著**
- LPO_fwdはPass@64で**スケールが非常によい**（mode-coveringが活きる）

小グループで効くのは、サンプルが少ないほど1次近似誤差が相対的に大きくなるため、それを**exact projectionで打ち消す**LPOの構造的メリットが際立つからだろう。

---

## 4. 考察

### 4.1 「Advantage正規化」議論からの脱却

ここ1〜2年のRLVR研究は、**「どのadvantage正規化が一番良いか」**という視点でGRPOを改善してきた。Dr.GRPO、RLOO、MaxRL、DAPOなど。ただLPOの視点から見れば、これらは**同じ暗黙ターゲットへの異なる温度スケジュール**に過ぎない。1次近似を使っている限り、どれだけ正規化を工夫しても誤差が消えない。

LPOの貢献は、この議論を**advantage正規化から「response simplex上の射影」へ**とパラダイムを移したこと。**ターゲット計算と射影メカニズムを分離**することで、

1. ターゲットは**閉形式で厳密解**が得られる
2. 射影は**forward/reverse KLを自由に選べる**（構造的性質が異なる）
3. 温度$\tau$は**統計量から自動決定**される

という設計自由度を獲得した。

### 4.2 Mode-coveringという設計選択

LLMの推論タスクで**「正解が複数経路ある」**のは普通のことだ。数学でもコードでも、別解が複数あってよい。GRPOのようなmode-seekingな最適化は、訓練中に1つの経路に確率質量を集中させ、他を潰しがち。**Pass@kのような「k本中どれか当たればいい」評価で性能が伸び悩む**のはこのためだ。

LPO_fwdのforward KLは、log-barrierで「正解経路が確率ゼロに潰れない」ことを保証する。これは**訓練の安定化とPass@k性能の両得**で、推論モデルの商品化（実用的にはtemperature > 0で複数サンプリングして多数決・自己一致を使う）ではかなり重要な性質だと思う。

### 4.3 実装面での扱いやすさ

LPOの副次的な美点は、**既存パイプラインにそのまま載る**こと。verlのような標準的なRLVRフレームワークを使っているなら、

1. advantageの代わりにターゲット$w^* = \mathrm{softmax}(R/\tau + s_t)$を計算
2. 勾配係数$c_k$をforward / reverseどちらかで計算
3. あとは既存の勾配更新ロジックで$\theta$を更新

の3ステップを差し替えるだけ。**ハイパーパラメータ探索の幅も増えない**（温度は基線の統計量から決まる）というのは、実装者にとって非常にありがたい。

### 4.4 限界と今後の展開

論文自身が認めている限界：

- **ステップレベルのlistwise射影は未開拓**。今回はシーケンスレベル（応答1本単位）の射影。Chain-of-Thoughtのステップ単位でターゲットを設定したらどうなるかは興味深い。
- **結果報酬設定に焦点**を当てている。プロセス報酬モデル（PRM）への拡張は次の研究テーマ。
- 現在は**KLダイバージェンスのみ**。f-divergenceやWassersteinなど他の距離への拡張でさらに構造が変わる可能性。

---

## 5. 関連研究

### 5.1 RLVR

DeepSeek-R1[^2]、OpenAI o1[^3]に代表される、verifier付き報酬での推論モデル後学習。GRPO[^4]を中心に、DAPO、Dr.GRPO、RLOO、MaxRL、ReMax、REINFORCE++など派生多数。本質的な改善は「advantage正規化」と「安定化」に集中してきた。

### 5.2 RL as Probabilistic Inference

Dayan & Hinton (1997)、Ziebart (2010)、Levine (2018)など、RLを確率推定として再解釈する古典研究。MPO[^5]、TRPOなど**明示的ターゲット射影**は連続行動空間では関数近似が必要で近似精度が課題だった。LPOの強みは**有限単体上で閉形式解が得られる**点。RLVR固有の構造（有限サンプリング）を活かした設計。

### 5.3 Listwise手法

学習ランキングの文脈で生まれたlistwise最適化（Cao et al., 2007; Luce et al., 1959）。LLM alignmentではLiPO[^6]がlistwise preference optimizationを提案しているが、これは**オフライン嗜好データ**ベース。LPOは**オンラインRLVR**設定で、**response simplex上のexact projection**という点が決定的に異なる。

### 5.4 並行研究

- **TPO（Kaddour, 2026）**: 同様のlistwise forward KL射影を実証面から検証
- **Shu et al., 2026**: 報酬ベースのGibbs分布としてターゲットを明示化する方向

---

## 6. まとめ

LPO（Listwise Policy Optimization）は、group-based RLVRを**「response simplex上の暗黙の1次近似射影」から「明示的なexact projection」へ**と格上げした研究。3つの柱がある。

1. **幾何学的再定式化**: 既存PG手法が「$w^* = \mathrm{softmax}(R/\tau)$への近似射影」だったことを明示
2. **Listwise Gibbs Target**: 有限単体上で閉形式にターゲットを計算
3. **Forward / Reverse KLの選択自由**: mode-covering（fwd）とmode-seeking（rev）の構造的トレードオフを明示的に制御

実験では3基線 × 5タスク = 15設定中、Pass@1で13勝、Pass@kでLPO_fwdが15/15全勝。**追加のハイパーパラメータ探索なし、既存パイプラインへの組み込みのみ**で、応答多様性・勾配安定性・CoT深度のすべてを改善した。

「**GRPOを改良するって、もうadvantage正規化の問題じゃないんだ**」と気づかせてくれる研究。RLVRの実装者・研究者にとって、2026年後半の必読文献だと思う。

---

## 7. 参考

[^1]: Qu, Y. et al. (2026). *Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex*. arXiv:2605.06139. <https://arxiv.org/abs/2605.06139>
[^2]: Guo, D. et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
[^3]: Jaech, A. et al. (2024). *OpenAI o1 System Card*. arXiv:2412.16720.
[^4]: Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
[^5]: Abdolmaleki, A. et al. (2018). *Maximum a Posteriori Policy Optimisation*. ICLR 2018.
[^6]: Liu, T. et al. (2025). *Listwise Preference Optimization for LLM Alignment*. arXiv:2503.04322.
