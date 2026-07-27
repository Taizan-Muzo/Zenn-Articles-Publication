---
title: "VIGOR：報酬分散が勾配を決める——漸進的Rollout配分でGRPOを2.3倍効率化"
emoji: "🎯"
type: "tech"
topics: ["LLM", "強化学習", "GRPO", "RLVR", "推論"]
published: false
---

## TL;DR

GRPOはLLM推論強化の標準的手法だが、各promptに固定数のrolloutを生成する設計が効率を阻害している。本論文は**報酬分散が勾配サイズを直接決定する**という理論的洞察（Theorem 1: $\|\nabla \mathcal{L}\| \leq M\sigma$）に基づき、**VIGOR**（VarIance Guided Online Rollout allocation）を提案する。少量のrolloutから始めて分散が高いpromptに段階的に予算を集中させる反復的割当手法で、数学推論で最大**2.3倍**、コーディングで**1.49倍**のrollout効率化を達成。Pareto分布仮定の下で閉形式の加速比（Theorem 2）も導出している。著者はHeyang Jiang、Henry Liu、Baharan Mirzasoleiman（UCLA）。

![VIGORのアルゴリズム概要](/images/vigor-variance-guided-rollout-allocation/fig1.png)

## 背景：GRPOのrollout問題

### RLVRとGRPO

強化学習 with verifiable rewards（RLVR）は、客観的に検証可能な報酬信号を用いてLLMの推論能力を向上させる手法として広く採用されている。中でもGRPO（Group Relative Policy Optimization）は、価値関数を不要とし、各promptに対して生成した複数ロールアウトの群内相対報酬で優劣を評価する設計で、DeepSeek-R1やDAPOなど産業規模の事後学習に標準的に使われている。

GRPOの更新式は次の通りである。各prompt $x$ に対して $G$ 個の応答 $\{y_1, \dots, y_G\}$ を生成し、群内相対優位性を計算する：

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \delta}$$

ここで $\bar{r}$、$\sigma_r$ は群内報酬の平均・標準偏差、$\delta$ は数値安定化項。この優位性を用いてPPO風のクリップ目的で更新する。

### 二つの致命的な観察

著者はGRPOの訓練プロセスを詳細に分析し、二つの重要な観察を行った。

**観察1：rollout生成が訓練コストの主要因である。** Qwen2.5-1.5Bでの測定では、rollout生成が総訓練時間の**50%以上**を占める。batch sizeが閾値を超えると、生成時間はrollout数にほぼ比例して増大し、長いCoTを生成する設定では顕著になる。

**観察2：訓練初期が性能向上の大部分を担う。** MATHデータセットでの実験で、1.5Bおよび3Bモデルでは最初の1エポックで最終検証スコア向上の**約80%**が達成される。7Bモデルでも最初のエポックの向上が以後6エポックの合計に匹敵する。つまり訓練初期の効率が極めて重要であり、既存手法（GRESO、DPSなど）が訓練履歴に依存して2エポック目以降でないと機能しないのは致命的な制約となる。

### 既存アプローチの限界

GRPOの効率化は大きく二つの方向で試みられてきた：

1. **困難promptサンプリング**（DAPO、GRESO、DPS）：多数のrolloutを生成して非零分散の困難promptを選ぶ。しかし生成過多による計算オーバーヘッドが大きく、履歴ベース手法は初期エポックで機能しない。
2. **rollout利用効率化**（RL-ZVP、PODS、M2PO）：零優位性サンプルに代替信号を与えるか、古いrolloutを再利用する。しかし訓練目標の変更や限定的な改善に留まる。

VIGORはこれらの限界を、**GRPOパラダイムを変更せずに**、オンラインの報酬分散信号のみで動的に予算を再配分することで解決する。

## 方法詳解

### 核心定理：分散が勾配を決める

VIGORの理論的基盤は、二元報酬設定下でのGRPO勾配の上界に関する定理である。

> **定理1**: KL項を除去したGRPOにおいて、二元報酬 $r_i \in \{+1, -1\}$、重要性比の勾配の一様有界 $\|\nabla_\theta \rho_i(\theta)\| \leq M$ を仮定すると、サンプルのGRPO目的の勾配は $\|\nabla_\theta \mathcal{L}_{\mathrm{GRPO}}\| \leq M\sigma$ を満たす。

証明は二段階で行われる。第一段階（補題1）では、優位性の絶対値の総和 $S = \sum_i \|A_i\|$ を用いて $\|\nabla \mathcal{L}\| \leq \frac{M}{G}S$ を導出。第二段階（補題2）では、正例の比率を $p$ としたとき、z-score正規化後の正負の優位性の絶対値の重み付き総和が $S = G\sigma$ となることを確認する。具体的には、正例の優位性 $A_+ = \frac{2(1-p)}{\sigma}$、負例の優位性 $A_- = \frac{-2p}{\sigma}$ であり：

$$S = G\left(p\|A_+\| + (1-p)\|A_-\|\right) = G \cdot \frac{4p(1-p)}{\sigma} = G\sigma$$

が成り立つ。$\sigma^2 = 4p(1-p)$ だから $\frac{4p(1-p)}{\sigma} = \sigma$ である。

![定理1の可視化](/images/vigor-variance-guided-rollout-allocation/fig2.png)

この定理の含意は明快である：**分散が最大のprompt（$p=0.5$、即ち半分成功・半分失敗）が最大の勾配信号を生む**。全問正解（$p=1$）や全問不正解（$p=0$）のpromptは分散ゼロで学習信号を生まない。これは「学習フロンティア」にある中難度promptこそがGRPO訓練のエンジンであることを意味する。

### VIGORアルゴリズム

VIGORは定理1の洞察をアルゴリズムに翻訳する。各訓練ステップで以下の手順を実行する：

**初期化**: バッチ $\mathcal{B}$ の全promptに初期rollout予算 $m_0$（例：2）を割り当てる。

**精炼反復** $t = 1, \dots, T$:

1. **rollout生成**: 活躍集合 $\mathcal{A}_t$ の各prompt $x$ に対し、$m_x^{(t)}$ 個のrolloutを生成
2. **分散計算**: $u_x = \mathrm{Var}(\{R(x,y) \mid y \in \mathcal{Y}_x\})$ を計算
3. **選択**: 分散上位 $\alpha$ 割合（例：50%）のpromptを次ラウンドに保持
4. **予算拡張**: 保持されたpromptの予算を $\gamma$ 倍（例：2倍）に拡大

最終的に保持された全rolloutでGRPO更新を実行する。重要なのは、**総rollout予算はGRPOと同一**であり、配分方法のみを変更する点である。

デフォルトハイパーパラメータは $T=4$、$m_0=2$、$\gamma=2$、$\alpha=0.5$。つまり8個のrollout予算を、2→4→8→16と段階的に高分散promptに集中させていく。

### 加速比の理論保証

著者はPareto分布仮定の下でVIGORのGRPOに対する閉形式の加速比を導出している。

> **定理2**: prompt級報酬分散 $u$ の上裾がPareto分布（閾値 $u_{\min}$、形状パラメータ $k > 1$）に従うと仮定する。予算保存ケース（$\alpha\gamma = 1$）での加速比は：
>
> $$\frac{\tau_{\mathrm{GRPO}}}{\tau_{\mathrm{VIGOR}}} = \left(\frac{(\alpha^{-1/k})^{T} - 1}{T(\alpha^{-1/k} - 1)}\right)^{1/3}$$

この加速比は精炼ラウンド数 $T$ とともに単調増加し、分布の裾が重いほど（$k \to 1^+$）大きくなる。直感的には、分散の分布に裾が重いほど高分散promptに予算を集中させる利益が大きいことを表している。計算偏重型（$\alpha\gamma > 1$）では $\Theta((\alpha^{-1/k})^{T/3})$ となり、やはり $T$ とともに指数的に増大する。

![加速比とカリキュラム学習](/images/vigor-variance-guided-rollout-allocation/fig4.png)

## 実験結果

### 設定

数学推論ではQwen2.5-1.5B/3B/7BおよびPhi-4-Mini-Instructを使用。1.5B/3BはMATHのみ、7B/PhiはMATH+DAPO混合で訓練。評価はMath500、AIME24、AMC、Minerva Math、Gaokao、Olympiad Benchの6ベンチマーク。コーディングではQwen3-8BをLiveCodeBench v6で訓練。

基線はGRPO、GRESO-r（同一rollout予算）、GRESO-g（同一勾配ステップ・約2倍rollout）、RL-ZVP。全手法で同一の有効batch size、同一stepあたりrollout予算、同一オプティマイザ設定を使用する。

### 数学推論：2.3倍のrollout効率化

VIGORは全4モデル規模で目標精度到達に必要なrolloutを一貫して削減した。最大の改善はQwen2.5-3Bで**2.3倍**。最終精度でも、公平なrollout予算マッチ（vs GRPO、GRESO-r、RL-ZVP）で全規模最多平均スコアを記録した：

| モデル | GRPO | GRESO-r | GRESO-g (2×rollout) | RL-ZVP | **VIGOR** |
|--------|------|---------|---------------------|--------|-----------|
| 1.5B | 29.5 | 28.4 | 30.7 | 29.8 | **30.1** |
| 3B | 36.6 | 36.7 | 37.7 | 37.3 | **37.9** |
| 7B | 49.8 | 49.0 | 50.3 | 49.0 | **51.0** |
| Phi-4 | 42.9 | 41.8 | 43.1 | 42.4 | **43.4** |

特に注目すべきは、GRESO-gが約2倍のrollout予算を使っているにもかかわらず、3B/7B/Phi-4ではVIGORが上回っている点である。

![実験結果の比較](/images/vigor-variance-guided-rollout-allocation/fig3.png)

### コーディング：1.49倍の効率化と3.4ptの精度向上

LiveCodeBench v6（Qwen3-8B、3種子平均）では：

- **Full Pass Rate**: GRPOの最終値に**1.49倍**少ないrolloutで到達。最終的には44.0%→45.7%（+1.7pt）
- **Average Test Pass Rate**: 63.4%→66.8%（**+3.4pt**）

### 追加時間オーバーヘッドなし

VIGORは反復的な分散計算と選択を伴うが、実際の生成・検証時間は基線とほぼ同等。GRESOは多数rollout生成のオーバーヘッドで顕著に遅くなる。Qwen2.5-3Bでは、VIGORが50歩で37.8%到達する間にGRESOは105歩を要し、**2.65倍**の実時間加速となった。

## 考察

### 涌現的カリキュラム学習

VIGORの最も興味深い性質の一つは、明示的な難度ラベルや手作りのスケジュールなしに**カリキュラム学習が自然発生**することである。MATHのLevel 1-5難度标注を用いた解析で、rollout重み付き平均難度が訓練進行とともに単調に増加することが確認された。

メカニズムは直感的である：訓練初期は中難度promptが高分散を示すが、モデルが学習して容易なpromptを解けるようになると、それらの分散が低下して選択から外れる。代わりに、より難しいpromptが新たな「学習フロンティア」として高分散を示し、予算が再配分される。この過程が訓練全体を通じて繰り返される。

### 分散選択の優位性

アブレーションで分散選択、難度選択、ランダム選択を比較した結果、分散選択が一貫して最善。難度（成功率）のみで選ぶより、分散（学習信号の大きさ）で選ぶ方が効率的。これは定理1の予測と一致する。

### 大規模rollout予算でも有効

候補プールサイズ $n=32$ での実験でもVIGORはGRPOに対し2.77倍の効率化を維持。事前に大プールを生成してから下サンプリングするPODSは、破棄される候補の生成コストがかさみ、VIGORの5.34倍のrolloutを要した。

### 低い偽陰性率

初期rollout数 $m_0=2$ で情報量の高いpromptを見逃す懸念に対し、実測で偽陰性率はわずか7.9%。見逃されたpromptも、戦略が変化する後続エポックで再び高分散を示し、選択され直す可能性がある。

## 関連研究

VIGORはGRPO効率化の文脈で以下の研究系譜に位置づけられる：

- **DAPO**: 非零分散の困難promptを動的サンプリングする先駆。しかし過剰rollout生成による計算コストが問題。
- **GRESO**: 訓練履歴から難度を推定し後続エポックで活用。初期エポックの恩恵がない。
- **RL-ZVP**: 零優位性サンプルにエントロピーベースのtoken級報酬を与える。訓練目標を変更する。
- **PODS**: 大規模候補プールから最も多様な報酬信号を提供する部分集合を選択。計算コストが高い。
- **VIP**: 埋め込みベースの成功率予測で難度を推定。ノイズに敏感。

VIGORの差別化要因は、(1) 訓練目標を変更しない、(2) 訓練初段階から機能する、(3) 理論的保証がある、(4) 追加の推論や予測モデルが不要、の4点である。

## まとめ

VIGORは「報酬分散が勾配サイズを決める」という理論的洞察を出発点に、GRPOのrollout予算を高分散promptに段階的に集中させる手法である。数学推論で2.3倍、コーディングで1.49倍の効率化を達成しつつ、訓練パラダイムを変更せず、追加オーバーヘッドもほぼない。Pareto分布仮定の下で加速比の閉形式表現も与えられており、理論と実験の両面から有効性が裏付けられている。

本論文が示唆するより大きなメッセージは、RLVRの効率化において「データ選択の設計空間」がまだ十分に探索されていないということである。VIGORは分散という直接的信号を用いるが、勾配ノルムや情報利得など他の信号も考えられる。また、本手法の非同期RLやマルチターンエージェントRLへの拡張も興味深い方向である。

## 参考

- Jiang, H., Liu, H., & Mirzasoleiman, B. "Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning." arXiv:2607.22002, 2026.
- Shao, Z. et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024. (GRPO)
- Yu, Q. et al. "DAPO: An Open-Source LLM Reinforcement Learning System." arXiv:2503.14476, 2025.
- Razin, E. et al. "Implicit Regularization in Tensor Factorization." 2025. (最適化タイムスケール理論)
