---
title: "RIPO：PPO-Clipの探索崩壊を「リーマン等長クリッピング」で克服"
emoji: "📐"
type: "tech"
topics: ["LLM強化学習", "PPO-Clip", "リーマン幾何", "探索崩壊", "信頼領域最適化", "推論モデル"]
published: true
---

## TL;DR

- PPO-Clip（およびGRPO/DAPOが継承するクリッピング）は、方策間のずれを**ユークリッド距離**で測っている。しかし方策空間の真の幾何はKLダイバージェンスが定める**リーマン多様体**であり、両者はそもそも整合しない。
- この「幾何のズレ」が、高確率トークンでは過剰に攻撃的になり、低確率トークンでは過剰に保守的になる原因。探索が起きるのは後者の領域なので、結果として**探索が潰れる（exploration collapse）**。
- 提案手法 **RIPO（Riemannian Isometric Policy Optimization）** は、トークンごとにクリップ幅を $\epsilon_{s,a}=\sqrt{\delta/\pi_{\mathrm{old}}}$ と動的に決める**リーマン等長クリッピング（RIC）**を導入。すべての更新が多様体上で等しい幾何距離 $\delta$ を消費する「等長」な信頼領域を実現する。
- 4モデル×7競技レベル数学ベンチマークでGRPOを一貫して上回り、平均で **+17〜+37%**、AIME24単体では最大 **+62%**（Qwen3-1.7B）の改善。ICML 2026。

---

## 背景：クリッピングはなぜ「効く」のか、そしてなぜ潰れるのか

LLMの推論能力を高める後学習の主役は、今や強化学習だ。その基盤エンジンとなっているのが **PPO-Clip** であり、GRPOやDAPOといった昨今の手法も、本質的にはこのクリップ機構を引き継いでいる。クリップの役割は「方策が一度の更新で大きく飛びすぎないよう、信頼領域（trust region）を保つ」こと。直感的にはよく効く。だが現場では、**エントロピーが早期に崩壊し、モデルが探索をやめて頭打ちになる**という現象が繰り返し報告される。

これまでの対策（DAPOのクリップ範囲拡大、GSPOの系列レベル比への移行、動的クリップなど）は、いずれも**経験的**な手直しだった。「なぜPPO-Clipは探索を潰すのか」という**根本原因**には踏み込めていなかった。本論文の問いはここにある。

### 方策多様体の幾何

方策間の違いを測る自然な尺度は、TRPO以来の**KLダイバージェンス**だ。これをパラメータ $\theta$ の周りで二次展開すると、

$$
D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta)\approx \frac12 \Delta\theta^\top F(\theta_{\mathrm{old}})\,\Delta\theta
$$

となる。$F(\theta)$ は**フィッシャー情報行列**であり、これが方策空間に**リーマン計量**を与える。つまり、方策空間はただのユークリッド空間ではなく、点によって「長さの目盛り」が変わる曲面（リーマン多様体）なのだ。

この幾何をトークンごとの重要度比 $r_{s,a}(\theta)=\pi_\theta(a|s)/\pi_{\mathrm{old}}(a|s)$ で書き直すと、状態 $s$ での真の幾何距離は

$$
d_{\mathrm{geom}}(\pi_{\mathrm{old}},\pi_\theta)\;\propto\;\frac12\,\pi_{\mathrm{old}}(a|s)\,(r_{s,a}-1)^2
$$

と、$\pi_{\mathrm{old}}$ に**依存する**形になる。重要なのは、高確率の行動ほど距離が「伸びやすく」、低確率の行動ほど距離が「縮む」ことだ。

### PPO-Clipの幾何的不整合

一方PPO-Clipは、制約 $\|r-1\|<\epsilon$ を課す。これは

$$
d_{\mathrm{clip}}=(r-1)^2
$$

すなわち **$\pi_{\mathrm{old}}$ を無視した一様なユークリッド距離**を暗に使っている。論文は命題3.1で、この不一致を以下のように総括している。

> PPO-Clip incorrectly employs a Euclidean metric to measure the discrepancy between policies... This leads to overly conservative updates in low-probability regions while aggressive in high-probability regions, ultimately causing exploration collapse.

![図1：PPO-Clipはユークリッド距離で方策のずれを測るため、多様体の幾何と食い違う。低確率トークンでは信頼領域予算がほとんど消費されない](/images/ripo-riemannian-isometric-policy-optimization/fig1.png)

具体例で見ると明快だ。$\pi_{\mathrm{old}}=0.8$ の行動で $r=1.2$ としたときの幾何距離は $0.5\times0.8\times0.2^2=0.016$ だが、$\pi_{\mathrm{old}}=0.01$ の行動で同じ $r=1.2$ でも幾何距離は $0.0002$ に過ぎない。PPO-Clipは両者を同じ $\epsilon$ で扱うから、**低確率側の予算は事実上使われず**、そこが「探索の源泉」であるにもかかわらず更新が抑止される。これが探索崩壊の構造的メカニズムだ。

---

## 方法詳解：Riemannian Isometric Policy Optimization

### リーマン等長クリッピング（RIC）

RIPOの核心は、信頼領域の予算を**幾何距離で均一**に消費させること。すなわち、すべての状態・行動・トークンについて

$$
d_{\mathrm{geom}}=\frac12\,\pi_{\mathrm{old}}(a|s)\,(r_{s,a}-1)^2\;\le\;\delta
$$

を満たすようにクリップ幅を決める。これを $r$ について解くと、

$$
|r_{s,a}-1|\;\le\;\sqrt{\frac{2\delta}{\pi_{\mathrm{old}}(a|s)}}
$$

が得られる。定数倍を超パラメータ $\delta$ に吸収して、実際の動的閾値は

$$
\epsilon_{s,a}(\pi_{\mathrm{old}})=\sqrt{\frac{\delta}{\pi_{\mathrm{old}}(a|s)}}
$$

となる。**低確率トークンほど大きな更新を許し、高確率トークンほど更新を絞る**。これで、一見逆説的に見えるが、多様体上ではどの更新も等しい幾何距離 $\delta$ を進む「等長（isometric）」なステップになる。

![図2：RIPOの動的クリップ幅は $\epsilon=\sqrt{\delta/\pi_{\mathrm{old}}}$ に従い、低確率トークンほど大きな更新を許容する](/images/ripo-riemannian-isometric-policy-optimization/fig2.png)

### 全体目的関数

RICをGRPOの系列相対アドバンテージと組み合わせると、RIPOの目的関数は

$$
\mathcal{J}_{\mathrm{RIPO}}(\theta)=\mathbb{E}\left[\frac{1}{\sum_i\|o_i\|}\sum_{i,t}\min\!\Big(r_{i,t}\hat A_{i,t},\;\mathrm{clip}\big(r_{i,t},\,1{-}\epsilon_{i,t},\,1{+}\epsilon_{i,t}\big)\hat A_{i,t}\Big)\right]
$$

となる。ここで $\epsilon_{i,t}$ は上の分布依存の動的閾値。論文はRIPOの読み方を「ripple（波紋）」とし、方策の改善が多様体上を滑らかに伝播する様を表現している。実装上のデフォルトは $\delta=0.05$、両側クリップ $[0.5,\,10]$、KLペナルティなし。つまり**既存コードのクリップ幅を「固定値」から「$\sqrt{\delta/\pi_{\mathrm{old}}}$ で計算する値」に差し替えるだけ**で導入できる。

### バイアス・バリアンスの観点

離尾（off-policy）重要度サンプリングの分散は、二次モーメント $\sum_x \pi_{\mathrm{old}}(x)\,r(x)^2 A(x)^2$ で支配される。低確率 $x$ では $r$ がいくらでも大きくなり得るため、無制限なISでは分散が爆発する。

- **PPO-Clip**： $r\le1+\epsilon$ で切るのでクリップ済みサンプルの寄与 $\pi_{\mathrm{old}}(1+\epsilon)^2\to0$ となり分散は下がるが、**クリップされたサンプルの目標寄与を丸ごと無視する**という深刻なバイアスを導入する。
- **RIC**： $r\le1+\sqrt{\delta/\pi_{\mathrm{old}}}$ で切る。クリップ境界上でも $\pi_{\mathrm{old}}(1+\sqrt{\delta/\pi_{\mathrm{old}}})^2\approx\mathcal O(\delta)$ となり、**密度に依存しない一定レベルの分散（同分散性 homoscedasticity）**が保証される。

つまりRICは「標準ISより分散が小さく、PPO-Clipよりバイアスがずっと小さい」という好ましいトレードオフを、幾何的に導くことができる（命題4.1）。

![図4：RICのクリップ則は分散寄与を密度によらず $\mathcal O(\delta)$ に抑え、バイアスとバイアンスを両立する](/images/ripo-riemannian-isometric-policy-optimization/fig4.png)

---

## 実験結果

### 設定

- **ベースモデル**：Llama3.2-3B-Instruct、Qwen3-1.7B/4B/8B-Base
- **比較対象**：GRPO、DAPO、GSPO、GMPO、DCPO、GPPO、Clip-Cov
- **ベンチマーク**：AMC23、AIME24、AIME25、HMMT25、BRUMO25、CMIMC25、SMT25 の7つ（いずれも汚染なしの競技レベル数学）

### 主結果（Table 1, Avg@8）

GRPOに対する相対改善は以下の通り（いずれもRIPOが上）：

| モデル | GRPO | RIPO | 相対改善 |
|--------|------|------|----------|
| Qwen3-1.7B-Base | 11.2 | 15.4 | **+37.2%** |
| Llama3.2-3B-Instruct | 6.4 | 8.6 | **+34.4%** |
| Qwen3-4B-Base | 25.7 | 30.1 | **+17.1%** |
| Qwen3-8B-Base | 28.5 | 38.5 | **+35.1%** |

![図3：4モデルすべてでRIPOがGRPOを上回る。平均改善は+17〜+37%](/images/ripo-riemannian-isometric-policy-optimization/fig3.png)

AIME24単体に絞ると、Qwen3-1.7BでGRPO 11.3 → RIPO 18.3 となり、要旨の「GRPO比で最大+60%」に相当する。DAPO/GSPO/GMPO/DCPOといった最新手法とも各モデルで比較され、RIPOが一貫して上回っている（例：Qwen3-8BでRIPO平均38.5 vs DCPO 34.5）。

### 汎化とPPO目標への移行

- **PPO目標への転送**（Table 4, GSM8k Avg@1）：RIPO-ClipはPPO-Clip/DAPO-Clip/DCPO-Clipを上回り、14Bで94.4（PPO-Clip 93.2）を記録。クリップ機構そのものの改良であるため、PPO系にもそのまま効く。
- **コード／検索タスク**（Table 6, Qwen3-8B）：コード平均44.9（GRPO 39.7, +13.2%）、検索平均43.4（GRPO 37.7, +15.1%）。数学以外へも汎化。
- **Pass@kの上限**（Table 5）：AIME25で $k=128$ 時に60.0%、HMMT25で45.3%に達し、GRPO/DAPO/DCPOを上回る。探索が保たれるため、多様な解を産み出す能力そのものが高まっている。

---

## 考察

### 「クリップ」を幾何の問題として再定式化した意義

本論文の最大の貢献は、クリップを**経験的なチューニング knobから、多様体上の幾何的必然性へと格上げ**した点にある。探索と安定性のトレードオフを手でいじるのではなく、「等長な信頼領域」という一つの原理から導出している。そのため実装が極めて薄く、かつ理論的な保証（同分散性によるバイアス・バリアンスの改善）を伴う。

### 対称性の必要性（アブレーション）

$\delta$ の感度実験（Table 2）は興味深い。**対称な** $\delta_{\mathrm{low}}=\delta_{\mathrm{high}}$ は頑健で、Qwen3-8BのAIME24 Avg@8は $\delta=0.05$ で41.7、非対称最適の0.05/0.04で43.8に達する。しかし **$\delta_{\mathrm{high}}\gg\delta_{\mathrm{low}}$ にすると一気に崩壊**する（0.05/0.02→28.8、0.08/0.02→27.5）。理由は明白で、幾何空間で信頼領域を非対称にするとエントロピーが爆発し、訓練が破綻するからだ。すなわち、**信頼領域は幾何空間で対称でなければならない**という制約が、実験を通じても裏付けられている。

### 最近の系列（GSPO・UP）との関係

直近で扱われることが多いGSPO（系列レベル比でGRPOの崩壊を防ぐ）やUP（正のアドバンテージに無制限勾配、負にクリップという非対称最適化）は、いずれも「トークンレベルの重要度比」や「探索―安定性ジレンマ」という別の軸にある。RIPOはそれらと**直交する軸**（ユークリッド vs リーマン幾何）を扱っており、組み合わせの余地が大きい。実際、RICはGRPOの枠組みにそのまま乗るため、GSPOやUPと併用すればさらなる安定化が期待できる。

---

## 関連研究

- **TRPO / Natural Policy Gradient**：KL制約による信頼領域の原点。本論文の「多様体幾何」の視点はここに直接連なるが、TRPOは二次計画を解くため重く、PPO-Clipがその安価な代替として普及した。RIPOは「PPOのクリップを幾何的に補正する」ことで、TRPOの精神をPPOの手軽さで回収している。
- **GRPO / DAPO**：系列相対アドバンテージと動的クリップ。クリップ機構そのものはユークリッドのまま。
- **GSPO**：長さ正規化された系列レベル比で崩壊を防ぐ。幾何の軸ではなく「比の単位」の軸の解決。
- **Clip-Cov / GPPO**：クリップ幅を共分散や勾配に応じて調整する試み。RIPOはこれらと同系統だが、理論的根拠が「等長性」という幾何原理に置かれている点が際立つ。

---

## まとめ

RIPOは、「PPO-Clipがなぜ探索を潰すのか」という長年の謎に対し、**方策多様体の幾何的不整合**という明快な答えを出した。そしてその解決を、固定 $\epsilon$ を $\sqrt{\delta/\pi_{\mathrm{old}}}$ という分布依存の動的閾値に置き換えるという、驚くほど薄い実装で実現した。結果は4モデル×7ベンチマークでGRPOを一貫して上回り、理論的にもバイアス・バリアンスの好ましいトレードオフを導く。LLMのRL後学習が「幾何を意識したクリッピング」へと一段階進むための、強力な一歩と言える。

---

## 参考

- Cai, Z., Guo, X., Wu, H., Wang, M., Ma, W.-Y., Zhang, Y.-Q., & Zhou, H. (2026). *Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization.* ICML 2026. arXiv:2607.10169
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* (PPO)
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation / TRPO.*
- Shao, Z., et al. (2024). *DeepSeekMath: GRPO.* 
- Zheng, C., et al. (2025). *Group Sequence Policy Optimization (GSPO).*
