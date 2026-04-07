---
title: "RLはLLMの推論を本当に引き出せるのか？"
emoji: "🔥"
type: "tech"
topics: ["LLM", "強化学習", "NeurIPS", "推論", "DeepSeek"]
published: true
---

## TL;DR

- DeepSeek-R1に代表される「RLVR（可検証報酬による強化学習）」は、LLMの推論力を向上させていると思われている
- しかし、**大規模サンプリング（pass@k, k大）で評価すると、Base ModelがRLVR Modelを逆転**する
- RLVRは「新しい推論パターン」を生み出しておらず、既存パターンの**サンプリング効率を上げているだけ**
- 6種のRLVRアルゴリズム（PPO, GRPO, DAPO等）はすべて似たような挙動で、Base Modelの限界にはるかに届いていない
- 一方、**知識蒸留（Distillation）はBase Modelの限界を真正面から突破できる**

## 1. なぜこの論文が熱いのか

2025年初頭、DeepSeek-R1が「強化学習（RL）だけでLLMを賢くできる」というパラダイムを切り開いた。以来、PPO、GRPO、DAPO、RLOO、ReMax、Reinforce++など、RLVR（Reinforcement Learning with Verifiable Rewards）と総称される手法が次々と登場し、AIMEやMATHのような数学ベンチマークで驚異的なスコアを叩き出した。

「RLがLLMに新しい推論能力を教えている」——これは2025年のLLM研究における**暗黙の前提**だった。

NeurIPS 2025で最優秀論文候補（Runner-up）に選ばれた本論文は、この前提に正面から疑問を投げかける。

> **「本当にそうなのか？」**

答えは、**部分的にイエス、しかし本質的にはノー**——だ。

## 2. 評価のトリック：pass@kを大きくしてみる

### 2-1. pass@kとは

LLMの推論能力を評価する際、通常は**pass@1**（1回生成して正解なら正解）を使う。しかし、推論タスクでは同じ問題に対して複数回サンプリングし、その中に正解が含まれる確率を測る**pass@k**が適している。

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

k=1なら「1回で当たる確率」、k=128なら「128回試して少なくとも1回正解する確率」だ。

### 2-2. 逆転するスコア

本論文の核心は、このkの値を大きくしたとき何が起きるかにある。

![fig1](/images/rlvr-reasoning-limit/fig1_pass_at_k.png)
*図1：左から数学（Minerva）、コード生成（LiveCodeBench）、蒸留比較。小kではRLVRが勝つが、大kではBase Modelが逆転する。*

**数学推論（Minerva, 32B）**:
- k=1: RLVR 8.0% vs Base 2.0% → **RLVRの圧勝**
- k=128: RLVR 29.0% vs Base 38.0% → **Base Modelが逆転**（9ポイント差）

**コード生成（LiveCodeBench, 7B）**:
- k=1: RLVR 28.1% vs Base 23.8%
- k=128: RLVR 42.8% vs Base ~50.0% → **逆転**

つまり、RLVRモデルが解ける問題は、**Base Modelが十分な回数サンプリングすれば「もっと高確率で」解ける**ということだ。RLVRは「新しい解法」を身につけたのではなく、「正解に近い道を最初の1発で当てやすくした」だけに過ぎない。

## 3. 推論境界の収窄

さらに驚くべきことに、RLVR訓練を進めるにつれて、**モデルが解ける問題の範囲自体が狭くなっている**。

![fig2](/images/rlvr-reasoning-limit/fig2_coverage_training.png)
*図2左：RLVRモデルはBase Modelに対して解ける問題のサブセットになっている。図2右：訓練が進むとpass@1は上がるが、pass@256は下がる。*

Coverage analysisの結果がこれを裏付ける。

- **AIME24（数学）**: RLVRモデルが解ける問題の78%は、Base Modelも解ける
- **LiveCodeBench（コード）**: RLVRモデルが解ける問題の85%は、Base Modelも解ける
- **MathVista（視覚推論）**: 同様にRLVR ≂ Baseのサブセット

RLVRは**推論の境界を広げておらず、むしろ狭めている**。訓練が進むとpass@1は上がるが、pass@256は下がる——これは典型的な**分布の尖鋭化（sharpening）**で、新しい能力を獲得したのではなく、特定の解法パターンに過学習している証拠だ。

## 4. Perplexity分析：RLVRの正解は「もうあった」

Perplexity分析がさらに明確な証拠を示す。

RLVRモデルが生成した「正解の推論パス」$Y_{\text{RL}}$を、Base Modelで評価したときのperplexity $PPL_{\text{Base}}(Y_{\text{RL}} \mid x)$を計算する。

![fig3](/images/rlvr-reasoning-limit/fig3_efficiency_ppl.png)
*図3左：6種のRLVRアルゴリズムはすべて最適サンプリングから大幅に離れている。図3右：RLVRの推論パスはBase Modelの分布内にすでに存在する。*

結果、RLVRの正解パスのperplexity分布は、Base Model自身の正解パスのそれと**高度に重なる**。つまり、RLVRが見つけた「正解ルート」は、Base Modelのサンプリング空間に**最初から含まれていた**。

RLVRは地図上の新しい道を切り開いたのではなく、**すでに存在していた道の「行き方」を覚えただけ**だ。

## 5. 6種のRLVRアルゴリズム、全部似たようなもの

論文はPPO、GRPO、ReMax、RLOO、Reinforce++、DAPOの6種を比較した。

**驚くべき結論**：6種ともほぼ同じ振る舞いをする。どれもBase Modelの限界には遠く及ばず、サンプリング効率のギャップ（$\Delta_{SE}$）はすべて40以上——最適値（0）から程遠い。

これは、現在のRLVRの性能が**アルゴリズムの違いではなく、Base Modelの能力に律速されている**ことを示している。PPOでもGRPOでもDAPOでも、Base Modelが持っている情報の範囲内で最適化しているに過ぎない。

## 6. では、何が「本当に」能力を拡張するのか？

### 蒸留（Distillation）との決定的な違い

論文で最も印象的な比較がこれだ。

DeepSeek-R1-Distill-Qwen-7B（知識蒸留モデル）と、RLVR訓練したQwen-2.5-Math-7Bを同じベースで比較する。

![fig1](/images/rlvr-reasoning-limit/fig1_pass_at_k.png)
*図1右：蒸留モデル（緑）はBase Modelの上限を突破している。RLVR（赤）はBase Model（青）の内側に収まる。*

**結果**: 蒸馏モデルのpass@kは、**全k値においてBase ModelとRLVRモデルの両方を上回る**。

なぜか？蒸留は**Teacher Modelが持つ「Base Modelには存在しない推論パターン」を直接注入する**からだ。新しい知識が外部から流入するため、Base Modelの限界を超えられる。

RLVRは「自分の庭（Base Modelの分布）の中で効率よく探す」だけ。蒸留は「よその庭から種を持ってくる」。これが根本的な違いだ。

## 7. 考察：RLVRの正しい位置づけ

この論文は「RLVRは無駄だ」と言っているわけではない。むしろ、**RLVRの正しい位置づけ**を示している。

**RLVRがやっていること（正確に）**：
- Base Modelが「たまに」生成できる正解パスを、「高確率で」生成できるようにする
- サンプリング効率を劇的に向上させる（k=128回サンプリングしなくても、k=1で当たるようになる）
- これは**実用上、非常に価値のある改善**

**RLVRがやっていないこと**：
- Base Modelの分布に存在しない、全く新しい推論スキルの創出
- 推論能力の上限の引き上げ
- 本質的な「知識の拡張」

言い換えれば、RLVRは**検索の最適化**であって**知識の獲得**ではない。Googleが「あなたが探しているものを1ページ目に持ってくる」ように、RLVRは「正解を1回目で持ってくる」——でも、その正解は最初からデータの中にあった。

## 8. では、どうすれば本当に能力を拡張できるのか

論文は3つの方向性を示唆している。

### ① 継続的スケーリング（Continual Scaling）
Base Modelを大きくし続けることが、現在確認されている唯一の「能力上限の引き上げ」手段。RLVRの効果もBase Modelの規模に比例して大きくなる。

### ② 多ターンのAgent-Environment相互作用
単発の問題解決ではなく、環境との相互作用を通じて「試行錯誤の経験」を蓄積する。これが従来のRLで「新しい戦略を学ぶ」メカニズムだった。現在のRLVRはこれをやっていない——1ターンで終わる問題に過度に特化している。

### ③ 蒸留との組み合わせ
強いTeacher Model（o1, o3, Claude等）からの蒸留と、RLVRを組み合わせるハイブリッドアプローチ。まず蒸留で能力を拡張し、その上でRLVRで効率を最大化する。

## 9. 限界と批判的な視点

この論文自体にも限界がある。

**pass@kの解釈**: 大kでのpass@kが高いことが「より賢い」ことを保証するわけではない。実用上はk=1〜k=8で高得点な方が価値が高い。RLVRの実用価値を低く評価しすぎている感はある。

**RLVRの真の価値**: k=1で8%→28%（コードタスク）という改善は、実用上は革命的に重要。Best-of-Nを1024回やるコストと、k=1で当てるコストの差は、本番環境では天と地ほど違う。

**「新しい推論」の定義**: そもそも「新しい推論」とは何か？Base Modelの分布に含まれない推論が本当に「新しい」のか、それとも「ノイズ」なのか。この境界自体が曖昧だ。

## 10. まとめ

この論文は、2025年のLLMブームの中心にあった**「RLVRで推論能力が向上する」という物語に、冷静なピリオドを打った**。

- RLVRはBase Modelの上限を超えない——**上限はBase Modelにある**
- RLVRの真の役割は「新しい能力の創出」ではなく「既存能力のサンプリング効率化」
- 6種のRLVRアルゴリズムはみな同じ限界にぶつかっている
- 蒸留だけが（現在のところ）Base Modelの限界を突破できる手法
- 本当の能力拡張には、新しいRLパラダイムが必要

とはいえ、この論文は**RLVRを否定しているのではなく、正しく位置づけている**。RLVRは実用上極めて有効だ。ただ、それを「モデルが賢くなった」と錯覚しないことが重要だ。

DeepSeek-R1の凄さは、**RLVRそのものではなく、蒸留とRLVRを組み合わせたパイプライン**にあるのだと、この論文を読むと改めて気づかされる。

---

## 参考

- 論文: [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (NeurIPS 2025 Runner-up, Oral)
- プロジェクトページ: [limit-of-RLVR.github.io](https://limit-of-RLVR.github.io)
- OpenReview: [openreview.net/forum?id=4OsgYD7em5](https://openreview.net/forum?id=4OsgYD7em5)
- 清華大学報道: [au.tsinghua.edu.cn](https://www.au.tsinghua.edu.cn/info/1062/4335.htm)
