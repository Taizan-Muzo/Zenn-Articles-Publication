---
title: "ContextPilot：32Kコンテキストで128Kを超える能動的文脈管理"
emoji: "🧭"
type: "tech"
topics: ["LLM", "Agent", "ReinforcementLearning", "ContextManagement"]
published: true
---

# TL;DR

長大なコンテキストを扱うAgenticタスクで、**モデル自身がコンテキストを編集する**「proactive context management」という考え方がある。本稿で取り上げるContextPilot（EMNLP 2026 Main Track）は、この枠組みに2つの改善を加えた。1つは**ツールセットの拡張**（planning・長期記憶・ソフトオフロード）、もう1つは**文脈管理に特化したRL**（どこで枝分かれ探索するか、どう信用割当するか）だ。結果は明快で、**32Kのコンテキスト窓しか使わないのに、128Kのネイティブコンテキストを持つベースモデルを大きく上回る**。Qwen3-8Bでは平均45.93→69.40に跳ね上がり、同規模のStateLM-8B-RLも3.55pt上回った。深層検索タスクでもBrowseComp系・GAIA・xBenchで一貫した改善を示している。

# 背景：コンテキストは誰が管理すべきか

ReAct的なAgentic推論では、思考・ツール呼び出し・観測が順次コンテキストに追加され、履歴が単調に増え続ける。この「コンテキスト過負荷」への対処は大きく2つに分かれる。

**受動的コンテキスト管理**は、閾値を超えたら切り詰める・要約するという人間設計のルールで履歴を処理する。シンプルだが、モデルは自分自身のコンテキストに対する決定権を持たず、シナリオごとの適応も難しい。

**能動的コンテキスト管理**は、MemGPT、Sculptor、StateLM、MemAct、AgentFoldといった研究が進めてきた枠組みで、削除・検索・要約といったコンテキスト編集ツールをモデルに渡し、自発的に管理させる。柔軟だが、著者らは3つの未解決課題を指摘する。

1. **ツールセットが貧弱**：検索・削除・要約しかなく、グローバルな計画立案、長期記憶、適応的圧縮をサポートしていない
2. **探索効率が悪い**：文脈管理アクションの最終結果への影響は不均質なのに、RL訓練では一律に扱われる
3. **信用割当が粗い**：最終的なtrajectory-level報酬を、中間の全コンテキスト編集アクションに一律配分してしまう

3点目は特に厄介だ。著者らの予備実験（Figure 3）が示すのは、正解した軌跡が実は「検索を繰り返すだけの非効率な文脈管理」だったり、不正解の軌跡が「note/delete/updateNoteを適切に使いこなした合理的な管理」だったりする、という事実である。最終報酬をそのまま配分すれば、非効率を強化し、適切な判断を罰することになる。

![fig1](/images/contextpilot-proactive-context-management/fig1_tool_ablation.png)

# 手法1：ツールセットの拡張

ContextPilotはStateLMの基本ツールセットを土台に、3カテゴリの新ツールを追加する。

| カテゴリ | ツール | 機能 |
|---------|-------|------|
| **Planning** | `plan` | 簡潔な計画を提案 |
| **長期記憶** | `memorize` | エンティティ・タイムスタンプ・イベントを抽出し、関連記憶項目間にエッジを張る |
| | `updateMemory` / `readMemory` | 記憶項目の更新／読み出し（隣接ノードも取得） |
| **ソフトオフロード** | `compressContext` | 軽量圧縮モデル（llmlingua-2）でメッセージを圧縮 |
| | `foldHistory` | 全履歴を廃棄して検索可能インデックスを構築。キーワードと要約に凝縮し、後から`searchContext`で復元 |

`foldHistory`の設計が面白い。単なる削除ではなく「キーワード＋要約」の形で履歴を畳み、後からキーワード検索で部分的に復元できる状態に保つ。ハードな削除と完全保持の中間を狙った「ソフト」なオフロードだ。

これらのうち、オフロード系・記憶書き込み系のツールは**履歴を書き換えるコンテキスト編集アクション**と見なされ、後述するスナップショット分割の境界になる。

## SFTデータの合成

Long-context QAのみSFTを実施（深層検索はベースモデルが既に検索能力を持つためSFTをスキップし、直接RL）。

教師モデルはQwen3.5-397B-A17B（thinking mode、温度0.6、top-p 0.95、最大出力4K）。ここで工夫されているのが**コンテキスト管理ハーネス**で、適切なタイミングでヒントと制約を与える。例えば`searchContext`の後は`readChunk`で内容を確認させる、コンテキスト長が閾値を超えたらオフロードツール以外を制限する、といった具合だ。

さらに**動的ツール提示**と**訂正・リトライ機構**が組み込まれている。メモリを書き込む前は`readMemory`を提示しない、`searchContext`前は`readChunk`を提示しない、といった前提条件管理に加え、教師が不正なmessage idを指定した場合は具体的なヒントを返してリトライさせる。重要なのは、**リトライ時は最終的に成功した試行のみを保持**し、失敗した中間呼び出しが模倣されないようにしている点と、これらヒント類は生成時の足場に過ぎず最終的なSFT軌跡には含まれない点だ。

3段階のフィルタリングも徹底している。

| 段階 | 内容 | 残存数 |
|------|------|--------|
| 出発点 | — | 3,196問 |
| 結果ベース | 完全一致。初回不正解は2回まで再試行を許可 | 3,114軌跡 |
| 過程ベース | GPT-OSS-120Bで不適切な文脈管理を除外 | −46 |
| ピーク長 | ピークコンテキスト32K超を破棄 | 3,068軌跡 |

これをコンテキスト編集操作で分割し、**51,469個のSFTスナップショット**を得る。

# 手法2：文脈管理に特化したRL

## Context-Aware Partial Rollout：どこで枝分かれするか

ARPOに着想を得た部分rolloutを使うが、分岐点の選び方が工夫されている。各コンテキスト管理アクションについて、2つの変化量を計算する。

**コンテキスト変化**：

$$\Delta C_t^{\mathrm{cm}} = \left| \frac{\text{len}(c_{t+1}^{\mathrm{cm}}) - \text{len}(c_t^{\mathrm{cm}})}{\text{len}(c_t^{\mathrm{cm}})} \right|$$

**エントロピー変化**：

$$\Delta H_t^{\mathrm{cm}} = H_t^{\mathrm{cm}} - H_{\mathrm{initial}}$$

ここで$H_t^{\mathrm{cm}}$はステップ$t$で観測を受け取った後に生成された$k$トークンの平均エントロピー、$H_{\mathrm{initial}}$は軌跡冒頭の$k$トークンの平均エントロピーだ。直前ステップではなく**初期状態を基準**にするのがミソで、隣接ステップ間の局所的な揺らぎではなく、初期クエリ状態から見た不確実性の大きな変化を捉えたいという意図がある。

これらを合成した**感度スコア**：

$$\mathcal{S}(a_t^{\mathrm{cm}}) = \alpha \cdot \Delta C_t^{\mathrm{cm}} + \beta \cdot \Delta H_t^{\mathrm{cm}}$$

（実験では$\alpha = \beta = 1$）

rollout時は、まずtrajectory-level rolloutを行って$M$個のスナップショットを得る。クエリあたりの予算$N$に対して$M < N$なら、残りを部分rolloutに回す。全コンテキスト管理アクションを感度スコアの降順に並べ、上位$N-M$個を分岐点として追加のサブ軌跡をサンプルする。影響の大きい判断に探索予算を集中させる仕組みだ。

実装では、1クエリあたり8本のtrajectory-level rolloutから最大64スナップショット（軌跡あたり最大8スナップショット）を得て、残りを部分rolloutで埋めて計128スナップショットを収集する。

![fig2](/images/contextpilot-proactive-context-management/fig2_rl_mechanism.png)

## スナップショット単位の信用割当：どう報酬を配るか

部分rolloutによって、各スナップショットを通る後続軌跡が複数得られる。これを利用して、中間スナップショットの価値を**その先の全分岐の報酬平均**で推定する。

終端スナップショット$S_M = T$の報酬は3要素の和：

$$R(S_M) = R_{\mathrm{out}} + R_{\mathrm{fmt}} + R_{\mathrm{pen}}$$

$R_{\mathrm{out}}$は予測と正解の一致、$R_{\mathrm{fmt}}$は出力のパース可能性、$R_{\mathrm{pen}}$は不正なツール呼び出し（メモリ構築前に`readMemory`を呼ぶ等）やコンテキスト長違反へのペナルティ。

中間スナップショット$S_i$の報酬は、$S_i$をprefixとして持つ全終端軌跡$\mathcal{T}(S_i)$の平均：

$$R(S_i) = \frac{1}{|\mathcal{T}(S_i)|} \sum_{T \in \mathcal{T}(S_i)} R(T)$$

あとは同一クエリ配下の全スナップショットをグループとして、平均と標準偏差でアドバンテージを標準化し、各スナップショットを独立サンプルとしてGRPO目的で最適化する。

## なぜ分散が下がるのか

付録で理論的な正当化が与えられている。スナップショット$S$の真のクレジットを

$$Q(S) \triangleq \mathbb{E}[R(T) \mid S \preceq T]$$

と定義すると、trajectory-level割当は1本の終端軌跡の報酬をそのまま使うのに対し、提案法は$n_S$本の平均を取る。条件付き分散$\sigma^2(S)$のもとで両者とも不偏だが、分散は

$$\operatorname{Var}[\widehat{Q}_{\mathrm{traj}}(S) \mid S] = \sigma^2(S), \qquad \operatorname{Var}[\widehat{Q}_{\mathrm{ours}}(S) \mid S] = \frac{\sigma^2(S)}{n_S}$$

となる。$n_S > 1$なら平均二乗誤差は厳密に小さくなる。しかもcontext-aware partial rolloutは感度の高いアクションの$n_S$を意図的に増やすので、この効果が増幅される仕掛けになっている。

# 実験結果

## Long-Context QA

ベースモデルはQwen3-8B、Qwen3-14B、Gemma4-E4B-it。評価はNovelQA（Copyright split）、∞Bench（En.MC）、LongMemEval-S、BrowseComp+の4ベンチマーク。3回実行の平均。

| Model | Length | NovelQA | ∞Bench | LongMemEval-S | BrowseComp+ | Avg |
|-------|--------|---------|--------|---------------|-------------|-----|
| Qwen3.5-397B-A17B (w/o tools) | 256K | 88.77 | 90.39 | 81.00 | 62.05 | 80.55 |
| Qwen3.5-397B-A17B (w/ tools) | 32K | 91.94 | 92.13 | 83.60 | 80.96 | **87.16** |
| RL-MemoryAgent-7B | 32K | 60.24 | 62.45 | 40.60 | — | — |
| ReadAgent-8B | 32K | 16.38 | 24.02 | 0.00 | — | — |
| StateLM-8B-RL | 32K | 84.15 | 73.07 | 59.73 | 46.44 | 65.85 |
| **Qwen3-8B (w/o tools)** | 128K | 65.74 | 66.96 | 45.20 | 5.82 | 45.93 |
| Qwen3-8B (w/ tools) | 32K | 38.09 | 39.59 | 24.47 | 8.28 | 27.61 |
| ContextPilot-8B | 32K | 82.56 | 71.03 | 60.67 | 48.84 | 65.78 |
| **ContextPilot-8B-RL** | 32K | 83.88 | 75.25 | 64.27 | 54.18 | **69.40** |
| StateLM-14B-RL | 32K | 84.85 | 78.46 | 64.47 | 52.67 | 70.11 |
| **ContextPilot-14B-RL** | 32K | 84.81 | 81.08 | 67.40 | 55.50 | **72.20** |
| **ContextPilot-E4B-RL** | 32K | 72.92 | 60.99 | 62.47 | 47.47 | **60.96** |

注目すべきは3点ある。

第一に、**32KのContextPilotが128Kのネイティブベースモデルを圧倒**していること。Qwen3-8Bの45.93から69.40へ、23.5ptの改善だ。

第二に、興味深い逆説がある。「w/ tools」の行を見てほしい。Qwen3-8Bにツールだけ渡してファインチューニングなしで使うと、平均27.61と**何もしない45.93より大幅に悪化**する。つまりコンテキスト管理ツールは、適切に訓練しないとむしろ有害だ。ContextPilot-8B（SFT）の65.78、RL後の69.40は、この「ツールを渡すだけでは役に立たない」状態から訓練で引き上げた結果である。

第三に、Gemma4-E4B-itでの伸びが最も大きい。SFTの54.74からRL後60.96へ、+6.22pt。ベースの能力が低いほど、文脈管理の訓練効果が大きいようだ。

## 深層検索

ベースモデルはWebSailor-7BとWebExplorer-8B。OpenSeekerから抽出した1KサンプルでRL。評価はBrowseComp、BrowseComp-ZH、GAIA、xBench-DeepSearch。

| Backbone | Method | BrowseComp | BrowseComp-ZH | GAIA | xBench-DS | Avg |
|----------|--------|-----------|---------------|------|----------|-----|
| WebSailor-7B | ReAct | 11.33 | 25.47 | 31.07 | 34.00 | 25.47 |
| | ReAct (w/ truncation) | 12.67 | 27.91 | 33.66 | 35.00 | 27.31 |
| | ReSum | 15.83 | 38.99 | 38.19 | 35.33 | 32.09 |
| | SUPO | 18.50 | 42.68 | 42.07 | 42.00 | 36.31 |
| | OpenSeeker | 16.50 | 41.87 | 41.42 | 43.33 | 35.78 |
| | **ContextPilot** | **21.17** | **43.14** | **45.31** | **43.67** | **38.32** |
| WebExplorer-8B | ReAct | 23.83 | 46.57 | 48.87 | 52.33 | 42.90 |
| | ReSum | 28.83 | 47.87 | 52.75 | 51.00 | 45.11 |
| | SUPO | 31.00 | 50.40 | 56.96 | 58.00 | 49.09 |
| | OpenSeeker | 29.17 | 48.56 | 57.28 | 56.67 | 47.92 |
| | **ContextPilot** | **32.17** | **53.63** | **57.93** | 56.67 | **50.10** |

両バックボーンで一貫してContextPilotが最高。SUPOに対して平均+1.51pt（2バックボーン平均）、単純なReActに対しては+7.20〜+10.06ptの改善だ。

![fig3](/images/contextpilot-proactive-context-management/fig3_main_results.png)

## 考察：なぜ効くのか

### トークン効率が劇的に改善する

15ターン以上の軌跡について、ターンあたりの平均入力トークン数を測ると違いが明確に出る。BrowseCompにおいて、WebExplorer-8Bはインタラクションが進むにつれ入力長がほぼ線形に増加し30Kトークン近くに達するのに対し、ContextPilot-8Bは**1ターンあたり8〜10Kトークンで安定**する。BrowseComp-ZHでも同様の傾向が確認されている。

これは単なるコスト削減ではない。コンテキストが膨張しないことで、同じコンテキスト予算（30K入力＋2K生成）でより多くのターンを回せるようになり、結果として長大な探索が可能になっている。

### RLがツール使用戦略を書き換える

RL訓練中のツール呼び出し分布を追跡すると、興味深い変化が見られる。訓練初期は情報検索ツールが全呼び出しの約半分を占めるが、訓練が進むにつれて検索の割合が減り、代わりに**planning/perception、長期記憶、コンテキストオフロードの割合が上昇**していく。

同時に、ツール実行失敗率も追跡されている。定義は「環境側でエラーが発生した呼び出し」（不正な形式、無効な引数、ツール前提条件の違反）。訓練初期は記憶系・オフロード系ツールの失敗率が検索系より顕著に高い。つまりSFTモデルは**これらのツールを呼ぶことはできても、使いどころを理解していない**。RLの過程で失敗率が大きく下がり、それに同期してタスク成功率が上がっていく。

この2つの観察は、SFTが「形式」を教え、RLが「判断」を教えるという分担を示唆している。

### ツール設計の寄与分解

Qwen3.5-397B-A17Bを使った累積的なアブレーションが示唆に富む。

| Tool Design | NovelQA | ∞Bench | LongMemEval-S | BrowseComp+ | Avg |
|-------------|---------|--------|---------------|-------------|-----|
| Original tools | 88.90 | 85.15 | 74.00 | 63.49 | 77.89 |
| + Planning | 89.76 | 87.23 | 78.50 | 65.66 | 80.29 |
| + Soft offloading | 91.28 | 89.63 | 80.20 | 71.20 | 83.08 |
| + Long-term memory | 91.94 | 92.13 | 83.60 | 80.96 | **87.16** |

BrowseComp+での伸びが最も大きく、63.49→80.96で**+17.47pt**。長期記憶の追加だけで+9.76ptも伸びている。入力長の観点がここで効いてくる。NovelQAの平均入力長が約119Kトークンなのに対し、BrowseComp+は552Kに達する。文脈が長大になるほど、記憶とオフロードの仕組みが効いてくるという構図だ。

### RL設計の寄与分解

Qwen3-8Bでのアブレーション（括弧内は前段からの変化）：

| Training method | NovelQA | ∞Bench | LongMemEval-S | BrowseComp+ | Avg |
|-----------------|---------|--------|---------------|-------------|-----|
| SFT | 82.56 | 71.03 | 60.67 | 48.84 | 65.78 |
| GRPO | 83.53 (+0.97) | 72.78 (+1.75) | 60.07 (−0.60) | 50.96 (+2.12) | 66.84 (+1.06) |
| + Entropy-based partial rollout | 82.52 (−1.01) | 73.07 (+0.29) | 62.13 (+2.06) | 49.64 (−1.32) | 66.84 (+0.00) |
| + Context-aware partial rollout | 83.05 (+0.53) | 73.94 (+0.87) | 61.40 (−0.73) | 51.08 (+1.44) | 67.37 (+0.53) |
| + Fine-grained credit assignment | 83.88 (+0.83) | 75.25 (+1.31) | 64.27 (+2.87) | 54.18 (+3.10) | **69.40 (+2.03)** |

ここで最も示唆的なのは、**エントロピーだけに基づく部分rolloutが平均で+0.00、BrowseComp+では−1.32ptと悪化**している点だ。不確実性の変化だけでは、本当に重要なコンテキスト編集操作を特定できない。コンテキスト変化量を足したcontext-aware版で安定した改善に転じ、さらにスナップショット単位の信用割当でBrowseComp+が+3.10ptと大きく伸びる。

Gemma4-E4B-itでも同様の傾向（GRPO 57.15 → +Entropy 58.99 → +Context 59.35 → +Fine-grained 60.96）が確認されており、2モデルで一貫している。

![fig4](/images/contextpilot-proactive-context-management/fig4_token_efficiency_ablation.png)

# 関連研究

- **StateLM**（Liu et al., ICLR 2026）：Pensieveパラダイム。本稿の直接的なベースラインであり、ツールセットの土台
- **Sculptor**（Li et al., 2025）：能動的コンテキスト管理の初期研究
- **MemAct**（Zhang et al., 2025）：Memory as Actionという定式化
- **AgentFold**（Ye et al., 2025）：マルチスケールな折りたたみ操作をSFTで学習
- **ARPO**（Dong et al., 2025）：Agentic Reinforced Policy Optimization。部分rolloutの着想元
- **SUPO**（Lu et al., 2025）：要約とエージェント能力をRLで同時訓練
- **ReSum**（Wu et al., 2025）：推論時要約による長期検索の実現

# まとめ

| 観点 | 結果 |
|------|------|
| コンテキスト効率 | 32K窓で128Kネイティブを大幅超過（Qwen3-8B 45.93→69.40） |
| トークン効率 | ターンあたり入力を8〜10Kで安定化（ReActは30K近くまで線形増加） |
| ツール拡張 | 長期記憶の追加だけでBrowseComp+ +9.76pt |
| 探索戦略 | エントロピー単独では不安定。コンテキスト変化の併用で安定改善 |
| 信用割当 | スナップショット単位でBrowseComp+ +3.10pt、分散は$\sigma^2/n_S$に低減 |

ContextPilotの本質的なメッセージは、**エージェントのコンテキスト管理は「ツールを渡す」だけでは成立しない**という点にある。Qwen3-8Bに同じツールを渡しても、訓練なしでは27.61とベースの45.93を大きく下回ってしまう。必要なのは、どの判断に探索予算を割くか（context-aware partial rollout）と、どの判断を評価するか（スナップショット単位の信用割当）を明示的に設計することだ。

この2つは、コンテキスト編集という操作の「影響が不均質である」という性質に正面から向き合った結果生まれた設計である。長大なコンテキストを扱うエージェント一般に通じる考え方として、覚えておいて損はない。

# 参考

- Pan, Z. et al. "ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL." arXiv:2608.28476, 2026. (EMNLP 2026 Main Track)
- Liu, X. et al. "The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context." ICLR 2026.
- Dong, G. et al. "Agentic Reinforced Policy Optimization." arXiv:2507.19849, 2025.
- Lu, M. et al. "Scaling LLM Multi-Turn RL with End-to-End Summarization-Based Context Management." arXiv:2510.06727, 2025.
- Wu, X. et al. "ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization." arXiv:2509.13313, 2025.
- コード: https://github.com/Tencent/ContextPilot
