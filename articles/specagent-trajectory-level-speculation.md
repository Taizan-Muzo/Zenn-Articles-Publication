---
title: "強いモデルは査読に回る──SpecAgentが軌跡単位の投機実行でエージェントの帯域と時間を半減する仕組み"
emoji: "🔍"
type: "tech"
topics: ["LLM", "エージェント", "推論高速化", "効率化", "ツール利用"]
published: true
---

## TL;DR

LLMエージェントは、ツールを何度も呼び、ユーザとのやり取りを挟みながら長い軌跡をたどってタスクを解いている。客室サービスでは20〜30ターン、業務ワークフローでは30〜80ターン、生活サービスでは50〜100ターンという世界がもう当たり前になっている。今の主流は「一つの強いモデルがすべてのステップを生成する」構成だ。しかしその多くは、注文を照会したり、レコードを読んだり、再度検索したりするような日常的なステップである。

LARK・香港科技大学（广州）・Yinwang・University of Cambridgeの合同チームが提案する **SpecAgent** は、強いモデルの役割を「著者」から「査読者」に転換する。小さな **Weak** モデルが軌跡の草稿を書き、状態を変えない read（検索・照会）ならその場で実行して実際の観測に接地させ、初めての状態変更アクションだけを保留にする。大きな **Strong** モデルは、履歴と草稿を **1回の prefill** で一括読み込み、意味的に妥当な最長プレフィックスを確定させ、初めて棄却されたステップだけを修正する。厳密一致ではなく「妥当な進捗かどうか」を判定するので、強モデルが選んだルートと違っていても正しい草稿は拾える。

τ²-Bench Airlineでは Strong-only と同じ **86.0%** を達成し、Retailでは 88.6% に対する **87.7%** と、Weak-Strong 間ギャップの89%を回復。同時に推定 HBM 読み出しは **43〜45%**、壁時間は **41%** に圧縮する。AutomationBenchではギャップの44%、VitaBenchでは75%を回復し、帯域・時間ともに Strong-only を下回る。MCP-Atlasを使ったエッジ・クラウド展開では、Qwen3.5-397B-A17B・GLM-5・DeepSeek-V4-Flash-0731の3つの Strong モデルに対して、**53〜75%** の品質ギャップを回復しながら Strong 出力トークンを **36〜50%**、API コストを **59〜81%** に削減した。τ²では1確定ステップあたり約7ステップに1回、Strong モデルが査読に入るだけで済んでいる。

> **Core insight**: 強いモデルは毎ステップ書く必要がない。接地された草稿を「査読」するだけで、ほとんどの品質を保ちながら、デコードを prefill に置き換えてメモリ帯域とレイテンシを半減できる。

---

## 背景：強いモデルがすべてのステップを書く代償

### エージェントの軌跡は長くなっている

LLMエージェントの性能は、モデル単体の推論力だけではスケールしなくなった。外部ツールを呼び、環境の応答を見て、方針を変えながら解く。問題は、その対話がどんどん長くなっていることだ。τ²-Benchの著者らの調査によると、実世界の MCP サービスでは業務ワークフローで30〜80ターン、生活サービスでは50〜100ターンにも達する（本論文 Table 1）。

この長さはそのまま serving コストに直結する。なぜなら、主流の設計では一つの強いモデルがステップごとに自己回帰的にトークンを生成しており、コストは軌跡の長さに比例する。しかも大部分のステップは「探す」「読む」「もう一度検索する」ような日常業務に過ぎない。小さなモデルでもここを書けるはずだ。

### 既存の分担では強モデルが書き続ける

コストを下げるために、モデル選択や協調の試みはあった。

- **ルーティング / カスケード**: クエリごとに弱いモデルか強いモデルを選ぶ。ただし、選ばれたモデルがそのステップを一から書く。強いモデルが選ばれたステップではコスト削減にならない。
- **協調エージェント（AgentCollab など）**: ステップごとに「次は誰が動くか」を自己評価で切り替える。ここでも、制御を持っているモデルが各行動を生成する。
- **投機的エージェント（Speculative Actions など）**: 強いモデルの「次の行動」を弱いモデルが予測し、重ね合わせてレイテンシを隠そうとする。しかし採用するかどうかの判定は、強いモデルがそのステップを生成したものと一致するかどうかが基本であり、強いモデルは結局すべての基準ステップを生成し続ける。

厳密一致はエージェントに向いていない。検索順序を変えたり、引数の指定を変えたりしても、最終的に正しく解けるケースは珍しくない。強いモデル自身が選んだ一つの経路に合わないからといって、有効な進捗を破棄するのはもったいない。

本論文が問うのは、**弱いモデルが数ステップまとめて草稿を書き、強いモデルがそれを「査読」して有効な部分を採用できないか**、ということだ。

---

## 方法詳解：SpecAgent の設計

SpecAgent は、行動の効果が取り消せるかどうかを第一原理に据える。read は状態を変えないので誤っても履歴から消せば元に戻る。write（変更・提出など）は取り消せない。だから投機は read の間なら自由に進められるし、write の直前だけで査読すれば十分だ。

![Strong-only と SpecAgent の役割対比](/images/specagent-trajectory-level-speculation/fig1.png)

### 可逆性に基づく 3 つの性質

エージェントを部分観測マルコフ決定過程として定式化する。環境には見えない状態 $s$ があり、エージェントは行動 $a_t$ を出すと観測 $o_t$ を得る。Weak 方策 $\pi_W$ と Strong 方策 $\pi_S$ が同じインターフェースを共有する。

行動は 2 クラスに分けられる。

- **read**: 検索・照会など、状態を変えない。観測を返すだけ。
- **state-changing action**: レコードの変更、最終提出など、状態を変える。

この区別が SpecAgent の根幹だ。read の誤りは履歴から削除すれば環境は元のまま。state-changing action の誤りは、履歴を編集しても取り消せない。したがって、**投機は可逆な間だけ許し、不可逆な境界でだけ査読する**のが最適だ。前もって read を査読してもムダだし、write 後に査読しても手遅れだ。

### Draft：接地された草稿を書く

Weak モデルは確定済み履歴 $H_t$ から最大 $K$ ステップの草稿チャンク $\hat{\tau}$ を生成する。構成は次のようになる。

$$
\hat{\tau} = \bigl((\hat{a}_1, \hat{o}_1), \ldots, (\hat{a}_{n-1}, \hat{o}_{n-1}), (\hat{a}_n, \bot)\bigr)
$$

途中の read はその場で環境に実行し、実際の観測を得る。後続のステップは空想ではなく、本物の観測に基づいて選ばれる。チャンクは、はじめて state-changing action が出たところで終わる。それは実行せずに保留（$\bot$）にする。

もし read だけで $K$ ステップに達したら、可逆性の原理から査読なしで確定する。Strong モデルを呼ばないケースもある、ということだ。

### Verify：1回の prefill でまとめて査読

Strong モデルは履歴 $H_t$ と草稿 $\hat{\tau}$ を **1回の prefill パス**で読み込み、評決を返す。

$$
(j, r, \tilde{a}) = V_S(H_t, \hat{\tau}), \qquad j \in \{0, \ldots, n\}
$$

$j$ は「意味的に妥当な最長プレフィックス」の長さ、$r$ は初めて棄却されたステップの理由、$\tilde{a}$ はその場で置き換える候補行動（任意）。受理は**プレフィックス**になる。なぜなら、各ステップは前のステップの観測に条件付けされているから、あるステップが誤れば、後続は正しく見えても前提が崩れている。$j = n$ ならチャンク全体を確定し、保留していた state-changing action を実行する。

ここで重要なのは、**厳密一致ではなく意味的妥当性**で判定する点だ。強いモデル自身が書いた行動と同じでなくても、タスクにとって合理的な進捗なら受理する。

### Repair：棄却は短く済ませる

棄却されたステップは、多くの場合、引数や対象が間違った一つのツール呼び出しに過ぎない。Verifier はその診断とともに $\tilde{a}$ を提示できる。これがスキーマ検証を通れば、追加の Strong 呼び出しなしで fused replacement として確定する。

修正が複数ステップにまたがる場合や、候補が無効な場合だけ、Strong モデルが履歴と理由 $r$ を受け取って1ターンの修正行動を生成する。その修正は Strong 自身の行動なので再査読は不要。棄却された接尾辞（保留していた write も含む）はすべて破棄され、次の草稿は確定済みステップだけの履歴から始まる。

![Draft-Verify-Repair プロトコル](/images/specagent-trajectory-level-speculation/fig2.png)

### コストモデル：なぜ prefill で速くなるのか

Strong-only では、確定ステップごとに Strong モデルがトークンをデコードする。デコードは重みをメモリから読み出す処理が律速で、1トークンごとに1回の重み読み出しが必要だ。一方、prefill は多くのトークンに対して重みを1回しか読まない。計算量は増えるが、メモリ帯域を節約できる。

この非対称性を使って、Strong モデルの1確定ステップあたりのメモリトラフィック比 $R_u$ を導出する。

$$
R_u = \frac{1 - p_u}{1 - p_u^n} \cdot \frac{v}{d} + \rho\,(1 - p_u)
$$

$p_u$ はタスク $u$ での Weak 草稿の受理率、$n$ はチャンク長、$v/d$ は評決の平均トークン数を Strong-only ステップのトークン数で割ったもの、$\rho$ は fused replacement では済まず追加の Strong ターンが必要な棄却の割合だ。

$p_u$ が高いほど $R_u$ は小さくなり、$p_u \to 1$ で $v/(nd)$ に漸近する。逆に $p_u$ が低いタスクでは、$\rho$ の項が大きくなり、コスト削減効果は小さくなる。これは後の実験で AutomationBench が改善しにくい理由を直接説明する。

---

## 実験結果

実験では Weak モデルに **Qwen3.5-4B**、Strong モデルに **Qwen3.5-397B-A17B**（MoE、15層フルアテンション＋45層 Gated DeltaNet）を使用。H20 GPU 上で SGLang を使い、Strong は 16-way tensor parallelism で展開した。比較対象は Weak-only、Strong-only、協調ベースライン AgentCollab、投機ベースライン Speculative Actions の4つ。

![メイン結果：品質と資源比率](/images/specagent-trajectory-level-speculation/fig3.png)

### τ²-Bench：品質はほぼ保たれる

τ²-Benchは航空会社と小売のカスタマーサービスをシミュレートし、データベースの read・mutation・返答が混在する。

- **Airline**: SpecAgent は **86.0%** で Strong-only と完全に同じ。Weak-only が 80.0% なので、6ポイントのギャップをすべて回復した。
- **Retail**: SpecAgent は **87.7%** で、Strong-only の 88.6% から **0.9ポイント** 差。これは Weak-Strong 間 8.2ポイントギャップの **89%回復**に相当する。ペアブートストラップの 95% 信頼区間は [−4.7, +2.9] で、統計的に区別しにくい。

AgentCollab も Airline で 85.3%、Retail で 85.7% と品質は近いが、Strong 呼び出しが Airline で 18.5回（Strong-only は 13.8回）と逆に増え、HBM読み出しも壁時間も増大する。Speculative Actions は Airline で 72.0%、Retail で 48.5% と、Weak-only を下回る品質で、かつ Strong 呼び出しも増える。SpecAgent は唯一、すべてのベンチマークで HBM読み出しと壁時間の**両方**を Strong-only より下回る協調手法になった。

### AutomationBench と VitaBench：難易度に応じた回復率

- **AutomationBench**: Weak 6.8%、Strong 15.2% に対し SpecAgent は **10.5%** と、ギャップの **44%** を回復。HBM読み出しは 78〜84%、壁時間は 57% に抑えたが、FLOPs は +38% 増えた。タスクが難しく Weak 草稿の棄却が多いため、repair コストが支配的になる。
- **VitaBench（Delivery）**: Weak 34.0%、Strong 48.0% に対し SpecAgent は **44.5%** と、ギャップの **75%** を回復。AgentCollab が 50.0% と Strong-only をわずかに上回ったが、HBM読み出し・壁時間ともに SpecAgent の 1.4倍だった。

### MCP-Atlas：エッジ・クラウド展開

Qwen3.5-4B をローカルの RTX 5090 に置き、Strong モデルを 3 つのクラウド API（Qwen3.5-397B-A17B、GLM-5、DeepSeek-V4-Flash-0731）に振り分けた。ここでは、API コストも含めて評価する。

| Strong モデル | Strong-only Pass@1 | SpecAgent Pass@1 | 回復したギャップ | Strong 出力トークン | API コスト |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-397B-A17B | 72.3 | **59.4** | 53% | 50% | 81% |
| GLM-5 | 76.2 | **67.3** | 72% | 46% | 62% |
| DeepSeek-V4-Flash-0731 | 84.2 | **74.3** | 75% | 36% | 59% |

Strong-only のトークンが長くなるモデルほど、相対的な削減率は大きくなる。DeepSeek-V4-Flash では Strong-only で 10.6k トークンを吐くところを SpecAgent では 3.8k トークンに抑えた。いずれのモデルでも、**回復した品質の割合は消費した Strong トークンの割合を上回る**。つまり、Strong モデルの1トークンあたりの品質購買力が、SpecAgent の中で高まっている。

ただし壁時間は改善しない。ローカル Weak モデルが、クラウド上の Strong モデルと同程度に遅いからだ（109秒 vs 105〜151秒）。この設定では、レイテンシよりもリモート利用量とコストの削減が主な利益となる。

---

## 考察

### なぜ時間は下がり、FLOPs は上がるのか

SpecAgent は品質を保ちながら壁時間を短縮する一方、FLOPs は 30〜87% 増える。この逆転は、デコードと prefill が異なる律速要因を持つことで説明される。

![コストモデルとMCP-Atlas結果](/images/specagent-trajectory-level-speculation/fig4.png)

Strong モデルの verify と repair は SpecAgent の FLOPs の **81〜85%** を占めるが、HBM 読み出しでは **39〜66%** にとどまる。逆に Weak モデルの草稿は FLOPs の **15〜19%** だが、HBM 読み出しの **34〜61%** を占める。デコードは重み読み出しが律速なので、prefill に切り替えることで帯域は節約される。同時に、verify と repair 用のプロンプトがキャッシュを共有せずに prefill されるため、VitaBench では prefill だけで 1.03 PFLOP 消費し、Strong-only の総計 0.72 PFLOP を超えた。

壁時間は、Strong 生成が元の 0.16〜0.34、Weak 草稿が 0.22〜0.40（VitaBench）に分解される。VitaBench では Weak 草稿の比率が大きく、全体として 69% に留まる。AutomationBench では個別の repair が Strong シェアを支配し、改善幅が小さくなる。

### 設計選択の消融実験

τ² Retail での消融実験は、SpecAgent の設計がどこに効いているかを明らかにする。

- **返答も保留する、あるいは read チャンクも含めてすべて査読する**: Strong 呼び出しが 3倍以上に増え、HBM読み出しが +63〜79%、壁時間が 2.5〜3.0倍に悪化。品質も下がる。
- **fused replacement を外す**: 棄却ごとに追加の Strong ターンが必ず発生し、Pass@1 は **82.5%** に落ちる。
- **受理を厳密一致にする**: Strong モデルが書いた行動と同じでないと棄却するようにすると、Pass@1 は **82.5%**、FLOPs は **+70%** 増える。

つまり、**「意味的に妥当な最長プレフィックスを受理する」**ことこそが、Weak モデルの進捗を保持しながらコストを下げる鍵だ。

### どんな状況で効くか

コストモデルは、導入する前に利益を予測できる枠組みを与える。

- **Weak モデルが十分に速い**: Proposition 2 のレイテンシ項 $L_u$ は、Weak ステップが Strong ステップより速ければ有意義だ。エッジ・クラウドの例では Weak が遅かったため、壁時間の改善は見られなかった。
- **Strong モデルのデコードが帯域律速**: H20 では batch サイズが約37シーケンス以下の時、デコードは帯域律速となる。これを超えるとデコードが計算律速になり、FLOPs 増加が壁時間に響く。
- **Weak モデルが草稿をある程度書ける**: $p_u$ が高いタスクほど削減効果が大きい。AutomationBench のように Weak-only が壊滅的に低いタスクでは、repair が支配的になり、品質回復もコスト削減も限定的になる。

### 限界

SpecAgent はあくまで **state-changing action** が実行される前に査読する。read の実行自体は取り消せない。外部サービスが read を記録していれば、履歴から削除しても痕跡は残る。論文はこれを明確に制限として述べている。

また、環境が read/write をどう分類するかは adapter に依存する。τ²-Bench や VitaBench では、ユーザへの返答はデータベースを変更しないので保留しない。AutomationBench や MCP-Atlas では、最終回答は評価対象なので write として保留する。これらはベンチマーク固有の設計選択だ。

---

## 関連研究

- **投機的デコーディング**: Leviathan et al. が開いたトークン単位の draft-and-verify は、SpecAgent と補完的だ。Osprey（本連載で前に取り上げた）は、投機デコードの drafter をターゲット非依存に事前学習する方向だった。SpecAgent は drafter を弱モデルに、verify を軌跡単位・意味的に拡張し、ツール利用エージェントに応用した。
- **ルーティングと弱強協調**: FrugalGPT や RouteLLM、Hybrid LLM はクエリ単位でモデルを選ぶ。AgentCollab はエージェント軌跡内で自己評価に基づいて制御を切り替える。これらは制御を持つモデルが各行動を生成するのに対し、SpecAgent は Strong モデルに「どこまで採用するか」を決定させ、生成は最小限にとどめる。
- **検証 vs 生成の非対称性**: Cobbe et al. の verifier や Lightman et al. の PRM は、数学推論で「生成より検証が容易」を示した。SpecAgent はこの非対称性をエージェント軌跡に拡張し、その非対称性の「コスト版」（prefill は帯域が節約できる）を活用する。
- **連載との接続**: 直前の HybridAgent は、思考を潜在空間に隠してテキスト行動だけを出すことで効率化した。SpecAgent は、思考ではなく「強モデルの生成コスト」を減らす。両方とも「強いモデルが全部を書く必要はない」という方向に沿っているが、切り口は異なる。

---

## まとめ

SpecAgent は、LLMエージェントにおける強いモデルの役割を「著者」から「査読者」に変えた。弱いモデルが read を即実行しながら草稿を書き、状態を変える write の直前でだけ強いモデルに査読を依頼する。査読は意味的な妥当性で行い、最長の有効プレフィックスを確定させ、必要なら fused replacement または1ターンの修正で補う。

結果として、τ²-Bench Airline では Strong-only と同点、Retail では 0.9ポイント差まで近づきながら、HBM読み出しを 43〜57%、壁時間を 41〜54% に圧縮。MCP-Atlas では 3 つのクラウド Strong モデルに対して、品質ギャップの 53〜75% を回復しながら API コストを 59〜81% に抑えた。

「強いモデルは書かなければならない」という暗黙の前提を緩めたことで、帯域律速な serving 環境では大きなレイテンシとコストの削減が可能になった。次に面白いのは、Weak drafter の能力向上と、Strong verifier の較正が、コストモデルのどのパラメータを動かすかだろう。

---

## 参考

- 論文: Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents (arXiv:2610.21463)  
  https://arxiv.org/abs/2610.21463
- HTML版:  
  https://arxiv.org/html/2610.21463v1
- コード・データ:  
  https://github.com/LARK-AI-Lab/SpecAgent
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding:  
  https://arxiv.org/abs/2211.17192
- Chen et al., FrugalGPT: How to Use Large Language Models while Reducing Cost and Improving Performance:  
  https://arxiv.org/abs/2305.05176
- Lightman et al., Let's Verify Step by Step:  
  https://arxiv.org/abs/2305.20050
- Ong et al., RouteLLM: Learning to Route LLMs with Preference Data:  
  https://arxiv.org/abs/2406.18665
- Gao et al., AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents:  
  https://arxiv.org/abs/2603.26034
- Ye et al., Speculative Actions: A Lossless Framework for Faster AI Agents (ICLR 2026):  
  https://openreview.net/forum?id=P0GOk5wslg
