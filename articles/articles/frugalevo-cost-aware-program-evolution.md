---
title: "50ドルの発見を1ドル未満で──FrugalEvoが強弱2モデルの分担でLLM進化探索のコストを1/30にする仕組み"
emoji: "💸"
type: "tech"
topics: ["機械学習", "LLM", "進化的計算", "最適化", "コスト効率"]
published: true
---

## TL;DR

AlphaEvolveに代表されるLLMガイド進化探索は、circle packingのような難しい最適化問題で人間の記録を塗り替えてきた。ただし評価軸が「固定イテレーション数後の性能」で、**同じ解を得るのにいくら払ったか**はほとんど問われてこなかった。NUS・UW・Stanfordの研究チームは、評価を「単位コストあたりのゲイン」に引き直すフレームワーク**FrugalEvo**を提案した。中身はシンプルで、**(1) 戦略の探索は高コストの強いモデル、実装と改善は低コストの安いモデルに分担させ、(2) プロンプトの構造をKVキャッシュのprefix再利用が最大化するように並べ替え、(3) 予算内の性能曲線の面積を測るBA-AUCという新指標を導入する**。circle packingでGLM-5.3 + Flashの構成なら**$0.55**、GPT-5.6 Terra + Lunaでも**$1.68**で半径和2.63599に到達し、平均約$50かけているCORALやSwarmResearchに匹敵・凌駕した。20タスク中19タスクでBA-AUCが最高か同等という結果は、「賢さは戦略の選定だけに使えばいい」という役割分担の有効性をはっきり示している。

**論文**: [FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution](https://arxiv.org/abs/2610.03675) (arXiv:2610.03675)
**著者**: Hui Chen, Xuan Qi, James Xu Zhao, Zhaopeng Feng, Shilong Liu, Kuang Xu, Pang Wei Koh, Bryan Hooi (NUS / UW / Princeton / Stanford)
**コード**: [GitHub](https://github.com/chchenhui/frugalevo)

![fig1](/images/frugalevo-cost-aware-program-evolution/fig1.png)

---

## 背景：「何イテレーション後か」から「いくらで」へ

FunSearchがcap set問題で新記録を出して以来、LLMと進化探索の組み合わせは急速に深まった。AlphaEvolveは個別関数からプログラム全体へ対象を拡大し、OpenEvolveやShinkaEvolveがオープンソース実装を提供し、AdaEvolveやEvoXが探索プロセス自体を適応化し、CORALやSwarmResearchが複数のコーディングエージェントを協調させている。いずれも基本ループは同じで、プログラムプールから親を選び、LLMに変異を提案させ、評価して戻す。

問題は評価軸だ。これらの研究は「Nイテレーション後のスコア」や「N個の候補生成後のスコア」で比較する。しかしLLM呼び出しにはAPI料金（やGPU計算コスト）がかかり、test-time scalingの効果が強いタスクほど、**高い性能は単にたくさん払った結果かもしれない**。1回の平均コストが違う手法同士をイテレーション数で並べるのは、燃費の違う車を「1時間でどこまで走ったか」で競わせるようなもので、実務上の意思決定（どれを動かすか）にほぼ結びつかない。

著者らの立場は、計算最適化における探索は**固定予算Bの下で解の品質s\*を最大化すべき**であり、比較は性能ーコスト曲線で行うべき、というものだ。そこで導入されたのがBA-AUCである。

![fig2](/images/frugalevo-cost-aware-program-evolution/fig2.png)

BA-AUC（Budget-Aware Area Under the Curve）は、累積LLMコスト$c$に対するbest-so-farスコア$q(c)$の、予算$B$までの面積として定義される：

$$\operatorname{BA-AUC}(B) = \int_0^B q(c)\, dc$$

走者ランが予算に達する前に終わった場合は最終スコアを予算まで水平に延長し、超えた場合は$B$で打ち切る。これで全手法が同じコスト区間$[0, B]$上で公平に比較できる。最終スコアが同じでも、**早く良い解に届いた手法の面積が大きくなる**。実務で本当に欲しいのはこっちの性質だ。

---

## 方法その1：戦略は強いモデル、実装は安いモデル

FrugalEvoは候補プログラムの生成を2段階に割り、別々のモデルに振り分ける。全体像が図1(a)である。

**Cold Start**。安いモデルだけで初期プログラムを、評価スコアが改善しなくなるまで反復改善する。ここでできた最良プログラムが最初のincumbent（現時点の親コード）になる。弱い初期プログラムを親にすると、強いモデルが出す戦略自体の質が下がり、予算を無駄にするからだ。

**Context Builder**。冒頭に高コストモデルを1回だけ呼び、課題記述と評価器コードを分析させて最適化目標と制約を抽出する。この分析結果は以後キャッシュされ、cold startのincumbentと評価フィードバックと合わせて探索コンテキストを構成する。探索が進むと、search memoryから直近のincumbent・試行済み戦略・失敗フィードバックを引き込んで更新していく。

**Strategy Explorer（高コストモデル）**。探索コンテキストを与えられ、**1回のAPI呼び出しで$K$個の異なる設計戦略を提案する**。各戦略は親プログラムのどこをどう変えるか、なぜ改善が期待できるか、どの計算制限を守るべきかを具体的に記述する。アルゴリズム設計の方向性決めが解の質を最も左右する場面であり、ここにだけ「強さ」を注ぎ込む。

**Solution Generator（低コストモデル）**。各戦略を1つずつ実装して試作し、評価スコアで戦略をランキングする。以後はランキングに従って戦略を選び、改善ラウンドに入る。1ラウンドは「1戦略 × 現incumbentを親にして最大$M$回の逐次試行」で、各試行の結果は評価され、失敗のフィードバックは次の試行のプロンプトに積まれる（親は固定）。**incumbentを更新する試行が出た瞬間にラウンドは終了**し、更新後の親で同じ戦略の次ラウンドへ。改善なしで終われば次の戦略へ移る。

**Evaluator / Search Memory**。評価器はタスク固有のスコア関数。search memoryはisland-based MAP-Elitesでプログラムの多様性を維持し、候補プログラムに加えて「提案済み戦略」「戦略ごとの試行結果」「失敗フィードバック」も記録する。

ポイントは、**強いモデルの呼び出し回数が戦略提案の1回（コンテキスト構築の分析を入れても2回）に圧縮される**ことだ。以後の大量の実装・改善・デバッグ的呼び出しはすべて安いモデルが担う。LLM推論が探索コストの大半を占めるこの種のタスクでは、この配分だけで単位コストあたりのゲインが変わってくる。

---

## 方法その2：プロンプトをキャッシュのために並べ替える

もう一つの工夫は地味だが効く。多くのLLM APIは**prompt caching**をサポートしており、同一のprompt prefixに対するKV状態を再利用すると入力トークンが安くなる。FrugalEvoはハーネスとプロンプト設計を、**進化ステップ間でprefixの共有が最大化されるように**組んでいる（図1(b)）。

プロンプトの並びは次の順序に固定される。

1. **固定セクション**：タスク指示・出力形式・課題分析（ほぼ変わらない）
2. **準固定セクション**：island-bestやeliteプログラム（変化は遅い）
3. **変動セクション**：ランダムサンプル、現incumbent（毎回変わりうる）
4. **追記セクション**：直前の試行のフィードバック（末尾にappend）

よくある実装では評価フィードバックをプロンプトの前の方に差し込むが、それをやるとキャッシュが全部無効になる。**「変わるものを末尾に追記する」だけでprefix再利用が保たれる**。強弱分担と合わせて、コスト削減はアルゴリズム側とシステム側の両面から攻めているわけだ。

---

## 実験結果：20タスク、2つのモデル構成

評価対象は計20タスクで、数学最適化5（circle packing、Heilbronn問題、signal processingなど）、システム最適化5（ADRS benchmarkのEPLB、GPU配置、LLM-SQL prefix caching、Cloudcast、トランザクションスケジューリング）、アルゴリズム最適化10（ALE-Bench-Lite、AtCoderヒューリスティックコンテスト由来）。

モデル構成は2つ。(1) 戦略にGPT-5.6 Terra、実装にGPT-5.6 Luna（対抗手法はTerraのみ）。(2) 戦略にGLM-5.3、実装にGLM-5.3 Flash（対抗手法はGLM-5.3のみ）。参考までに価格はTerraが入力$2.00/出力$12.00（1M tokensあたり）、Lunaが$0.20/$1.20、GLM-5.3が$1.15/$3.50、Flashが$0.075/$0.25。**LunaはTerraの1/10、FlashはGLM-5.3の約1/15〜1/46**という価格差をフルに使う設計だ。予算は数学タスクが$2（GPT構成）/$1（GLM構成）、システムとアルゴリズムが$1（GPT構成）/$0.5（GLM構成）。

結果の骨子は次の通り。

- **数学5タスク**：両モデル構成で全タスクの平均BA-AUCが最高。GPT構成では5タスクすべて、GLM構成では4タスクで平均性能が最高か同等。クローズドなAlphaEvolveの報告値とも、circle packing、Heilbronn Convex (13)、Heilbronn Triangle、Min-Max-3の4タスクで同等以上。
- **システム5タスク**：両構成で全タスクが最高か同等の性能、BA-AUCは4タスクで最高。EPLBでは貪欲なexpert replication＋load-aware block assignment＋pairwise swapの組み合わせで0.1473を達成。
- **ALE-Bench-Lite 10タスク**：平均privateスコア1924.9で、OpenEvolve（1887.5）やAdaEvolve（1881.9）を上回る（図3(b)）。

目玉はcircle packingである（図3(a)）。

| 手法 | 半径の和 | コスト |
|---|---|---|
| AlphaEvolve（報告値） | 2.635000 | 非公開 |
| CORAL（マルチエージェント） | 2.635985 | 平均 約$50 |
| SwarmResearch（マルチエージェント） | 2.635996 | 平均 約$50 |
| **FrugalEvo (GPT-5.6 Terra+Luna)** | **2.635996** | **$1.68** |
| **FrugalEvo (GLM-5.3+Flash)** | **2.635990** | **$0.55** |

**$50級のマルチエージェント手法と同水準の解を、$0.55で**出している。1/30〜1/90のコストである。得られた解も定性的に興味深く、5行レイアウトに非対称変位を加え、制約付きSLSQP最適化と固定中心の半径線形計画で精緻化し、評価器の許容差の範囲内で半径をわずかに膨らませる、という複合戦略に進化している。Signal ProcessingではSavitzky–Golay、Butterworth、スペクトルフィルタを状況に応じてブレンドしつつ軽微な反転を抑制するフィルタ構成に到達し、best 0.78912を記録した。

コスト曲線の形状も重要だ。Signal Processing、Heilbronn Convex、Transaction Schedulingの3タスクで、FrugalEvoは**最初の$0.2以内に優位を確立し、以後ずっと維持する**。「強いモデルを後からたっぷり使う」手法は立ち上がりが遅く、面積（BA-AUC）で追いつけない。

![fig3](/images/frugalevo-cost-aware-program-evolution/fig3.png)

---

## アブレーション：分担は本当に効いているのか

GPT-5.6 Terra/Luna構成でのアブレーション結果が図4である。

![fig4](/images/frugalevo-cost-aware-program-evolution/fig4.png)

- **強モデルのみ（Terra+Terra）**：Circle Packingの平均は2.632278、Signal Processingでは0.68534まで落ちる。実装・改善の呼び出しを高コストモデルで行うと予算がすぐ枯渇し、探索回数が減る。平均性能への打撃が最も大きい。
- **弱モデルのみ（Luna+Luna）**：Circle Packingは意外に粘るが（2.634565）、Signal Processingのbestが0.75973まで下がる。**有望な戦略を選び抜く力が弱い**ことがbest性能に出る。
- **cold startなし**：初期プログラムをそのまま親にすると両タスクで平均・bestとも低下。安いモデルでincumbentを作ってから強いモデルに見せる、という前処理がコストパフォーマンス上合理的。
- **逐次フィードバックなし**：改善ラウンド内で失敗のフィードバックを次の試行に渡さないと、平均低下に加えてrun間の分散が増える。フィードバックは安定性にも効いている。

つまり「戦略選定の強さ」と「実装の安さ」は**代替ではなく補完**で、どちらか片方だけでは最終性能もコスト効率も落ちる。この非対称性が論文の主張の核である。

---

## 考察：何が削れているのか

読んでいて最も刺さったのは、コスト意識の評価軸そのものを変えた点だ。BA-AUCは指標としては素朴だが、既存手法の比較表が根底から崩れる。CORALやSwarmResearchが$50かけて到達した解を、モデル2つの直列分担が$0.55で再現したという事実は、「マルチエージェントで探索の多様性を確保する」という近年の潮流への静かな反証になっている。多様性はIslandモデルのMAP-Elitesでも保てるし、高価な能力は本当に必要な瞬間（戦略の方向づけ）にだけ呼べばいい。

設計として美しいのは、役割分担が**タスクのエラー構造と噛み合っている**点だと考える。プログラム進化では、失敗の多くは「良い戦略のまずい実装」であり、評価器が即座に数値で教えてくれる。つまり実装段階のエラーは安価に検出・修復できるため、そこに高い知性を注ぐ限界利益が低い。一方、戦略選定の失敗は「その方向自体が望み薄」ということで、検出に多くの探索コストがかかる。だから知的リソースは未検出コストの大きい側に置く、という資源配分の直感に忠実なのである。

注意すべき限界も論文が明示している。第一に、この枠組みは**解の生成と評価がすべてコード環境内で完結する**タスク向けで、湿式実験のように自動実行できない評価を含む科学領域には素直に拡張できない。第二に、戦略探索を導くプロンプト（指示）は計算最適化向けに手作業で設計されており、タスクから自動導出する仕組みは今後の課題だ。第三に、BA-AUCの曲線はrun間分散が大きい領域では3回平均の意味が薄くなりうるので、繰り返し数を増やした再検証はあってもよい。

実務的な含意も大きい。LLMの性能差は縮まる一方なので、「強いモデルの呼び出し回数を構造的に最小化するパイプライン設計」は、モデルのアップグレードがあってもそのまま資産になる。コスト効率はモデル選択（routing）だけでなく、**ワークフロー設計で決まる**というのが本論文の一番伝えたいことだろう。

---

## 関連研究

- **FunSearch** (Romera-Paredes et al., 2024, Nature)：LLM生成と自動評価を組み合わせて数学的解を進化させた先駆。本稿の問題設定（評価はコストの外）の土台。
- **AlphaEvolve** (Novikov et al., 2025)：関数からプログラム全体への進化に拡張。FrugalEvoはその4数学タスクで報告値に同等以上。
- **ShinkaEvolve** (Lange et al., 2026, ICLR)：サンプル効率に焦点を当てた進化。ただし「サンプル数」であり「ドル」ではない。
- **OpenEvolve / CodeEvolve**：オープンソース実装。FrugalEvoの比較対象であり、数学タスクの実装元。
- **AdaEvolve** (Cemri et al., 2026) / **EvoX** (Liu et al., 2026)：探索プロセスの適応化・メタ進化。
- **CORAL / SwarmResearch**：複数コーディングエージェントの協調。性能は出すがコストが高く、本論文の対比軸。
- **FrugalGPT / RouteLLM**：モデルカスケードやroutingによるコスト削減。FrugalEvoは訓練不要でtoken確率も使わず、タスク特性に基づく役割分担で攻める点が違う。
- **Squeeze Evolve** (Maheswaran et al., 2026)：verifierなしで複数モデルを協調させる試み。log-prob等のモデル内部信号を使う点が対照的。
- **SimpleTES** (Ye et al., 2026)：科学発見向けに評価駆動のスケーリングを行う研究。長期探索のコスト効率という文脈で近い問題意識。

---

## まとめ

FrugalEvoは、LLMガイド進化探索の評価を「固定イテレーション後の性能」から「固定コスト予算内の性能曲線（BA-AUC）」に引き直し、その評価軸で勝つための最小構成——**強いモデルに戦略、安いモデルに実装、キャッシュに優しいプロンプト**——を示した。20タスクで既存手法に匹配または凌駕し、circle packingでは$0.55で$50級のマルチエージェント手法と同等の新記録級の解に到達した。印象的なのは手法の素朴さで、特別な訓練もtoken確率へのアクセスも要しない。「どの段階に知性を注ぐか」の設計次第で、同じモデル群から30倍のコスト差が生まれる。LLMで何かを探索させるパイプラインを書く人には、評価軸と役割分担の両方を見直す良いきっかけになる論文だ。

---

## 参考

- Hui Chen, Xuan Qi, James Xu Zhao, Zhaopeng Feng, Shilong Liu, Kuang Xu, Pang Wei Koh, Bryan Hooi. [FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution](https://arxiv.org/abs/2610.03675). arXiv:2610.03675, 2026.
- [GitHub: chchenhui/frugalevo](https://github.com/chchenhui/frugalevo)
- A. Novikov et al. [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131). arXiv:2506.13131, 2025.
- B. Romera-Paredes et al. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024.
- R. Lange, Y. Imajuku, E. Cetin. [ShinkaEvolve: Towards open-ended and sample-efficient program evolution](https://arxiv.org/abs/2401.02051). ICLR 2026.
- L. Chen, M. Zaharia, J. Zou. [FrugalGPT: How to use large language models while reducing cost and improving performance](https://arxiv.org/abs/2305.05176). arXiv:2305.05176, 2023.
