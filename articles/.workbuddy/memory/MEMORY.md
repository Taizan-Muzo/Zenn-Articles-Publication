# Zenn 论文精读发布 SOP

## 发布流程
1. 读取本文件获取SOP与候选池
2. 检查daily log（.workbuddy/memory/YYYY-MM-DD.md）确认今日是否已发文
3. 从候选池选题或搜索最新arXiv论文（已发布的arXiv ID勿重复，见下表）
4. 日语撰写精读 → matplotlib配图2-4张（images/<slug>/figX.png）→ articles/<slug>.md（published: false）
5. git add + commit + push origin main → 成功后published: true → 再次commit + push
6. 更新daily log、本文件（已发文+候选池）、.workbuddy/automations/zenn/memory.md（简执行记录）

## 文章格式
- Front matter: title(日语), emoji, type("tech"), topics(日语标签数组), published
- 结构: TL;DR → 背景 → 方法详解 → 実験結果 → 考察 → 関連研究 → まとめ → 参考
- 仓库: /Users/Zhuanz/Desktop/Zenn-Articles-Publication/

## 写作与配图要点（累积教训）
- 日语地道自然，避免翻译腔/AI味；学术表达：提案する、示した、達成した、有効性を検証
- 图表说明用日语，技术术语保留英文原词
- matplotlib日文字体：Hiragino Sans + fontweight（W3/W6名称未注册）；「≥」未収録→「≧」或mathtext
- mathtext不支持\big、\ge、\le → 用\geq/\leq；上付きマイナス(U+207B)无字形→mathtext表記
- FancyArrowPatch的linestyle用ls="-"（元组指定会报错）
- push前跑简体字检查check_simplified.py；JIS收录字（辞随静声面雷等）会误报，用Shift_JIS编码判定兜底
- fig出现异常大图片（数万像素）→ 坐标/比例不匹配，需重新设计子图布局

## 已发布文章（日期: arXiv ID 关键词）
- 09-04 2609.04194 CoT可読性≠解釈可能性（COLM 2026）
- 09-05 2609.04108 On-Policy Distillation vs RLVR
- 09-06 2609.04199 Compile by Training（EMNLP Demo）
- 09-07 2609.04180 補助ビュー事前学習（EMNLP Findings）
- 09-08 2609.04198 LLMジャッジ共有エンドポイント信頼性
- 09-09 2609.04010 Uno dAR無損失高速化
- 09-10 2609.04773 PTA ツール蒸留分布ズレ（EMNLP Main）
- 09-11 2609.05395 EDGE 韓国API多段ツール（EMNLP Industry）
- 09-12 2609.05275 Layer Dropout（ICML 2026）
- 09-13 2609.10036 BSE POMDP最適性保証
- 09-14 2609.07821 A*-Thought-V2 幾何力学
- 09-17 2609.06702 PARSER 並列読み+逐次推論
- 09-18 2609.09233 サブエージェントvsスキル
- 09-19 2609.09153 Procedural Graphs 自己進化
- 09-20 2609.20715 ActObs 観測監督
- 09-23 2609.07036 Aha-Flow Distillation
- 09-24 2609.20089 UnifiedPlayers 協調RL
- 09-25 2609.20612 AMPLE-Math OPSD特権情報
- 09-27 2609.05295 RISE 外挿教師
- 10-04 2609.11163 LILA 校準フリープルーニング
- 10-05 2610.03665 Pivot-SD 自己蒸留10倍（EMNLP Oral）
- 10-05 2610.03695 Queen 棋力+説明（Princeton）
- 10-06 2610.03675 FrugalEvo 1/30コスト
- 10-07 2610.01415 PoS 明示的信念状態
- 10-07 2610.03632 World Embedding物理忠実性
- 10-08 2610.07948 CRG 主張木校準
- 10-09 2610.08452 Agentic AutoRAG 失敗帰属
- 10-10 2609.28385 UECR-GRPO 蒸留+GRPO統一
- 10-12 2610.07247 BiToK-SD 双レベルTop-K
- 10-13 2609.09338 Osprey ドラフター事前学習（EMNLP）
- 10-14 2610.07835 DHCG 動的協調グラフ
- 10-17 2610.18861 HybridAgent 潜在+テキスト（Stanford）
- 10-18 2610.21463 SpecAgent 軌跡投機（LARK）
- 10-19 2610.21731 Retrieved but Not Read
- 10-20 2610.22160 EMO 閉ループTTC
- 10-21 2610.22067 LURCH 長文脈3性質
- 10-22 2610.22019 Priming Base Models（NVIDIA）
- 10-23 2610.22174 Mosaic マルチセッションRL
- 10-24 2610.21496 OPRD リマインダー蒸留
- 10-25 2610.31314 StateTrack-Bench 状態整合性（EMNLP Main）
- 10-26 2610.32871 J-Comm 潜在通信
- 11-02 2610.39372 Cord reveal順序残差
- 11-03 2610.42148 Flide 状態フォーク9.5倍
- 11-05 2611.04152 PACT token認証
- 11-06 2611.06876 SpecCompact スロット並列（NYU）
- 11-08 2611.07554 The Memory Trap（Princeton+Anthropic）
- 11-09 2611.07845 Masked diffusion as RSA理論（QMUL）
- 11-10 2611.08905 From Procedures to Permissions サブエージェント権限（清華+Samsung）
- 11-11 2611.09042 FPRT 失敗プロファイリング+到達可能軌跡（NeurIPS 2026）
- 11-13 2611.12127 Lost in Rounding bf16再帰状態長考化（PayPal AI）
- 11-14 2610.40350 CORAL 認知的オフロード文脈整理（COLM 2026）

dLLM系譜（関連研究用）: Uno→Pivot-SD→Cord→PACT→RSA理論→（候補: DiVR再生成）。投機系: Osprey→SpecAgent→SpecCompact。数値精度系: Lost in Rounding。長期エージェント記憶系: StateTrack→Memory Trap→CORAL→（候補: MAPE）。

## 候选论文池（23本）
- Judge Robustness（2610.31172）LLM-as-a-Judge多言語堅牢性
- ReflectPO（2610.30933）長期エージェント反思ガイド付き方針最適化
- Prospective Memory（2611.05731）見越し記憶で先回りするエージェント
- HyCo（2610.21436）モバイル・クラウド投機デコーディング・ハイブリッドドラフティング
- Raincheck（2610.39562）マルチターンMLLM遅延prefill、TTFT最大24.1%削減
- ZHA（2610.42112）階層木インデクサでサブ二次attention、1M超で優位
- TraceSpec（2610.41177）LLMエージェント投機デコーディングの理解と改善
- Character-Length Control dLLM（2611.06371, EMNLP 2026 Main）訓練不要の文字数制御要約
- SpecCTMC（2611.08362）dLLM連続時間CTMC投機化、逐次評価1.73-4.38倍削減
- Measured Coupling（2611.08307, Tel Aviv）dLLM並列デコーディング実測カップリング、訓練不要
- O2PD（2611.07731）overlap-gatedオンポリシー蒸留、prefix 20%のみ教師監督が信頼できる
- Reality Check of Inference-Time Context Augmentation（2610.30983）
- Beyond K/V Caches / HST（2611.08133）ハイブリッド線形attention状態ハンドオフ、保持率89.88%・21.66倍
- RecSA（2611.07232）再帰疎attention、選択O(log L)、32K訓練で2M外挿、RULER 1M 60.5
- DUET（2611.09239）block diffusion LM双スパース推論（南京大）
- RidgeFill（2611.09709）訓練不要の後層キャッシュ充填でprefill高速化、50頁
- QGPM（2611.09554）クエリ誘導RoPE変調でlost-in-the-middle緩和（USC）
- MemoryDojo（2611.09721）LLMエージェントメモリ経由のポイズニング・ベンチマーク
- AgentOPSD（2611.12587）再帰的信念改訂（log-odds空間ベイズ更新）でターン単位信用割当て、ALFWorld 89.1%
- Overlapping Block Denoising（2611.14387）dLLMのブロック境界を重複させてデノイズ、効率化（dLLM系譜候補）
- On-Policy Mid-Training Distillation（2611.13855）受動的模倣を超えるオンポリシー・ミッドトレーニング蒸留
- DiVR（2611.17320, 11-12投稿）dLLMのremasking再生成タイミングを学習するverifier、隠れ状態の誤り情報+孤立remasking再現性の診断、LLaDA-8B 35.3→43.4%（greedy比1.9-3.5倍コスト）
- MAPE（2611.17068, 11-12投稿）プログラム進化でエージェントメモリ設計を最小シードから進化、LoCoMo/LongMemEval最大+12%・servingコスト35%、raw会話保持+クエリ意図合成の2設計パターン

## 次回の候補補充メモ
- 次回実行時に11-14/11-15付けarXiv新着から1-2本補充すること（11-14実行時点で11-13付けアナウンス分（135件）の上位50件まで走査済み。残り85件も覗いてよい）
- 2026-11-14付け発行でCORAL (2610.40350) を選定・発行済み。長期エージェント・文脈管理を発行したため次回は多様性を意識（dLLM系が候補池に6本と偏重気味→Judge Robustness=評価系・QGPM=長文脈・Reality Check=測定系・Raincheck=MLLMなどが良候補）

