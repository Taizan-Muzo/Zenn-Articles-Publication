# Zenn 论文精读发布 SOP

## 发布流程
1. 读取本文件获取SOP与候选池
2. 检查daily log确认今日是否已发文
3. 从候选池选题或搜索最新论文
4. 日语撰写精读 → matplotlib配图 → Zenn格式md
5. 两阶段Git推送：published:false → published:true
6. 更新daily log与MEMORY.md

## 文章格式
- Front matter: title(日语), emoji, type("tech"), topics(日语标签数组), published
- 结构: TL;DR → 背景 → 方法详解 → 実験結果 → 考察 → 関連研究 → まとめ → 参考
- 配图路径: `/images/<slug>/figX.png`
- 仓库: /Users/Zhuanz/Desktop/Zenn-Articles-Publication/

## 日语写作要点
- 地道自然，避免翻译腔和AI味
- 使用学术论文常见的日语表达：提案する、示した、達成した、有効性を検証
- 图表说明用日语，技术术语保留英文原词

## 已发布文章
- 2026-09-04: 可読性は解釈可能性にあらず──CoT推論の判定重要性と実際の重要性の比較 (arXiv:2609.04194, COLM 2026)
- 2026-09-05: 順次実行が同時最適化に勝つ──On-Policy DistillationとRLVRの相互作用を解き明かす (arXiv:2609.04108)
- 2026-09-06: 自然言語仕様をコンパイルする──Compile by Trainingで再利用可能な神経関数を生成する (arXiv:2609.04199, EMNLP 2026 System Demos)
- 2026-09-07: 多様な視点が暗記に勝つ──補助ビューがLLMの事前学習を加速する仕組み (arXiv:2609.04180, EMNLP 2026 Findings)
- 2026-09-08: 工学が完璧でも測定は崩れる──LLMジャッジの信頼性が共有エンドポイントで破れる仕組み (arXiv:2609.04198)
- 2026-09-09: 速さを捨てずに速くなる──Unoが離散拡散でARモデルの無損失高速化を実現する仕組み (arXiv:2609.04010)
- 2026-09-10: 教師が書く前に生徒は実行できない──PTAがツール使用蒸留の分布ズレを断つ仕組み (arXiv:2609.04773, EMNLP 2026 Main)
- 2026-09-11: 実行が嘘を見抜く──EDGEが韓国公開APIの多段ツール呼び出しで9Bを27Bに迫る仕組み (arXiv:2609.05395, EMNLP 2026 Industry)
- 2026-09-12: 捨ててはいけない──Layer DropoutがLLMの訓練効率と推論弾性を同時に開く仕組み (arXiv:2609.05275, ICML 2026)
- 2026-09-13: 見えないものを信じる──BSEがLLMエージェントにPOMDPの最適性保証を取り戻す仕組み (arXiv:2609.10036)
- 2026-09-14: 角度が推論の宿命を決める──A*-Thought-V2が幾何力学でCoTの明示と潜在を自動で切り分ける仕組み (arXiv:2609.07821)
- 2026-09-17: 読むことと推論を分ける──PARSERが長文脈Agentの並列読みと深い推論を両立する仕組み (arXiv:2609.06702)
- 2026-09-18: 知識は使い方で決まる──サブエージェントとスキルの実行方式が長期タスクの性能を分ける仕組み (arXiv:2609.09233)
- 2026-09-19: 暗黙の知識を明示する──Procedural GraphがLLMエージェントの手続きを自己進化させる仕組み (arXiv:2609.09153)
- 2026-09-20: 環境を隠すな──ActObsが観測監督でエージェントの探索を変える仕組み (arXiv:2609.20715)
- 2026-09-23: 「待って」より「やってる」が推論に効く──Aha-Flow DistillationがFlow Markerの見落しを正す仕組み (arXiv:2609.07036)
- 2026-09-24: 三つの役が互いに磨き合う──UnifiedPlayersが計画・実行・評価の協調でツール推論を自己増強する仕組み (arXiv:2609.20089)
- 2026-09-25: 成功と有用は別物──AMPLE-MathがOPSDの蒸留改善と特権情報の寄与を切り分ける仕組み (arXiv:2609.20612)
- 2026-09-27: 外挿が教師を作る──RISEがRLVRの訓練軌跡から逐次改善の教師を自前で構築する仕組み (arXiv:2609.05295)
- 2026-10-04: 校準データなしで切る──LILAが特異値分布のKS距離だけでLLMを構造化プルーニングする仕組み (arXiv:2609.11163)
- 2026-10-05: 10個のPivotが256個のtokenを動かす──Pivot-SDがMasked Diffusion LMの自己蒸留を10倍効率化する仕組み (arXiv:2610.03665, EMNLP 2026 Main Oral)
- 2026-10-05: 沈黙の名人に言葉を与える──Queenが4Bでグランドマスター級の棋力と説明を両立する仕組み (arXiv:2610.03695, Princeton)
- 2026-10-06: 50ドルの発見を1ドル未満で──FrugalEvoが強弱2モデルの分担でLLM進化探索のコストを1/30にする仕組み (arXiv:2610.03675, NUS/UW/Stanford)
- 2026-10-07: 記憶の先に信念を置く──PoSが明示的信念状態で長期エージェントの文脈管理を変える仕組み (arXiv:2610.01415)
- 2026-10-07: 映像の中の物理は取り出せるか──World Embedding Benchmarkがマルチモーダル埋め込みの物理忠実性を測り直す (arXiv:2610.03632)
- 2026-10-08: 信じる前に分解する──CRGがLLMエージェントの信頼性を主張の木に分解して推論時校準を実現する仕組み (arXiv:2610.07948)
- 2026-10-09: 失敗を診て次を指す──Agentic AutoRAGが検索と生成の失敗帰属でRAGパイプラインを自動最適化する仕組み (arXiv:2610.08452, EMNLP 2026 REALM + NeurIPS 2026 ML4Sys)
- 2026-10-10: 教師を信じる場所と信じない場所──UECR-GRPOがオンポリシー蒸留とGRPOをエントロピー校準信用再分配で統一する仕組み (arXiv:2609.28385)
- 2026-10-12: どこに蒸留するかは学んで決める──BiToK-SDがTop-Kトークン選択を双レベル最適化で解く仕組み (arXiv:2610.07247)
- 2026-10-14: チームは必要になってから増やす──DHCGが実行フィードバックで協調グラフを動的に育てる仕組み (arXiv:2610.07835)
- 2026-10-17: 黙って考え、言葉で動く──HybridAgentが潜在推論とテキスト行動を単一の生成過程で切り替える仕組み (arXiv:2610.18861, Stanford/UIUC/Together AI)
- 2026-10-13: 加速する側だけが世間知らず──Ospreyが投機デコードのドラフター事前学習を再利用可能にする仕組み (arXiv:2609.09338, EMNLP 2026)

## 候选论文池
- LLM-as-a-Judge / 評価信頼性 系最新研究
  - CRG: エージェント信頼度の主張分解 (arXiv:2610.07948) 既発
- Reasoning / Planning 系最新研究
- LLM Agent / Tool-use / SFT初始化 相关
  - UECR-GRPO: 蒸留とGRPOの統一 (arXiv:2609.28385) 既発
  - HybridAgent: 潜在推論とテキスト行動の統合 (arXiv:2610.18861, Stanford/UIUC/Together AI) 既発
- RAG / Retrieval 増強
  - Agentic AutoRAG: 推論駆動エージェントでRAG最適化 (arXiv:2610.08452, EMNLP 2026 REALM + NeurIPS 2026 ML4Sys) 既発
- Multilingual / Cross-lingual NLP
- Efficient Inference / 推論高速化
  - Osprey: 投機デコードの汎用ドレフター (arXiv:2609.09338, EMNLP 2026) 既発
  - BiToK-SD: 二段階Top-Kトークン選択自己蒸留 (arXiv:2610.07247) 既発
- Instruction Tuning / Alignment
- Long Context / Context Window 拡張
- Agent Behavior / Programmatic Specification
  - ABCAgent: Agent Behavior as Code (arXiv:2610.04824) 既発
- Multi-Agent Reasoning
  - DHCG: 動的階層協調グラフ (arXiv:2610.07835) 既発
