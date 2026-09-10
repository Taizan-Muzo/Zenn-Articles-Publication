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

## 候选论文池
- LLM-as-a-Judge / 評価信頼性 系最新研究
- Reasoning / Planning 系最新研究
- LLM Agent / Tool-use 相关（KOPA-Bench: arXiv:2609.05395 EMNLP 2026 Industry）
- RAG / Retrieval 增强
- Multilingual / Cross-lingual NLP
- Efficient Inference / 推论高速化（Don't Drop Dropout: arXiv:2609.05275 同系）
- Instruction Tuning / Alignment
- Long Context / Context Window 拡張
