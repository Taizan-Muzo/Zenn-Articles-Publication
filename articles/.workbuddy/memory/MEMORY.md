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
（随发文更新）

## 候选论文池
- Reasoning / Planning 系最新研究
- LLM Agent / Tool-use 相关
- RAG / Retrieval 增强
- Multilingual / Cross-lingual NLP
- Efficient Inference / 推论高速化
- Instruction Tuning / Alignment
- Long Context / Context Window 拡張
