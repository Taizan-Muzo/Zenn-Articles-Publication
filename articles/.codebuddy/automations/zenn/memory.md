# 2026-04-03 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `rlvr-reasoning-limit.md` — "LLMの強化学習は「推論能力」を本当に引き出しているのか？"
- 论文: Does RL Really Incentivize Reasoning in LLMs Beyond the Base Model? (NeurIPS 2025 Runner-up)
- 状态: ✅ 已发布 (published: true)
- 配图: 3张 (fig1_pass_at_k, fig2_coverage_training, fig3_efficiency_ppl)
- git: 2次commit (下書き + 公開), push成功
- 备注: 文章和配图为之前session准备，本次仅执行发布流程

# 2026-04-06 Zenn发文 - 自动化执行记录 (第1篇)

## 执行摘要
- 文章: `dapo-open-source-rl-llm.md` — "大規模LLM強化学習をオープンソースで実現する4つの鍵技術"
- 论文: DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476)
- 机构: ByteDance Seed, 清华大学, 香港大学
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_decoupled_clip, fig2_dynamic_sampling, fig3_training_efficiency, fig4_techniques_overview)
- git: 2次commit (下書き + 公開), push成功

# 2026-04-06 Zenn发文 - 自动化执行记录 (第2篇)

## 执行摘要
- 文章: `reasoning-shift-context-llm.md` — "Reasoning Shift: コンテキストがLLMの推論を「静かに」短くしてしまう問題"
- 论文: Reasoning Shift: How Context Silently Shortens LLM Reasoning (arXiv:2604.01161)
- 作者: Gleb Rodionov (2026-04-01发布)
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_reasoning_length, fig2_self_verification, fig3_performance_impact, fig4_three_scenarios)
- git: 2次commit (下書き + 公開), push成功
- 备注: 用户要求发更新的论文，选择了4天前刚发布的最新论文

# 2026-04-07 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `skill0-agent-rl.md` — "SKILL0: Agentのスキルをパラメータに「内化」する新たなRLフレームワーク"
- 论文: SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization (arXiv:2604.02268)
- 作者: Zhengxi Lu et al. (浙江大学), 2026-04-02公开
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_comparison, fig2_pipeline, fig3_performance, fig4_helpfulness)
- git: 2次commit (下書き + 公開), push成功
- 亮点: 动态课程学习 + 视觉上下文渲染，ALFWorld +9.7%

# 2026-04-07 标题修复 - 自动化执行记录

## 执行摘要
- 修复了17篇文章标题（全部压缩到70字符以内，Zenn限制）
- git: 1次commit, push成功 (18 files changed)
- 已更新MEMORY.md中的文章格式规范，添加标题上限70字符的注意事项

# 2026-04-08 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `neureasoner-mon.md` — "NeuReasoner: ニューロン白盒解析でLLM推論エラーを検出・修正する統合フレームワーク"
- 论文: NeuReasoner (arXiv:2604.02972), 北京大学, 2026-04-03
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4)
- git: 2次commit + push成功
- 核心: MoN白盒解析 + 轻量MLP O(1)监控 + 特殊token自我修正，最大+27.0% / -63.3% token

# 2026-04-09 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `code-stop-confidence-dynamics.md` — "CoDE-Stop: 推論モデルの「考えすぎ」を置信度の動的で止める手法"
- 论文: CoDE-Stop (arXiv:2604.04930), UMD/USC, 2026-04-06
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4)
- git: 2次commit + push成功
- 核心: 学習不要の推論早停止、置信度動的+退化スコア2条件、最大63%トークン削減

# 2026-04-10 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `triattention-kv-compression.md` — "TriAttention: 三角関数でKVキャッシュを10.7倍圧縮する長距離推論の効率化"
- 论文: TriAttention: Efficient Long Reasoning with Trigonometric KV Compression (arXiv:2604.04921)
- 作者: Weian Mao et al. (MIT, NVIDIA, ZJU), 2026-04-06公开
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4)
- git: 2次commit + push成功
- 核心: Pre-RoPE空間Q/K集中現象、三角級数によるKV重要度予測、10.7倍KV削減、AIME25でFull Attention同等精度

# 2026-04-11 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `cog-drift-rl-curriculum.md` — "難問を「選択肢」に変える：RLVRの探索限界を突破するCog-DRIFT"
- 论文: Cog-DRIFT (arXiv:2604.04767), UNC Chapel Hill, Mohit Bansal Lab, 2026-04-06公开
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4) - 变形难度阶梯、课程动态、基准对比、消融研究
- git: 2次commit (下書き + 公開), push成功
- 核心: タスク変形(4択MCQ→OEQ) + インスタンスレベル適応型カリキュラム、BigMath-Hard 0%→10.11%、6ベンチマーク平均+4.72%

# 2026-04-12 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `ragen2-template-collapse.md` — "Agentic RLの「見えない崩壊」：エントロピーでは検出できないTemplate Collapse"
- 论文: RAGEN-2 (arXiv:2604.06268), Northwestern University等, 2026-04-07公开
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_template_collapse, fig2_mi_vs_entropy, fig3_snr_filtering, fig4_performance)
- git: 2次commit (下書き + 公開), push成功
- 核心: Template Collapse新失效模式、MI vs Entropy(r=+0.39 vs -0.14)、SNR-Aware Filtering、全タスク20-30%向上

# 2026-04-12 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `gemma-phi-qwen3-reasoning-benchmark.md` — "Dense vs MoE推論モデルの実力比較"
- 论文: Gemma 4, Phi-4, and Qwen3: Accuracy–Efficiency Tradeoffs (arXiv:2604.07035)
- 作者: Md Motaleb Hossen Manik, Ge Wang (RPI), 2026-04-08
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4) - 散布図・ヒートマップ・プロンプト感度・レーダーチャート
- git: 2次commit (下書き + 公開), push成功
- 核心: 7モデル×4ベンチマーク×3プロンプト=8,400回統制実験、Gemma-4-E4Bが最良トレードオフ(精度0.675, VRAM 14.9GB)、MoEのVRAMオーバーヘッド警鐘、Phi-4 GSM8Kプロンプト崩壊(0.670→0.110)

# 2026-04-13 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `sppo-sequence-level-ppo.md` — "SPPO: PPOを超える長距離推論RL"
- 论文: SPPO (arXiv:2604.08865), ACL 2026 Main Conference, SUSTech/Microsoft, 2026-04-10
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_comparison, fig2_benchmarks_7b, fig3_training_efficiency, fig4_resource_efficiency)
- git: 2次commit (下書き + 公開), push成功
- 核心: Sequence-Level Contextual Bandit再定式化、N=1でGRPO(N=8)同等性能、5.9倍訓練加速、12.8%メモリ削減、7BモデルAverage 58.56

# 2026-04-14 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `decomposing-delta-preference.md` — "偏好対からLLMは何を学ぶか：Delta分解でDPOを効率化"
- 论文: Decomposing the Delta: What Do Models Actually Learn from Preference Pairs? (arXiv:2604.08723)
- 作者: Chia-Hsuan Lee et al. (Capital One), 2026-04-09
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_two_dimensions, fig2_generator_scaling, fig3_data_efficiency, fig4_dual_strategy)
- git: 2次commit (下書き + 公開), push成功
- 核心: DeltaをGenerator-LevelとSample-Levelの2次元に分解、Generator Delta最大化でOOD+9.0%、Sample Delta上位30%で全データ訓練同等、Step Coherenceが最良フィルタ基準
