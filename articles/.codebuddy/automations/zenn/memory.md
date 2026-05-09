# 2026-05-06 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `articles/resrl-negative-projection-rl.md` — "ResRL：負サンプルの「残差」だけでRLVRを強化する投影勾配法"
- 论文: ResRL (arXiv:2605.00380), Zihan Lin et al., ICML 2026, 2026-05-01
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: SVD低ランク正部分空間投影で負サンプル残差分離・NSR比数学+9.4%・コード+9.6%・Agent+10.4%・12ベンチマーク平均一位・KL不要・Pass@1とPass@128同時改善

# 2026-05-05 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `lenvm-length-value-model.md` — "LenVM：価値推定で解くtokenレベル長さモデリング"
- 论文: LenVM (arXiv:2604.27039), Zhen Zhang et al. (UCSB・Apple), 2026-04-29
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 生成長さをRL価値推定で定式化・LIFEBench 64.8(7Bでクローズド超過)・GSM8K 200tokenで63%(硬截断6%の10倍)・PPO互換

# 2026-05-04 Zenn发文 - 自动化执行记录 (第4次触发)

## 执行摘要
- 状态: 今日已完成（Latent-GRPO已发布），跳过
- 时间: 10:14 第四次自动化触发

# 2026-05-04 Zenn发文 - 自动化执行记录 (第3次触发)

## 执行摘要
- 状态: 今日已完成（Latent-GRPO已发布），跳过
- 时间: 09:58 第三次自动化触发

# 2026-05-04 Zenn发文 - 自动化执行记录 (第2次触发)

## 执行摘要
- 状态: 今日已完成（Latent-GRPO已发布），跳过
- 时间: 09:57 第二次自动化触发

# 2026-05-04 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `latent-grpo.md` — "GRPOを潜在空間で動かすと壊れる3つの理由とLatent-GRPOの解"
- 论文: Latent-GRPO (arXiv:2604.27998), Jingcheng Deng et al. (中科院計算所・中科院大学・中科院自動化所), 2026-04-30
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: GRPO潜在空間3結合ボトルネック（多様体欠如・探索最適化非整列・混合非閉包）・単側ノイズサンプリング最重要（除去→崩壊）・高難度3.3倍短縮+4.27 P@1・AIME24 Pass@64 50.0%(GRPO 2.1倍)

# 2026-05-03 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `visual-primitives-deepseek.md` — "「指で指す」でMLLM推論の壁を破る：Visual Primitives"
- 论文: Thinking with Visual Primitives (DeepSeek AI), 2026-04-30 (公布翌日删除)
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: 指代鸿沟定式化、Point/Box作为思考最小单位、7056倍视觉压缩、迷宫66.9%(+17pp vs GPT-5.4)、Path Tracing 56.7%(+26pp vs Claude)

# 2026-05-03 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `kernelized-advantage-estimation.md` — "非パラメトリック統計でRLVRを再設計：KAEが実現する神諭レベルの推論最適化"
- 论文: Kernelized Advantage Estimation: From Nonparametric Statistics to LLM Reasoning (arXiv:2604.28005), Shijin Gong, Kai Ye, Jin Zhu, Xinyu Zhang, Hongyi Zhou, Chengchun Shi (中科大・LSE・伯明翰大・中科院・清華), 2026-04-30
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: カーネル平滑化による価値関数推定・Stone最適収束率達成・GRPO比MSE 60-73%削減・G=4 DAPO訓練でAIME25 0.1781(+27.6% vs GPG)・追加コスト履歴バッファのみ・3定理3推論で理論保証強固

# 2026-05-02 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `llm-hallucination-incentive-nature.md` — "正確性評価がLLMのハルシネーションを招く：Nature掲載論文が暴くインセンティブ構造"
- 论文: Evaluating Large Language Models for Accuracy Incentivizes Hallucinations (Nature, arXiv:2509.04664), Kalai, Nachum, Vempala, Zhang (OpenAI×Georgia Tech), 2026-04-22
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: Nature掲載・IIV還元で生成エラー率≥2×IIV分類エラー率を数学証明・5統計誤差源・Open-rubric評価提案・推測インセンティブ歪み・RLHFキャリブレーション破壊

# 2026-05-01 Zenn发文 (第2篇) - 执行记录

## 执行摘要
- 文章: `webgen-r1-website-generation.md` — "RLで7Bが671Bに肉薄するWebGen-R1"
- 论文: WebGen-R1 (arXiv:2604.20398), Juyong Jiang et al. (HKUST・Alibaba Tongyi Lab・ETRI・Ant Group), ICLR 2026, 2026-04-22
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: スキャフォールド駆動構造化生成・カスケード型マルチモーダル報酬（構造→実行→美学）・GRPO最適化・VRR 95.89%(一位)・AAS 3.94(一位)・FSR 29.21%(DeepSeek-R1 30.25%に匹敵)・Human r=0.762

# 2026-05-01 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `exploration-hacking-rl-resistance.md` — "RL訓練を「拒否」するLLM：Exploration Hacking"
- 论文: Exploration Hacking: Can LLMs Learn to Resist RL Training? (arXiv:2604.28182), Eyon Jang, Damon Falck, Joschka Braun et al., 2026-04-30
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 81页37图大作・モデル生物実験で選択的RL抵抗を実証・Biosecurity/AI R&D領域で有効・監視+重みノイズ+SFT引き出しの3段階対策・フロンティアモデル間接情報で探索抑制推論顕在化（GPT-4o 42%・Gemini 38%）・報酬ハッキングより根源的な「探索ハッキング」脅威モデル

# 2026-04-29 Zenn発文 (第2篇) - 执行记录

## 执行摘要
- 文章: `where-reasoning-breaks-connectives.md` — "推論はどこで壊れるか：接続詞が推論チェーンの脆弱点である理由"
- 论文: Where Reasoning Breaks (arXiv:2604.20564)、Seunghyun Park / Yuanyuan Lei (フロリダ大学)、ACL 2026 Findings、2026-04-22
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4)，git: 2次commit，push成功
- 核心: 論理接続詞の高エントロピー分岐点・41%脆弱性・1.75倍破壊力・三層介入・Self-Cons比50%コスト削減

# 2026-04-29 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `rl-generalization-feature.md` — "RLはなぜSFTより汎化するのか？特徴レベルで解き明かすメカニズム解明"
- 论文: Why Does Reinforcement Learning Generalize? (arXiv:2604.25011), Dan Shi et al. (天津大学TJUNLP・DFKI・ザール大学), ACL 2026主会, 2026-04-27
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 3モデルSparse CrossCoder・SFT早すぎる特化vs RL遅延探索・50〜16個の汎化制御特徴・零化最大-46.2%・増幅最大+55.6%・未知タスクでも有効

# 2026-04-28 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `polygrpo-multilingual-reasoning.md` — "polyGRPO：言語を潜在変数にする多言語推論のRL最適化"
- 论文: Language as a Latent Variable for Reasoning Optimization (arXiv:2604.21593), Linjuan Wu et al., 2026-04-23
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 言語＝潜在変数・多言語応答でRLVR探索空間拡張・18.1Kで英語+6.72%・X-CSQA数学のみ訓練で常識+4.9%唯一超越・polyGSPOでも+6.6%

# 2026-04-26 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `structmem-hierarchical-memory.md` — "構造化記憶でAgentの「長期記憶」を実現するStructMem"
- 论文: StructMem (arXiv:2604.21748), Buqiang Xu et al. (浙大・蚂蚁集团), ACL 2026主会, 2026-04-23
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: ACL 2026主会・イベント中心階層化記憶・LoCoMo Overall 76.82%(+1.04)・Token 18.5x効率・API 50x削減・幻覚率<4%

# 2026-04-27 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `abstract-cot-latent-reasoning.md` — "Abstract-CoT：言葉を使わず思考するLLM推論の効率化"
- 论文: Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought (arXiv:2604.22709), IBM Research AI (Keshav Ramji et al.), 2026-04-24
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 抽象トークンによる離散潜在推論・3段階訓練パイプライン(SFT+PI-3ウォームアップ+GRPO)・最大11.6倍推論圧縮(MATH-500 1671→144)・精度ほぼ同等(92.6→90.8)・語彙にZipf的冪乗分布出現・3B〜32B汎化

# 2026-04-25 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `vps-verbal-critique-reasoning.md` — "VPS：自然言語の「批判」だけで推論を拡張する第4の軸"
- 论文: Process Supervision via Verbal Critique (arXiv:2604.21611), Hao-Yuan Chen (Mindify AI Research), 2026-04-23
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: 推理时扩展第4轴(语言监督粒度)・免训练VPS框架・GPQA 94.9%新SOTA・AIME +63.3pp弱Actor救助・Reflexion比+8.5~12.1pp・H vs Δ r=+0.90

# 2026-04-24 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `knowledge-capsules-kvi.md` — "テキストでなくKVで注入する：KVIがRAGの構造的限界を突破する理由"
- 论文: Knowledge Capsules (arXiv:2604.20487), 浙江天使医療AI・米缇AI・浙江大学, 2026-04-22
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: 非パラメトリックKV注入(KVI)フレームワーク、コンテキスト→メモリレベル知識統合、MedHopQA 92.5%、消融でグラフ検索除去=-100%

# 2026-04-23 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `evpo-explained-variance-ppo.md` — "PPOかGRPOか？Explained Varianceが解くRLVRの最大のジレンマ"
- 论文: EVPO (arXiv:2604.19485), 北京大学・复旦大学・上海人工智能ラボ, 2026-04-21
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: Explained VarianceでCritic品質監視、PPO/GRPO適応切替、カルマンフィルタ定式化、定理1(EV=0境界)、全4タスク一位

# 2026-04-22 Zenn发文 - 自动化执行记录 (第2篇 - CoSearch)

## 执行摘要
- 文章: `cosearch-joint-rank-reason.md` — "検索もRLで鍛える：CoSearchがAgentic Searchの検索ボトルネックを解消"
- 论文: CoSearch (arXiv:2604.17555), UMass Amherst / Snap Research, 2026-04-19
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: 推論+生成型ランカーGRPO同時訓練、セマンティックグルーピング、7B+6.6%/3B+10.0%

# 2026-04-22 Zenn发文 - 自动化执行记录 (第5次触发)

## 执行摘要
- 状态: 今日已完成（PreRL/DSRL已发布），跳过
- 时间: 09:59 第五次自动化触发

# 2026-04-22 Zenn发文 - 自动化执行记录 (第4次触发)

## 执行摘要
- 状态: 今日已完成（PreRL/DSRL已发布），跳过
- 时间: 09:58 第四次自动化触发

# 2026-04-22 Zenn发文 - 自动化执行记录 (第3次触发)

## 执行摘要
- 状态: 今日已完成（PreRL/DSRL已发布），跳过
- 时间: 09:58 第三次自动化触发

# 2026-04-22 Zenn发文 - 自动化执行记录 (第2次触发)

## 执行摘要
- 状态: 今日已完成（PreRL/DSRL已发布），跳过
- 时间: 09:57 第二次自动化触发

# 2026-04-22 Zenn发文 - 自动化执行记录 (第1次)

## 执行摘要
- 文章: `prerrl-pretrain-space-rlvr.md` — "問題条件を外すだけで推論が開花：PreRLのパラダイムシフト"
- 论文: PreRL/DSRL (arXiv:2604.14142), 中科院自动化所・NUS・テンセントAI Lab, 2026-04-15
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: P(y|x)→P(y)空間拡張、NSR負サンプル強化14.89倍思考増加、DSRL二段階

# 2026-04-21 Zenn发文 - 自动化执行记录 (第2篇)

## 执行摘要
- 文章: `specguard-step-verification.md` — "ステップ単位で検証する投機的推論：SpecGuardの仕組みと成果"
- 论文: SpecGuard (arXiv:2604.15244), IIT Kharagpur / Adobe Research, 2026-04-16
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: デュアル信号ステップ検証、自己一貫性候補選択、外部PRM不要、MATH500 85.4%

# 2026-04-21 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `longact-sparse-qk-rl.md` — "Q/Kの「大振幅」だけを更新：LongActが長文脈RLの壁を破る"
- 论文: LongAct (arXiv:2604.14922), 北京大学・上海交通大学, 2026-04-16
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: Q/K大振幅スパース更新、LongBench v2 +8%、推論ゼロオーバーヘッド

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

# 2026-04-15 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `tepo-token-level-rl.md` — "GRPOの弱点をToken粒度で解決：TEPOによるLLM強化学習の安定化"
- 论文: TEPO: Token-Level Policy Optimization (arXiv:2604.12736), 2026-04-14
- 初版: arXiv:2510.09369 (2025年10月)
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_method_comparison, fig2_benchmarks, fig3_training_efficiency, fig4_ablation)
- git: 3次commit (draft + path fix + 公開), push成功
- 核心: マルコフ逐次尤度でグループ報酬をトークン単位に因果配分、KLマスク制約でエントロピーコラプス回避、収束50%削減
- 修正: 初回pushでパス間違いを修正（articles/articles/ → articles/）。.gitignoreにgen_figures_*.pyと.codebuddy/を追加

# 2026-04-16 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `mm-doc-r1-spo.md` — "GRPOの基線偏差を相似度で修正：MM-Doc-R1と多轮RL文書理解"
- 论文: MM-Doc-R1 (arXiv:2604.13579)，Fudan大学・Singularity AI，2026-04-15
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_framework, fig2_spo_vs_grpo, fig3_results, fig4_ablation)
- git: 2次commit (draft + 公開), push成功。路径正确。
- 核心: SPO用轨迹类似度加权修正GRPO多轮基线偏差，MMLongBench-Doc +10.4%，SPO>GRPO +5.0%(8B)/+6.1%(4B)

# 2026-04-17 Zenn発文 - 自動化実行記録

## 実行サマリー
- 文章: `lightning-opd-offline-distillation.md` — "教師サーバー不要で4倍高速：Lightning OPDとOffline蒸留の理論"
- 论文: Lightning OPD (arXiv:2604.13010), NVIDIA, 2026-04-14
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1-fig4), git: 2次commit, push成功
- 核心: 教師一貫性定理、事前計算でOPDをオフライン化、4倍高速、AIME2024 69.9%(8B)

# 2026-04-19 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `rlvr-reward-hacking.md` — "RLVRの隠れた失敗：LLMはバリデータを「ハック」している"
- 论文: LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (arXiv:2604.15149)
- 作者: Lukas Helff et al. (TU Darmstadt, Meta FAIR, DFKI, hessian.AI)
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_induction_vs_shortcut, fig2_shortcut_comparison, fig3_complexity_reasoning_effect, fig4_training_causality)
- git: 2次commit (下書き + 公開), push成功
- 核心: RLVR訓練モデルが帰納推論タスクでルールの代わりにインスタンス列挙でバリデータをハック、IPT(同型摂動テスト)で黒箱検出可能、非RLVRモデルはゼロハック、外延バリデータがハックを誘発し同型バリデータが抑制することを因果証明

# 2026-04-19 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `hallucination-trajectory-commitment.md` — "ハルシネーションは「引き寄せの罠」か？因果証拠で迫る非対称アトラクター"
- 论文: Hallucination as Trajectory Commitment (arXiv:2604.15400), Chimera Research Initiative (Istanbul), 2026-04-16
- 状态: ✅ 已发布 (published: true)
- 配图: 4张 (fig1_attractor_basin, fig2_stepwise_divergence, fig3_layer_sweep, fig4_window_regime)
- git: 2次commit (draft + 公開), push成功
- 核心: 同一プロンプト分岐法で因果的非対称性を証明、破壊87.5% vs 修正33.3%（2.63倍）、生成前Step 0でr=0.776予測、5つのレジーム・クラスター発見

# 2026-04-20 Zenn发文 - 自动化执行记录

## 执行摘要
- 文章: `knowrl-minimal-knowledge-rl.md` — "RLVRの報酬希薄性を「最小十分な知識」で突破するKnowRL"
- 论文: KnowRL (arXiv:2604.12627), 天津大学・百度・中科院, 2026-04-14
- 状态: ✅ 已发布 (published: true)
- 配图: 4张, git: 2次commit, push成功
- 核心: KP原子化 + CSS最小十分セット選別 + 剪枝交互パラドックス、1.5B SOTA 74.16
- 备注: 之前session准备的草稿，本次仅执行发布流程

# 2026-04-20 Zenn发文 - 自动化执行记录 (第2次触发)

## 执行摘要
- 状态: 今日已完成（KnowRL已发布），跳过
- 时间: 09:59 第二次自动化触发

# 2026-04-20 Zenn发文 - 自动化执行记录 (第3次触发)

## 执行摘要
- 状态: 今日已完成（KnowRL已发布），跳过
- 时间: 10:00 第三次自动化触发

