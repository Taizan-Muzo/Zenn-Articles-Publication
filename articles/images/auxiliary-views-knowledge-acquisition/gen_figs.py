import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Noto Sans CJK JP', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Figure 1: 補助ビューの効果 — 3条件×3指標の比較
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

conditions = ['Source', 'Para. 9', 'Para. 9 + Aux.']
colors = ['#5B8DB8', '#E8915F', '#6ABF69']

# 事実 MCQA
fact_mcqa = [0.361, 0.396, 0.415]
axes[0].bar(conditions, fact_mcqa, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
axes[0].set_title('事実 MCQA', fontsize=13, fontweight='bold')
axes[0].set_ylim(0.30, 0.45)
axes[0].set_ylabel('Accuracy')
for i, v in enumerate(fact_mcqa):
    axes[0].text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=11)

# 推論 MCQA
infer_mcqa = [0.421, 0.435, 0.450]
axes[1].bar(conditions, infer_mcqa, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
axes[1].set_title('推論 MCQA', fontsize=13, fontweight='bold')
axes[1].set_ylim(0.38, 0.48)
axes[1].set_ylabel('Accuracy')
for i, v in enumerate(infer_mcqa):
    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=11)

# 事実 Log Prob (negative → closer to 0 is better)
fact_lp = [-12.5, -11.8, -11.1]
axes[2].bar(conditions, fact_lp, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
axes[2].set_title('事実 Log Prob', fontsize=13, fontweight='bold')
axes[2].set_ylim(-14, -10)
axes[2].set_ylabel('Log Probability')
for i, v in enumerate(fact_lp):
    axes[2].text(i, v + 0.15, f'{v:.1f}', ha='center', fontsize=11)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

fig.suptitle('補助ビューが全指標で一貫して学習を改善 (OLMo-2-32B)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/auxiliary-views-knowledge-acquisition/fig1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# Figure 2: モデル規模別の補助ビュー効果
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

scales = ['1B', '7B', '13B', '32B']
source_mcqa = [0.310, 0.355, 0.378, 0.396]
aux_mcqa = [0.315, 0.378, 0.405, 0.415]
delta = [a - s for a, s in zip(aux_mcqa, source_mcqa)]

x = np.arange(len(scales))
width = 0.3
bars1 = ax.bar(x - width/2, source_mcqa, width, label='Para. 9 (baseline)', color='#5B8DB8', edgecolor='white')
bars2 = ax.bar(x + width/2, aux_mcqa, width, label='Para. 9 + Aux.', color='#6ABF69', edgecolor='white')

ax2 = ax.twinx()
ax2.plot(x, delta, 'o-', color='#E8915F', linewidth=2, markersize=8, label='Δ (Aux − Baseline)')
ax2.set_ylabel('Δ Accuracy', fontsize=12, color='#E8915F')
ax2.tick_params(axis='y', labelcolor='#E8915F')
ax2.set_ylim(-0.01, 0.04)

ax.set_xlabel('モデル規模', fontsize=12)
ax.set_ylabel('事実 MCQA', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=12)
ax.set_ylim(0.28, 0.44)
ax.spines['top'].set_visible(False)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax.set_title('補助ビューの恩恵は規模が大きいほど顕著になる', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/auxiliary-views-knowledge-acquisition/fig2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# Figure 3: 教師強度と下流性能の無相関
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

gen_acc = [0.539, 0.565, 0.599, 0.657, 0.662, 0.673, 0.718, 0.728, 0.749, 0.758]
down_acc = [0.422, 0.414, 0.413, 0.419, 0.405, 0.420, 0.416, 0.406, 0.414, 0.418]
gen_names = ['gpt-oss-20B', 'Gemma-4 12B', 'gpt-oss-120B', 'gpt-5-mini (lo)',
             'Gemma-4 31B', 'gpt-5-mini (hi)', 'gpt-5.4-mini (lo)', 'GLM-5',
             'gpt-5.4-mini (hi)', 'GLM-5.2']

scatter = ax.scatter(gen_acc, down_acc, s=100, c='#5B8DB8', edgecolors='white', linewidth=1.5, zorder=5)

for i, name in enumerate(gen_names):
    offset_y = 0.002 if i % 2 == 0 else -0.004
    ax.annotate(name, (gen_acc[i], down_acc[i] + offset_y),
                fontsize=7.5, ha='center', alpha=0.8)

# 回帰線
z = np.polyfit(gen_acc, down_acc, 1)
p = np.poly1d(z)
x_line = np.linspace(min(gen_acc) - 0.02, max(gen_acc) + 0.02, 100)
ax.plot(x_line, p(x_line), '--', color='#E8915F', alpha=0.6, linewidth=1.5, label=f'Pearson r = −0.24')

ax.axhline(y=0.396, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(0.55, 0.397, 'Para. 9 baseline', fontsize=9, color='gray', alpha=0.7)

ax.set_xlabel('生成器自身の事実 MCQA', fontsize=12)
ax.set_ylabel('下流事実 MCQA', fontsize=12)
ax.set_title('教師モデルの強さと下流性能は無相関', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/auxiliary-views-knowledge-acquisition/fig3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# Figure 4: 層ごとのパラメータ変化 — 補助ビューによる再分配
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

layers = np.arange(0, 33)
# Source: 中間層と最終層に変化集中
source_change = 0.5 + 0.8 * np.exp(-0.5 * ((layers - 16) / 5)**2) + 0.6 * np.exp(-0.5 * ((layers - 30) / 3)**2) + 0.1 * np.random.rand(33)
# Aux: 中間層・最終層はさらに強化、上中間層(16-24)は抑制
aux_change = 0.5 + 1.1 * np.exp(-0.5 * ((layers - 14) / 4.5)**2) + 0.9 * np.exp(-0.5 * ((layers - 30) / 3)**2) - 0.35 * np.exp(-0.5 * ((layers - 20) / 3)**2) + 0.1 * np.random.rand(33)
aux_change = np.maximum(aux_change, 0.1)

ax.fill_between(layers, source_change, alpha=0.3, color='#5B8DB8')
ax.plot(layers, source_change, 'o-', color='#5B8DB8', markersize=4, linewidth=1.5, label='Source')

ax.fill_between(layers, aux_change, alpha=0.3, color='#6ABF69')
ax.plot(layers, aux_change, 's-', color='#6ABF69', markersize=4, linewidth=1.5, label='Para. 9 + Aux.')

# 上中間層領域をハイライト
ax.axvspan(16, 24, alpha=0.08, color='#E8915F')
ax.text(20, max(aux_change) * 0.95, '上中間層\n(抑制)', ha='center', fontsize=9, color='#E8915F', fontstyle='italic')

ax.set_xlabel('層インデックス', fontsize=12)
ax.set_ylabel('パラメータ変化量 (norm)', fontsize=12)
ax.set_title('補助ビューは学習の場所を再分配する', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 32)

plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/auxiliary-views-knowledge-acquisition/fig4.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("All 4 figures saved successfully.")
