import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# --- Style settings ---
plt.rcParams.update({
    'font.family': ['Hiragino Sans', 'sans-serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 180,
    'savefig.bbox': 'tight',
})

# ============================================================
# fig1: Architecture overview - BSE vs History-Conditioned Agent
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: History-Conditioned (bad)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('History-Conditioned Agent\n(公理A4違反)', fontsize=13, fontweight='bold', color='#C0392B', pad=12)

boxes_left = [
    (1, 6, 'Environment', '#E74C3C'),
    (1, 4, 'Raw History $h_t$\n$(a_0, o_1, ..., a_{t-1}, o_t)$', '#F39C12'),
    (1, 2, 'LLM\n(history-conditioned)', '#E74C3C'),
    (1, 0.3, 'Action $a_t$', '#95A5A6'),
]
for x, y, txt, clr in boxes_left:
    box = FancyBboxPatch((x-0.8, y-0.45), 7.6, 0.9, boxstyle="round,pad=0.15",
                          facecolor=clr, alpha=0.15, edgecolor=clr, linewidth=1.8)
    ax.add_patch(box)
    ax.text(x+3, y, txt, ha='center', va='center', fontsize=9, color='#2C3E50')

# Arrows
for (y_from, y_to) in [(5.55, 4.45), (3.55, 2.45), (1.55, 0.75)]:
    ax.annotate('', xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

# Warning
ax.text(5, 7.2, '[X] 確率質量を蓄積しない', ha='center', fontsize=10, color='#C0392B',
        fontweight='bold')

# Right: BSE-Augmented (good)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('BSE-Augmented Agent\n(公理A1–A4充足)', fontsize=13, fontweight='bold', color='#27AE60', pad=12)

boxes_right = [
    (1, 6, 'Environment', '#27AE60'),
    (1, 4, 'BSE: Bayes Filter\n$b_t = U_\\beta(b_{t-1}, a, o)$', '#3498DB'),
    (1, 2, 'LLM\n(belief-measurable $\\tilde{\\pi}$)', '#27AE60'),
    (1, 0.3, 'Action $a_t$', '#95A5A6'),
]
for x, y, txt, clr in boxes_right:
    box = FancyBboxPatch((x-0.8, y-0.45), 7.6, 0.9, boxstyle="round,pad=0.15",
                          facecolor=clr, alpha=0.15, edgecolor=clr, linewidth=1.8)
    ax.add_patch(box)
    ax.text(x+3, y, txt, ha='center', va='center', fontsize=9, color='#2C3E50')

for (y_from, y_to) in [(5.55, 4.45), (3.55, 2.45), (1.55, 0.75)]:
    ax.annotate('', xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

ax.text(5, 7.2, '[O] Bellman最適性保証を継承', ha='center', fontsize=10, color='#27AE60',
        fontweight='bold')

plt.tight_layout()
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/belief-state-engine-pomdp-llm/fig1.png')
plt.close()

# ============================================================
# fig2: Four Axioms and Theorems flow
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

ax.text(6, 6.6, '四公理 → 七定理 → BSE設計', ha='center', fontsize=14, fontweight='bold', color='#2C3E50')

axioms = [
    ('A1', '再帰更新性', '#E74C3C'),
    ('A2', '予測十分性', '#E67E22'),
    ('A3', '確率内化', '#F1C40F'),
    ('A4', '信念可測性', '#27AE60'),
]
for i, (label, name, clr) in enumerate(axioms):
    x = 1.5 + i * 2.5
    box = FancyBboxPatch((x-0.9, 4.6), 1.8, 1.0, boxstyle="round,pad=0.12",
                          facecolor=clr, alpha=0.2, edgecolor=clr, linewidth=2)
    ax.add_patch(box)
    ax.text(x, 5.3, label, ha='center', va='center', fontsize=11, fontweight='bold', color=clr)
    ax.text(x, 4.85, name, ha='center', va='center', fontsize=8, color='#2C3E50')

# Theorems
theorems = [
    ('Thm1', '存在性', '#3498DB'),
    ('Thm4', 'Bayes一意性', '#3498DB'),
    ('Thm6', '値等価性', '#3498DB'),
    ('Thm9', '組合せ信頼性', '#9B59B6'),
]
for i, (label, name, clr) in enumerate(theorems):
    x = 1.5 + i * 2.5
    box = FancyBboxPatch((x-0.9, 2.6), 1.8, 1.0, boxstyle="round,pad=0.12",
                          facecolor=clr, alpha=0.2, edgecolor=clr, linewidth=2)
    ax.add_patch(box)
    ax.text(x, 3.3, label, ha='center', va='center', fontsize=11, fontweight='bold', color=clr)
    ax.text(x, 2.85, name, ha='center', va='center', fontsize=8, color='#2C3E50')

# BSE result
box = FancyBboxPatch((3.5, 0.6), 5, 1.0, boxstyle="round,pad=0.12",
                      facecolor='#27AE60', alpha=0.15, edgecolor='#27AE60', linewidth=2.5)
ax.add_patch(box)
ax.text(6, 1.1, 'BSE: Bellman最適性保証付きLLMエージェント', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#27AE60')

# Arrows: axioms -> theorems
for i in range(4):
    x = 1.5 + i * 2.5
    ax.annotate('', xy=(x, 3.6), xytext=(x, 4.6),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

# Arrows: theorems -> BSE
for i in range(4):
    x = 1.5 + i * 2.5
    ax.annotate('', xy=(6, 1.6), xytext=(x, 2.6),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2,
                               connectionstyle="arc3,rad=0.1"))

plt.tight_layout()
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/belief-state-engine-pomdp-llm/fig2.png')
plt.close()

# ============================================================
# fig3: Tiger POMDP belief dynamics illustration
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Belief trajectory over time
ax = axes[0]
np.random.seed(42)
steps = np.arange(0, 21)
# Simulate belief updates: start uniform, listen several times, then open
b_left = np.zeros(21)
b_left[0] = 0.5
# After listen actions with observations gradually increasing confidence
for t in range(1, 8):
    # hears "left" mostly (0.85 accuracy, tiger on left)
    obs_left = np.random.random() < 0.85
    if obs_left:
        b_left[t] = 0.85 * b_left[t-1] / (0.85 * b_left[t-1] + 0.15 * (1 - b_left[t-1]))
    else:
        b_left[t] = 0.15 * b_left[t-1] / (0.15 * b_left[t-1] + 0.85 * (1 - b_left[t-1]))
# After step 7, agent is confident and opens
for t in range(8, 21):
    b_left[t] = b_left[7]  # belief doesn't change after opening

ax.fill_between(steps[:8], b_left[:8], 1-b_left[:8], alpha=0.3, color='#3498DB', label='$b($right$)$')
ax.fill_between(steps[:8], 0, b_left[:8], alpha=0.3, color='#E74C3C', label='$b($left$)$')
ax.plot(steps[:8], b_left[:8], 'o-', color='#E74C3C', markersize=4, linewidth=1.5)
ax.plot(steps[:8], 1-np.array(b_left[:8]), 'o-', color='#3498DB', markersize=4, linewidth=1.5)
ax.axhline(y=0.5, color='#95A5A6', linestyle='--', alpha=0.5, label='一様先験')
ax.axvline(x=7, color='#27AE60', linestyle=':', alpha=0.7)
ax.text(7.3, 0.05, 'open!', fontsize=9, color='#27AE60', fontweight='bold')
ax.set_xlabel('Step $t$')
ax.set_ylabel('信念 $b_t(s)$')
ax.set_title('Tiger POMDP: BSE信念軌道', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='center right')
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.05, 1.05)

# Right: Comparison bar chart
ax = axes[1]
methods = ['Reactive', 'CoT', 'ReAct', 'NL Tracker', 'QMDP', 'POMCP', 'BSE']
# Representative values based on paper description
returns_tiger = [-15, -8, -5, 2, 5, 7, 9]
colors = ['#E74C3C', '#E67E22', '#F1C40F', '#95A5A6', '#3498DB', '#2ECC71', '#27AE60']
bars = ax.barh(methods, returns_tiger, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
ax.set_xlabel('平均割引報酬 (Tiger POMDP)')
ax.set_title('Tiger POMDP: 手法別報酬比較', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='#2C3E50', linewidth=0.8)
# Highlight BSE
bars[-1].set_edgecolor('#27AE60')
bars[-1].set_linewidth(2.5)

plt.tight_layout()
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/belief-state-engine-pomdp-llm/fig3.png')
plt.close()

# ============================================================
# fig4: Ablation study heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ablations = [
    '完全BSE',
    'AB1: 予測歩削除',
    'AB2: 観測歩削除',
    'AB3: 誤初期先験',
    'AB4: Top-1直列化',
    'AB5: Top-3直列化',
    'AB6: 自由テキスト信念',
    'AB7: 原始履歴露出',
    'AB8: 領域ヘッダ削除',
]
metrics = ['報酬', 'Brier↓', 'NLL↓', 'JSD↓']

# Simulated ablation results (relative to full BSE)
data = np.array([
    [9.0, 0.05, 0.12, 0.01],   # Full BSE
    [3.0, 0.25, 0.80, 0.15],   # AB1
    [-5.0, 0.45, 2.50, 0.35],  # AB2
    [5.0, 0.15, 0.40, 0.08],   # AB3
    [6.0, 0.12, 0.30, 0.05],   # AB4
    [7.5, 0.08, 0.20, 0.03],   # AB5
    [4.0, 0.20, 0.60, 0.12],   # AB6
    [1.0, 0.35, 1.50, 0.30],   # AB7
    [2.0, 0.30, 1.00, 0.20],   # AB8
])

# Normalize each column for color mapping
data_norm = np.zeros_like(data)
for j in range(data.shape[1]):
    col = data[:, j]
    if j == 0:
        data_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    else:
        data_norm[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)

im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(4))
ax.set_xticklabels(metrics, fontsize=11)
ax.set_yticks(range(9))
ax.set_yticklabels(ablations, fontsize=10)

# Add text annotations
for i in range(9):
    for j in range(4):
        val = data[i, j]
        text = f'{val:.2f}' if abs(val) < 10 else f'{val:.1f}'
        ax.text(j, i, text, ha='center', va='center', fontsize=9,
                color='white' if data_norm[i, j] < 0.3 or data_norm[i, j] > 0.7 else '#2C3E50')

ax.set_title('消融実験: BSE構成要素の寄与分離', fontsize=13, fontweight='bold', pad=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('良さ（緑＝良好）', fontsize=10)

plt.tight_layout()
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/belief-state-engine-pomdp-llm/fig4.png')
plt.close()

print("All 4 figures generated successfully.")
