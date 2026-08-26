import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

colors = {
    'grpo': '#4C72B0',
    'erpo': '#DD8452',
    'base': '#55A868',
    'grpo_qw': '#C44E52',
    'grpo_qkl': '#8172B3',
    'erpo_low': '#64B5CD',
    'erpo_high': '#E377C2',
}

# ============================================================
# Figure 1: Main results - grouped bar chart (Avg@32, 6 benchmarks)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

benchmarks = ['AIME24', 'AIME25', 'AMC', 'MATH500', 'Minerva', 'OlympiadBench']
grpo_avg32 = [0.471, 0.287, 0.768, 0.850, 0.516, 0.558]
erpo_avg32 = [0.509, 0.342, 0.820, 0.904, 0.500, 0.593]

x = np.arange(len(benchmarks))
width = 0.32

bars1 = ax.bar(x - width/2, [v*100 for v in grpo_avg32], width, label='GRPO', color=colors['grpo'], alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, [v*100 for v in erpo_avg32], width, label='ERPO', color=colors['erpo'], alpha=0.85, edgecolor='white', linewidth=0.5)

for b1, b2 in zip(bars1, bars2):
    diff = b2.get_height() - b1.get_height()
    if diff > 0:
        ax.annotate(f'+{diff:.1f}%', xy=(b2.get_x() + b2.get_width()/2, b2.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color='#C44E52', fontweight='bold')

ax.set_ylabel('Pass@32 (%)')
ax.set_title('ERPO vs GRPO on Qwen2.5-Math-7B (Mean across temps 0.1-1.5)')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.legend(loc='upper left')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/erpo-environment-regularized-policy-optimization/fig1_main_results.png')
plt.close()

# ============================================================
# Figure 2: Temperature stability - line chart (MATH500, Qwen-7B, n=8)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

temps = [0.1, 0.6, 1.0, 1.5]
grpo_temp = [66.8, 68.4, 73.8, 0.4]
erpo_temp = [79.4, 80.6, 75.2, 8.6]
erpo_high_temp = [78.8, 81.0, 76.0, 15.0]  # alpha=5e-2

ax.plot(temps, grpo_temp, 'o-', color=colors['grpo'], linewidth=2.5, markersize=8, label='GRPO')
ax.plot(temps, erpo_temp, 's-', color=colors['erpo'], linewidth=2.5, markersize=8, label='ERPO (alpha=1e-2)')
ax.plot(temps, erpo_high_temp, 'D-', color=colors['erpo_high'], linewidth=2.5, markersize=8, label='ERPO (alpha=5e-2)')

ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.55, 2, 'Standard\ntemp range', ha='center', fontsize=9, color='gray', style='italic')
ax.text(1.25, 2, 'High\ntemp', ha='center', fontsize=9, color='gray', style='italic')

ax.set_xlabel('Sampling Temperature')
ax.set_ylabel('MATH500 Pass@1 (%)')
ax.set_title('Temperature Stability: ERPO Maintains Performance at High Temps')
ax.set_xticks(temps)
ax.legend(loc='upper left')
ax.set_ylim(-5, 95)

# Highlight the collapse region
ax.fill_between([1.0, 1.5], [-5, -5], [95, 95], alpha=0.05, color='red')

plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/erpo-environment-regularized-policy-optimization/fig2_temperature_stability.png')
plt.close()

# ============================================================
# Figure 3: KL & Entropy decomposition (ablation mechanism)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

methods = ['GRPO', 'GRPO\nw/ QW', 'GRPO\nw/ QKL', 'ERPO']
query_kl = [0.9679, 0.5933, 0.0041, 0.0828]
policy_kl = [0.0601, 0.0113, 0.1001, 0.0728]
entropy = [0.5063, 0.2782, 0.5674, 0.4244]
bar_colors = [colors['grpo'], colors['grpo_qw'], colors['grpo_qkl'], colors['erpo']]

for ax_i, vals, title, ylabel in zip(axes, [query_kl, policy_kl, entropy],
                                        ['Query-KL', 'Policy-KL', 'Response Entropy'],
                                        ['KL divergence', 'KL divergence', 'Entropy']):
    bars = ax_i.bar(methods, vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_i.set_title(title, fontweight='bold')
    ax_i.set_ylabel(ylabel)
    ax_i.tick_params(axis='x', labelsize=10)
    for bar, val in zip(bars, vals):
        ax_i.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Mechanism Decomposition: What Each Component Does', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/erpo-environment-regularized-policy-optimization/fig3_kl_entropy_decomposition.png')
plt.close()

# ============================================================
# Figure 4: Reward hacking analysis - training vs eval gap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

steps = [0, 40, 80, 120, 160, 200, 240]

# Left: Training accuracy
grpo_train = [44.48, 76.41, 76.09, 78.29, 78.95, 79.44, 76.70]
erpo_train = [44.55, 75.63, 76.53, 78.66, 80.63, 78.41, 81.46]

axes[0].plot(steps, grpo_train, 'o-', color=colors['grpo'], linewidth=2, markersize=6, label='GRPO Train')
axes[0].plot(steps, erpo_train, 's-', color=colors['erpo'], linewidth=2, markersize=6, label='ERPO Train')
axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Training Accuracy')
axes[0].legend(loc='lower right')
axes[0].set_ylim(40, 90)

# Right: Eval accuracy (TP1)
grpo_eval = [31.20, 73.20, 76.00, 73.40, 72.20, 73.60, 58.40]
erpo_eval = [31.20, 74.00, 77.20, 77.00, 78.40, 78.60, 78.40]

axes[1].plot(steps, grpo_eval, 'o-', color=colors['grpo'], linewidth=2, markersize=6, label='GRPO Eval')
axes[1].plot(steps, erpo_eval, 's-', color=colors['erpo'], linewidth=2, markersize=6, label='ERPO Eval')
axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Eval Accuracy @ Temp=1.0')
axes[1].legend(loc='lower right')
axes[1].set_ylim(25, 90)

# Annotate the collapse
axes[1].annotate('Reward\nHacking!', xy=(240, 58.4), xytext=(200, 48),
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                 fontsize=11, color='#C44E52', fontweight='bold', ha='center')
axes[1].annotate('Stable\n+47.2pp', xy=(240, 78.4), xytext=(200, 85),
                 arrowprops=dict(arrowstyle='->', color='#55A868', lw=1.5),
                 fontsize=11, color='#55A868', fontweight='bold', ha='center')

fig.suptitle('Reward Hacking: GRPO Collapses at Step 240, ERPO Remains Stable', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/erpo-environment-regularized-policy-optimization/fig4_reward_hacking.png')
plt.close()

print('All 4 figures generated successfully.')