import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ========== Figure 1: Advantage Concept Diagram ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})

# Left: Value trajectory with consequential step highlighted
ax = axes[0]
steps = np.arange(0, 11)
values = [0.45, 0.47, 0.46, 0.48, 0.49, 0.52, 0.85, 0.88, 0.87, 0.90, 0.91]

ax.plot(steps, values, 'o-', color='#4A90D9', linewidth=2, markersize=7, zorder=3)
ax.axvspan(4.5, 5.5, alpha=0.15, color='#E74C3C', zorder=1)

ax.annotate('', xy=(5, 0.85), xytext=(5, 0.49),
            arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2.5))
ax.text(5.3, 0.67, r'$\hat{A}(s_t, a_t)$' + '\n= +0.36', fontsize=11, color='#E74C3C', fontweight='bold')

ax.set_xlabel('Reasoning Step $t$', fontsize=12)
ax.set_ylabel(r'Value $\hat{V}(s_t)$', fontsize=12)
ax.set_title('Value Trajectory with Consequential Step', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(steps)
ax.set_ylim(0.3, 1.0)

# Right: Two similar-looking steps with different advantages
ax2 = axes[1]
categories = ['Step A\n(self-check:\n"maybe I\nmade error")', 'Step B\n(self-check:\n"check coords\nof point A")']
advantages = [0.02, 0.42]
colors = ['#95A5A6', '#E74C3C']

bars = ax2.bar(categories, advantages, color=colors, width=0.5, edgecolor='white', linewidth=2)
ax2.set_ylabel('Advantage', fontsize=12)
ax2.set_title('Similar Text, Drastically\nDifferent Importance', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 0.55)
ax2.axhline(y=0.1, color='#E67E22', linestyle='--', linewidth=1.5, alpha=0.7, label=r'$\delta = 0.1$ threshold')
ax2.legend(fontsize=10, loc='upper left')

for bar, val in zip(bars, advantages):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.015, f'{val:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

fig.tight_layout(pad=2)
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/legibility-is-not-interpretability-cot/fig1.png')
plt.close()
print("Figure 1 saved.")

# ========== Figure 2: Judge & Critic Performance ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: PR-AUC comparison
ax = axes[0]
models = ['Random\nBaseline', 'OOB Judge\n(Qwen3-8B)', 'OOB Judge\n(Qwen3.6-27B)', 'Fine-tuned\nCritic (1.7B)', 'Noise\nCeiling']
correct_vals = [0.02, 0.04, 0.06, 0.085, 0.58]
incorrect_vals = [0.02, 0.08, 0.12, 0.29, 0.60]

x = np.arange(len(models))
width = 0.32

bars1 = ax.bar(x - width/2, correct_vals, width, label='Correct responses', color='#3498DB', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, incorrect_vals, width, label='Incorrect responses', color='#E74C3C', edgecolor='white', linewidth=1.5)

ax.set_ylabel('PR-AUC', fontsize=12)
ax.set_title('Judge & Critic: PR-AUC\n(In-Distribution)', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.72)

# Right: Precision@k% budget curve
ax2 = axes[1]
k_pct = np.array([0.5, 1, 2, 5, 10])
random_p = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
critic_correct = np.array([0.12, 0.08, 0.05, 0.03, 0.03])
critic_incorrect = np.array([0.58, 0.35, 0.18, 0.08, 0.05])
ceiling = np.array([0.65, 0.55, 0.42, 0.28, 0.15])

ax2.plot(k_pct, ceiling, 's--', color='#2ECC71', linewidth=2, markersize=7, label='Noise ceiling', alpha=0.7)
ax2.plot(k_pct, critic_incorrect, 'o-', color='#E74C3C', linewidth=2, markersize=7, label='Critic (incorrect)')
ax2.plot(k_pct, critic_correct, 'o-', color='#3498DB', linewidth=2, markersize=7, label='Critic (correct)')
ax2.plot(k_pct, random_p, '^:', color='#95A5A6', linewidth=1.5, markersize=6, label='Random baseline')

ax2.set_xlabel('Inspection Budget (% of steps)', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision@k% Budget\n(Fine-tuned Critic, ID)', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 0.75)

fig.tight_layout(pad=2)
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/legibility-is-not-interpretability-cot/fig2.png')
plt.close()
print("Figure 2 saved.")

# ========== Figure 3: Reasoning Patterns by Model Type ==========
fig, ax = plt.subplots(figsize=(10, 5))

patterns = ['Always High', 'Gradual\nClimb', 'Sudden\nClimb', 'Recovery\nfrom Dip', 'Once High\nthen Low', 'Never High']
non_thinking = [24, 15, 22, 12, 8, 19]
thinking = [61, 12, 8, 6, 5, 8]

x = np.arange(len(patterns))
width = 0.32

bars1 = ax.bar(x - width/2, non_thinking, width, label='Non-thinking\n(Qwen3-1.7B)', color='#3498DB', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, thinking, width, label='Thinking\n(Qwen3-1.7B)', color='#E67E22', edgecolor='white', linewidth=1.5)

for bar, val in zip(bars1, non_thinking):
    if val > 5:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f'{val}%', ha='center', fontsize=9, color='#3498DB', fontweight='bold')
for bar, val in zip(bars2, thinking):
    if val > 5:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f'{val}%', ha='center', fontsize=9, color='#E67E22', fontweight='bold')

ax.set_ylabel('Proportion of Responses (%)', fontsize=12)
ax.set_title('Reasoning Patterns: Thinking vs Non-Thinking Models', fontsize=14, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(patterns, fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(0, 72)

# Add annotation
ax.annotate('Thinking mode shifts mass\nto "Always High" pattern',
            xy=(0, 61), xytext=(2, 65),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', edgecolor='#E67E22'))

fig.tight_layout(pad=2)
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/legibility-is-not-interpretability-cot/fig3.png')
plt.close()
print("Figure 3 saved.")

# ========== Figure 4: Consequential Step Types & Faithfulness ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Consequential step types
ax = axes[0]
step_types = ['Active\nComputation', 'Uncertainty\nMgmt', 'Self-\nChecking', 'Final Answer\nEmission', 'Setup /\nPlanning']
correct_freq = [0.28, 0.24, 0.20, 0.08, 0.12]
incorrect_freq = [0.10, 0.08, 0.12, 0.38, 0.14]

x = np.arange(len(step_types))
width = 0.3

bars1 = ax.bar(x - width/2, correct_freq, width, label='Correct responses', color='#2ECC71', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, incorrect_freq, width, label='Incorrect responses', color='#E74C3C', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Fraction of Consequential Steps', fontsize=11)
ax.set_title('Consequential Step Types\n(Qwen3-1.7B, Thinking)', fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(step_types, fontsize=9)
ax.legend(fontsize=9)

# Right: Faithfulness test
ax2 = axes[1]
conditions = ['No Cue', 'With Cue']
has_consequential = [58, 15]
no_consequential = [42, 85]

bars1 = ax2.bar(conditions, has_consequential, color='#E74C3C', edgecolor='white', linewidth=1.5, label='Has consequential step')
bars2 = ax2.bar(conditions, no_consequential, bottom=has_consequential, color='#BDC3C7', edgecolor='white', linewidth=1.5, label='No consequential step')

for bar, val in zip(bars1, has_consequential):
    ax2.text(bar.get_x() + bar.get_width()/2, val/2, f'{val}%', ha='center', va='center', fontsize=13, fontweight='bold', color='white')

ax2.set_ylabel('Proportion of Responses (%)', fontsize=11)
ax2.set_title('Self-Advantage Faithfulness Test\n(Scruples Dataset)', fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylim(0, 105)

fig.tight_layout(pad=2)
fig.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/legibility-is-not-interpretability-cot/fig4.png')
plt.close()
print("Figure 4 saved.")

print("All 4 figures generated successfully!")
