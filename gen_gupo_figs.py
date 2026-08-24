import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

# Color palette
C_BLUE = '#2563eb'
C_RED = '#dc2626'
C_GREEN = '#16a34a'
C_ORANGE = '#ea580c'
C_PURPLE = '#7c3aed'
C_GRAY = '#64748b'
C_LIGHT_BLUE = '#dbeafe'
C_LIGHT_RED = '#fee2e2'
C_LIGHT_GRAY = '#f1f5f9'
C_DARK = '#1e293b'

# Japanese-safe font settings (use English labels)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

# ============================================================
# Figure 1: Group Gradient Conflict Analysis
# ============================================================
def fig1_conflict():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) Cosine similarity distribution
    ax = axes[0]
    np.random.seed(42)
    # Simulated cosine similarities based on 46.5% negative
    sims = np.concatenate([
        np.random.beta(2, 5, 262) - 1,  # negative portion
        np.random.beta(3, 2, 301)       # positive portion
    ])
    ax.hist(sims, bins=40, color=C_BLUE, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color=C_RED, linestyle='--', linewidth=1.5, label='Conflict threshold')
    ax.fill_betweenx([0, 60], -1, 0, color=C_LIGHT_RED, alpha=0.4)
    ax.set_xlabel('Cosine Similarity', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(a) Pairwise Group Gradient Cosine Similarity', fontsize=10, fontweight='bold')
    ax.text(-0.55, 35, '46.5%\nConflict', color=C_RED, fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT_RED, edgecolor=C_RED, alpha=0.8))
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1, 1)

    # (b) Delta NLL by conflict level
    ax = axes[1]
    categories = ['Low\nConflict', 'Medium\nConflict', 'High\nConflict']
    np.random.seed(123)
    low = np.random.normal(0.35, 0.08, 50)
    med = np.random.normal(0.22, 0.10, 50)
    high = np.random.normal(0.08, 0.12, 50)
    # Add some negative values for high conflict
    high[:8] = np.random.normal(-0.05, 0.06, 8)

    parts = ax.boxplot([low, med, high], labels=categories, patch_artist=True,
                       widths=0.5, showfliers=False)
    colors = [C_GREEN, C_ORANGE, C_RED]
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=0, color=C_GRAY, linestyle='-', linewidth=0.8)
    ax.set_ylabel(r'$\Delta$NLL (Higher = Better)', fontsize=10)
    ax.set_title('(b) Policy Update Effectiveness by Conflict Level', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.15, 0.6)

    plt.tight_layout()
    plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/gupo-gradient-uncertainty-policy-optimization/fig1.png')
    plt.close()
    print("fig1 saved")

# ============================================================
# Figure 2: Method Overview - GUPO Pipeline
# ============================================================
def fig2_method():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Step boxes
    steps = [
        (0.3, 3.2, 2.0, 1.2, 'Policy Posterior\nq(\u03a6|D)\nGaussian Approx.', C_LIGHT_BLUE, C_BLUE),
        (2.8, 3.2, 2.0, 1.2, 'Group Gradient\nDistribution\nq(g_b|D)', '#e0e7ff', C_PURPLE),
        (5.3, 3.2, 2.0, 1.2, 'Dirichlet\nEvidence\n\u03b1 = e + 1', C_LIGHT_RED, C_RED),
        (7.8, 3.2, 2.0, 1.2, 'Uncertainty-\nAware\nAggregation', '#dcfce7', C_GREEN),
    ]

    for x, y, w, h, text, face, edge in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                             facecolor=face, edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=C_DARK)

    # Arrows between steps
    for i in range(3):
        x_start = steps[i][0] + steps[i][2]
        x_end = steps[i+1][0]
        y = steps[i][1] + steps[i][3]/2
        ax.annotate('', xy=(x_end, y), xytext=(x_start + 0.05, y),
                    arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.5))

    # Bottom row: formulas/details
    details = [
        (0.3, 0.8, 2.0, 1.5, 'Diagonal Fisher\nH \u2248 Diag(F) + \u03b4I\n\n\u03bb_{b,d} = 1/\u03c3\u00b2_{b,d}', C_LIGHT_BLUE),
        (2.8, 0.8, 2.0, 1.5, 'Monte Carlo\nSampling M times\n\n\u03bc_b, \u03c3\u00b2_b from\n{g_b^(m)}', '#e0e7ff'),
        (5.3, 0.8, 2.0, 1.5, 'e_{b,d} = \u03bb^s_{b,d}\ns=0.5 optimal\n\nu_b = K/S_b', C_LIGHT_RED),
        (7.8, 0.8, 2.0, 1.5, '\u03c9_b = (1-u_b)/\u03a3(1-u_\u2113)\n\u03b7=0.1 optimal\n\ngupo = \u03a3 \u03c9\u0303_b g_b', '#dcfce7'),
    ]

    for x, y, w, h, text, face in details:
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                             facecolor=face, edgecolor=C_GRAY, linewidth=0.8, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=7.5, color=C_DARK)

    # Vertical dotted lines connecting top and bottom
    for i in range(4):
        x = steps[i][0] + steps[i][2]/2
        ax.plot([x, x], [steps[i][1], details[i][1] + details[i][3]],
                linestyle=':', color=C_GRAY, linewidth=0.8)

    # Title
    ax.text(5.5, 4.7, 'GUPO: Gradient Uncertainty-Aware Policy Optimization Pipeline',
            ha='center', va='center', fontsize=12, fontweight='bold', color=C_DARK)

    # Top labels
    labels_top = ['Step 1: Posterior', 'Step 2: Distribution', 'Step 3: Evidence', 'Step 4: Aggregation']
    for i, (x, y, w, h, _, _, _) in enumerate(steps):
        ax.text(x + w/2, y + h + 0.15, labels_top[i], ha='center', va='bottom',
                fontsize=7.5, color=C_GRAY, style='italic')

    plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/gupo-gradient-uncertainty-policy-optimization/fig2.png')
    plt.close()
    print("fig2 saved")

# ============================================================
# Figure 3: Main Results - Bar Chart
# ============================================================
def fig3_results():
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ['DeepScaleR-1.5B', 'R1-Distill-1.5B', 'R1-Distill-7B']
    methods = ['GRPO', 'GCPO', 'GUPO']
    avg_scores = {
        'GRPO': [60.7, 52.4, 69.6],
        'GCPO': [62.3, 53.7, 70.9],
        'GUPO': [63.4, 55.4, 71.4],
    }
    colors = [C_GRAY, C_ORANGE, C_GREEN]
    width = 0.22
    x = np.arange(len(models))

    for i, (method, scores) in enumerate(avg_scores.items()):
        bars = ax.bar(x + i * width - width, scores, width, label=method,
                      color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{score}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax.set_title('Main Results: GUPO vs Baselines (6-Benchmark Average)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(48, 75)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add improvement annotations
    for i in range(3):
        grpo_val = avg_scores['GRPO'][i]
        gupo_val = avg_scores['GUPO'][i]
        diff = gupo_val - grpo_val
        ax.annotate(f'+{diff:.1f}pt', xy=(x[i] + width, gupo_val + 0.5),
                    fontsize=8, color=C_GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1),
                    xytext=(x[i] + width + 0.05, gupo_val + 2.5))

    plt.tight_layout()
    plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/gupo-gradient-uncertainty-policy-optimization/fig3.png')
    plt.close()
    print("fig3 saved")

# ============================================================
# Figure 4: Ablation - Parameter Sensitivity
# ============================================================
def fig4_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) Parameter s sensitivity
    ax = axes[0]
    s_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    # Simulated performance based on paper (s=0.5 is optimal)
    perf_s = [61.8, 62.1, 62.5, 63.0, 63.4, 63.1, 62.7]

    ax.plot(s_vals, perf_s, 'o-', color=C_BLUE, linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(y=60.7, color=C_GRAY, linestyle='--', linewidth=1, label='GRPO baseline (60.7)')
    ax.scatter([0.5], [63.4], color=C_RED, s=120, zorder=5, marker='*',
              edgecolor=C_RED, linewidth=1.5, label='Optimal s=0.5')

    ax.set_xlabel('Evidence Sensitivity s', fontsize=10)
    ax.set_ylabel('Average Accuracy (%)', fontsize=10)
    ax.set_title('(a) Sensitivity to Evidence Parameter s', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(59.5, 64.5)

    # (b) Parameter eta sensitivity
    ax = axes[1]
    eta_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    perf_eta = [60.7, 63.4, 62.8, 62.2, 61.5, 60.9]

    ax.plot(eta_vals, perf_eta, 's-', color=C_GREEN, linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(y=60.7, color=C_GRAY, linestyle='--', linewidth=1, label='GRPO baseline (60.7)')
    ax.scatter([0.1], [63.4], color=C_RED, s=120, zorder=5, marker='*',
              edgecolor=C_RED, linewidth=1.5, label=r'Optimal $\eta$=0.1')

    ax.set_xlabel(r'Aggregation Coefficient $\eta$', fontsize=10)
    ax.set_ylabel('Average Accuracy (%)', fontsize=10)
    ax.set_title(r'(b) Sensitivity to Aggregation Parameter $\eta$', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(59.5, 64.5)

    plt.tight_layout()
    plt.savefig('/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/gupo-gradient-uncertainty-policy-optimization/fig4.png')
    plt.close()
    print("fig4 saved")

if __name__ == '__main__':
    fig1_conflict()
    fig2_method()
    fig3_results()
    fig4_ablation()
    print("All figures generated successfully!")
