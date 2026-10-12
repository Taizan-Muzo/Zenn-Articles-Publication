# -*- coding: utf-8 -*-
"""BiToK-SD (arXiv:2610.07247) 記事用の図を生成する"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SLUG = "bitok-sd-bilevel-topk-self-distillation"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "images", SLUG)
os.makedirs(OUT, exist_ok=True)

JP = "Hiragino Sans"
JPB = "Hiragino Sans"
plt.rcParams["font.family"] = JP
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#5b6470"
plt.rcParams["text.color"] = "#1f2933"
plt.rcParams["xtick.color"] = "#3e4c59"
plt.rcParams["ytick.color"] = "#3e4c59"

BLUE = "#2e86c1"
TEAL = "#16a085"
RED = "#c0392b"
ORANGE = "#d68910"
PURPLE = "#7d3c98"
GRAY = "#8895a3"
DARK = "#34495e"


def box(ax, x, y, w, h, text, fc, fs=10.5, bold=False, alpha=0.14, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=fc, facecolor=fc, alpha=alpha, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontname=JP, fontweight=("bold" if bold else "normal"), zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#5b6470", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=4, shrinkA=1, shrinkB=1,
        connectionstyle=f"arc3,rad={rad}"))


# ============================================================ fig1
# 全体構造：双レベル最適化としての自己蒸留
fig, ax = plt.subplots(figsize=(13.6, 6.9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.6)
ax.axis("off")

ax.text(0.15, 7.25, "BiToK-SD の全体構造──下位問題がしきい値を選び、上位問題が学生を更新する",
        fontsize=13.5, fontname=JP, fontweight="bold")

# 上位レベルの帯
ax.add_patch(FancyBboxPatch(
    (0.35, 4.15), 13.3, 2.85,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1.6, edgecolor=BLUE, facecolor=BLUE, alpha=0.06, zorder=1))
ax.text(0.55, 6.72, "上位問題（学生 θ の更新）", fontsize=11.5, fontname=JP, fontweight="bold", color=BLUE)

# 下位レベルの帯
ax.add_patch(FancyBboxPatch(
    (0.35, 0.35), 13.3, 3.2,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1.6, edgecolor=PURPLE, facecolor=PURPLE, alpha=0.06, zorder=1))
ax.text(0.55, 3.28, "下位問題（適応しきい値 ρ の最適化）", fontsize=11.5, fontname=JP, fontweight="bold", color=PURPLE)

# 上位レベル内の箱
box(ax, 2.0, 5.85, 2.7, 1.15, "入力問題 x\n＋ 自己教師用\n参照解（feedback）", DARK, fs=9.5)
arrow(ax, 3.35, 5.85, 3.95, 5.85)
box(ax, 5.6, 5.85, 2.5, 1.15, "学生 policy θ\nQwen3-1.7/4/8B\non-policy rollout", BLUE, fs=9.5, bold=True)
arrow(ax, 6.85, 5.85, 7.5, 5.85)
box(ax, 8.9, 5.85, 2.2, 1.15, "推論軌跡\n（CoT tokens）\nℓ=KL(pT‖pS) が定義される位置", TEAL, fs=9.5)

box(ax, 12.35, 5.85, 2.15, 1.15, "重み付き蒸留損失\nL=Σ z̃·ℓ / Σz̃\n→ θ を更新", BLUE, fs=9.5, bold=True)
arrow(ax, 10.0, 5.85, 11.25, 5.85)
# θ更新のループ（上部を通して）
arrow(ax, 12.35, 6.55, 5.6, 6.55, color=BLUE, lw=1.6, rad=0.18)
ax.text(8.9, 7.05, "θ の更新 → 次ステップの rollout 分布も変化（u と ρ は毎ステップ再計算）",
        fontsize=9.5, fontname=JP, color=BLUE, ha="center")

# 下位レベル内の箱
box(ax, 2.0, 1.95, 3.3, 1.15, "ユーティリティスコア（Soft-OR）\nu = ĥ + d̂ − ĥ·d̂\n（学生エントロピー ĥ ＋ 逆KL d̂）", PURPLE, fs=9.5)
arrow(ax, 3.65, 1.95, 4.3, 1.95)
box(ax, 6.15, 1.95, 3.4, 1.15, "微分可能なしきい値緩和\nmin Ψ(ρ; u)\nsoftplus項＋二次正則\n（モメンタム付き勾配で近似解）", PURPLE, fs=9.3, bold=True)
arrow(ax, 7.85, 1.95, 8.55, 1.95)
box(ax, 10.4, 1.95, 2.5, 1.15, "ソフトマスク（連続重み）\nz̃ = σ((u − ρ)/τm)\n上位50%付近に集中", TEAL, fs=9.5, bold=True)

# 上下の接続
arrow(ax, 10.4, 2.55, 12.35, 5.25, color=TEAL, lw=1.6, rad=0.25)
ax.text(13.45, 3.55, "連続重み z̃ が\n上位損失へ", fontsize=9.5, color=TEAL, fontname=JP, ha="center")
ax.text(3.7, 3.25, "学生 pS / 自己教師 pT の\n分布から u を毎ステップ計算",
        fontsize=9, fontname=JP, color="#5b6470", ha="center")
arrow(ax, 5.6, 5.3, 3.65, 2.55, color="#5b6470", lw=1.2, rad=0.12)

ax.text(0.15, 0.02, "BiToK-SD: bilevel最適化──下位問題が適応しきい値 ρ を決め、上位問題が選ばれた位置の重み付き蒸留で θ を更新する。しきい値は detach して扱うが、u と ρ は学生の進化に合わせて毎ステップ再計算される。",
        fontsize=9, fontname=JP, color="#5b6470")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1.png"), dpi=160, facecolor="white")
plt.close(fig)
print("fig1 done")

# ============================================================ fig2
# (a) ハードTop-K vs ソフトマスク  (b) しきい値ρの訓練ダイナミクス
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

ax = axes[0]
np.random.seed(7)
N = 48
u_sorted = np.sort(np.random.beta(1.6, 9.0, N) * 0.24 + 0.005)  # utilityの例（正規化後を想定）
rho = np.median(u_sorted)
tau_m = 0.035
soft = 1.0 / (1.0 + np.exp(-(u_sorted - rho) / tau_m))
hard = (u_sorted >= rho).astype(float)

x = np.arange(N)
ax.bar(x, u_sorted, color=GRAY, alpha=0.55, width=0.8, label="トークン別ユーティリティ $u_t$（例）", zorder=2)
ax.axvline(N * 0.5 - 0.5, color=PURPLE, lw=1.6, ls="--", zorder=3)
ax.text(N * 0.5 - 1.0, 0.235, "適応しきい値 $\\hat\\rho$", color=PURPLE, fontsize=10.5, fontname=JP, fontweight="bold", ha="right")

ax.plot(x, soft, color=TEAL, lw=2.6, marker="o", ms=3.5, zorder=5, label="BiToK-SD: ソフトマスク $\\tilde z$（連続重み）")
ax.plot(x, hard * 0.28 + 0.012, color=RED, lw=2.2, ls=":", zorder=4, label="ハードTop-K（0/1の等重み）")

ax.set_xlabel("ユーティリティ順に並べたトークン位置（ミニバッチ内）", fontname=JP)
ax.set_ylabel("ユーティリティ / マスク値", fontname=JP)
ax.set_ylim(0, 0.3)
ax.set_title("(a) ハード選択は0/1の等重み、ソフトマスクは連続的な強さを持つ", fontname=JP, fontweight="bold", fontsize=12)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax.annotate("境界付近の「効きやすさ」\nが連続に変わる", xy=(N * 0.5 + 1.5, 0.52), xytext=(N * 0.68, 0.5),
            fontsize=9.5, fontname=JP, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))

ax = axes[1]
steps = np.linspace(0, 200, 41)
# 1.7B: 0.061 → 上昇 → 0.063 (範囲 0.044-0.105)
r17 = 0.061 + 0.034 * np.sin(np.pi * steps / 260.0) + 0.004 * np.sin(steps / 14.0)
r17[0] = 0.061
r17[-1] = 0.063
# 4B: 0.030 → 0.102 (範囲 0.029-0.151, 後半に山)
r4 = 0.030 + 0.072 * (steps / 200.0) ** 1.1 + 0.045 * np.exp(-((steps - 168) / 26.0) ** 2)
r4[0] = 0.030
r4[-1] = 0.102
# 8B: 0.069 → 0.159 (範囲 0.051-0.217, 後半に山)
r8 = 0.069 + 0.09 * (steps / 200.0) ** 1.0 + 0.055 * np.exp(-((steps - 172) / 24.0) ** 2)
r8[0] = 0.069
r8[-1] = 0.159

ax.plot(steps, r8, color=RED, lw=2.4, label="Qwen3-8B（0.069 → 0.159）")
ax.plot(steps, r4, color=ORANGE, lw=2.4, label="Qwen3-4B（0.030 → 0.102）")
ax.plot(steps, r17, color=BLUE, lw=2.4, label="Qwen3-1.7B（0.061 → 0.063）")
for ys, col in [(0.061, BLUE), (0.030, ORANGE), (0.069, RED)]:
    ax.plot([0], [ys], "o", color=col, ms=6, zorder=5)
for ye, col in [(0.063, BLUE), (0.102, ORANGE), (0.159, RED)]:
    ax.plot([200], [ye], "s", color=col, ms=6, zorder=5)

ax.set_xlabel("訓練ステップ", fontname=JP)
ax.set_ylabel("緩和しきい値 ρ", fontname=JP)
ax.set_xlim(-6, 210)
ax.set_ylim(0.0, 0.23)
ax.set_title("(b) しきい値 ρ は訓練中に適応（開始・終了は論文報告値）", fontname=JP, fontweight="bold", fontsize=12)
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax.text(105, 0.008, "曲線は論文の報告値（開始・終了・変動範囲）に基づく概略", fontsize=8.5,
        fontname=JP, color="#5b6470", ha="center")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2.png"), dpi=160, facecolor="white")
plt.close(fig)
print("fig2 done")

# ============================================================ fig3
# 主結果：3スケール × 6手法 + OPSDとの差分
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.75, 1.0]})

ax = axes[0]
methods = ["Base", "SFT", "GRPO", "OPSD", "TIP", "BiToK-SD"]
data = {
    "1.7B": [35.64, 24.95, 36.74, 40.85, 35.71, 41.48],
    "4B":   [60.87, 46.69, 61.14, 61.49, 59.46, 62.07],
    "8B":   [63.42, 60.95, 62.99, 64.64, 62.15, 64.85],
}
colors = [GRAY, "#aab4be", ORANGE, BLUE, RED, TEAL]
nb = len(methods)
group_w = 0.82
bw = group_w / nb
xs = np.arange(3)
for gi, (scale, vals) in enumerate(data.items()):
    for mi, v in enumerate(vals):
        xx = xs[gi] - group_w / 2 + bw * (mi + 0.5)
        is_top = (mi == nb - 1)
        ax.bar(xx, v, width=bw * 0.92, color=colors[mi],
               alpha=1.0 if is_top else 0.88,
               edgecolor=DARK if is_top else "none",
               linewidth=1.1 if is_top else 0, zorder=3)
        ax.text(xx, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.8,
                fontname=JP, fontweight=("bold" if is_top else "normal"), rotation=90,
                color=DARK if is_top else "#3e4c59")
ax.set_xticks(xs)
ax.set_xticklabels(["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"], fontname=JP, fontweight="bold")
ax.set_ylabel("平均精度 Avg@64（%）", fontname=JP)
ax.set_ylim(0, 82)
ax.set_title("(a) AIME24 / AIME25 / HMMT25 の平均（thinkingモード, 200ステップ最終値）", fontname=JP, fontweight="bold", fontsize=11.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, methods, loc="upper left", fontsize=9.5, framealpha=0.95, ncol=2)

ax = axes[1]
scales = ["1.7B", "4B", "8B"]
tip_d = [-5.14, -2.03, -2.49]
bitok_d = [0.63, 0.58, 0.21]
x2 = np.arange(3)
w2 = 0.34
ax.bar(x2 - w2 / 2 - 0.02, tip_d, width=w2, color=RED, alpha=0.9, label="TIP（ハード選択）", zorder=3)
ax.bar(x2 + w2 / 2 + 0.02, bitok_d, width=w2, color=TEAL, zorder=3, label="BiToK-SD（学習した軟選択）")
for i, v in enumerate(tip_d):
    ax.text(x2[i] - w2 / 2 - 0.02, v - 0.16, f"{v:+.2f}", ha="center", va="top", fontsize=9.5, color=RED, fontname=JP, fontweight="bold")
for i, v in enumerate(bitok_d):
    ax.text(x2[i] + w2 / 2 + 0.02, v + 0.09, f"+{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=TEAL, fontname=JP, fontweight="bold")
ax.axhline(0, color=DARK, lw=1.2)
ax.set_xticks(x2)
ax.set_xticklabels(scales, fontname=JP, fontweight="bold")
ax.set_ylabel("OPSDに対する平均差分（点）", fontname=JP)
ax.set_ylim(-6.2, 1.8)
ax.set_title("(b) 同じ信号でも、選び方で正負が分かれる", fontname=JP, fontweight="bold", fontsize=11.5)
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax.text(1.0, 1.35, "TIPはハードマスク＋逆KLで全9セル劣化\nBiToK-SDは双レベル軟選択で全スケール改善",
        ha="center", fontsize=9.5, fontname=JP, color=DARK)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3.png"), dpi=160, facecolor="white")
plt.close(fig)
print("fig3 done")

# ============================================================ fig4
# (a) Top-K比アブレーション (b) スコア別 (c) 反射マーカー重み比 (d) トークン数
fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4))

ax = axes[0][0]
ks = [20, 30, 40, 50, 60, 70]
avg = [53.59, 53.70, 54.22, 55.99, 54.95, 54.53]
pass64 = [83.33, 86.67, 80.00, 90.00, 76.67, 86.67]
ax.plot(ks, avg, "o-", color=BLUE, lw=2.4, ms=8, label="Avg@64", zorder=4)
ax.axhline(55.10, color=GRAY, ls="--", lw=1.6, label="Uniform（＝OPSD）55.10")
for k, v in zip(ks, avg):
    ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontname=JP)
ax2 = ax.twinx()
ax2.plot(ks, pass64, "s--", color=ORANGE, lw=1.8, ms=6, alpha=0.9, label="Pass@64（右軸）")
ax2.set_ylabel("Pass@64（%）", fontname=JP, color=ORANGE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax2.set_ylim(70, 96)
ax.set_xlabel("Top-K 比率 k（Qwen3-1.7B, AIME24）", fontname=JP)
ax.set_ylabel("Avg@64（%）", fontname=JP)
ax.set_ylim(52.5, 57.0)
ax.set_title("(a) 比率 k は逆U字──中間の50%でピーク", fontname=JP, fontweight="bold", fontsize=11.5)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9, framealpha=0.95)

ax = axes[0][1]
names = ["Uniform\n(=OPSD)", "Student\nEnt.", "Teacher\nEnt.", "KL\n(順方向)", "RKL\n(逆方向)", "Soft-OR\n(既定)"]
vals = [55.10, 54.79, 54.69, 55.78, 53.85, 55.99]
cols = [GRAY, "#aab4be", "#aab4be", BLUE, RED, TEAL]
bars = ax.bar(np.arange(6), vals, color=cols, width=0.68, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=9.5,
            fontname=JP, fontweight=("bold" if i in (5,) else "normal"), color=DARK if i == 5 else "#3e4c59")
ax.set_xticks(np.arange(6))
ax.set_xticklabels(names, fontsize=8.8, fontname=JP)
ax.set_ylim(53.0, 56.6)
ax.set_ylabel("Avg@64（%）", fontname=JP)
ax.set_title("(b) ユーティリティスコアの選択（k=50%固定）", fontname=JP, fontweight="bold", fontsize=11.5)
ax.text(2.5, 53.3, "順KL系が有効・逆KLとエントロピー単体はUniformを下回る", ha="center", fontsize=9, fontname=JP, color="#5b6470")

ax = axes[1][0]
cats = ["反射マーカー位置\n（wait / hmm / maybe …）", "その他の位置"]
ratio = [1.63, 1.00]
bars = ax.bar([0, 1], ratio, color=[TEAL, GRAY], width=0.5, zorder=3)
for i, v in enumerate(ratio):
    ax.text(i, v + 0.035, f"×{v:.2f}", ha="center", fontsize=12, fontname=JP, fontweight="bold", color=DARK)
ax.set_xticks([0, 1])
ax.set_xticklabels(cats, fontsize=9.5, fontname=JP)
ax.set_ylim(0, 2.0)
ax.set_ylabel("平均ソフトマスク重み（比）", fontname=JP)
ax.set_title("(c) セレクタは反射に関わる位置を1.63倍重く蒸留", fontname=JP, fontweight="bold", fontsize=11.5)
ax.text(0.5, 1.85, "訓練全体を通した平均比。OPSDは全位置等重み・TIPは二値のため\nこのような「強度の配分」は原理的に生まれない",
        ha="center", fontsize=9, fontname=JP, color="#5b6470")

ax = axes[1][1]
sc = ["1.7B", "4B", "8B"]
diff = [-3.7, -1.6, -4.8]
ax.bar(np.arange(3), diff, color=BLUE, width=0.5, zorder=3)
for i, v in enumerate(diff):
    ax.text(i, v - 0.22, f"{v:+.1f}%", ha="center", fontsize=11, fontname=JP, fontweight="bold", color=BLUE)
ax.axhline(0, color=DARK, lw=1.2)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(sc, fontname=JP, fontweight="bold")
ax.set_ylabel("OPSDに対する生成トークン数の変化（%）", fontname=JP)
ax.set_ylim(-6.2, 0.8)
ax.set_title("(d) 反射マーカーは増えるのに長さはむしろ減る（AIME24）", fontname=JP, fontweight="bold", fontsize=11.5)
ax.text(1.0, 0.45, "4-gram反復率と打ち切り率も全スケールで低下──「長文化」ではない",
        ha="center", fontsize=9, fontname=JP, color="#5b6470")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4.png"), dpi=160, facecolor="white")
plt.close(fig)
print("fig4 done")

print("ALL DONE ->", OUT)
