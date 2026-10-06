# -*- coding: utf-8 -*-
"""FrugalEvo 記事用の図を生成する"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SLUG = "frugalevo-cost-aware-program-evolution"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", SLUG)
os.makedirs(OUT, exist_ok=True)

JP = "Hiragino Kaku Gothic ProN W3"
JPB = "Hiragino Kaku Gothic ProN W6"
plt.rcParams["font.family"] = JP
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#5b6470"
plt.rcParams["axes.labelcolor"] = "#1f2933"
plt.rcParams["text.color"] = "#1f2933"
plt.rcParams["xtick.color"] = "#3e4c59"
plt.rcParams["ytick.color"] = "#3e4c59"

STRONG = "#c0392b"   # 高コストモデル
CHEAP = "#2e86c1"    # 低コストモデル
GRAY = "#8895a3"
ACC = "#16a085"
BG = "#ffffff"


def box(ax, x, y, w, h, text, fc, tc="#1f2933", fs=10.5, bold=False, lw=1.2, ec=None):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=ec if ec else fc, facecolor=fc, alpha=0.14, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontname=JPB if bold else JP, zorder=3, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="#5b6470", style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls, zorder=4,
        shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- fig1
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4),
                         gridspec_kw={"width_ratios": [1.35, 1.0]})
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.15, 9.6, "(a) FrugalEvo の探索ループ", fontsize=13, fontname=JPB,
        color="#1f2933")

# 1列目: cold start -> context -> explorer
box(ax, 1.35, 8.35, 2.3, 1.15, "初期プログラム\n$+$\nCold Start", CHEAP, bold=False)
box(ax, 4.2, 8.35, 2.3, 1.15, "Context Builder\n（課題分析・履歴統合）", "#7f8c8d")
box(ax, 7.1, 8.35, 2.6, 1.15, "Strategy Explorer\n高コスト LLM（探索）", STRONG, bold=True)
arrow(ax, 2.5, 8.35, 3.05, 8.35)
arrow(ax, 5.35, 8.35, 5.8, 8.35)

# explorer -> generator
box(ax, 7.1, 6.05, 2.6, 1.15, "Solution Generator\n低コスト LLM（実装・改善）", CHEAP, bold=True)
arrow(ax, 7.1, 7.78, 7.1, 6.62)
ax.text(7.25, 7.2, "$K$ 個の戦略を\n1 回の呼び出しで提案", fontsize=9.5,
        color="#5b6470", va="center", linespacing=1.4)

# generator -> evaluator
box(ax, 4.2, 6.05, 2.3, 1.15, "Evaluator\nスコア $s(x)$・フィードバック", "#7f8c8d")
arrow(ax, 5.7, 6.05, 5.35, 6.05)

# evaluator -> memory
box(ax, 1.35, 6.05, 2.3, 1.15, "Search Memory\n（island MAP-Elites）", ACC)
arrow(ax, 3.05, 6.05, 2.5, 6.05)

# loop back
arrow(ax, 1.35, 5.48, 1.35, 4.85, color=GRAY)
arrow(ax, 1.35, 4.85, 4.2, 4.85, color=GRAY)
arrow(ax, 4.2, 4.85, 4.2, 5.48, color=GRAY)
ax.text(2.78, 4.6, "予算 $B$ が尽きるまで反復", fontsize=9.5, ha="center", color="#5b6470")

# refinement sub-loop
ax.add_patch(FancyBboxPatch((5.5, 1.3), 4.5, 2.5,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            linewidth=1.1, edgecolor=CHEAP, facecolor=CHEAP,
                            alpha=0.06, zorder=1, linestyle="--"))
ax.text(7.75, 3.5, "改善ラウンド（最大 $M$ 回の試行）", fontsize=9.5, ha="center",
        color=CHEAP, fontname=JPB)
for i, (xx, lab) in enumerate(zip([6.15, 7.75, 9.35], ["試行 1", "試行 2", "試行 $M$"])):
    box(ax, xx, 2.7, 1.2, 0.68, lab, CHEAP, fs=9.5)
    if i < 2:
        arrow(ax, xx + 0.62, 2.7, xx + 0.98, 2.7, color=CHEAP, lw=1.1)
ax.text(7.75, 1.85, "成功したら即ラウンド終了 → incumbent 更新\n失敗のフィードバックは次の試行へ（親は固定）",
        fontsize=9, ha="center", color="#5b6470", linespacing=1.5)
arrow(ax, 7.1, 5.48, 7.75, 3.85, color=CHEAP, lw=1.1)

# ---- (b) キャッシュ効率プロンプト
ax = axes[1]
ax.set_xlim(0, 6.4); ax.set_ylim(0.2, 10.6); ax.axis("off")
ax.text(0.1, 10.35, "(b) キャッシュ効率プロンプト", fontsize=13, fontname=JPB)

layers = [
    ("固定セクション", "タスク指示・出力形式・課題分析", "#16a085", 1.55),
    ("準固定セクション", "island-best / elite プログラム", "#2e86c1", 1.35),
    ("変動セクション", "ランダムサンプル・incumbent", "#e67e22", 1.35),
    ("追記セクション", "直前の試行のフィードバック", "#c0392b", 1.1),
]
y = 9.0
for name, sub, col, h in layers:
    ax.add_patch(FancyBboxPatch((0.9, y - h), 4.7, h,
                                boxstyle="round,pad=0.012,rounding_size=0.04",
                                linewidth=1.3, edgecolor=col, facecolor=col,
                                alpha=0.15, zorder=2))
    ax.text(3.25, y - h / 2 + 0.22, name, ha="center", va="center", fontsize=10.5,
            fontname=JPB, color=col, zorder=3)
    ax.text(3.25, y - h / 2 - 0.28, sub, ha="center", va="center", fontsize=9,
            color="#5b6470", zorder=3)
    y -= h + 0.34

ax.annotate("", xy=(0.62, 1.3), xytext=(0.62, 9.0),
            arrowprops=dict(arrowstyle="-|>", color="#5b6470", lw=1.4))
ax.text(0.5, 5.2, "変化\nしにくい", fontsize=9.5, ha="center", va="center",
        color="#5b6470", rotation=0, linespacing=1.4)
ax.text(0.5, 2.2, "変化\nしやすい", fontsize=9.5, ha="center", va="center",
        color="#5b6470", linespacing=1.4)
ax.text(3.25, 0.75, "前方ほど prefix が共有される → KV キャッシュ再利用",
        fontsize=9.5, ha="center", color="#1f2933", fontname=JPB)

fig.suptitle("図1: FrugalEvo の全体像", fontsize=14, fontname=JPB, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(os.path.join(OUT, "fig1.png"), dpi=155, facecolor=BG)
plt.close(fig)


# ---------------------------------------------------------------- fig2
fig, ax = plt.subplots(figsize=(9.6, 5.4))
c = [0.0, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
frugal = [0.96, 2.20, 2.48, 2.58, 2.620, 2.631, 2.6350, 2.63599]
base = [0.96, 1.35, 1.85, 2.20, 2.42, 2.52, 2.58, 2.618]

ax.plot(c, frugal, marker="o", ms=5, lw=2.4, color=STRONG, label="FrugalEvo（BA-AUC が大きい）")
ax.fill_between(c, frugal, color=STRONG, alpha=0.13)
ax.plot(c, base, marker="s", ms=5, lw=2.4, color=GRAY, ls="--", label="既存手法（立ち上がりが遅い）")
ax.fill_between(c, base, color=GRAY, alpha=0.13)

ax.axhline(2.63599, color=STRONG, lw=0.9, alpha=0.45, ls=":")
ax.axhline(2.618, color=GRAY, lw=0.9, alpha=0.45, ls=":")
ax.axvline(1.0, color="#1f2933", lw=1.2, alpha=0.7)
ax.text(1.0, 1.02, " 予算 $B$", fontsize=11, color="#1f2933", va="bottom", fontname=JPB)

ax.annotate("この面積が BA-AUC\n（予算内でどれだけ早く\n良い解に到達したか）",
            xy=(0.3, 2.35), xytext=(0.06, 1.62), fontsize=10.5,
            fontname=JPB, color=STRONG,
            arrowprops=dict(arrowstyle="->", color=STRONG, lw=1.3))
ax.annotate("最終スコアは近くても\n道中の面積が違う", xy=(0.72, 2.30),
            xytext=(0.42, 1.62), fontsize=10.5, fontname=JPB, color="#5b6470",
            arrowprops=dict(arrowstyle="->", color="#5b6470", lw=1.2))

ax.set_xlabel("累積 LLM コスト（USD）", fontsize=11.5)
ax.set_ylabel("best-so-far スコア $q(c)$", fontsize=11.5)
ax.set_title("図2: Budget-Aware AUC（BA-AUC）の概念", fontsize=13.5, fontname=JPB, pad=14)
ax.set_xlim(0, 1.12); ax.set_ylim(0.9, 2.78)
ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.95)
ax.grid(alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2.png"), dpi=155, facecolor=BG)
plt.close(fig)


# ---------------------------------------------------------------- fig3
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

ax = axes[0]
pts = [
    ("FrugalEvo\n(GLM-5.3 + Flash)", 0.55, 2.635990, STRONG, "o"),
    ("FrugalEvo\n(GPT-5.6 Terra + Luna)", 1.68, 2.635996, STRONG, "o"),
    ("SwarmResearch", 50.0, 2.635996, "#7f8c8d", "s"),
    ("CORAL", 50.0, 2.635985, "#7f8c8d", "s"),
    ("AlphaEvolve", 50.0, 2.635000, "#b2bec3", "^"),
]
for name, cost, score, col, mk in pts:
    ax.scatter([cost], [score], s=130 if mk == "o" else 95, marker=mk,
               color=col, edgecolor="white", linewidth=1.2, zorder=3)
ann = {
    0: ("FrugalEvo GLM-5.3 + Flash\n$0.55", (13, -26), "left", STRONG),
    1: ("FrugalEvo GPT-5.6 Terra + Luna\n$1.68", (12, 28), "left", STRONG),
    2: ("SwarmResearch\n~$50", (-13, 14), "right", "#5b6470"),
    3: ("CORAL\n~$50", (-13, -20), "right", "#5b6470"),
    4: ("AlphaEvolve", (12, -6), "left", "#7f8c8d"),
}
for i, (name, cost, score, col, mk) in enumerate(pts):
    lab, (ox, oy), halign, lcol = ann[i]
    ax.annotate(lab, (cost, score), textcoords="offset points",
                xytext=(ox, oy), fontsize=9.5, color=lcol,
                va="center", ha=halign, fontname=JPB if mk == "o" else JP,
                linespacing=1.4)
ax.axhspan(2.6359, 2.63605, color=STRONG, alpha=0.07, zorder=0)
ax.set_xscale("log")
ax.set_xlim(0.32, 160)
ax.set_ylim(2.6344, 2.6366)
ax.set_xticks([0.5, 1, 5, 50])
ax.set_xticklabels(["$0.5", "$1", "$5", "$50"])
ax.set_xlabel("Circle Packing 1 回あたりのコスト（USD, 対数軸）", fontsize=11)
ax.set_ylabel("半径の和（大きいほど良い）", fontsize=11.5)
ax.set_title("(a) 同品質を 1/30 以下のコストで", fontsize=12.5, fontname=JPB)
ax.grid(alpha=0.28, ls=":", which="both")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
names = ["FrugalEvo", "OpenEvolve", "AdaEvolve", "ShinkaEvolve", "EvoX"]
vals = [1924.9, 1887.5, 1881.9, 1856.0, 1842.0]
cols = [STRONG] + ["#aeb9c4"] * 4
bars = ax.barh(range(len(names)), vals, color=cols, height=0.62,
               edgecolor="white", linewidth=1.1)
for i, v in enumerate(vals):
    ax.text(v + 6, i, f"{v:.1f}", va="center", fontsize=10.5,
            color="#1f2933", fontname=JPB if i == 0 else JP)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(1780, 1975)
ax.set_xlabel("ALE-Bench-Lite 10 課題の平均 private スコア", fontsize=11)
ax.set_title("(b) アルゴリズム最適化（予算 $1）", fontsize=12.5, fontname=JPB)
ax.grid(axis="x", alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("図3: コストと性能のトレードオフ", fontsize=14, fontname=JPB, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "fig3.png"), dpi=155, facecolor=BG)
plt.close(fig)


# ---------------------------------------------------------------- fig4
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
labels = ["FrugalEvo", "Terra+Terra", "Luna+Luna", "cold start なし", "逐次FB なし"]
short = ["FrugalEvo", "強モデルのみ", "弱モデルのみ", "cold start\nなし", "逐次FB\nなし"]

data = [
    ("(a) Circle Packing（平均, ↑）", [2.635989, 2.632278, 2.634565, 2.632417, 2.633659],
     2.6315, 2.6369, "{:.6f}"),
    ("(b) Signal Processing（平均, ↑）", [0.77791, 0.68534, 0.74865, 0.75576, 0.76072],
     0.0, 0.90, "{:.5f}"),
]
for ax, (title, vals, lo, hi, fmt) in zip(axes, data):
    cols = [STRONG] + ["#aeb9c4"] * 4
    ax.bar(range(5), vals, color=cols, width=0.62, edgecolor="white", linewidth=1.1)
    for i, v in enumerate(vals):
        ax.text(i, v + (hi - lo) * 0.018, fmt.format(v), ha="center", fontsize=9.8,
                color="#1f2933", fontname=JPB if i == 0 else JP)
    ax.set_xticks(range(5))
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=12.5, fontname=JPB)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("図4: アブレーション（GPT-5.6 Terra / Luna, 同一予算）",
             fontsize=14, fontname=JPB, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "fig4.png"), dpi=155, facecolor=BG)
plt.close(fig)

print("saved to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f, os.path.getsize(os.path.join(OUT, f)))
