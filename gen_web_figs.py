# -*- coding: utf-8 -*-
"""World Embedding Benchmark 記事用の図を生成する"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SLUG = "world-embedding-benchmark-physical-fidelity"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "images", SLUG)
os.makedirs(OUT, exist_ok=True)

JP = "Hiragino Kaku Gothic ProN W3"
JPB = "Hiragino Kaku Gothic ProN W6"
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
            fontname=JPB if bold else JP, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#5b6470", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=4, shrinkA=1, shrinkB=1,
        connectionstyle=f"arc3,rad={rad}"))


# ============================================================ fig1
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2),
                         gridspec_kw={"width_ratios": [1.45, 1.0]})
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
ax.text(0.1, 8.3, "(a) ベンチマーク構築パイプライン", fontsize=13, fontname=JPB)

box(ax, 1.55, 7.35, 2.6, 1.1, "物理ソルバー\nOpenFOAM / DOLFINx\nChrono / HCIPy / Meep", DARK, fs=9, bold=True)
arrow(ax, 2.9, 7.35, 3.4, 7.35)
box(ax, 4.8, 7.35, 2.3, 1.1, "場・軌跡\nおよび物理量", BLUE, fs=10)
arrow(ax, 5.95, 7.35, 6.45, 7.35)
box(ax, 8.05, 7.35, 2.9, 1.1, "ParaView → FFmpeg\nPNG フレーム → MP4", TEAL, fs=9.5)
ax.text(4.8, 6.55, "Sim2Video", fontsize=9.5, ha="center", color="#5b6470")

box(ax, 1.15, 5.4, 1.9, 0.95, "80 家族\n× 100 事例", DARK, fs=10, bold=True)
arrow(ax, 1.15, 6.78, 1.15, 5.9, color=DARK)

annots = [
    ("NL クエリ", "テンプレート\n＋スロット値", PURPLE),
    ("構成メタデータ", "材料・幾何\n境界条件", GRAY),
    ("物理応答", "Re 数・応力\n塑性ひずみ・波長", ORANGE),
]
for i, (t, s, col) in enumerate(annots):
    box(ax, 3.9 + i * 2.3, 5.4, 2.1, 1.35, f"{t}\n{s}", col, fs=8.6)
    arrow(ax, 3.9 + i * 2.3, 4.72, 3.9 + i * 2.3, 4.15, color=GRAY, lw=1.1)

box(ax, 6.0, 3.55, 6.0, 1.05, "1 事例 = 動画 ＋ 注釈 ＋ クエリ", DARK, fs=10.5,
    bold=True, alpha=0.10)
arrow(ax, 1.15, 4.9, 1.15, 3.55, color=DARK)
arrow(ax, 1.15, 3.55, 2.85, 3.55, color=DARK)
arrow(ax, 8.05, 6.78, 8.05, 5.9, color=TEAL, lw=1.2)
arrow(ax, 6.0, 3.0, 6.0, 2.35)
box(ax, 6.4, 1.7, 3.4, 1.05, "World Embedding Benchmark\n8,000 事例", RED, fs=10.5,
    bold=True, alpha=0.13)

ax = axes[1]
ax.set_xlim(-0.4, 3.4); ax.set_ylim(-0.4, 5.2); ax.axis("off")
ax.text(-0.4, 5.0, "(b) 4 分野・80 家族の内訳", fontsize=13, fontname=JPB)

branches = [
    ("Fluid Mechanics\n流体力学", 7, 700, "OpenFOAM", "#2980b9"),
    ("Solid Mechanics\n固体力学", 27, 2700, "DOLFINx (FEniCSx)", "#16a085"),
    ("Dynamics\n动力学", 19, 1900, "Chrono + 解析/ODE", "#d68910"),
    ("Optics & EM\n光学・電磁気学", 27, 2700, "解析 + HCIPy + Meep", "#8e44ad"),
]
y = 4.2
for name, fam, case, solver, col in branches:
    ax.barh([y], [fam], color=col, height=0.55, alpha=0.85)
    ax.text(fam + 0.5, y + 0.16, f"{fam} 家族 / {case:,} 事例", fontsize=10,
            va="center", color=col, fontname=JPB)
    ax.text(fam + 0.5, y - 0.20, solver, fontsize=8.8, va="center", color="#5b6470")
    ax.text(-0.3, y, name.replace("\n", " "), fontsize=10, va="center", ha="right",
            fontname=JPB)
    y -= 1.1
ax.set_xlim(-4.6, 42)
ax.axvline(0, color="#5b6470", lw=1)
ax.text(20, 0.1, "合計 80 家族 / 8,000 事例", fontsize=10.5, ha="center",
        color="#1f2933", fontname=JPB)

fig.suptitle("図1: World Embedding Benchmark の構成", fontsize=14, fontname=JPB, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(os.path.join(OUT, "fig1.png"), dpi=155, facecolor="#ffffff")
plt.close(fig)


# ============================================================ fig2
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9))
cols = [BLUE, ORANGE, PURPLE]

# (a) retrieval
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
ax.text(0.15, 8.3, "(a) Text-Video Retrieval", fontsize=12.5, fontname=JPB)
ax.text(0.15, 7.75, "分野内双方向検索・Recall@10", fontsize=9.5, color="#5b6470")
box(ax, 2.1, 6.3, 2.4, 1.1, "クエリ\n（自然言語）", cols[0], fs=10)
box(ax, 6.9, 6.3, 2.4, 1.1, "動画\n（8,000 本中）", cols[0], fs=10)
arrow(ax, 3.35, 6.45, 5.65, 6.45, color=cols[0], rad=-0.25)
arrow(ax, 5.65, 6.15, 3.35, 6.15, color=cols[0], rad=0.25)
ax.text(4.5, 6.85, "埋め込み空間で照合", fontsize=9.5, ha="center", color=cols[0])
box(ax, 4.5, 4.5, 4.4, 1.1, "意味的整合ではなく\n数値条件まで一致するか", DARK, fs=9.5)
box(ax, 4.5, 2.7, 4.6, 1.35, "タスク趣旨: 物理状態と\n物理状態の整合（数値含む）", cols[0], fs=10, bold=True)
arrow(ax, 4.5, 3.95, 4.5, 3.4)
ax.text(4.5, 1.3, "評価軸: cross-modal physical alignment\n（物理的アラインメント）", fontsize=10,
        ha="center", color=cols[0], fontname=JPB, linespacing=1.5)

# (b) regression
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
ax.text(0.15, 8.3, "(b) Property Regression", fontsize=12.5, fontname=JPB)
ax.text(0.15, 7.75, "10 課題の macro nRMSE（↓）", fontsize=9.5, color="#5b6470")
box(ax, 2.0, 6.3, 2.3, 1.15, "凍結動画\n埋め込み", cols[1], fs=10)
arrow(ax, 3.2, 6.3, 4.35, 6.3, color=cols[1])
box(ax, 5.8, 6.3, 2.5, 1.15, "線形プローブ\n（学習は probe のみ）", cols[1], fs=9.5)
arrow(ax, 7.1, 6.3, 8.0, 6.3, color=cols[1])
box(ax, 9.0, 6.3, 1.6, 1.15, "ŷ", cols[1], fs=11, bold=True)
box(ax, 5.8, 4.4, 6.6, 1.5, "予測対象の例\n重力 g / レイノルズ数 Re\n波長 λ / 弾性率 E",
    cols[1], fs=9.8)
arrow(ax, 5.8, 5.5, 5.8, 5.15)
arrow(ax, 9.0, 5.65, 9.0, 5.15, color=cols[1])
box(ax, 5.0, 2.7, 4.8, 1.35, "定量物理量がどこまで\n取り出せるかを測る", cols[1], fs=10, bold=True)
arrow(ax, 5.0, 3.4, 5.0, 3.65)
ax.text(5.0, 1.3, "評価軸: quantitative recoverability\n（定量情報の復元可能性）", fontsize=10,
        ha="center", color=cols[1], fontname=JPB, linespacing=1.5)

# (c) pair classification
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
ax.text(0.15, 8.3, "(c) Pair Classification", fontsize=12.5, fontname=JPB)
ax.text(0.15, 7.75, "二択の正解率（chance = 50%）", fontsize=9.5, color="#5b6470")
box(ax, 3.4, 6.7, 2.6, 0.95, "anchor 動画", DARK, fs=10, bold=True)
box(ax, 1.5, 5.1, 2.4, 1.05, "正解記述\n（同一事例）", TEAL, fs=9.5)
box(ax, 5.1, 5.1, 2.7, 1.05, "同家族負例\n（パラメータのみ違う）", RED, fs=9.3)
box(ax, 8.2, 5.1, 2.6, 1.05, "異家族負例\n（別の物理系）", GRAY, fs=9.3)
arrow(ax, 2.9, 6.35, 2.1, 5.62, color=TEAL, lw=1.2)
arrow(ax, 3.9, 6.35, 4.6, 5.62, color=RED, lw=1.2)
arrow(ax, 4.0, 6.35, 7.6, 5.62, color=GRAY, lw=1.2)
ax.text(5.1, 3.9, "within-family", fontsize=10.5, ha="center", color=RED, fontname=JPB)
ax.text(8.2, 3.9, "cross-family", fontsize=10.5, ha="center", color="#5b6470", fontname=JPB)
box(ax, 5.1, 2.7, 4.6, 1.15, "数値・パラメータ差への\n細やかな感度を問う", RED, fs=9.8)
box(ax, 5.1, 1.35, 5.4, 0.85, "粗い系の見分け（こちらは容易）", GRAY, fs=9.8)

fig.suptitle("図2: 3 つのタスクと評価軸", fontsize=14, fontname=JPB, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig2.png"), dpi=155, facecolor="#ffffff")
plt.close(fig)


# ============================================================ fig3
models = ["Video-\nCLIP-XL", "Qwen3-VL\nEmb-2B", "Qwen3-VL\nEmb-8B",
          "Omni-Embed\nNemotron-3B", "LCO-Emb\n-3B", "LCO-Emb\n-3B-2605",
          "LCO-Emb\n-7B"]
short = ["VideoCLIP\n-XL", "Qwen3VL\n-2B", "Qwen3VL\n-8B",
         "Omni\n-3B", "LCO\n-3B", "LCO\n-3B-2605", "LCO\n-7B"]
fluid = [3.3, 3.9, 6.6, 3.1, 3.9, 4.6, 4.9]
solid = [1.4, 1.6, 1.9, 1.4, 1.5, 1.5, 1.6]
dyn = [2.3, 7.3, 7.4, 3.1, 5.1, 4.5, 5.7]
opt = [0.6, 2.3, 3.1, 0.7, 0.7, 1.2, 1.9]

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

ax = axes[0]
x = np.arange(len(models)); w = 0.2
for i, (vals, lab, col) in enumerate(zip(
        [fluid, solid, dyn, opt],
        ["Fluid", "Solid", "Dynamics", "Optics & EM"],
        ["#2980b9", "#16a085", "#d68910", "#8e44ad"])):
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col,
           edgecolor="white", linewidth=0.8)
ax.axhline(0.37, color=RED, ls="--", lw=1.2)
ax.text(-0.42, 0.62, "ランダム ≈ 0.4", fontsize=9, color=RED, ha="left")
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.8)
ax.set_ylabel("Recall@10（%）", fontsize=11)
ax.set_title("(a) 検索はほぼ全滅に近い", fontsize=12.5, fontname=JPB)
ax.legend(fontsize=9.5, ncol=2, framealpha=0.95)
ax.set_ylim(0, 8.6)
ax.grid(axis="y", alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
emb_w = [48.1, 51.5, 53.1, 52.5, 51.5, 50.5, 46.8]
emb_c = [66.1, 80.8, 86.2, 71.2, 68.5, 72.9, 76.4]
mllm_w = [61.3, 58.8, 60.2, 59.0]
mllm_c = [92.0, 91.2, 91.9, 90.9]
ax.scatter(emb_w, emb_c, s=110, color=GRAY, edgecolor="white", linewidth=1.2,
           label="埋め込みモデル（類似度比較）", zorder=3)
ax.scatter(mllm_w, mllm_c, s=130, color=RED, marker="^", edgecolor="white",
           linewidth=1.2, label="MLLM（直接プロンプト）", zorder=3)
# JEPA is not applicable; keep focus
ax.axvline(50, color=RED, ls="--", lw=1.2)
ax.fill_betweenx([60, 100], 50, 65, color=RED, alpha=0.05, zorder=0)
ax.text(50.4, 96, "chance = 50%", fontsize=9.5, color=RED)
ax.annotate("粗い見分けはできるが\n細かな数値差はほぼランダム",
            xy=(51.5, 68.5), xytext=(40, 82), fontsize=10, color="#5b6470",
            fontname=JPB, arrowprops=dict(arrowstyle="->", color="#5b6470", lw=1.2),
            linespacing=1.5)
ax.annotate("生成モデルは\n同条件を言い当てられる", xy=(61.3, 92.0), xytext=(61.5, 82.5),
            fontsize=10, color=RED, fontname=JPB,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2), linespacing=1.5)
ax.set_xlabel("within-family 正解率（%）", fontsize=11)
ax.set_ylabel("cross-family 正解率（%）", fontsize=11)
ax.set_xlim(42, 70); ax.set_ylim(60, 98)
ax.set_title("(b) 能力はあるのに類似度に出ない", fontsize=12.5, fontname=JPB)
ax.legend(fontsize=9.8, loc="lower right", framealpha=0.95)
ax.grid(alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("図3: 事前学習済みモデルの素の実力", fontsize=14, fontname=JPB, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(os.path.join(OUT, "fig3.png"), dpi=155, facecolor="#ffffff")
plt.close(fig)


# ============================================================ fig4
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

ax = axes[0]
labels = ["Fluid", "Solid", "Dynamics", "Optics\n& EM", "Pair\nwithin", "Pair\ncross"]
before = [4.9, 1.6, 5.7, 1.9, 46.8, 76.4]
after = [17.9, 9.4, 21.0, 9.4, 59.6, 98.4]
x = np.arange(len(labels)); w = 0.36
ax.bar(x - w / 2, before, w, label="LCO-Emb-7B（素）", color=GRAY, edgecolor="white")
ax.bar(x + w / 2, after, w, label="+ Physics Adaptation", color=RED, edgecolor="white")
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - w / 2, b + 1.2, f"{b}", ha="center", fontsize=8.6, color="#5b6470")
    ax.text(i + w / 2, a + 1.2, f"{a}", ha="center", fontsize=8.6, color=RED,
            fontname=JPB)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_yscale("symlog", linthresh=5)
ax.set_ylim(0, 130)
ax.set_yticks([0, 5, 10, 50, 100])
ax.set_yticklabels(["0", "5", "10", "50", "100"])
ax.set_ylabel("Recall@10 / 正解率（%）", fontsize=10.5)
ax.set_title("(a) 物理適応で検索・判別は急上昇", fontsize=12, fontname=JPB)
ax.legend(fontsize=9.3, loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
names = ["VideoCLIP\n-XL", "LCO-3B\n-2605", "Omni\n-3B", "LCO\n-7B",
         "LCO-3B", "V-JEPA2\nViT-L", "V-JEPA2\nViT-g"]
vals = [0.81, 1.85, 1.41, 2.02, 2.47, 0.94, 1.02]
cols_ = ["#2980b9", GRAY, GRAY, GRAY, GRAY, TEAL, TEAL]
after_map = {"LCO-7B": 2.89, "LCO-3B": 3.95}
y = np.arange(len(names))
ax.barh(y, vals, color=cols_, height=0.6, edgecolor="white")
for i, v in enumerate(vals):
    key = names[i].replace("\n", "")
    if key in after_map:
        a = after_map[key]
        ax.plot([v, a], [i, i], color=RED, lw=1.6, zorder=4)
        ax.scatter([a], [i], s=55, color=RED, zorder=5, edgecolor="white", linewidth=1)
        ax.text(a + 0.08, i + 0.22, f"適応後 {a}", fontsize=9.2, color=RED,
                fontname=JPB)
    ax.text(v + 0.06, i - 0.28, f"{v:.2f}", fontsize=9.2, color="#5b6470")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 4.9)
ax.set_xlabel("回帰 macro nRMSE（×100, ↓ が良い）", fontsize=10.5)
ax.set_title("(b) だが定量復元は悪化する", fontsize=12, fontname=JPB)
ax.grid(axis="x", alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
sysnames = ["No RAG", "VideoCLIP\n-XL", "LCO\n-Omni-3B", "+ Ft.\nphysics", "+ Ft.\nphys+gen"]
avg = [0.60, 0.59, 0.63, 0.65, 0.66]
optics = [0.67, 0.63, 0.65, 0.65, 0.68]
thermal = [0.53, 0.54, 0.61, 0.64, 0.63]
x = np.arange(len(sysnames)); w = 0.27
ax.bar(x - w, optics, w, label="Optics", color="#8e44ad", edgecolor="white")
ax.bar(x, thermal, w, label="Thermal", color=ORANGE, edgecolor="white")
ax.bar(x + w, avg, w, label="Average", color=RED, edgecolor="white")
for i, v in enumerate(avg):
    ax.text(i + w, v + 0.012, f"{v:.2f}", ha="center", fontsize=9.2, color=RED,
            fontname=JPB)
ax.axhline(0.60, color=GRAY, ls="--", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(sysnames, fontsize=8.4)
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("PhyGenBench スコア", fontsize=10.5)
ax.set_title("(c) 検索強化で生成の物理性も上がる", fontsize=12, fontname=JPB)
ax.legend(fontsize=9.3, framealpha=0.95)
ax.grid(axis="y", alpha=0.28, ls=":")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("図4: 物理適応の効果・トレードオフ・下流への効き方", fontsize=14,
             fontname=JPB, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(OUT, "fig4.png"), dpi=155, facecolor="#ffffff")
plt.close(fig)

print("saved:", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f, os.path.getsize(os.path.join(OUT, f)))
