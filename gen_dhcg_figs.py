# -*- coding: utf-8 -*-
"""DHCG (arXiv:2610.07835) 記事用の図を生成する"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SLUG = "dhcg-dynamic-hierarchical-collaboration-graphs"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "images", SLUG)
os.makedirs(OUT, exist_ok=True)

JP = "Hiragino Sans"
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


def box(ax, x, y, w, h, text, fc, fs=10.5, bold=False, alpha=0.14, lw=1.3, tc=None):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=fc, facecolor=fc, alpha=alpha, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontname=JP, fontweight=("bold" if bold else "normal"),
            color=(tc if tc else "#1f2933"), zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>", ms=14, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, zorder=1, shrinkA=2, shrinkB=2, linestyle=ls))


# ============================================================
# fig1: 3つのMAS構築パラダイム比較
# ============================================================
fig, ax = plt.subplots(figsize=(12.6, 6.6))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 6.6)
ax.axis("off")

# (a) graph-based
box(ax, 2.1, 5.9, 3.4, 0.62, "(a) グラフ型 MAS", GRAY, fs=11.5, bold=True, alpha=0.10)
box(ax, 2.1, 4.85, 3.4, 1.1, "ノード・エッジを\n最適化 / 剪定 / 生成\n(G-Designer, GPTSwarm)", GRAY, fs=9.3)
box(ax, 2.1, 3.2, 3.4, 1.6, "ノード集合は固定\n⇒ 構成の柔軟性がない\n未解決の推理ニーズに\n役割を足せない", RED, fs=9.0, alpha=0.10)
arrow(ax, 2.1, 4.28, 2.1, 4.03, GRAY, lw=1.2, ms=10)

# (b) workflow-based
box(ax, 6.3, 5.9, 3.4, 0.62, "(b) ワークフロー型 MAS", GRAY, fs=11.5, bold=True, alpha=0.10)
box(ax, 6.3, 4.85, 3.4, 1.1, "探索・進化・フィードバックで\n実行可能な手続きを設計\n(AFlow, ADAS)", GRAY, fs=9.3)
box(ax, 6.3, 3.2, 3.4, 1.6, "依存関係はテンプレートや\n大域目的で決まる\n⇒ 下流の情報要求とズレる\n規模も実行前に固定", RED, fs=9.0, alpha=0.10)
arrow(ax, 6.3, 4.28, 6.3, 4.03, GRAY, lw=1.2, ms=10)

# (c) feedback-driven (ours)
box(ax, 10.7, 5.9, 3.6, 0.62, "(c) 提案: フィードバック駆動 (DHCG)", TEAL, fs=11.5, bold=True, alpha=0.12)
box(ax, 10.7, 4.85, 3.6, 1.1, "実行フィードバックを見ながら\nグラフをゼロから漸進構築\n(Planner が逐次意思決定)", TEAL, fs=9.3)
box(ax, 10.7, 3.2, 3.6, 1.6, "役割をその場で生成\n必要な情報だけを経路指定\n十分なら早期確定・\n不足なら層を追加", TEAL, fs=9.0, alpha=0.16)
arrow(ax, 10.7, 4.28, 10.7, 4.03, TEAL, lw=1.2, ms=10)

# 共通の問題ラベル
ax.text(6.3, 2.45, "既存手法の3つの限界", fontsize=11.5, fontname=JP, fontweight="bold",
        color=RED, ha="center")
box(ax, 2.1, 1.55, 3.4, 0.85, "① 限られた構成\n(役割の被覆が不足)", RED, fs=9.3, alpha=0.08)
box(ax, 6.3, 1.55, 3.4, 0.85, "② ズレた依存関係\n(冗長 or 欠落)", RED, fs=9.3, alpha=0.08)
box(ax, 10.7, 1.55, 3.6, 0.85, "③ 硬直した規模\n(足りない / 無駄に続く)", RED, fs=9.3, alpha=0.08)

ax.text(6.3, 0.55, "DHCG は①②③すべてを実行時に動的な判断で解く──構成・依存・規模を決めるのは実行前ではなく実行中",
        fontsize=9.6, fontname=JP, color=DARK, ha="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ============================================================
# fig2: DHCG フレームワーク概要（漸進的なグラフ構築ループ）
# ============================================================
fig, ax = plt.subplots(figsize=(12.6, 7.6))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 7.6)
ax.axis("off")

# Planner / Workers / Generator の3モジュール帯
box(ax, 3.1, 6.9, 2.2, 0.72, "Planner (Qwen3-8B 訓練済)", PURPLE, fs=10.0, bold=True, alpha=0.16)
box(ax, 8.2, 6.9, 2.4, 0.72, "Workers (Qwen3-8B)", BLUE, fs=10.0, bold=True, alpha=0.14)
box(ax, 11.3, 6.9, 1.9, 0.72, "Generator", TEAL, fs=10.0, bold=True, alpha=0.16)

# Step 1: 初期層構築
box(ax, 3.1, 5.5, 2.9, 1.15, "Step 1: 初期層の生成\nV0 ~ pi_init(q)\n独立・互補な役割 n0 個\n(初期層は辺なし)", PURPLE, fs=9.0)
box(ax, 8.2, 5.5, 2.4, 1.15, "並列実行\n(同一層は並列・\n層間のみ辺を許可)", BLUE, fs=9.3)
arrow(ax, 4.6, 5.5, 6.95, 5.5, PURPLE)
ax.text(5.75, 5.68, "役割 (rho_i, p_i)", fontsize=8.6, fontname=JP, color=PURPLE, ha="center")

# Step 2: 動的計画ループ
box(ax, 3.1, 3.6, 3.2, 1.55, "Step 2: 動的計画\n観測 o_t = (q, Z_{t-1})\n直前層の (役割, 出力) 対集合", PURPLE, fs=9.2)
arrow(ax, 8.2, 4.85, 8.2, 4.42, BLUE)
ax.text(9.0, 4.62, "Z_{t-1} を Planner へ", fontsize=8.6, fontname=JP, color=PURPLE, ha="center")

# 行動空間の2択
box(ax, 1.85, 1.85, 2.9, 1.35, "EXPAND(V_t, E_t)\n新しい層を追加\n必要な源だけ選択\n(疎なルーティング)", ORANGE, fs=9.0, alpha=0.16)
box(ax, 5.6, 1.85, 2.9, 1.35, "FINALIZE(C_{t-1})\n構築を打ち切り\n選んだ出力だけで\n解答合成へ", TEAL, fs=9.0, alpha=0.16)
arrow(ax, 2.9, 2.85, 2.35, 2.58, PURPLE)
arrow(ax, 3.6, 2.85, 5.1, 2.58, PURPLE)
ax.text(3.7, 2.42, "a_t", fontsize=10.5, fontname=JP, color=PURPLE, ha="center", style="italic")

# EXPAND → 実行 → 観測のループ
box(ax, 8.2, 3.3, 2.4, 1.1, "新しい層 V_t を\nWorkers で実行", BLUE, fs=9.3)
arrow(ax, 3.3, 1.85, 7.0, 2.9, ORANGE, lw=1.3)
arrow(ax, 8.2, 2.75, 8.2, 2.42, BLUE)

# Step 3: 解答合成
box(ax, 11.3, 3.3, 1.9, 1.1, "Step 3:\n選ばれた出力だけ\nで最終解答", TEAL, fs=9.0)
arrow(ax, 7.05, 1.85, 10.4, 3.0, TEAL, lw=1.5)
arrow(ax, 9.42, 3.3, 10.32, 3.3, TEAL)

# 階層DAGのミニ図
dag_x0, dag_y0 = 9.3, 6.9
n_layer0 = [(-0.55, 5.62), (0.0, 5.62), (0.55, 5.62)]
n_layer1 = [(-0.3, 6.15), (0.3, 6.15)]
for (dx, y) in n_layer0:
    box(ax, 10.9 + dx, y, 0.42, 0.34, "", BLUE, alpha=0.25, lw=1.0)
for (dx, y) in n_layer1:
    box(ax, 10.9 + dx, y, 0.42, 0.34, "", ORANGE, alpha=0.30, lw=1.0)
arrow(ax, 10.6, 5.79, 10.7, 5.97, DARK, lw=1.0, ms=7)
arrow(ax, 10.9, 5.79, 10.9, 5.97, DARK, lw=1.0, ms=7)
arrow(ax, 11.2, 5.79, 11.1, 5.97, DARK, lw=1.0, ms=7)
ax.text(10.9, 5.30, "階層 DAG: 層内は並列・層間のみ依存", fontsize=8.4, fontname=JP, color=DARK, ha="center")

ax.text(6.3, 0.6, "Planner は毎層「広げるか・確定するか」を選べる──最大層数 L に達するまで、あるいは十分と判断するまで構築が続く",
        fontsize=9.6, fontname=JP, color=DARK, ha="center")
ax.text(4.75, 0.98, "難問なら大きいグラフ、易問なら小さいグラフ（difficulty-aware）", fontsize=9.2, fontname=JP,
        color=ORANGE, ha="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ============================================================
# fig3: Action-aware Preference Optimization
# ============================================================
fig, ax = plt.subplots(figsize=(12.6, 7.0))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 7.0)
ax.axis("off")

# 観測ノード
box(ax, 2.0, 5.6, 2.6, 0.8, "観測 o_t\n(q, Z_{t-1})", PURPLE, fs=10.0, bold=True, alpha=0.16)

# 木構造の軌跡
cand = [
    (5.3, 5.6, "EXPAND A", ORANGE),
    (5.3, 4.0, "EXPAND B", ORANGE),
    (5.3, 2.4, "FINALIZE", TEAL),
]
for (x, y, label, c) in cand:
    box(ax, x, y, 2.0, 0.7, label, c, fs=9.6, alpha=0.14)
    arrow(ax, 3.3, 5.6, 4.28, y, PURPLE, lw=1.2, ms=9)

# 各候補の子（再帰実行）
kids = [
    (7.9, 6.3, "a", 0.42), (7.9, 5.0, "b", 0.10),   # EXPAND A の子
    (7.9, 4.3, "c", 0.30), (7.9, 3.15, "d", 0.10),  # EXPAND B の子
]
for (x, y, label, r) in kids:
    box(ax, x, y, 1.3, 0.55, "子: r=%.2f" % r, GRAY, fs=8.4, alpha=0.08)
for src_y in (5.6, 4.0):
    arrow(ax, 6.32, src_y, 7.23, src_y + 0.5, GRAY, lw=1.0, ms=7)
    arrow(ax, 6.32, src_y, 7.23, src_y - 0.5, GRAY, lw=1.0, ms=7)
arrow(ax, 6.32, 2.4, 7.23, 2.4, GRAY, lw=1.0, ms=7)
box(ax, 7.9, 2.4, 1.3, 0.55, "合成: r=0.05", GRAY, fs=8.4, alpha=0.08)

# 価値計算
box(ax, 10.9, 5.4, 3.0, 2.4, "価値の再帰推定\n\nV_T = r - λC_T\nV_t = γ V̄_{t+1} - λC_t\n\nr: 最終スコア, C: token コスト\nλ: 精度と効率のトレードオフ", DARK, fs=9.2, alpha=0.08)
arrow(ax, 8.6, 6.3, 9.38, 6.0, GRAY, lw=1.0, ms=8)
arrow(ax, 8.6, 2.4, 9.38, 4.9, GRAY, lw=1.0, ms=8)

# 好対構築（クロスアクション / 同一アクション）
box(ax, 5.3, 1.35, 4.6, 0.85, "クロスアクション対 (EXPAND vs FINALIZE):\nタスクのみ価値で比較（λを無効化）\n→ 低コストの FINALIZE に偏らない", RED, fs=8.8, alpha=0.10)
box(ax, 10.6, 1.35, 3.6, 0.85, "同一アクション対 (EXPAND vs EXPAND):\nコスト込み価値で比較\n→ 実行効率も区別", BLUE, fs=8.8, alpha=0.10)

# 重み付きDPO
box(ax, 6.3, 0.35, 9.2, 0.62, "重み付き DPO: L_APO = -1/N Σ w̃_i log σ( β Δρ_θ^(i) )　　w_i は価値ギャップをクリップして正規化", PURPLE, fs=9.6, bold=True, alpha=0.14)

ax.text(2.9, 3.5, "① 軌跡木の収集\n（少ない軌跡から\n多数の選好対を作る）", fontsize=9.0, fontname=JP,
        color=PURPLE, ha="center", linespacing=1.5)
ax.text(2.9, 2.3, "② 下流成果で代替行動を評価\n→ オンラインRL不要の\nオフライン学習", fontsize=9.0, fontname=JP,
        color=PURPLE, ha="center", linespacing=1.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ============================================================
# fig4: 実験結果（7ベンチマーク比較 + 消融）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

# --- 左: 7ベンチマークの平均比較 ---
ax1 = axes[0]
methods = ["IO", "CoT", "CoT-SC", "LLM-Debate", "MacNet", "G-Designer", "ARG-Designer", "AFlow", "MaAS", "DHCG"]
avg = [60.37, 63.78, 68.90, 68.67, 67.18, 70.10, 67.43, 65.41, 70.66, 73.43]
colors = [GRAY, GRAY, GRAY, BLUE, BLUE, ORANGE, ORANGE, "#8e7cc3", "#8e7cc3", RED]
y = np.arange(len(methods))
bars = ax1.barh(y, avg, color=colors, alpha=0.82, height=0.62)
for i, v in enumerate(avg):
    ax1.text(v + 0.35, i, "%.2f" % v, va="center", fontsize=9.3,
             color=(RED if methods[i] == "DHCG" else "#3e4c59"),
             fontweight=("bold" if methods[i] == "DHCG" else "normal"))
ax1.set_yticks(y)
ax1.set_yticklabels(methods, fontsize=9.8)
ax1.set_xlim(55, 76)
ax1.set_xlabel("7ベンチマーク平均精度 (%)", fontsize=10.5, fontname=JP)
ax1.set_title("ベースライン比較（Qwen3-8B 共通バックボーン）", fontsize=11.5, fontname=JP, fontweight="bold")
ax1.invert_yaxis()
ax1.grid(axis="x", alpha=0.28, linestyle="--")
ax1.text(60.0, 8.9, "静的MAS", fontsize=8.8, fontname=JP, color=BLUE)
ax1.text(60.0, 7.9, "グラフ型", fontsize=8.8, fontname=JP, color=ORANGE)
ax1.text(60.0, 6.9, "ワークフロー型", fontsize=8.8, fontname=JP, color="#8e7cc3")
ax1.annotate("+13.06 vs IO\n+2.77〜+8.02 vs MAS", xy=(73.5, 9.3), xytext=(69.8, 9.9),
             fontsize=9.0, fontname=JP, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# --- 右: 消融 ---
ax2 = axes[1]
abl_labels = ["Base Planner", "SFT", "Vanilla DPO", "w/o Dyn.\nRouting", "w/o Dyn.\nScale", "w/o Dyn.\nRole", "DHCG\n(APO)"]
abl_vals = [84.01, 82.80, 84.16, 84.19, 82.39, 83.09, 85.02]
abl_colors = [GRAY, RED, PURPLE, ORANGE, RED, ORANGE, TEAL]
x = np.arange(len(abl_labels))
bars2 = ax2.bar(x, abl_vals, color=abl_colors, alpha=0.82, width=0.62)
for i, v in enumerate(abl_vals):
    ax2.text(i, v + 0.18, "%.2f" % v, ha="center", fontsize=9.0,
             color="#1f2933", fontweight=("bold" if i == 6 else "normal"))
ax2.set_xticks(x)
ax2.set_xticklabels(abl_labels, fontsize=8.6, fontname=JP)
ax2.set_ylim(81.5, 86.0)
ax2.set_ylabel("3ベンチマーク平均 (%)", fontsize=10.5, fontname=JP)
ax2.set_title("消融実験（Planner訓練 / グラフ構築）", fontsize=11.5, fontname=JP, fontweight="bold")
ax2.grid(axis="y", alpha=0.28, linestyle="--")
ax2.annotate("SFT は逆効果\n(-1.21)", xy=(1, 82.8), xytext=(1.7, 82.0),
             fontsize=8.8, fontname=JP, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
ax2.annotate("動的スケール除去が\n最大の打撃 (-2.63)", xy=(4, 82.39), xytext=(4.3, 85.2),
             fontsize=8.8, fontname=JP, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("done:", OUT)
