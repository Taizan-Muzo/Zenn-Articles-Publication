# -*- coding: utf-8 -*-
"""Osprey (arXiv:2609.09338) 記事用の図を生成する"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SLUG = "osprey-target-agnostic-drafter-pretraining"
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


def box(ax, x, y, w, h, text, fc, fs=10.5, bold=False, alpha=0.14, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=fc, facecolor=fc, alpha=alpha, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontname=JP, fontweight=("bold" if bold else "normal"), zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, zorder=1, shrinkA=2, shrinkB=2))


# ============================================================
# fig1: 4段階パイプライン（target-agnostic / per-target の2相）
# ============================================================
fig, ax = plt.subplots(figsize=(12.6, 6.4))
ax.set_xlim(0, 12.6)
ax.set_ylim(0, 6.4)
ax.axis("off")

# フェーズ帯
ax.add_patch(FancyBboxPatch((0.25, 0.35), 5.0, 5.7,
             boxstyle="round,pad=0.02,rounding_size=0.05",
             linewidth=1.4, edgecolor=BLUE, facecolor=BLUE, alpha=0.05, zorder=0))
ax.add_patch(FancyBboxPatch((6.55, 0.35), 5.8, 5.7,
             boxstyle="round,pad=0.02,rounding_size=0.05",
             linewidth=1.4, edgecolor=TEAL, facecolor=TEAL, alpha=0.05, zorder=0))
ax.text(2.75, 5.78, "target-agnostic 相（一度だけ実行）", fontsize=12.5, fontname=JP,
        fontweight="bold", color=BLUE, ha="center")
ax.text(9.45, 5.78, "per-target 相（対象ごとに実行）", fontsize=12.5, fontname=JP,
        fontweight="bold", color=TEAL, ha="center")

# Stage 1: Prune
box(ax, 1.5, 4.55, 2.3, 1.15, "Stage 1: Prune\nQwen3-4B の先頭2層を残す\n(≈591M param)", BLUE, fs=9.8)
# Stage 2: Pre-train
box(ax, 4.05, 4.55, 2.3, 1.15, "Stage 2: Pre-train\nFineWeb で次トークン予測\n(汎用NTP能力を復元)", BLUE, fs=9.8)
arrow(ax, 2.68, 4.55, 2.88, 4.55, BLUE)

# 再利用可能なバックボーン
box(ax, 2.75, 2.7, 4.4, 1.05, "再利用可能な shallow NTP backbone\n(全 target 共通・2層・265M body)", BLUE, fs=10.2, bold=True, alpha=0.20)
arrow(ax, 4.05, 3.95, 4.05, 3.24, BLUE)
ax.text(4.62, 3.60, "1回\n作れば\n十分", fontsize=8.6, fontname=JP, color=BLUE, ha="center", va="center", linespacing=1.3)

# Stage 3: Convert（3つの処理）
box(ax, 7.9, 3.55, 3.1, 1.85, "Stage 3: Convert\n・語彙整列（subtoken平均）\n・zero-init QKV拡張\n・隠れ状態注入 (low/mid/high)", TEAL, fs=9.6)
# Stage 4: Distill
box(ax, 11.15, 3.55, 2.1, 1.85, "Stage 4: Distill\nTTT で\non-policy蒸留\n(λ=0.8)", TEAL, fs=9.6)
arrow(ax, 9.48, 3.55, 10.08, 3.55, TEAL)
arrow(ax, 4.98, 2.7, 6.32, 3.42, TEAL)

# ターゲットモデルたち
box(ax, 7.9, 1.35, 2.6, 0.85, "Qwen3-8B\n+16.1%", TEAL, fs=10.0, alpha=0.18)
box(ax, 10.05, 1.35, 1.8, 0.85, "Llama-3.3-70B\n+21.2%", TEAL, fs=10.0, alpha=0.18)
box(ax, 11.9, 1.35, 1.2, 0.85, "MiniMax\nM2.5\n+22.7%", TEAL, fs=9.0, alpha=0.18)
ax.text(8.1, 0.62, "mean accepted length の改善（EAGLE-3比）", fontsize=9.0, fontname=JP, color="#3e4c59", ha="center")
for tx in (7.9, 10.05, 11.9):
    arrow(ax, tx, 3.55, tx, 1.82, TEAL, lw=1.0, ms=9)

ax.text(6.55, 2.7, "───", fontsize=13, color=TEAL, ha="center", va="center")
ax.text(2.75, 0.62, "同じ1個のバックボーンから\n3系統・8B〜229B のドラフターが生える", fontsize=9.5, fontname=JP,
        color=DARK, ha="center", linespacing=1.4)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig1 done")

# ============================================================
# fig2: ドメインシフトで崩れる acceptance length と Osprey の改善
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))

# 左: math で適応した EAGLE-3 の5ドメイン評価（脆弱性）
ax = axes[0]
domains = ["math\n(同ドメイン)", "chat", "commonsense", "finance", "code"]
al_math = [5.06, 2.53, 2.64, 3.09, 1.86]
colors = [RED if d.startswith("math") else "#d98880" for d in domains]
bars = ax.bar(domains, al_math, color=colors, edgecolor=RED, linewidth=1.0, width=0.62)
for b, v in zip(bars, al_math):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.2f}",
            ha="center", fontsize=10, fontname=JP, fontweight="bold", color=DARK)
ax.set_ylabel("mean accepted length", fontname=JP)
ax.set_title("EAGLE-3 を math データだけで適応した場合\n（Qwen3-8B・5ドメイン評価）", fontsize=11.5, fontname=JP)
ax.set_ylim(0, 5.8)
ax.axhline(1.0, color=GRAY, lw=0.9, ls="--")
ax.text(4.42, 1.12, "AL=1（加速なし）", fontsize=8.5, fontname=JP, color=GRAY, ha="right")
ax.grid(axis="y", alpha=0.25, ls="--")
for lbl in ax.get_xticklabels():
    lbl.set_fontname(JP)

# 右: ID / OOD での Osprey vs EAGLE-3
ax = axes[1]
cats = ["in-domain (ID)", "out-of-domain (OOD)", "全25組平均"]
eagle = [3.971, 2.977, 3.176]
osprey = [4.465, 3.493, 3.686]
x = np.arange(len(cats))
w = 0.34
b1 = ax.bar(x - w / 2, eagle, w, label="EAGLE-3 (from scratch)", color=GRAY, edgecolor="#5d6d7e", linewidth=1.0)
b2 = ax.bar(x + w / 2, osprey, w, label="Osprey", color=TEAL, edgecolor=TEAL, linewidth=1.2, alpha=0.85)
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.07, f"{v:.3f}", ha="center", fontsize=9.8,
                fontname=JP, color=DARK, fontweight="bold")
gains = ["+12.4%", "+17.3%", "+16.1%"]
for i, g in enumerate(gains):
    ax.text(i, max(eagle[i], osprey[i]) + 0.62, g, ha="center", fontsize=10.5,
            fontname=JP, color=TEAL, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontname=JP)
ax.set_ylabel("mean accepted length", fontname=JP)
ax.set_title("Osprey の優位は OOD で最大になる\n（Qwen3-8B・5×5 クロスドメイン行列）", fontsize=11.5, fontname=JP)
ax.set_ylim(0, 5.6)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.25, ls="--")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig2 done")

# ============================================================
# fig3: zero-init QKV 拡張と語彙整列（Stage 3 Convert の中身）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

# 左: QKV 拡張
ax = axes[0]
ax.set_xlim(0, 6.3)
ax.set_ylim(0, 5.2)
ax.axis("off")
ax.set_title("zero-init QKV 拡張", fontsize=12.5, fontname=JP, fontweight="bold", color=DARK)

box(ax, 1.35, 4.35, 2.0, 0.85, "target の隠れ状態\n$[h^{low}; h^{mid}; h^{high}]$", PURPLE, fs=9.3)
box(ax, 1.35, 3.1, 2.0, 0.75, "線形射影 $W_c$", PURPLE, fs=9.5)
arrow(ax, 1.35, 3.92, 1.35, 3.48, PURPLE)
box(ax, 1.35, 2.0, 1.7, 0.72, "$c_{t-1}$\n(前位置の表現)", PURPLE, fs=9.5)
arrow(ax, 1.35, 2.72, 1.35, 2.38, PURPLE)

box(ax, 3.6, 2.0, 1.7, 0.72, "$z_t$\n(トークン側表現)", BLUE, fs=9.5)

# 結合後の拡張QKV
box(ax, 3.6, 3.6, 3.9, 1.5, "拡張された射影\n$[\\;0\\;\\;|\\;\\;W_{qkv}\\;]$$(c_{t-1} と z_t を連結して入力)", DARK, fs=10.5, bold=True, alpha=0.10)
arrow(ax, 2.22, 2.0, 2.75, 3.15, DARK)
arrow(ax, 4.47, 2.0, 4.30, 2.86, DARK)
ax.text(3.6, 1.15, "初期化時: $W_{qkv} \\cdot z_t$ のみ有効\n→ 事前学習済み挙動がそのまま保存される", fontsize=9.5,
        fontname=JP, color=DARK, ha="center", linespacing=1.5)
ax.text(3.6, 0.35, "訓練中にゼロ列が学習され、target 情報を徐々に取り込む", fontsize=9.5, fontname=JP,
        color=TEAL, ha="center")

# 右: 語彙整列
ax = axes[1]
ax.set_xlim(0, 6.3)
ax.set_ylim(0, 5.2)
ax.axis("off")
ax.set_title("語彙整列（vocabulary alignment）", fontsize=12.5, fontname=JP, fontweight="bold", color=DARK)

ax.text(3.15, 4.85, "source 語彙 → target 語彙 の初期化規則", fontsize=10.5, fontname=JP,
        ha="center", color=DARK)
box(ax, 3.15, 3.75, 5.2, 1.35, "共通トークン v ∈ $V_S$\n→ pretrained 行をそのままコピー\n$W~[v] = W_S[v]$", BLUE, fs=10.0)
box(ax, 3.15, 2.15, 5.2, 1.35, "target 専用トークン v ∉ $V_S$\n→ 文字列を source tokenizer で分割し\nsubtoken 埋め込みの平均で初期化", ORANGE, fs=10.0)
ax.text(3.15, 1.15, "対象: 埋め込み行列 と LM head の両方\n（target 側ではなく drafter 側の重みから初期化）", fontsize=9.5,
        fontname=JP, ha="center", color=DARK, linespacing=1.5)
ax.text(3.15, 0.35, "事前学習で得た言語知識を語彙変換後も保持", fontsize=9.5, fontname=JP,
        color=TEAL, ha="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig3 done")

# ============================================================
# fig4: スケール間の成果と損益分岐
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))

# 左: 3 target の AL 改善率 + 多言語
ax = axes[0]
settings = ["Qwen3-8B\n(5×5行列)", "Llama-3.3-70B\n(reproduce比)", "MiniMax-M2.5\n(229B MoE)", "G-MMLU\n(42言語)", "MGSM\n(11言語)"]
gains = [16.1, 21.2, 22.7, 23.8, 12.4]
colors = [BLUE, TEAL, TEAL, PURPLE, PURPLE]
bars = ax.bar(settings, gains, color=colors, edgecolor=colors, linewidth=1.2, width=0.6, alpha=0.85)
for b, v in zip(bars, gains):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"+{v:.1f}%", ha="center",
            fontsize=10.5, fontname=JP, fontweight="bold", color=DARK)
ax.axhline(0, color="#5b6470", lw=0.8)
ax.set_ylabel("mean accepted length の改善率 (%)", fontname=JP)
ax.set_title("同じ1個の backbone が 8B〜229B・多言語まで転送する\n(MiniMax はコードデータのみで適応)", fontsize=11.5, fontname=JP)
ax.set_ylim(0, 28.5)
ax.grid(axis="y", alpha=0.25, ls="--")
for lbl in ax.get_xticklabels():
    lbl.set_fontname(JP)

# 右: 損益分岐（10k steps vs 55k steps）
ax = axes[1]
tokens = np.linspace(0, 5.0, 200)  # 10億トークン単位
saving_per_bn = 0.88  # 10億トークンあたり 880 H100h
ax.plot(tokens, saving_per_bn * tokens, color=TEAL, lw=2.2, label="サービス時の節約（累計）")
ax.plot(tokens, np.full_like(tokens, 0.576), color=ORANGE, lw=2.0, ls="--", label="Stage 2 事前学習コスト（10k steps = 576 H100h）")
ax.plot(tokens, np.full_like(tokens, 3.040), color=RED, lw=2.0, ls="--", label="Stage 2 事前学習コスト（55k steps = 3,040 H100h）")
ax.axvline(0.65, color=ORANGE, lw=1.0, ls=":")
ax.axvline(3.5, color=RED, lw=1.0, ls=":")
ax.scatter([0.65, 3.5], [0.576, 3.040], s=55, color=[ORANGE, RED], zorder=5, edgecolor="white", linewidth=1.2)
ax.annotate("損益分岐\n0.65B tokens", xy=(0.65, 0.576), xytext=(1.35, 1.55),
            fontsize=9.8, fontname=JP, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax.annotate("損益分岐\n3.5B tokens", xy=(3.5, 3.040), xytext=(3.9, 1.9),
            fontsize=9.8, fontname=JP, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.set_xlabel("生成トークン総量（10億トークン単位）", fontname=JP)
ax.set_ylabel("H100 GPU-hours", fontname=JP)
ax.set_title("事前学習コストの回収（MiniMax-M2.5・EAGLE-3比）\n10k steps の backbone でほぼ全恩恵が得られる", fontsize=11.5, fontname=JP)
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 4.4)
ax.legend(fontsize=9.3, loc="upper left", framealpha=0.92)
ax.grid(alpha=0.25, ls="--")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig4 done")

print("ALL DONE:", OUT)
