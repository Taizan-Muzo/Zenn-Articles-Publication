# -*- coding: utf-8 -*-
"""QGPM (arXiv:2611.09554) 記事用配図生成スクリプト."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# 日本語フォント設定（Hiragino Sans）
plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/Zhuanz/Desktop/Zenn-Articles-Publication/images/qgpm-query-guided-rope-modulation"
os.makedirs(OUT, exist_ok=True)

C_BLUE = "#2E86AB"
C_ORANGE = "#E07A2F"
C_GREEN = "#7FB069"
C_RED = "#C75146"
C_GRAY = "#8A9BA8"
C_PURPLE = "#9B7EBD"
C_DARK = "#3D4A55"

# ============================================================
# fig1: RoPE周波数面の長距離減衰
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

# 左: 距離で一貫して運べる平面数 m(Δp) = d·ln(Δp/2π)/(2·lnθ)
d, theta = 128, 1e7
dp = np.logspace(0.7, np.log10(65536), 400)
m_boundary = d * np.log(dp / (2 * np.pi)) / (2 * np.log(theta))
usable = 64 - m_boundary
usable = np.clip(usable, 0, 64)

ax1.plot(dp, usable, color=C_BLUE, lw=2.5)
ax1.axhline(64, color=C_GRAY, lw=0.8, ls=":")
ax1.fill_between(dp, 0, usable, alpha=0.10, color=C_BLUE)

ax1.scatter([100, 32768], [53, 30], s=70, zorder=5,
            color=C_RED, edgecolors="white", linewidths=1.2)
ax1.annotate("100トークン\n53/64面", xy=(100, 53), xytext=(180, 60),
             fontsize=10, color=C_DARK,
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
ax1.annotate("32Kトークン\n30/64面", xy=(32768, 30), xytext=(9000, 14),
             fontsize=10, color=C_DARK,
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

ax1.set_xscale("log")
ax1.set_xlabel("クエリ-証拠間の距離 Δp（トークン, logスケール）")
ax1.set_ylabel("距離を一貫して運べる平面数")
ax1.set_title("波長が距離を超える「遅い面」だけが距離信号を保持する\n（$\\theta=10^7$, d=128, 全64面）",
              fontsize=11.5, fontweight="bold", color="#444444", loc="left")
ax1.set_ylim(0, 68)
ax1.grid(alpha=0.25, which="both")
ax1.spines[["top", "right"]].set_visible(False)

# 右: 速い面/遅い面の分業とQGPMの介入
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("QGPMの介入: 圧縮してよいのは速い面×低関連テキストだけ",
              fontsize=11.5, fontweight="bold", color="#444444", loc="left")

# 遅い面（上）: 触らない
ax2.add_patch(plt.Rectangle((0.6, 6.4), 8.8, 2.6, fc="#EAF3EE", ec=C_GREEN, lw=1.6))
ax2.text(1.0, 8.35, "遅い面（λ > 文脈長）", fontsize=11, fontweight="bold", color="#3A6B3F")
ax2.text(1.0, 7.35, "文書レイアウトを担う → 全トークンの stride = 1（触らない）",
         fontsize=10, color=C_DARK)

# 速い面（下）: 低関連のみ圧縮
ax2.add_patch(plt.Rectangle((0.6, 3.3), 8.8, 2.6, fc="#FDF1E7", ec=C_ORANGE, lw=1.6))
ax2.text(1.0, 5.25, "速い面（λ < 距離）", fontsize=11, fontweight="bold", color="#9C5313")
ax2.text(1.0, 4.25, "局所的な語順を担う → 低関連テキストの stride を 1 未満に圧縮",
         fontsize=10, color=C_DARK)

# トークン列の概念
y0 = 1.6
segs = [(0.6, 1.6, C_GRAY, "前置き\n(保護)"), (2.4, 1.3, C_BLUE, "関連文\nstride=1"),
        (3.9, 1.7, C_GRAY, "低関連 stride<1"), (5.8, 1.3, C_BLUE, "関連文\nstride=1"),
        (7.3, 2.1, C_PURPLE, "質問")]
for x, w, c, lab in segs:
    ax2.add_patch(plt.Rectangle((x, y0), w, 1.1, fc="white", ec=c, lw=1.6))
    ax2.text(x + w / 2, y0 + 0.55, lab, ha="center", va="center", fontsize=8.2, color=C_DARK)
ax2.annotate("", xy=(7.3, 0.85), xytext=(3.05, 0.85),
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8))
ax2.text(5.1, 0.35, "累積位置が縮み、証拠が質問の「近く」へ", ha="center",
         fontsize=9.5, color=C_RED)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1.png"), dpi=160, facecolor="white")
plt.close(fig)

# ============================================================
# fig2: QGPM 2パス パイプライン
# ============================================================
fig, ax = plt.subplots(figsize=(12.2, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 62)
ax.axis("off")

def box(x, y, w, h, fc, ec, text, fs=9.5, tc=C_DARK, fw="normal"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5,
                               joinstyle="round"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=fw)

def arrow(x1, y1, x2, y2, color=C_GRAY, lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

ax.set_title("QGPM: Pass 1 で関連性を測り、Pass 2 で位置だけを積み直す",
             fontsize=13, fontweight="bold", color="#444444", loc="left", pad=10)

# Pass 1
ax.text(2, 56.5, "Pass 1: plainなprefill（1回だけ余分に実行）", fontsize=11,
        fontweight="bold", color=C_BLUE)
box(2, 45, 20, 9, "#EAF3F9", C_BLUE, "長文脈 + 質問を\n素のモデルで1回読む", 9.5)
box(26, 45, 24, 9, "#EAF3F9", C_BLUE, "キャリブレーション済み\n上位ヘッド集合 H から\nRoPE logit を読む", 9.5)
box(54, 45, 22, 9, "#EAF3F9", C_BLUE, "文単位の関連性 z_i\n→ 順位のみ使用", 9.5)
box(80, 45, 18, 9, "#EAF3F9", C_BLUE, "sigmoid\nf_i = σ((K-0.5-ρ)/T)", 8.8)
arrow(22, 49.5, 26, 49.5); arrow(50, 49.5, 54, 49.5); arrow(76, 49.5, 80, 49.5)

# 2つの地図
box(14, 32, 30, 8, "#F3EEFA", C_PURPLE, "広い地図（下層用）\n大きいK + 軟らかいT: 候補を広く生かす", 9.2)
box(56, 32, 30, 8, "#F3EEFA", C_PURPLE, "鋭い地図（上層用）\n小さいK + 硬いT: 上位だけ残す", 9.2)
arrow(87, 45, 74, 40); arrow(83, 45, 34, 40)
ax.text(50, 41.5, "同じ順位から2枚の地図", ha="center", fontsize=9.5, color=C_PURPLE,
        fontweight="bold")
ax.text(29, 27.6, "保護規則: 最初の関連文より前は stride=1 で保護（前置きの圧縮は証拠の符号化を壊す）",
        ha="center", fontsize=8.8, color=C_RED)

# Pass 2
ax.text(2, 23.5, "Pass 2: 応答パス（attentionカーネルは無変更、RoPEテーブルのみ書き換え）",
        fontsize=11, fontweight="bold", color=C_ORANGE)
box(2, 11, 22, 9, "#FDF1E7", C_ORANGE, "平面ごとのゲート\ng_m = σ((m-m*)/τ_m)", 9.2)
box(28, 11, 24, 9, "#FDF1E7", C_ORANGE, "stride\nδ_m(j) = max(s+(1-s)g_m, ε)\n低関連だけ圧縮", 8.6)
box(56, 11, 20, 9, "#FDF1E7", C_ORANGE, "累積位置\np'_{m,j} = Σ δ_m(k)\n（厳密に増加）", 8.6)
box(80, 11, 18, 9, "#FDF1E7", C_ORANGE, "証拠が質問の\n近くへ引き寄せ\nられる", 9.2)
arrow(24, 15.5, 28, 15.5); arrow(52, 15.5, 56, 15.5); arrow(76, 15.5, 80, 15.5)

# 地図→Pass2 の接続
arrow(29, 32, 13, 20.2, color=C_PURPLE)
arrow(71, 32, 66, 20.2, color=C_PURPLE)

# 下部の注記
ax.text(50, 5.2, "層スケジュール: 下層は広い地図・上層は鋭い地図（各層のKVキャッシュは自分の地図の座標系）",
        ha="center", fontsize=9.5, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.45", fc="#F5F6F7", ec=C_GRAY, lw=1.0))
ax.text(50, 1.4, "トークンは1つも削られず・並べ替えられず・重みは不変。全スコアを1にすれば素のモデルと同一出力",
        ha="center", fontsize=9.5, color=C_GREEN, fontweight="bold")

fig.savefig(os.path.join(OUT, "fig2.png"), dpi=160, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

# ============================================================
# fig3: 主結果（NoLiMa + 5タスク平均）
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

models = ["Qwen2.5\n-7B", "Qwen3\n-4B", "Ministral\n-8B", "Granite\n-3.3-8B"]
vanilla_nolima = [28.5, 16.1, 11.5, 18.6]
qgpm_nolima = [46.1, 25.1, 24.1, 35.6]
vanilla_avg = [68.6, 69.2, 65.0, 59.4]
qgpm_avg = [75.4, 71.6, 69.1, 68.3]

x = np.arange(4); w = 0.36
b1 = ax1.bar(x - w / 2, vanilla_nolima, w, label="Vanilla", color=C_GRAY)
b2 = ax1.bar(x + w / 2, qgpm_nolima, w, label="QGPM", color=C_BLUE)
for b in list(b1) + list(b2):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
             f"{b.get_height():.1f}", ha="center", fontsize=8.5, color=C_DARK)
for i in range(4):
    diff = qgpm_nolima[i] - vanilla_nolima[i]
    ax1.text(x[i] + w / 2, qgpm_nolima[i] + 5.5, f"+{diff:.1f}",
             ha="center", fontsize=9, color=C_RED, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9.5)
ax1.set_ylabel("精度（%）")
ax1.set_ylim(0, 60)
ax1.set_title("NoLiMa（語彙的重複を排した検索）\n4バックボーンすべてで +9.0〜+17.6pt",
              fontsize=11.5, fontweight="bold", color="#444444", loc="left")
ax1.legend(fontsize=9, frameon=False)
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)

b3 = ax2.bar(x - w / 2, vanilla_avg, w, label="Vanilla", color=C_GRAY)
b4 = ax2.bar(x + w / 2, qgpm_avg, w, label="QGPM", color=C_BLUE)
for b in list(b3) + list(b4):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
             f"{b.get_height():.1f}", ha="center", fontsize=8.5, color=C_DARK)
for i in range(4):
    diff = qgpm_avg[i] - vanilla_avg[i]
    ax2.text(x[i] + w / 2, qgpm_avg[i] + 4.2, f"+{diff:.1f}",
             ha="center", fontsize=9, color=C_RED, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9.5)
ax2.set_ylabel("精度（%）")
ax2.set_ylim(0, 88)
ax2.set_title("5タスク平均（NoLiMa + RULER 4タスク）\nVanillaを下回らない唯一の手法",
              fontsize=11.5, fontweight="bold", color="#444444", loc="left")
ax2.legend(fontsize=9, frameon=False)
ax2.grid(axis="y", alpha=0.25)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3.png"), dpi=160, facecolor="white")
plt.close(fig)

# ============================================================
# fig4: 入力編集・Oracle との比較
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8),
                               gridspec_kw={"width_ratios": [1.35, 1]})

methods = ["Vanilla", "STRING", "Reorder", "Top-K", "QGPM"]
vals = [28.5, 33.1, 39.9, 41.1, 46.1]
cols = [C_GRAY, "#B8C4CC", "#E8B96A", "#E8A04A", C_BLUE]
bars = ax1.bar(methods, vals, 0.62, color=cols)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}",
             ha="center", fontsize=9.5, color=C_DARK, fontweight="bold")
ax1.set_ylabel("NoLiMa 精度（%）")
ax1.set_ylim(0, 56)
ax1.set_title("関連性の使い方の比較（Qwen2.5-7B, NoLiMa）\nTop-K=上位5チャンク保持 / Reorder=関連順に並べ替え / QGPM=位置のみ変調",
              fontsize=11, fontweight="bold", color="#444444", loc="left")
ax1.grid(axis="y", alpha=0.25)
ax1.spines[["top", "right"]].set_visible(False)
ax1.annotate("全トークンを保持したまま\n最も高い", xy=(4, 46.1), xytext=(2.7, 51),
             fontsize=9.5, color=C_BLUE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3))

ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("Oracle との差 = 関連性推定の限界",
              fontsize=11.5, fontweight="bold", color="#444444", loc="left")
rows = [
    ("Oracle（人手で証拠位置をマーク）", "NoLiMa全長で Vanilla +17.6〜+21.2", C_GREEN, 8.6),
    ("QGPM vs Oracle の差", "16Kまで 2.0pt以内 / 32Kで 5.7pt", C_ORANGE, 7.0),
    ("QA-1 の結果", "Oracleでも Vanilla比 +2.0pt（位置はQAのボトルネックでない）", C_GRAY, 5.0),
    ("NIAH-MV（64K）", "QGPM 93.4 > Reorder 92.2 > Top-K 91.1", C_BLUE, 3.0),
]
for title, desc, c, y in rows:
    ax2.add_patch(plt.Rectangle((0.3, y - 0.9), 9.4, 1.7, fc="white", ec=c, lw=1.4))
    ax2.text(0.6, y + 0.3, title, fontsize=9.5, fontweight="bold", color=C_DARK, va="center")
    ax2.text(0.6, y - 0.45, desc, fontsize=8.4, color="#5A6B76", va="center")
ax2.text(5, 1.0, "32Kでの頭打ちは位置変調ではなく\n関連性推定側に起因", ha="center",
         fontsize=9.5, color=C_RED, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4.png"), dpi=160, facecolor="white")
plt.close(fig)

print("done:", os.listdir(OUT))
