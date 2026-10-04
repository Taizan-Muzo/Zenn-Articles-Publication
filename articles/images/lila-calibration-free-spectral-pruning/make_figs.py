# LILA (arXiv:2609.11163) figure generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
fm.fontManager.addfont(font_path)
jp = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = jp.get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/Zhuanz/Desktop/Zenn-Articles-Publication/articles/images/lila-calibration-free-spectral-pruning"

C_BLUE = "#2563EB"
C_RED = "#DC2626"
C_GRAY = "#9CA3AF"
C_GREEN = "#059669"
C_ORANGE = "#EA580C"

# ---------------- fig1: pipeline overview ----------------
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis("off")

boxes = [
    ("FFNの重み $W_{up}$\n(各層)", "#EFF6FF", C_BLUE),
    ("非負行列 $M$ を構成\nAbs / Split / Act", "#EFF6FF", C_BLUE),
    ("NMF分解\n(ANLS + rSVD初期化)", "#EFF6FF", C_BLUE),
    ("ニューロン $j$ を消融\n特異値分布を再計算\n(rank-1 downdate)", "#FEF3C7", C_ORANGE),
    ("KS距離で採点\n$s_j=\\sup_t|F_\\sigma(t)-F_\\sigma^{(-j)}(t)|$", "#FEE2E2", C_RED),
    ("Top-K選択で構造化\nプルーニング", "#ECFDF5", C_GREEN),
]
x = 0.0
widths = [0.13, 0.15, 0.15, 0.17, 0.21, 0.15]
for (label, fc, ec), w in zip(boxes, widths):
    ax.add_patch(plt.Rectangle((x, 0.25), w, 0.5, facecolor=fc, edgecolor=ec, lw=2, zorder=2))
    ax.text(x + w / 2, 0.5, label, ha="center", va="center", fontsize=10.5, zorder=3)
    x += w
    if x < 0.97:
        ax.annotate("", xy=(x + 0.015, 0.5), xytext=(x - 0.005, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.8))
        x += 0.02

ax.text(0.5, 0.06, "校準データ不要・学習不要・閉形式 ― アーキテクチャは元のまま保持",
        ha="center", fontsize=12, color="#111827", fontweight="bold")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------- fig2: KS distance concept ----------------
rng = np.random.default_rng(42)
# synthetic singular values: full spectrum vs ablated (rank slightly reduced)
sig_full = np.sort(rng.lognormal(0, 1.1, 200))[::-1]
sig_abl = np.sort(np.delete(sig_full, rng.choice(200, 8, replace=False)) * (0.98 + 0.02 * rng.random(192)))[::-1]

def ecdf(s, t):
    return np.array([np.mean(s <= ti) for ti in t])

t = np.linspace(np.min(sig_abl) * 0.8, np.max(sig_full) * 1.05, 500)
F1, F2 = ecdf(sig_full, t), ecdf(sig_abl, t)
diff = np.abs(F1 - F2)
imax = np.argmax(diff)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(t, F1, color=C_BLUE, lw=2.5, label="完全スペクトルのECDF $F_\\sigma(t)$")
ax.plot(t, F2, color=C_RED, lw=2.5, ls="--", label="ニューロン $j$ 消融後のECDF $F_\\sigma^{(-j)}(t)$")
ax.vlines(t[imax], F1[imax], F2[imax], color="#111827", lw=2.5)
ax.plot([t[imax]] * 2, [F1[imax], F2[imax]], "o", color="#111827", ms=6)
ax.annotate("$s_j^{spec}=\\sup_t|F_\\sigma(t)-F_\\sigma^{(-j)}(t)|$\n（KS統計量＝重要性スコア）",
            xy=(t[imax], (F1[imax] + F2[imax]) / 2), xytext=(t[imax] + 0.02 * t.max(), 0.25),
            fontsize=11, arrowprops=dict(arrowstyle="->", color="#374151"))
ax.set_xlabel("特異値 $t$", fontsize=12)
ax.set_ylabel("累積分布関数", fontsize=12)
ax.set_title("消融による特異値分布の変位が大きいニューロンほど「譜的に不可欠」", fontsize=12.5)
ax.legend(fontsize=10.5, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_ks_distance.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------- fig3: main results grouped bars ----------------
sp = ["20%", "25%", "30%"]
dense = 69.00
methods = {
    "Dense (69.0)": [dense, dense, dense],
    "LILA-Spectrum, Abs\n(校準なし, noRFT)": [63.35, 60.20, 57.50],
    "PruneNet (noRFT)": [61.67, 58.63, 55.45],
    "SliceGPT (WT2校準, noRFT)": [58.18, 55.48, 51.50],
    "Wanda (128校準)": [58.14, 54.98, 51.21],
}
colors = [C_GRAY, C_BLUE, C_GREEN, C_RED, C_ORANGE]

fig, ax = plt.subplots(figsize=(11.5, 5.2))
x = np.arange(len(sp))
n = len(methods)
w = 0.15
for i, ((name, vals), c) in enumerate(zip(methods.items(), colors)):
    bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=c,
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=8.5)
ax.axhline(dense, color=C_GRAY, ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f"LLaMA-2-7B スパース度 {s}" for s in sp], fontsize=11.5)
ax.set_ylabel("Zero-shot 平均精度 (%)", fontsize=12)
ax.set_ylim(45, 72)
ax.set_title("校準ゼロ・微調整ゼロのLILAがRLポリシー45MパラメータのPruneNetや\n校準済みSliceGPTをnoRFTで上回る（PIQA/HellaSwag/ARC-E/ARC-C/WinoGrande平均）",
             fontsize=12)
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_main_results.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close()

# ---------------- fig4: uniform vs adaptive sparsity (PPL) ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# left: PPL comparison at 25%
models = ["LLaMA-2-7B", "Phi-2"]
uni = [11.14, 54.07]
ada = [9.50, 40.03]
x = np.arange(2)
w = 0.32
for i, (vals, lab, c) in enumerate([(uni, "Uniform 25%", C_BLUE), (ada, "Adaptive (KS-score)", C_GREEN)]):
    bars = axes[0].bar(x + (i - 0.5) * w, vals, w, label=lab, color=c, edgecolor="white")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}", ha="center", fontsize=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=11.5)
axes[0].set_ylabel("WikiText-2 PPL（低いほど良い）", fontsize=11)
axes[0].set_title("25% 圧縮：Adaptive KS予算が生成品質を改善", fontsize=11.5)
axes[0].legend(fontsize=10)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, 62)

# right: layer-wise KS score conceptual + 30% collapse
layers = np.arange(32)
ks = 0.35 + 0.25 * np.exp(-layers / 4.5) + 0.2 * np.exp(-(31 - layers) / 3.5) + 0.03 * np.sin(layers * 1.3)
ks = ks / ks.max()
axes[1].plot(layers, ks, color=C_BLUE, lw=2.2)
axes[1].fill_between(layers, ks, alpha=0.15, color=C_BLUE)
axes[1].axhline(0.65, color=C_RED, ls="--", lw=1.6)
axes[1].text(16, 0.68, "生存下界 $m_{min}=0.65$（単層最大35%スパース）", fontsize=9.5, color=C_RED)
axes[1].annotate("30%でAdaptiveは崩壊\nPPL 281.47 / Acc 40.0", xy=(28, 0.55), xytext=(12, 0.30),
                 fontsize=10, color="#111827",
                 arrowprops=dict(arrowstyle="->", color=C_RED))
axes[1].set_xlabel("FFN層のインデックス", fontsize=11)
axes[1].set_ylabel("層別平均KSスコア（正規化）", fontsize=11)
axes[1].set_title("初期層と末尾層が譜的に敏感 → 高圧縮では単層ボトルネックが破綻", fontsize=11.5)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_adaptive_sparsity.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close()

print("done")
