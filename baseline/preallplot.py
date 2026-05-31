import os

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 全局风格配置
# =========================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["font.size"] = 23
plt.rcParams["axes.titlesize"] = 27
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 21
plt.rcParams["ytick.labelsize"] = 21
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.titlesize"] = 29

# =========================
# 配置
# =========================
_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_BASELINE_DIR, "results")
SAVE_NAME_PNG = os.path.join(_BASELINE_DIR, "baseline_combined.png")

METHODS = [
    ("AdamW_metrics.csv", "32-bit AdamW"),
    ("8bit_AdamW_metrics.csv", "8-bit AdamW"),
    ("MDQ_metrics.csv", "MDQ"),
    ("Adam_mini_metrics.csv", "Adam-mini"),
    ("GaLore_metrics.csv", "GaLore"),
]

COLORS = ["#4C78A8", "#9A9A9A", "#F58518", "#54A24B", "#E45756"]
SMOOTHING = 0.85


def load_method_data(filename: str) -> pd.DataFrame | None:
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        print(f"警告: 找不到 {file_path}")
        return None
    return pd.read_csv(file_path)


def plot_metric(ax, metric: str, ylabel: str):
    found_any = False
    alpha = 1 - SMOOTHING

    for i, (filename, display_name) in enumerate(METHODS):
        df = load_method_data(filename)
        if df is None or metric not in df.columns:
            continue

        found_any = True
        x = df["step"]
        y = df[metric]
        y_smooth = y.ewm(alpha=alpha, adjust=False).mean()

        ax.plot(
            x,
            y,
            color=COLORS[i],
            alpha=0.12,
            linewidth=0.8,
            label="_nolegend_",
        )
        ax.plot(
            x,
            y_smooth,
            label=display_name,
            color=COLORS[i],
            linewidth=2.6 if display_name == "MDQ" else 2.3,
            alpha=0.98 if display_name == "MDQ" else 0.95,
        )

    if not found_any:
        print(f"错误: 未找到任何可用的 {metric} 数据。")
        return

    ax.set_xlabel("Step", fontsize=25, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=25, labelpad=9)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(
        fontsize=18,
        frameon=True,
        loc="upper right",
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.9,
    )


def plot_combined_figure():
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.3), dpi=300)

    plot_metric(axes[0], metric="loss", ylabel="Loss")
    plot_metric(axes[1], metric="grad_norm", ylabel="Gradient Norm")

    plt.tight_layout(w_pad=3.2)
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.25,
        0.08,
        "(a) Loss vs Step on GPT2-1.5B (XL)",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )
    fig.text(
        0.765,
        0.08,
        "(b) Gradient Norm vs Step on GPT2-1.5B (XL)",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
    )

    fig.savefig(SAVE_NAME_PNG, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"图片已保存至: {SAVE_NAME_PNG}")


if __name__ == "__main__":
    plot_combined_figure()
