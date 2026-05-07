import pandas as pd
import matplotlib.pyplot as plt
import sys
import shutil

# =========================
# USER CONFIG
# =========================
USE_MM = True
THOU_PER_MM = 39.3701

USE_LATEX = False
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.6
DPI = 300

LINE_WIDTH = 1.0
MARKER_SIZE = 4
STD_ALPHA = 0.2  # shading transparency

CLOSED_LOOP = False  # True -> fix y-axis to 0 to 0.9 mm (or equivalent thou), False -> auto

COLOR_CYCLE = [
    "#0072B2",
    "#FC6666",
    "#E69F00",
    "#CC79A7",
    "#009E73",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def configure_plot_style(use_latex=True):
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Only enable LaTeX if it is actually usable
    if use_latex and shutil.which("latex") is not None:
        try:
            plt.rcParams["text.usetex"] = True
        except Exception:
            plt.rcParams["text.usetex"] = False
    else:
        plt.rcParams["text.usetex"] = False

    # Always enforce a clean serif look (no LaTeX dependency)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def compute_stats(csv_path, use_mm=True):
    df = pd.read_csv(csv_path)

    if use_mm:
        df["abs_error"] = df["abs_delta_thou"] / THOU_PER_MM
    else:
        df["abs_error"] = df["abs_delta_thou"]

    stats = (
        df.groupby("noise_mm")["abs_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("noise_mm")
    )

    stats["std"] = stats["std"].fillna(0.0)

    avg_std = stats["std"].mean()
    stats["std_from_avg_std"] = (stats["std"] - avg_std).abs()

    return stats


def print_stats_table(stats, label, use_mm=True):
    unit = "mm" if use_mm else "thou"

    print("\n" + "=" * 80)
    print(f"Curve: {label}")
    print(f"Units: {unit}")
    print("=" * 80)
    print(f"noise_mm\tmean_{unit}\tstd_{unit}\tstd_from_avg_std_{unit}")

    for _, row in stats.iterrows():
        print(
            f"{int(row['noise_mm'])}\t"
            f"{row['mean']:.6f}\t"
            f"{row['std']:.6f}\t"
            f"{row['std_from_avg_std']:.6f}"
        )


def plot_multiple(csv_label_pairs, use_mm=True):
    configure_plot_style(use_latex=USE_LATEX)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    if use_mm:
        y_label = "Absolute Deviation (mm)"
        unit_str = "mm"
    else:
        y_label = "Absolute Deviation (thou)"
        unit_str = "thou"

    all_noise_vals = set()

    for i, pair in enumerate(csv_label_pairs):
        if ":" not in pair:
            print(f"Skipping invalid input: {pair}")
            continue

        csv_path, label = pair.split(":", 1)

        stats = compute_stats(csv_path, use_mm)
        print_stats_table(stats, label, use_mm)

        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        x = stats["noise_mm"].to_numpy()
        y = stats["mean"].to_numpy()
        err = stats["std_from_avg_std"].to_numpy()

        all_noise_vals.update(x.tolist())

        # Mean line
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
            color=color,
        )

        # Shaded region instead of error bars
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=color,
            alpha=STD_ALPHA,
            linewidth=0,
        )

    ax.set_xlabel("Vibration Amplitude (mm)")
    ax.set_ylabel(y_label)
    ax.set_title("Mean Absolute Deviation vs Vibration Amplitude")
    ax.grid(True)

    if all_noise_vals:
        ax.set_xticks(sorted(all_noise_vals))

    # Closed-loop plots use fixed y-axis, open-loop uses matplotlib autoscaling
    if CLOSED_LOOP:
        if use_mm:
            ax.set_ylim(0, 0.9)
        else:
            ax.set_ylim(0, 0.9 * THOU_PER_MM)

    ax.legend()

    fig.tight_layout()

    out_path = f"error_vs_noise_shaded_avgstd_{unit_str}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plot.py file1.csv:Label1 file2.csv:Label2 ...")
        sys.exit(1)

    csv_label_pairs = sys.argv[1:]
    plot_multiple(csv_label_pairs, use_mm=USE_MM)