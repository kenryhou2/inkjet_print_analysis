import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import shutil

# =========================
# USER CONFIG
# =========================
USE_MM = False
THOU_PER_MM = 39.3701

USE_LATEX = False
FIG_WIDTH_IN = 4.0
FIG_HEIGHT_IN = 4.5
DPI = 300

LINE_WIDTH = 1.0
MARKER_SIZE = 4
STD_ALPHA = 0.2

EXP_FREQ_HZ = 10

# Add this near your USER CONFIG section

# Fixed y-axis limits in mm
Y_LIM_MIN_MM = 0
Y_LIM_MAX_MM = 9.0

# Automatically matched y-axis limits in thou
Y_LIM_MIN_THOU = Y_LIM_MIN_MM * THOU_PER_MM
Y_LIM_MAX_THOU = Y_LIM_MAX_MM * THOU_PER_MM

# Fixed x-axis limits in mm
X_LIM_MIN_MM = 0
X_LIM_MAX_MM = 15.0

# Automatically matched x-axis limits in thou
X_LIM_MIN_THOU = X_LIM_MIN_MM * THOU_PER_MM
X_LIM_MAX_THOU = X_LIM_MAX_MM * THOU_PER_MM
# Options:
#   "std"              -> regular standard deviation
#   "sem"              -> standard error of the mean
#   "std_from_avg_std" -> absolute deviation from average std
STD_MODE = "sem"

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


def apply_x_axis_limits(ax, use_mm=True):
    if use_mm:
        x_min = X_LIM_MIN_MM
        x_max = X_LIM_MAX_MM
    else:
        x_min = X_LIM_MIN_THOU
        x_max = X_LIM_MAX_THOU

    if x_min is not None or x_max is not None:
        ax.set_xlim(x_min, x_max)


def configure_plot_style(use_latex=True):
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if use_latex and shutil.which("latex") is not None:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


def compute_stats(csv_path, use_mm=True):
    df = pd.read_csv(csv_path)

    # -------------------------
    # Y-axis unit conversion
    # -------------------------
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

    # -------------------------
    # X-axis unit conversion
    # -------------------------
    # Original CSV stores vibration amplitude as noise_mm.
    # If USE_MM is False, convert the x-axis from mm to thou.
    if use_mm:
        stats["noise_x"] = stats["noise_mm"]
    else:
        stats["noise_x"] = stats["noise_mm"] * THOU_PER_MM

    stats["std"] = stats["std"].fillna(0.0)

    # Standard error of the mean
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    # Absolute deviation of each std from the average std of that curve
    avg_std = stats["std"].mean()
    stats["std_from_avg_std"] = (stats["std"] - avg_std).abs()

    return stats


def print_stats_table(stats, label, loop_type, use_mm=True):
    unit = "mm" if use_mm else "thou"

    print("\n" + "=" * 80)
    print(f"Curve: {label}")
    print(f"Loop Type: {loop_type}")
    print(f"Units: {unit}")
    print(f"Shaded Region Mode: {STD_MODE}")
    print("=" * 80)
    print(
        f"noise_{unit}\tmean_{unit}\tstd_{unit}\tsem_{unit}\t"
        f"std_from_avg_std_{unit}"
    )

    for _, row in stats.iterrows():
        print(
            f"{row['noise_x']:.6f}\t"
            f"{row['mean']:.6f}\t"
            f"{row['std']:.6f}\t"
            f"{row['sem']:.6f}\t"
            f"{row['std_from_avg_std']:.6f}"
        )


def parse_args(args):
    if "--open-loop" in args:
        split_idx = args.index("--open-loop")
        closed_loop_pairs = args[:split_idx]
        open_loop_pairs = args[split_idx + 1:]
    else:
        closed_loop_pairs = args
        open_loop_pairs = []

    return closed_loop_pairs, open_loop_pairs


def get_std_column():
    valid_modes = ["std", "sem", "std_from_avg_std"]

    if STD_MODE not in valid_modes:
        raise ValueError(
            f"Invalid STD_MODE='{STD_MODE}'. "
            f"Expected one of: {valid_modes}"
        )

    return STD_MODE


def apply_y_axis_limits(ax, use_mm=True):
    if use_mm:
        y_min = Y_LIM_MIN_MM
        y_max = Y_LIM_MAX_MM
    else:
        y_min = Y_LIM_MIN_THOU
        y_max = Y_LIM_MAX_THOU

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)


def plot_group(
    ax,
    csv_label_pairs,
    loop_type,
    line_style,
    marker_style,
    use_mm=True,
):
    all_noise_vals = set()
    std_column = get_std_column()

    for i, pair in enumerate(csv_label_pairs):
        if ":" not in pair:
            print(f"Skipping invalid input: {pair}")
            continue

        csv_path, label = pair.split(":", 1)

        stats = compute_stats(csv_path, use_mm)
        print_stats_table(stats, label, loop_type, use_mm)

        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        x = stats["noise_x"].to_numpy()
        y = stats["mean"].to_numpy()
        err = stats[std_column].to_numpy()

        lower = np.maximum(y - err, 0.0)
        upper = y + err

        all_noise_vals.update(x.tolist())

        ax.plot(
            x,
            y,
            marker=marker_style,
            linestyle=line_style,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
            color=color,
        )

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=STD_ALPHA,
            linewidth=0,
        )

    return all_noise_vals


def plot_multiple(closed_loop_pairs, open_loop_pairs, use_mm=True):
    configure_plot_style(use_latex=USE_LATEX)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    if use_mm:
        y_label = "Absolute Mean Error (mm)"
        x_label = "Vibration Amplitude (mm)"
        unit_str = "mm"
    else:
        y_label = "Absolute Mean Error (thou)"
        x_label = "Vibration Amplitude (thou)"
        unit_str = "thou"

    all_noise_vals = set()

    all_noise_vals.update(
        plot_group(
            ax,
            closed_loop_pairs,
            loop_type="Closed Loop",
            line_style=":",
            marker_style="^",
            use_mm=use_mm,
        )
    )

    all_noise_vals.update(
        plot_group(
            ax,
            open_loop_pairs,
            loop_type="Open Loop",
            line_style="-",
            marker_style="o",
            use_mm=use_mm,
        )
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True)

    if all_noise_vals:
        ax.set_xticks(sorted(all_noise_vals))

    ax.relim()
    ax.autoscale_view()

    apply_y_axis_limits(ax, use_mm=use_mm)
    apply_x_axis_limits(ax, use_mm=use_mm)
    ax.set_title(f"Structured Vibration Amplitude Sweep: \n Compensated vs. Open Loop at {EXP_FREQ_HZ} Hz")

    ax.legend()

    fig.tight_layout()

    out_path = f"closed_vs_open_loop_error_vs_noise_{unit_str}_{STD_MODE}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python3 amplitude_sweep_plot.py "
            "closed1.csv:Label closed2.csv:Label closed3.csv:Label "
            "--open-loop open1.csv:Label open2.csv:Label open3.csv:Label"
        )
        sys.exit(1)

    closed_loop_pairs, open_loop_pairs = parse_args(sys.argv[1:])

    plot_multiple(
        closed_loop_pairs=closed_loop_pairs,
        open_loop_pairs=open_loop_pairs,
        use_mm=USE_MM,
    )