import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# =========================
# Paths
# =========================
BASE_DIR = r"/home/hkou/work/cmu_biorobotics/boeing_inkjet/experiment/scripts/test_02082026"

CSV_MAP = {
    ("a", 8):  os.path.join(BASE_DIR, "linear_8mm", "output_csv", "combined_deviation_all_images.csv"),
    ("a", 11): os.path.join(BASE_DIR, "linear_11mm", "output_csv", "combined_deviation_all_images.csv"),

    ("b", 8):  os.path.join(BASE_DIR, "cyl_8mm",    "output_csv", "combined_deviation_all_images.csv"),
    ("b", 11): os.path.join(BASE_DIR, "cyl_11mm",   "output_csv", "combined_deviation_all_images.csv"),

    ("c", 8):  os.path.join(BASE_DIR, "rc_8mm",     "output_csv", "combined_deviation_all_images.csv"),
    ("c", 11): os.path.join(BASE_DIR, "rc_11mm",    "output_csv", "combined_deviation_all_images.csv"),
}


# =========================
# User Config
# =========================
USE_ABSOLUTE_DEVIATION = True   # True -> abs_delta_thou, False -> delta_thou

N_TOTAL = 200
RANDOM_SEED = 42
SAMPLE_WITH_REPLACEMENT = False

USE_LATEX = False
SAVE_PDF = True
SAVE_PNG = True

FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 4.5
DPI = 300

LINE_WIDTH = 1.0
FILL_ALPHA = 0.25
ZERO_LINE_WIDTH = 0.8

X_TICK_SPACING_MM = 1.0


# =========================
# Constants
# =========================
THOU_TO_MM = 0.0254
N_PER_METHOD = N_TOTAL // 2

rng = np.random.default_rng(RANDOM_SEED)

if USE_ABSOLUTE_DEVIATION:
    DEVIATION_COL_THOU = "abs_delta_thou"
    DEVIATION_COL_MM = "abs_delta_mm"
    X_LABEL = "Absolute Error (mm)"
    OUTPUT_SUFFIX = "abs"
else:
    DEVIATION_COL_THOU = "delta_thou"
    DEVIATION_COL_MM = "signed_delta_mm"
    X_LABEL = "Error (mm)"
    OUTPUT_SUFFIX = "signed"

OUT_FIG_PNG = os.path.join(BASE_DIR, f"figure10_2x3_mm_N200_{OUTPUT_SUFFIX}_latex.png")
OUT_FIG_PDF = os.path.join(BASE_DIR, f"figure10_2x3_mm_N200_{OUTPUT_SUFFIX}_latex.pdf")
OUT_TABLE = os.path.join(BASE_DIR, f"table1_2x3_mm_N200_{OUTPUT_SUFFIX}.csv")


# =========================
# Plot Style
# =========================
COLOR_ADJUSTED = "#0072B2"
COLOR_UNADJ = "#E69F00"


def configure_plot_style(use_latex=True):
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,

        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,

        "axes.linewidth": 0.8,
        "lines.linewidth": LINE_WIDTH,

        "grid.alpha": 0.3,
        "legend.frameon": False,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    if use_latex and shutil.which("latex") is not None:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


# =========================
# Helpers
# =========================
def case_to_label(case):
    if case == "a":
        return "Type I"
    if case == "b":
        return "Type II"
    if case == "c":
        return "Type III"
    return str(case)


def kde_curve(values_mm, n=500):
    values_mm = np.asarray(values_mm, dtype=float)
    values_mm = values_mm[np.isfinite(values_mm)]

    if len(values_mm) < 2:
        return None, None

    kde = gaussian_kde(values_mm)

    if USE_ABSOLUTE_DEVIATION:
        x_min = 0.0
        x_max = values_mm.max() * 1.1
    else:
        x_range = values_mm.max() - values_mm.min()
        pad = 0.1 * x_range if x_range > 0 else 1.0
        x_min = values_mm.min() - pad
        x_max = values_mm.max() + pad

    x = np.linspace(x_min, x_max, n)
    y = kde(x)

    return x, y


def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)


def uniform_sample(values, n, label=""):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f"No finite samples available for {label}")

    if len(values) < n and not SAMPLE_WITH_REPLACEMENT:
        print(
            f"[warning] Only {len(values)} samples available for {label}. "
            f"Using all available samples instead of {n}."
        )
        return values

    return rng.choice(values, size=n, replace=SAMPLE_WITH_REPLACEMENT)


def improvement_percent(baseline, adjusted):
    if USE_ABSOLUTE_DEVIATION:
        return 100 * (np.mean(baseline) - np.mean(adjusted)) / np.mean(baseline)

    denom = abs(np.mean(baseline))
    if denom < 1e-12:
        return float("nan")

    return 100 * (abs(np.mean(baseline)) - abs(np.mean(adjusted))) / denom


# =========================
# Load + sample data
# =========================
configure_plot_style(use_latex=USE_LATEX)

data = {}
table_rows = []

for (case, noise), path in CSV_MAP.items():
    case_label = case_to_label(case)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found for case ({case_label}), {noise}mm:\n  {path}"
        )

    df = pd.read_csv(path)

    if "method" not in df.columns:
        raise ValueError(f"CSV missing required column 'method':\n  {path}")

    if DEVIATION_COL_THOU not in df.columns:
        raise ValueError(
            f"CSV missing required column '{DEVIATION_COL_THOU}'.\n"
            f"For signed deviation mode, the CSV must contain 'delta_thou'.\n"
            f"Path:\n  {path}"
        )

    df[DEVIATION_COL_MM] = df[DEVIATION_COL_THOU] * THOU_TO_MM

    adj_all = df[df["method"] == "adjusted"][DEVIATION_COL_MM].to_numpy()
    unadj_all = df[df["method"] == "unadjusted"][DEVIATION_COL_MM].to_numpy()

    adj = uniform_sample(
        adj_all,
        N_PER_METHOD,
        label=f"{case_label}, {noise}mm, Compensated"
    )

    unadj = uniform_sample(
        unadj_all,
        N_PER_METHOD,
        label=f"{case_label}, {noise}mm, Open Loop"
    )

    data[(case, noise)] = {
        "adjusted": adj,
        "unadjusted": unadj
    }

    mean_unadj = np.mean(unadj)
    mean_adj = np.mean(adj)

    table_rows.append({
        "Case": case_label,
        "Max Vibration Amplitude (mm)": noise,
        "Quantity": "Absolute Deviation" if USE_ABSOLUTE_DEVIATION else "Signed Deviation",
        "N": len(adj) + len(unadj),
        "N Compensated": len(adj),
        "N Open Loop": len(unadj),

        "Mean Open Loop (mm)": float(mean_unadj),
        "Mean Compensated (mm)": float(mean_adj),
        "Mean Improvement (%)": float(improvement_percent(unadj, adj)),

        "Mean Abs Open Loop (mm)": float(np.mean(np.abs(unadj))),
        "Mean Abs Compensated (mm)": float(np.mean(np.abs(adj))),

        "STD Open Loop (mm)": float(np.std(unadj, ddof=1)) if len(unadj) > 1 else float("nan"),
        "STD Compensated (mm)": float(np.std(adj, ddof=1)) if len(adj) > 1 else float("nan"),

        "IQR Open Loop (mm)": float(iqr(unadj)),
        "IQR Compensated (mm)": float(iqr(adj)),

        "Median Open Loop (mm)": float(np.median(unadj)),
        "Median Compensated (mm)": float(np.median(adj)),
    })


# =========================
# Figure 10: 2x3 KDE plots
# =========================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
    sharey=True,
)
fig.suptitle(
    "Unstructured Vibration Distributions of Error: \n Compensated vs. Open Loop",
    fontsize=10,
    y=0.9
)
case_order = ["a", "b", "c"]
noise_order = [8, 11]

legend_handles = [
    Patch(
        facecolor=COLOR_ADJUSTED,
        edgecolor=COLOR_ADJUSTED,
        linewidth=LINE_WIDTH,
        alpha=FILL_ALPHA,
        label="Compensated"
    ),
    Patch(
        facecolor=COLOR_UNADJ,
        edgecolor=COLOR_UNADJ,
        linewidth=LINE_WIDTH,
        alpha=FILL_ALPHA,
        label="Open Loop"
    ),
]

for col, case in enumerate(case_order):
    for row, noise in enumerate(noise_order):
        ax = axes[row, col]
        d = data[(case, noise)]

        x, y = kde_curve(d["adjusted"])
        if x is not None:
            ax.plot(x, y, color=COLOR_ADJUSTED, linewidth=LINE_WIDTH)
            ax.fill_between(x, y, color=COLOR_ADJUSTED, alpha=FILL_ALPHA, linewidth=0)

        x, y = kde_curve(d["unadjusted"])
        if x is not None:
            ax.plot(x, y, color=COLOR_UNADJ, linewidth=LINE_WIDTH)
            ax.fill_between(x, y, color=COLOR_UNADJ, alpha=FILL_ALPHA, linewidth=0)

        if not USE_ABSOLUTE_DEVIATION:
            ax.axvline(
                0.0,
                color="black",
                linewidth=ZERO_LINE_WIDTH,
                linestyle="--",
                alpha=0.5,
            )

        N = len(d["adjusted"]) + len(d["unadjusted"])
        case_label = case_to_label(case)

        ax.set_title(f"{case_label}, {noise}mm Max Vibration\n$N={N}$")
        ax.set_xlabel(X_LABEL)

        # Force x-axis labels/ticks to increment by 1 mm
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SPACING_MM))

        if col == 0:
            ax.set_ylabel("Probability Density")

        ax.grid(True)
        ax.legend(handles=legend_handles, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, 0.93])

if SAVE_PNG:
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight")
    print(f"[saved] {OUT_FIG_PNG}")

if SAVE_PDF:
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    print(f"[saved] {OUT_FIG_PDF}")

plt.show()


# =========================
# Table I
# =========================
table_df = pd.DataFrame(table_rows).sort_values(
    by=["Max Vibration Amplitude (mm)", "Case"]
)

table_df.to_csv(OUT_TABLE, index=False)

print(f"[saved] {OUT_TABLE}")

print("\nTab-separated stats table:\n")
print(table_df.to_csv(sep="\t", index=False))