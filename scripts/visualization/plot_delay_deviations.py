#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import MultipleLocator

DELAYS = [0, 10, 20]

def kde_curve(values, n=800):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    kde = gaussian_kde(values)
    x = np.linspace(0, values.max() * 1.1, n)
    y = kde(x)
    return x, y

def kde_mode(values, n=800):
    x, y = kde_curve(values, n=n)
    if x is None:
        return float("nan"), (None, None)
    mode_val = float(x[np.argmax(y)])
    return mode_val, (x, y)

def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "N": int(len(values)),
        "mean_thou": float(np.mean(values)),
        "std_thou": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "min_thou": float(np.min(values)),
        "p25_thou": float(q25),
        "median_thou": float(q50),
        "p75_thou": float(q75),
        "iqr_thou": float(q75 - q25),
        "p90_thou": float(np.percentile(values, 90)),
        "p95_thou": float(np.percentile(values, 95)),
        "max_thou": float(np.max(values)),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_delay_distributions_thou.py /path/to/combined.csv")
        sys.exit(1)

    combined_csv = sys.argv[1]
    if not os.path.exists(combined_csv):
        raise FileNotFoundError(f"Missing CSV:\n  {combined_csv}")

    out_dir = os.path.dirname(combined_csv)
    out_fig = os.path.join(out_dir, "kde_overlay_delays_thou_mean_mode.png")
    out_stats = os.path.join(out_dir, "delay_stats_thou_with_mode.csv")

    df = pd.read_csv(combined_csv)
    required = {"abs_delta_thou", "delay_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {missing}.\nColumns are: {list(df.columns)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    stats_rows = []

    for delay in DELAYS:
        vals = df.loc[df["delay_ms"] == delay, "abs_delta_thou"].to_numpy()
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            print(f"WARNING: no samples for delay={delay}ms. Skipping.")
            continue

        # Summary stats
        s = summarize(vals)
        mode_val, (x, y) = kde_mode(vals)

        s["delay_ms"] = delay
        s["mode_thou"] = float(mode_val)
        stats_rows.append(s)

        if x is None:
            print(f"WARNING: not enough samples for KDE delay={delay}ms (N={len(vals)}).")
            continue

        mean_val = float(np.mean(vals))

        # KDE curve
        (line,) = ax.plot(x, y, linewidth=2, label=f"{delay} ms (N={len(vals)})")
        ax.fill_between(x, y, alpha=0.20)

        # Mean line (dashed)
        # ax.axvline(mean_val, color=line.get_color(), linestyle="--", linewidth=2, alpha=0.9)

        # Mode line (dotted)
        # ax.axvline(mode_val, color=line.get_color(), linestyle=":", linewidth=2, alpha=0.9)

    ax.set_title("Absolute Error vs Additional Mocap Delay")
    ax.set_xlabel("Absolute Error (thou)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    ax.set_xlim(left=0)
    # Major ticks every 10 thou, minor ticks every 1 thou
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # Optional: make minor grid visible (light)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)

    # Optional: make minor tick marks shorter
    ax.tick_params(axis="x", which="minor", length=3, labelsize=8)
    ax.tick_params(axis="both", which="major", length=6, labelsize=12)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.show()
    print(f"[saved] {out_fig}")

    stats_df = pd.DataFrame(stats_rows).sort_values("delay_ms")
    stats_df.to_csv(out_stats, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\nPer-delay statistics (thou) incl. KDE mode:")
        print(stats_df)

    print(f"\n[saved] {out_stats}")

if __name__ == "__main__":
    main()