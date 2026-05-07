# Visualization CLI Usage

These scripts read CSV files produced by the line scanner scripts and write figure files in the current working directory or next to the input CSV, depending on the script.

Install the plotting dependencies in the project environment:

```bash
python3 -m pip install pandas matplotlib numpy scipy
```

Run commands from this directory unless you adjust the relative paths:

```bash
cd scripts/visualization
```

## Input Argument Format

Amplitude sweep plotting scripts expect each dataset as:

```text
/path/to/combined_deviation_all_images.csv:Legend Label
```

Quote the whole `csv:label` argument when the label contains spaces:

```bash
"/path/to/combined_deviation_all_images.csv:Roller Coaster"
```

## Amplitude Sweep Plot

Script:

```text
amplitude_sweep_plot.py
```

Usage:

```bash
python3 amplitude_sweep_plot.py \
  "/path/to/rc/output_csv/combined_deviation_all_images.csv:Roller Coaster" \
  "/path/to/cylinder/output_csv/combined_deviation_all_images.csv:Cylinder" \
  "/path/to/linear/output_csv/combined_deviation_all_images.csv:Linear"
```

Open-loop-only example:

```bash
python3 amplitude_sweep_plot.py \
  "/path/to/open_loop/10_Hz/rc/output_csv/combined_deviation_all_images.csv:Roller Coaster" \
  "/path/to/open_loop/10_Hz/cyl/output_csv/combined_deviation_all_images.csv:Cylinder" \
  "/path/to/open_loop/10_Hz/lin/output_csv/combined_deviation_all_images.csv:Linear"
```

Output:

```text
error_vs_noise_shaded_avgstd_mm.pdf
```

The script also prints a summary table for each curve. The `CLOSED_LOOP` constant inside the script controls whether the plot uses fixed closed-loop y-axis limits.

## Closed Loop vs Open Loop Comparison

Script:

```text
amplitude_sweep_closed_v_open_comparison.py
```

Usage:

```bash
python3 amplitude_sweep_closed_v_open_comparison.py \
  "/path/to/closed_loop/5_Hz/rc/output_csv/combined_deviation_all_images.csv:Roller Coaster-Comp." \
  "/path/to/closed_loop/5_Hz/cyl/output_csv/combined_deviation_all_images.csv:Cylinder-Comp." \
  "/path/to/closed_loop/5_Hz/lin/output_csv/combined_deviation_all_images.csv:Linear-Comp." \
  --open-loop \
  "/path/to/open_loop/5_Hz/rc/output_csv/combined_deviation_all_images.csv:Roller Coaster-OL" \
  "/path/to/open_loop/5_Hz/cyl/output_csv/combined_deviation_all_images.csv:Cylinder-OL" \
  "/path/to/open_loop/5_Hz/lin/output_csv/combined_deviation_all_images.csv:Linear-OL"
```

Figure-label example:

```bash
python3 amplitude_sweep_closed_v_open_comparison.py \
  "/path/to/closed_loop/15_Hz/rc/output_csv/combined_deviation_all_images.csv:Type III-Comp." \
  "/path/to/closed_loop/15_Hz/cyl/output_csv/combined_deviation_all_images.csv:Type II-Comp." \
  "/path/to/closed_loop/15_Hz/lin/output_csv/combined_deviation_all_images.csv:Type I-Comp." \
  --open-loop \
  "/path/to/open_loop/15_Hz/rc/output_csv/combined_deviation_all_images.csv:Type III-OL" \
  "/path/to/open_loop/15_Hz/cyl/output_csv/combined_deviation_all_images.csv:Type II-OL" \
  "/path/to/open_loop/15_Hz/lin/output_csv/combined_deviation_all_images.csv:Type I-OL"
```

10 Hz example with alternate bitmap subfolders:

```bash
python3 amplitude_sweep_closed_v_open_comparison.py \
  "/path/to/closed_loop/10_Hz/rc/output_csv/combined_deviation_all_images.csv:Type III-Comp." \
  "/path/to/closed_loop/10_Hz/cyl/output_csv/combined_deviation_all_images.csv:Type II-Comp." \
  "/path/to/closed_loop/10_Hz/lin/output_csv/combined_deviation_all_images.csv:Type I-Comp." \
  --open-loop \
  "/path/to/open_loop/10_Hz/rc/output_csv/combined_deviation_all_images.csv:Type III-OL" \
  "/path/to/open_loop/10_Hz/cyl/17mm_bitmap/output_csv/combined_deviation_all_images.csv:Type II-OL" \
  "/path/to/open_loop/10_Hz/lin/17mm_bitmap/output_csv/combined_deviation_all_images.csv:Type I-OL"
```

Output:

```text
closed_vs_open_loop_error_vs_noise_thou_sem.pdf
```

The script prints a summary table for every closed-loop and open-loop curve. Closed-loop inputs go before `--open-loop`; open-loop inputs go after it.

## Delay Deviation Distributions

Script:

```text
plot_delay_deviations.py
```

Usage:

```bash
python3 plot_delay_deviations.py /path/to/output_csv/combined_deviation_all_delays.csv
```

Outputs are saved next to the input CSV:

```text
/path/to/output_csv/kde_overlay_delays_thou_mean_mode.png
/path/to/output_csv/delay_stats_thou_with_mode.csv
```

The input CSV must include:

```text
abs_delta_thou
delay_ms
```
