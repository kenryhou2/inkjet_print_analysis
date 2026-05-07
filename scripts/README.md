# Line Scanner Scripts

These scripts process line-scan JPG images into per-image deviation CSV files and combined CSV summaries. They are interactive OpenCV scripts: run the script from a terminal, enter the requested folder path, then select the ROI for each image in the OpenCV window.

## Requirements

Use the project Python environment with:

```bash
python3 -m pip install opencv-python numpy
```

The scripts open GUI windows for ROI selection and scan review, so they must be run in an environment with display access.

## Amplitude Sweep Line Scanner

Script:

```text
scripts/amplitude_sweep_linescanner.py
```

Expected input folder:

```text
<image_folder>/
  lin_8mm_1.jpg
  cyl_12mm_2.jpg
  rc_15mm_3.jpg
```

Filename format:

```text
<case>_<amplitude_mm>mm_<trial>.jpg
```

Valid cases:

```text
lin -> case a
cyl -> case b
rc  -> case c
```

Run:

```bash
cd scripts
python3 amplitude_sweep_linescanner.py
```

When prompted, enter the image folder:

```text
Enter folder path containing JPG files: /path/to/amplitude_sweep_images
```

Interactive controls:

```text
ROI window:     drag ROI, then Enter to confirm; Esc cancels/skips the image
Preview window: Enter accepts scan; r redoes the ROI/scan; Esc skips the image
```

Outputs are written under the input folder:

```text
<image_folder>/output_csv/
  <image_stem>_deviation.csv
  combined_deviation_all_images.csv
```

The combined CSV is the input used by the amplitude sweep visualization scripts.

## Delay Linear Line Scanner

Script:

```text
scripts/delay_linear_linescanner.py
```

Expected input parent folder:

```text
<delay_parent>/
  0ms/
    linear_8mm_adjusted_0ms_1.jpg
  10ms/
    linear_8mm_adjusted_10ms_1.jpg
  20ms/
    linear_8mm_adjusted_20ms_1.jpg
```

Filename format:

```text
<case>_<amplitude>_<method>_<delay>_<trial>.jpg
```

Valid values:

```text
case:      linear, rc, cyl
amplitude: 8mm, 11mm
method:    adjusted, unadjusted
delay:     0ms, 10ms, 20ms
```

Run:

```bash
cd scripts
python3 delay_linear_linescanner.py
```

When prompted, enter the parent folder containing `0ms`, `10ms`, and `20ms`:

```text
Enter parent folder path containing "0ms/10ms/20ms": /path/to/delay_parent
```

Notes:

- The current batch path processes only images whose parsed `method` is `adjusted`.
- Missing delay folders are skipped with a warning.
- ROI selection uses the same OpenCV flow as the amplitude sweep scanner.

Outputs:

```text
<delay_parent>/<delay_folder>/output_csv/
  <image_stem>_deviation.csv

<delay_parent>/output_csv/
  combined_deviation_all_delays.csv
```

The combined delay CSV is the input used by `visualization/plot_delay_deviations.py`.
