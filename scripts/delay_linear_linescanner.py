import os
import re
import csv
from glob import glob

import cv2
import numpy as np


"""
Batch Linear Line Scanner (folder -> per-image CSV + combined CSV)

This version matches the *working single-image* parameter set you validated:

ROI selection:
  - scale = 0.25 (UI only)

Preprocess:
  - Adaptive threshold: blockSize=11, C=2  (kept for consistency/debug)
  - Canny: low=30, high=120

HoughLinesP:
  - threshold=20
  - minLineLength=10
  - maxLineGap=4

Vertical filter:
  - angle tolerance = ±7 degrees

Clustering (merge duplicate Hough detections of same physical line):
  - cluster threshold = 20 px  (CRITICAL to avoid collapsing adjacent true lines)

Calibration:
  - THOU_PER_PIXEL = 1000/600
  - IDEAL_SPACING_THOU = 125.0

Filename parsing (expected):
  linear_8mm_adjusted1.jpg
  linear_8mm_unadjusted2.jpg
  rc_11mm_adjusted3.jpg
  rc_11mm_unadjusted1.jpg
"""


# =========================
# Config (MATCHES WORKING CODE)
# =========================

THOU_PER_PIXEL = 1000 / 600
IDEAL_SPACING_THOU = 433#125.0

# Canny (MATCH)
CANNY_LOW = 30
CANNY_HIGH = 120

# HoughLinesP (MATCH)
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 10
MAX_LINE_GAP = 4

# Near-vertical filter (MATCH)
ANGLE_TOL_DEG = 7

# Clustering threshold (MATCH)
CLUSTER_THRESH_PX = 20


# =========================
# Utility functions
# =========================

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def process_parent(parent_dir: str):
    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(f"Not a folder: {parent_dir}")

    delay_dirs = ["0ms", "10ms", "20ms"]

    combined_rows = []
    global_idx = 1

    # Put ONE output_csv at the parent level
    out_dir = os.path.join(parent_dir, "output_csv")
    os.makedirs(out_dir, exist_ok=True)

    for d in delay_dirs:
        folder_path = os.path.join(parent_dir, d)
        if not os.path.isdir(folder_path):
            print(f"WARNING: missing folder {folder_path}, skipping.")
            continue

        images = sorted(glob(os.path.join(folder_path, "*.jpg")))
        print(f"\nDelay folder {d}: found {len(images)} JPGs")

        for img_path in images:
            meta = parse_filename(img_path)  # now includes delay_ms

            # Optional: enforce adjusted only
            if meta["method"] != "adjusted":
                continue

            print(f"---\nProcessing {meta['base']} (delay {meta['delay_ms']} ms, trial {meta['trial']})")

            img = load_image(img_path)

            roi = select_roi(img)
            if roi is None:
                print("  Skipped (no ROI selected).")
                continue

            edges, binary, gray, roi_img = preprocess_roi(img, roi)

            x_candidates = detect_vertical_lines(edges)
            if not x_candidates:
                print("  WARNING: No vertical lines detected. Skipping.")
                continue

            averaged_x = cluster_x_positions(x_candidates)
            if len(averaged_x) < 2:
                print("  WARNING: <2 line clusters found. Skipping.")
                continue

            spacing_px, spacing_thou, delta_thou = compute_spacing_and_deviation(averaged_x)

            # Per-image CSV (still saved next to that delay folder)
            per_out_dir = os.path.join(folder_path, "output_csv")
            os.makedirs(per_out_dir, exist_ok=True)
            per_csv = os.path.join(per_out_dir, f"{meta['stem']}_deviation.csv")

            with open(per_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "image",
                    "case",
                    "noise_mm",
                    "method",
                    "delay_ms",
                    "trial",
                    "spacing_index_in_image",
                    "x1_px",
                    "x2_px",
                    "spacing_px",
                    "spacing_thou",
                    "delta_thou",
                    "abs_delta_thou",
                ])

                for i in range(len(spacing_px)):
                    row = [
                        meta["base"],
                        meta["case"],
                        meta["noise_mm"],
                        meta["method"],
                        meta["delay_ms"],
                        meta["trial"],
                        i + 1,
                        averaged_x[i],
                        averaged_x[i + 1],
                        spacing_px[i],
                        spacing_thou[i],
                        delta_thou[i],
                        abs(delta_thou[i]),
                    ]
                    w.writerow(row)

                    combined_rows.append([global_idx] + row)
                    global_idx += 1

            print(f"  Wrote: {per_csv}")

    # ONE combined CSV for ALL delays
    combined_csv = os.path.join(out_dir, "combined_deviation_all_delays.csv")
    with open(combined_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "global_spacing_index",
            "image",
            "case",
            "noise_mm",
            "method",
            "delay_ms",
            "trial",
            "spacing_index_in_image",
            "x1_px",
            "x2_px",
            "spacing_px",
            "spacing_thou",
            "delta_thou",
            "abs_delta_thou",
        ])
        w.writerows(combined_rows)

    print("\n==============================")
    print("DONE.")
    print(f"Combined CSV across 0/10/20ms:\n  {combined_csv}")
    print("==============================\n")


def parse_filename(path: str):
    """
    Expected examples:
      linear_8mm_adjusted1.jpg
      linear_8mm_unadjusted2.jpg
      rc_11mm_adjusted3.jpg
      rc_11mm_unadjusted1.jpg

    Returns:
      case: 'a' or 'c'
      noise_mm: 8 or 11
      method: 'adjusted' or 'unadjusted'
      trial: int
      stem: filename without extension
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)

    # m = re.match(r"^(linear|rc|cyl)_(8mm|11mm)_(adjusted|unadjusted)(\d+)$", stem, re.IGNORECASE)
    m = re.match(
        r"^(linear|rc|cyl)_(8mm|11mm)_(adjusted|unadjusted)_(0ms|10ms|20ms)_(\d+)$",
        stem,
        re.IGNORECASE
    )
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern: {base}\n"
            f"Expected like: linear_8mm_adjusted1.jpg"
        )

    prefix = m.group(1).lower()
    noise_str = m.group(2).lower()
    method = m.group(3).lower()
    delay_str = m.group(4).lower()      # "0ms"
    delay_ms = int(delay_str.replace("ms", ""))  # 0,10,20
    trial = int(m.group(5))

    case = "a" if prefix == "linear" else "c"
    noise_mm = 8 if noise_str.startswith("8") else 11

    return {
        "prefix": prefix,
        "case": case,
        "noise_mm": noise_mm,
        "method": method,
        "delay_ms": delay_ms,
        "trial": trial,
        "stem": stem,
        "base": base,
    }


def select_roi(img):
    """
    Shows a scaled image for faster ROI selection; maps ROI back to original pixels.
    ESC/Cancel -> returns None.
    """
    scale = 0.25
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (int(w * scale), int(h * scale)))

    roi_scaled = cv2.selectROI(
        "Select ROI (ENTER to confirm, ESC to skip image)",
        scaled,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if roi_scaled is None or roi_scaled[2] == 0 or roi_scaled[3] == 0:
        return None

    x, y, rw, rh = (int(roi_scaled[i] / scale) for i in range(4))
    return (x, y, rw, rh)


def preprocess_roi(img, roi):
    """
    Matches working code:
      - crop to ROI
      - grayscale
      - adaptive threshold (kept)
      - Canny with (30,120)
    Returns edges, binary, gray, roi_img
    """
    x, y, w, h = roi
    roi_img = img[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    return edges, binary, gray, roi_img


def detect_vertical_lines(edges):
    """
    Matches working code defaults:
      HoughLinesP(... threshold=20, minLineLength=10, maxLineGap=4)
      near-vertical filter with ANGLE_TOL_DEG
    Returns sorted x candidates (may contain duplicates for same physical line).
    """
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )

    if lines is None:
        return []

    x_candidates = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * (180 / np.pi)

        if (90 - ANGLE_TOL_DEG) <= angle <= (90 + ANGLE_TOL_DEG):
            x_center = (x1 + x2) // 2
            x_candidates.append(int(x_center))

    x_candidates.sort()
    return x_candidates


def cluster_x_positions(x_list, thresh_px=CLUSTER_THRESH_PX):
    """
    Matches working code logic:
      - cluster consecutive x detections if within threshold_distance
      - output one averaged x per cluster
    """
    if not x_list:
        return []

    x_list = sorted(x_list)
    averaged_positions = []
    temp_cluster = []

    for i, x in enumerate(x_list):
        if i == 0 or abs(x - x_list[i - 1]) < thresh_px:
            temp_cluster.append(x)
        else:
            if temp_cluster:
                averaged_positions.append(int(np.mean(temp_cluster)))
            temp_cluster = [x]

    if temp_cluster:
        averaged_positions.append(int(np.mean(temp_cluster)))

    return averaged_positions


def compute_spacing_and_deviation(averaged_x):
    """
    Computes pairwise spacing and signed deviation from IDEAL_SPACING_THOU.
    """
    spacing_px = []
    spacing_thou = []
    delta_thou = []

    for i in range(len(averaged_x) - 1):
        spx = averaged_x[i + 1] - averaged_x[i]
        sth = spx * THOU_PER_PIXEL
        dth = sth - IDEAL_SPACING_THOU

        spacing_px.append(spx)
        spacing_thou.append(sth)
        delta_thou.append(dth)

    return spacing_px, spacing_thou, delta_thou


# =========================
# Main batch processing
# =========================

def process_folder(folder_path: str):
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    images = sorted(glob(os.path.join(folder_path, "*.jpg")))
    if not images:
        raise FileNotFoundError(f"No JPG files found in: {folder_path}")

    out_dir = os.path.join(folder_path, "output_csv")
    os.makedirs(out_dir, exist_ok=True)

    combined_rows = []
    global_idx = 1

    print(f"\nFound {len(images)} JPGs in:\n  {folder_path}\n")

    for img_path in images:
        meta = parse_filename(img_path)
        print(f"---\nProcessing {meta['base']}  (case {meta['case']}, {meta['noise_mm']}mm, {meta['method']}, trial {meta['trial']})")

        img = load_image(img_path)

        roi = select_roi(img)
        if roi is None:
            print("  Skipped (no ROI selected).")
            continue

        edges, binary, gray, roi_img = preprocess_roi(img, roi)

        x_candidates = detect_vertical_lines(edges)
        if not x_candidates:
            print("  WARNING: No vertical lines detected. Skipping.")
            continue

        averaged_x = cluster_x_positions(x_candidates)
        if len(averaged_x) < 2:
            print("  WARNING: <2 line clusters found. Skipping.")
            continue

        spacing_px, spacing_thou, delta_thou = compute_spacing_and_deviation(averaged_x)

        # Write per-image CSV
        per_csv = os.path.join(out_dir, f"{meta['stem']}_deviation.csv")
        with open(per_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "image",
                "case",
                "noise_mm",
                "method",
                "trial",
                "spacing_index_in_image",
                "x1_px",
                "x2_px",
                "spacing_px",
                "spacing_thou",
                "delta_thou",
                "abs_delta_thou"
            ])

            for i in range(len(spacing_px)):
                row = [
                    meta["base"],
                    meta["case"],
                    meta["noise_mm"],
                    meta["method"],
                    meta["trial"],
                    i + 1,
                    averaged_x[i],
                    averaged_x[i + 1],
                    spacing_px[i],
                    spacing_thou[i],
                    delta_thou[i],
                    abs(delta_thou[i]),
                ]
                w.writerow(row)

                combined_rows.append([global_idx] + row)
                global_idx += 1

        print(f"  Wrote: {per_csv}")
        print(f"  Lines detected (clusters): {len(averaged_x)}  -> spacings: {len(spacing_px)}")

    # Write combined CSV
    combined_csv = os.path.join(out_dir, "combined_deviation_all_images.csv")
    with open(combined_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "global_spacing_index",
            "image",
            "case",
            "noise_mm",
            "method",
            "trial",
            "spacing_index_in_image",
            "x1_px",
            "x2_px",
            "spacing_px",
            "spacing_thou",
            "delta_thou",
            "abs_delta_thou"
        ])
        w.writerows(combined_rows)

    print("\n==============================")
    print("DONE.")
    print(f"Per-image + combined CSVs saved in:\n  {out_dir}")
    print(f"Combined CSV:\n  {combined_csv}")
    print("==============================\n")


if __name__ == "__main__":
    parent_dir = input('Enter parent folder path containing "0ms/10ms/20ms": ').strip().strip('"')
    process_parent(parent_dir)
