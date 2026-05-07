import os
import re
import csv
from glob import glob

import cv2
import numpy as np


"""
Batch Line Scanner (folder -> per-image CSV + combined CSV)

Expected filename format:
    [case]_[noise_mm]_[trial].jpg

Examples:
    cyl_12mm_1.jpg
    rc_8mm_2.jpg
    lin_11mm_3.jpg

Case mapping:
    lin -> a   (case a)
    cyl -> b   (case b)
    rc  -> c   (case c)

Interactive flow per image:
  1. Select ROI
  2. Detect / cluster vertical lines
  3. Show preview with averaged_x overlay
  4. Choose:
       r     -> redo scan for same image
       Esc   -> skip image
       Enter -> accept and save results
"""


# =========================
# Config
# =========================

THOU_PER_PIXEL = 1000 / 600
# IDEAL_SPACING_THOU = 669.291 #17mm  # change if needed
IDEAL_SPACING_THOU = 590.551 #15mm
# Canny
CANNY_LOW = 30
CANNY_HIGH = 120

# HoughLinesP
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 10
MAX_LINE_GAP = 4

# Near-vertical filter
ANGLE_TOL_DEG = 9

# Clustering threshold
CLUSTER_THRESH_PX = 12

# Preview display scale (UI only)
PREVIEW_SCALE = 0.75


# =========================
# Utility functions
# =========================

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def parse_filename(path: str):
    """
    Expected:
        [case]_[noise_mm]_[trial].jpg

    Examples:
        cyl_12mm_1.jpg
        rc_8mm_2.jpg
        lin_11mm_3.jpg
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)

    m = re.match(r"^(lin|cyl|rc)_(\d+)mm_(\d+)$", stem, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern: {base}\n"
            f"Expected like: cyl_12mm_1.jpg"
        )

    prefix = m.group(1).lower()
    noise_mm = int(m.group(2))
    trial = int(m.group(3))

    case_map = {
        "lin": "a",
        "cyl": "b",
        "rc": "c",
    }

    return {
        "prefix": prefix,
        "case": case_map[prefix],
        "noise_mm": noise_mm,
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
    cv2.destroyWindow("Select ROI (ENTER to confirm, ESC to skip image)")

    if roi_scaled is None or roi_scaled[2] == 0 or roi_scaled[3] == 0:
        return None

    x, y, rw, rh = (int(roi_scaled[i] / scale) for i in range(4))
    return (x, y, rw, rh)


def preprocess_roi(img, roi):
    """
    Crop to ROI, grayscale, adaptive threshold, Canny.
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
    Detect near-vertical lines with HoughLinesP.
    Returns sorted x candidates.
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
    Cluster nearby x detections and output one averaged x per cluster.
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


def make_preview_image(roi_img, averaged_x, x_candidates=None):
    """
    Draw averaged_x as green vertical lines.
    Optionally draw raw Hough x candidates as faint red lines.
    """
    if len(roi_img.shape) == 2:
        preview = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    else:
        preview = roi_img.copy()

    h, w = preview.shape[:2]

    if x_candidates is not None:
        for x in x_candidates:
            cv2.line(preview, (int(x), 0), (int(x), h - 1), (0, 0, 255), 1)

    for i, x in enumerate(averaged_x):
        cv2.line(preview, (int(x), 0), (int(x), h - 1), (0, 255, 0), 2)
        cv2.putText(
            preview,
            str(i + 1),
            (int(x) + 4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    instructions = "ENTER=accept | r=redo | ESC=skip"
    cv2.putText(
        preview,
        instructions,
        (10, max(25, h - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if PREVIEW_SCALE != 1.0:
        preview = cv2.resize(
            preview,
            (int(preview.shape[1] * PREVIEW_SCALE), int(preview.shape[0] * PREVIEW_SCALE))
        )

    return preview


def review_scan(roi_img, averaged_x, x_candidates=None):
    """
    Show preview and wait for:
      Enter -> accept
      r     -> redo
      Esc   -> skip

    Returns one of: 'accept', 'redo', 'skip'
    """
    window_name = "Scan Preview"
    preview = make_preview_image(roi_img, averaged_x, x_candidates=x_candidates)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)

    while True:
        key = cv2.waitKey(0) & 0xFF

        # Enter
        if key == 13 or key == 10:
            cv2.destroyWindow(window_name)
            return "accept"

        # r
        if key == ord('r') or key == ord('R'):
            cv2.destroyWindow(window_name)
            return "redo"

        # Esc
        if key == 27:
            cv2.destroyWindow(window_name)
            return "skip"


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
        try:
            meta = parse_filename(img_path)
        except ValueError as e:
            print(f"Skipping file: {os.path.basename(img_path)}")
            print(f"  Reason: {e}")
            continue

        print(
            f"---\nProcessing {meta['base']} "
            f"(prefix {meta['prefix']}, case {meta['case']}, "
            f"{meta['noise_mm']}mm, trial {meta['trial']})"
        )

        img = load_image(img_path)

        accepted = False
        skipped = False

        while True:
            roi = select_roi(img)
            if roi is None:
                print("  Skipped (no ROI selected).")
                skipped = True
                break

            edges, binary, gray, roi_img = preprocess_roi(img, roi)

            x_candidates = detect_vertical_lines(edges)
            if not x_candidates:
                print("  WARNING: No vertical lines detected.")
                action = review_scan(roi_img, [], x_candidates=[])
                if action == "redo":
                    continue
                elif action == "skip":
                    skipped = True
                    break
                else:
                    print("  Cannot accept: no lines detected. Redoing.")
                    continue

            averaged_x = cluster_x_positions(x_candidates)
            if len(averaged_x) < 2:
                print("  WARNING: <2 line clusters found.")
                action = review_scan(roi_img, averaged_x, x_candidates=x_candidates)
                if action == "redo":
                    continue
                elif action == "skip":
                    skipped = True
                    break
                else:
                    print("  Cannot accept: fewer than 2 averaged lines. Redoing.")
                    continue

            action = review_scan(roi_img, averaged_x, x_candidates=x_candidates)

            if action == "redo":
                continue
            elif action == "skip":
                print("  Skipped by user.")
                skipped = True
                break
            elif action == "accept":
                accepted = True
                break

        if skipped or not accepted:
            continue

        spacing_px, spacing_thou, delta_thou = compute_spacing_and_deviation(averaged_x)

        per_csv = os.path.join(out_dir, f"{meta['stem']}_deviation.csv")
        with open(per_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "image",
                "prefix",
                "case",
                "noise_mm",
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
                    meta["prefix"],
                    meta["case"],
                    meta["noise_mm"],
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
        print(f"  Lines detected (clusters): {len(averaged_x)} -> spacings: {len(spacing_px)}")

    combined_csv = os.path.join(out_dir, "combined_deviation_all_images.csv")
    with open(combined_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "global_spacing_index",
            "image",
            "prefix",
            "case",
            "noise_mm",
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

    cv2.destroyAllWindows()

    print("\n==============================")
    print("DONE.")
    print(f"Per-image + combined CSVs saved in:\n  {out_dir}")
    print(f"Combined CSV:\n  {combined_csv}")
    print("==============================\n")


if __name__ == "__main__":
    folder_path = input('Enter folder path containing JPG files: ').strip().strip('"')
    process_folder(folder_path)