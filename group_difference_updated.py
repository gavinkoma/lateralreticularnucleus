#!/usr/bin/env python3
# ---- slit-aligned kinematics plotter (MCP + Wrist + Elbow Angle) ----
# 2x2 design:
#   phase: pre_ablate vs post_ablate
#   control_experimental: control vs experimental
#
# Outputs:
#  (1) "All four curves" plots (4 mean±SEM traces): control-pre, control-post, exp-pre, exp-post
#      - For each part:
#          * mcp, wrist: 2 rows (X_mm, Y_mm), pellet-centered, aligned to pass_slit (t=0)
#          * elbow_angle: 1 row (angle_deg), aligned to pass_slit (t=0)
#      - Optional faint individual trial traces
#      - Smoothed mean/SEM shading (visualization only)
#
#  (2) Delta plots (post - pre) for each group (control vs experimental)
#      - For each part:
#          * mcp, wrist: 2 rows (ΔX_mm, ΔY_mm)
#          * elbow_angle: 1 row (Δangle_deg)
#      - Prefer *paired* delta across subjects (best), falling back to unpaired mean-difference delta
#
# Key behaviors:
# - Reads metadata XLSX (must include: filename_no_ext, hand_lift_off, pass_slit, camera_label, phase, control_experimental)
# - Optionally uses pellet_post_x, pellet_post_y, mm_per_px to pellet-center and convert to mm for mcp/wrist.
#   If pellet/mm fields are missing for a given row, we will still compute elbow_angle but skip pellet-centered coords.
# - Finds each cleaned DLC CSV in CLEANED_DIR using filename_no_ext + CLEANED_SUFFIX
# - Mirrors LEFT camera (cam4) into RIGHT coordinates: x' = IMG_W - x
# - Windowing:
#     end = pass_slit + post_frames
#     start = max(hand_lift_off, pass_slit - pre_frames)   (if pre_frames is set)
#     otherwise start = min(hand_lift_off, pass_slit)      (original behavior)
# - Time index: t = frame - pass_slit
# - Bands: SEM (mean ± SEM), smoothed for visualization

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

# =========================
# DEFAULT USER CONFIG (override with CLI)
# =========================
DEFAULT_META_XLSX = "/Users/gavin/Documents/lateralreticularnucleus/compiled_metadata_sorted_cleaned.xlsx"

DEFAULT_EXAMPLE_CLEANED_CSV = "/Users/gavin/Documents/lateralreticularnucleus/analyzed_videos/iter03/postablate_cam4_1_9_26_5B-rat6_2026-01-09_14_46_19_camcommerb4DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"
DEFAULT_CLEANED_DIR = os.path.dirname(DEFAULT_EXAMPLE_CLEANED_CSV)

DEFAULT_CLEANED_SUFFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"

# Camera geometry (for mirroring)
DEFAULT_IMG_W, DEFAULT_IMG_H = 2048, 700

# Windowing defaults
DEFAULT_POST_FRAMES = 100
DEFAULT_PRE_FRAMES = None  # None => start=min(hand_lift_off, pass_slit)

# Which camera labels correspond to LEFT vs RIGHT view
LEFT_CAMERA_TOKENS = ["camcommerb4", "cam4"]   # left side view
RIGHT_CAMERA_TOKENS = ["camcommerb5", "cam5"]  # right side view

# Parts to plot
PARTS = ["mcp", "wrist", "elbow_angle"]
RIGHT_PREFIX = "right_"
LEFT_PREFIX = "left_"

# Joint bodyparts for elbow angle
# (these names MUST match your DLC bodypart names)
RIGHT_SHOULDER = "right_shoulder"
RIGHT_ELBOW = "right_elbow"
RIGHT_WRIST = "right_wrist"

LEFT_SHOULDER = "left_shoulder"
LEFT_ELBOW = "left_elbow"
LEFT_WRIST = "left_wrist"

# Group labels in metadata
PHASE_PRE = "pre_ablate"
PHASE_POST = "post_ablate"
GROUP_CONTROL = "control"
GROUP_EXPERIMENTAL = "experimental"

# Plot options
DEFAULT_PLOT_INDIVIDUAL_TRIALS = True
DEFAULT_INDIVIDUAL_ALPHA = 0.15
DEFAULT_INDIVIDUAL_LW = 0.8

DEFAULT_MEAN_LW = 2.5
DEFAULT_SEM_ALPHA = 0.25

# Smoothing (visualization only)
DEFAULT_SMOOTH_SIGMA = 2

# Debug verbosity
DEFAULT_DEBUG_FIRST_N = 3
DEFAULT_PRINT_SKIPS = 20

# Optional output
DEFAULT_SAVE_ALIGNED_TABLE = False


# =========================
# HELPERS
# =========================
def smooth_curve(y, sigma=2):
    """Gaussian smoothing for visualization only."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or sigma is None or sigma <= 0:
        return y
    if np.all(np.isnan(y)):
        return y

    nans = np.isnan(y)
    if np.any(nans):
        x = np.arange(len(y))
        y2 = y.copy()
        y2[nans] = np.interp(x[nans], x[~nans], y[~nans])
        ys = gaussian_filter1d(y2, sigma=sigma)
        ys[nans] = np.nan
        return ys

    return gaussian_filter1d(y, sigma=sigma)


def detect_side(camera_label: str, filename_no_ext: str) -> str:
    blob = f"{camera_label} {filename_no_ext}".lower()
    if any(tok in blob for tok in LEFT_CAMERA_TOKENS):
        return "left"
    if any(tok in blob for tok in RIGHT_CAMERA_TOKENS):
        return "right"
    raise ValueError(f"Could not infer side from camera_label='{camera_label}' / filename='{filename_no_ext}'")


def build_cleaned_path(cleaned_dir: str, filename_no_ext: str, cleaned_suffix: str) -> str:
    return os.path.join(cleaned_dir, str(filename_no_ext) + cleaned_suffix)


def read_dlc_multiindex_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def pick_scorer(df: pd.DataFrame) -> str:
    return pd.unique(df.columns.get_level_values(0))[0]


def extract_xy(df: pd.DataFrame, bodypart: str):
    scorer = pick_scorer(df)
    sub = df.loc[:, (scorer, bodypart, ["x", "y"])]
    x = pd.to_numeric(sub[(scorer, bodypart, "x")], errors="coerce")
    y = pd.to_numeric(sub[(scorer, bodypart, "y")], errors="coerce")
    return x, y


def slice_by_frame_iloc(series_a: pd.Series, series_b: pd.Series, start: int, end: int):
    """
    Robust slicing by positional frame number.
    Returns sliced series_a, series_b plus clamped start/end actually used.
    """
    start = int(start)
    end = int(end)
    n = len(series_a)
    if n == 0:
        return series_a.iloc[0:0], series_b.iloc[0:0], start, end

    start2 = max(0, min(start, n - 1))
    end2 = max(0, min(end, n - 1))

    if start2 > end2:
        return series_a.iloc[0:0], series_b.iloc[0:0], start2, end2

    return series_a.iloc[start2:end2 + 1], series_b.iloc[start2:end2 + 1], start2, end2


def normalize_subject_key(filename_no_ext: str) -> str:
    """
    Best-effort subject key from filename_no_ext.
    Removes leading pre/post ablate tokens so pre & post versions map to the same key.
    """
    s = str(filename_no_ext)

    s = re.sub(r'^(pre[_-]?ablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(post[_-]?ablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(preablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(postablate[_-]?)', '', s, flags=re.IGNORECASE)

    s = re.sub(r'^(pre[_-]?ablation[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(post[_-]?ablation[_-]?)', '', s, flags=re.IGNORECASE)

    return s


def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def compute_elbow_angle_deg(df: pd.DataFrame, side: str, img_w: int) -> pd.Series:
    """
    Compute elbow angle (degrees) from shoulder–elbow–wrist markers.
    If side=='left', mirror X for all three markers so angles are computed
    in the same right-facing coordinate frame as the position traces.
    """
    bodyparts = set(df.columns.get_level_values(1))

    if side == "right":
        sh, el, wr = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    else:
        sh, el, wr = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST

    missing = [bp for bp in (sh, el, wr) if bp not in bodyparts]
    if missing:
        raise RuntimeError(f"Missing markers for elbow_angle: {missing}")

    sx, sy = extract_xy(df, sh)
    ex, ey = extract_xy(df, el)
    wx, wy = extract_xy(df, wr)

    if side == "left":
        sx = img_w - sx
        ex = img_w - ex
        wx = img_w - wx

    # vectors (S - E) and (W - E)
    v1x = sx - ex
    v1y = sy - ey
    v2x = wx - ex
    v2y = wy - ey

    dot = v1x * v2x + v1y * v2y
    mag1 = np.sqrt(v1x**2 + v1y**2)
    mag2 = np.sqrt(v2x**2 + v2y**2)

    # avoid divide-by-zero
    denom = mag1 * mag2
    cos_theta = dot / denom
    cos_theta = cos_theta.where(np.isfinite(cos_theta), np.nan)

    # clip for numerical stability
    cos_theta = np.clip(cos_theta.to_numpy(dtype=float), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    return pd.Series(angle, index=sx.index)


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Slit-aligned kinematics plotter with pellet-centering + elbow angle.")
    ap.add_argument("--meta_xlsx", default=DEFAULT_META_XLSX)
    ap.add_argument("--cleaned_dir", default=DEFAULT_CLEANED_DIR)
    ap.add_argument("--cleaned_suffix", default=DEFAULT_CLEANED_SUFFIX)

    ap.add_argument("--img_w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--img_h", type=int, default=DEFAULT_IMG_H)

    # Window control
    ap.add_argument(
        "--pre_frames",
        type=int,
        default=DEFAULT_PRE_FRAMES,
        help="If set: start = max(hand_lift_off, pass_slit - pre_frames). If not set: start = min(hand_lift_off, pass_slit).",
    )
    ap.add_argument(
        "--post_frames",
        type=int,
        default=DEFAULT_POST_FRAMES,
        help="end = pass_slit + post_frames (default 100).",
    )

    # Display-only zoom (does NOT change aggregation)
    ap.add_argument("--zoom_pre", type=int, default=None, help="Display-only zoom: xlim left = -zoom_pre.")
    ap.add_argument("--zoom_post", type=int, default=None, help="Display-only zoom: xlim right = +zoom_post.")

    # Plot options
    ap.add_argument("--plot_individual_trials", action="store_true", default=DEFAULT_PLOT_INDIVIDUAL_TRIALS)
    ap.add_argument("--no_plot_individual_trials", action="store_true", default=False)
    ap.add_argument("--individual_alpha", type=float, default=DEFAULT_INDIVIDUAL_ALPHA)
    ap.add_argument("--individual_lw", type=float, default=DEFAULT_INDIVIDUAL_LW)

    ap.add_argument("--mean_lw", type=float, default=DEFAULT_MEAN_LW)
    ap.add_argument("--sem_alpha", type=float, default=DEFAULT_SEM_ALPHA)
    ap.add_argument("--smooth_sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)

    ap.add_argument("--debug_first_n", type=int, default=DEFAULT_DEBUG_FIRST_N)
    ap.add_argument("--print_skips", type=int, default=DEFAULT_PRINT_SKIPS)

    ap.add_argument("--save_aligned_table", action="store_true", default=DEFAULT_SAVE_ALIGNED_TABLE)
    ap.add_argument("--out_aligned_csv", default=None)

    return ap.parse_args()


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    META_XLSX = args.meta_xlsx
    CLEANED_DIR = args.cleaned_dir
    CLEANED_SUFFIX = args.cleaned_suffix
    IMG_W, IMG_H = args.img_w, args.img_h

    PRE_FRAMES = args.pre_frames
    POST_FRAMES = args.post_frames

    PLOT_INDIVIDUAL_TRIALS = args.plot_individual_trials and (not args.no_plot_individual_trials)
    INDIVIDUAL_ALPHA = args.individual_alpha
    INDIVIDUAL_LW = args.individual_lw

    MEAN_LW = args.mean_lw
    SEM_ALPHA = args.sem_alpha
    SMOOTH_SIGMA = args.smooth_sigma

    DEBUG_FIRST_N = args.debug_first_n
    PRINT_SKIPS = args.print_skips

    SAVE_ALIGNED_TABLE = args.save_aligned_table
    OUT_ALIGNED_CSV = args.out_aligned_csv
    if OUT_ALIGNED_CSV is None:
        OUT_ALIGNED_CSV = os.path.join(
            CLEANED_DIR,
            "slit_aligned_mcp_wrist_elbowangle_long_pellet_centered_mm.csv",
        )

    ZOOM_PRE = args.zoom_pre
    ZOOM_POST = args.zoom_post

    # -------------------------
    # LOAD METADATA
    # -------------------------
    meta = pd.read_excel(META_XLSX).copy()
    meta["meta_row"] = meta.index

    required_cols = [
        "filename_no_ext",
        "hand_lift_off",
        "pass_slit",
        "camera_label",
        "phase",
        "control_experimental",
    ]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise KeyError(f"Metadata missing required columns: {missing}\nFound: {list(meta.columns)}")

    # pellet/mm columns are optional per-row (we'll skip mcp/wrist if missing)
    has_pellet = all(c in meta.columns for c in ["pellet_post_x", "pellet_post_y"])
    has_mm = "mm_per_px" in meta.columns

    records = []
    skipped = []
    loaded_debug = 0

    for _, row in meta.iterrows():
        i = int(row["meta_row"])
        fname_base = str(row["filename_no_ext"])
        cam_label = str(row["camera_label"])
        phase = str(row["phase"]).strip() if not pd.isna(row["phase"]) else ""
        group = str(row["control_experimental"]).strip() if not pd.isna(row["control_experimental"]) else ""
        subject_key = normalize_subject_key(fname_base)

        if pd.isna(row["hand_lift_off"]) or pd.isna(row["pass_slit"]):
            skipped.append((i, fname_base, "NaN hand_lift_off/pass_slit", None))
            continue

        hand = int(row["hand_lift_off"])
        slit = int(row["pass_slit"])

        end = int(slit + POST_FRAMES)
        if PRE_FRAMES is None:
            start = min(hand, slit)
        else:
            start = max(hand, int(slit - PRE_FRAMES))

        try:
            side = detect_side(cam_label, fname_base)
        except Exception as e:
            skipped.append((i, fname_base, f"side infer failed: {repr(e)}", None))
            continue

        cleaned_path = build_cleaned_path(CLEANED_DIR, fname_base, CLEANED_SUFFIX)
        if not os.path.exists(cleaned_path):
            skipped.append((i, fname_base, "missing cleaned csv", cleaned_path))
            continue

        # read pellet/mm if available
        pellet_x = safe_float(row["pellet_post_x"]) if has_pellet else None
        pellet_y = safe_float(row["pellet_post_y"]) if has_pellet else None
        mm_per_px = safe_float(row["mm_per_px"]) if has_mm else None

        try:
            df = read_dlc_multiindex_csv(cleaned_path)
            scorer = pick_scorer(df)
            bodyparts = set(df.columns.get_level_values(1))

            if loaded_debug < DEBUG_FIRST_N:
                print(f"\n--- DEBUG row {i} ---")
                print("trial:", fname_base)
                print("subject_key:", subject_key)
                print("phase:", phase, "| group:", group)
                print("camera_label:", cam_label, "| inferred side:", side)
                print("cleaned_path:", cleaned_path)
                print("frames (rows):", len(df))
                print("hand_lift_off:", hand, "pass_slit:", slit, "start_used:", start, "end_used:", end)
                print("scorer:", scorer)
                print("pellet_post_x/y:", pellet_x, pellet_y, "| mm_per_px:", mm_per_px)
                loaded_debug += 1

            for part in PARTS:
                # -------------------------
                # ELBOW ANGLE (deg)
                # -------------------------
                if part == "elbow_angle":
                    try:
                        angle_deg = compute_elbow_angle_deg(df, side=side, img_w=IMG_W)
                        a_s, _, start_used, end_used = slice_by_frame_iloc(
                            angle_deg.reset_index(drop=True),
                            angle_deg.reset_index(drop=True),
                            start, end
                        )
                        if a_s.empty:
                            skipped.append((i, fname_base, "empty slice for elbow_angle (range issue)", cleaned_path))
                            continue

                        frames = np.arange(start_used, end_used + 1, dtype=int)
                        t = frames - slit

                        tmp = pd.DataFrame({
                            "meta_row": i,
                            "trial": fname_base,
                            "subject_key": subject_key,
                            "phase": phase,
                            "control_experimental": group,
                            "camera_label": cam_label,
                            "side_original": side,
                            "part": "elbow_angle",
                            "frame": frames,
                            "t": t,
                            "angle_deg": a_s.to_numpy(dtype=float),
                            "hand_lift_off": hand,
                            "pass_slit": slit,
                            "end_frame": end,
                            "cleaned_path": cleaned_path,
                        })
                        records.append(tmp)
                    except Exception as e:
                        skipped.append((i, fname_base, f"elbow_angle failed: {repr(e)}", cleaned_path))
                    continue

                # -------------------------
                # MCP / WRIST (pellet-centered mm)
                # -------------------------
                if pellet_x is None or pellet_y is None or mm_per_px is None:
                    skipped.append((i, fname_base, f"missing pellet/mm -> skipping {part}", cleaned_path))
                    continue

                if side == "right":
                    bp = RIGHT_PREFIX + part
                    if bp not in bodyparts:
                        skipped.append((i, fname_base, f"missing bodypart {bp}", cleaned_path))
                        continue
                    x_px, y_px = extract_xy(df, bp)
                    pellet_x_px = pellet_x
                    pellet_y_px = pellet_y
                else:
                    bp = LEFT_PREFIX + part
                    if bp not in bodyparts:
                        skipped.append((i, fname_base, f"missing bodypart {bp}", cleaned_path))
                        continue
                    x_px, y_px = extract_xy(df, bp)

                    # mirror into right-facing coordinates
                    x_px = IMG_W - x_px
                    # mirror pellet x the same way
                    pellet_x_px = IMG_W - pellet_x
                    pellet_y_px = pellet_y

                xs_px, ys_px, start_used, end_used = slice_by_frame_iloc(
                    x_px.reset_index(drop=True),
                    y_px.reset_index(drop=True),
                    start, end
                )

                if xs_px.empty:
                    skipped.append((i, fname_base, f"empty slice for {bp} (range issue)", cleaned_path))
                    continue

                frames = np.arange(start_used, end_used + 1, dtype=int)
                t = frames - slit

                x_mm = (xs_px.to_numpy(dtype=float) - float(pellet_x_px)) * float(mm_per_px)
                y_mm = -(ys_px.to_numpy(dtype=float) - float(pellet_y_px)) * float(mm_per_px)

                tmp = pd.DataFrame({
                    "meta_row": i,
                    "trial": fname_base,
                    "subject_key": subject_key,
                    "phase": phase,
                    "control_experimental": group,
                    "camera_label": cam_label,
                    "side_original": side,
                    "part": part,
                    "frame": frames,
                    "t": t,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "hand_lift_off": hand,
                    "pass_slit": slit,
                    "end_frame": end,
                    "pellet_x_px_used": float(pellet_x_px),
                    "pellet_y_px_used": float(pellet_y_px),
                    "mm_per_px": float(mm_per_px),
                    "cleaned_path": cleaned_path,
                })
                records.append(tmp)

        except Exception as e:
            skipped.append((i, fname_base, f"exception: {repr(e)}", cleaned_path))
            continue

    aligned = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    print("\n=========================")
    print("Aligned trials:", aligned["trial"].nunique() if not aligned.empty else 0)
    print("Aligned rows:", len(aligned))
    if not aligned.empty:
        print("\nParts counts (rows):")
        print(aligned["part"].value_counts(dropna=False))
        print("\nPhase counts (rows):")
        print(aligned["phase"].value_counts(dropna=False))
        print("\nGroup counts (rows):")
        print(aligned["control_experimental"].value_counts(dropna=False))
    print("\nSkipped:", len(skipped))
    if skipped:
        print("\nFirst skipped reasons:")
        for item in skipped[:PRINT_SKIPS]:
            print(item)

    if SAVE_ALIGNED_TABLE and not aligned.empty:
        aligned.to_csv(OUT_ALIGNED_CSV, index=False)
        print("\nSaved aligned long table to:", OUT_ALIGNED_CSV)

    # =========================
    # PLOTTING
    # =========================
    if aligned.empty:
        print("\n[ERROR] aligned is empty -> nothing to plot. Check skipped reasons above.")
        return

    COLOR_BY_GROUP = {
        GROUP_CONTROL: "royalblue",
        GROUP_EXPERIMENTAL: "firebrick",
    }
    LS_BY_PHASE = {
        PHASE_PRE: "-",
        PHASE_POST: "--",
    }

    def apply_display_zoom(ax_list):
        if ZOOM_PRE is None and ZOOM_POST is None:
            return
        cur_left, cur_right = ax_list[0].get_xlim()
        left = -int(ZOOM_PRE) if ZOOM_PRE is not None else cur_left
        right = int(ZOOM_POST) if ZOOM_POST is not None else cur_right
        for ax in ax_list:
            ax.set_xlim(left, right)

    def plot_four_curves(aligned_df: pd.DataFrame, part: str):
        d = aligned_df[aligned_df["part"] == part].copy()
        if d.empty:
            print(f"[WARN] No data for part='{part}'.")
            return

        d = d[
            d["phase"].isin([PHASE_PRE, PHASE_POST]) &
            d["control_experimental"].isin([GROUP_CONTROL, GROUP_EXPERIMENTAL])
        ].copy()
        if d.empty:
            print(f"[WARN] No data for part='{part}' in required phase/group labels.")
            return

        d = d.sort_values(["trial", "t"])

        if part == "elbow_angle":
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 4.5))

            coord = "angle_deg"

            if PLOT_INDIVIDUAL_TRIALS:
                for (grp, ph), sub in d.groupby(["control_experimental", "phase"], sort=False):
                    c = COLOR_BY_GROUP.get(grp, None)
                    for _, g in sub.groupby("trial", sort=False):
                        ax.plot(g["t"], g[coord], alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW, color=c)

            for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
                for ph in [PHASE_PRE, PHASE_POST]:
                    sub = d[(d["control_experimental"] == grp) & (d["phase"] == ph)]
                    if sub.empty:
                        continue

                    pivot = sub.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")
                    mean = pivot.mean(axis=1)
                    sem = pivot.sem(axis=1)

                    mean_s = smooth_curve(mean.values, sigma=SMOOTH_SIGMA)
                    sem_s = smooth_curve(sem.values, sigma=SMOOTH_SIGMA)

                    ax.plot(
                        mean.index, mean_s,
                        linewidth=MEAN_LW,
                        linestyle=LS_BY_PHASE.get(ph, "-"),
                        color=COLOR_BY_GROUP.get(grp, None),
                        label=f"{grp} | {ph}",
                    )
                    ax.fill_between(mean.index, mean_s - sem_s, mean_s + sem_s,
                                    alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

            ax.axvline(0, linestyle="--", linewidth=1, color="black")
            ax.set_title("ELBOW ANGLE (deg) (aligned to pass_slit; t=0) — 4 curves (group × phase)")
            ax.set_ylabel("Angle (deg)")
            ax.set_xlabel("t (frames relative to pass_slit)")
            ax.legend(ncol=2)

            apply_display_zoom([ax])
            plt.tight_layout()
            plt.show()
            return

        # mcp/wrist: two axes (x_mm and y_mm)
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
        axx, axy = axes

        for ax, coord, ylab in [(axx, "x_mm", "X (mm)"), (axy, "y_mm", "Y (mm)")]:
            if PLOT_INDIVIDUAL_TRIALS:
                for (grp, ph), sub in d.groupby(["control_experimental", "phase"], sort=False):
                    c = COLOR_BY_GROUP.get(grp, None)
                    for _, g in sub.groupby("trial", sort=False):
                        ax.plot(g["t"], g[coord], alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW, color=c)

            for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
                for ph in [PHASE_PRE, PHASE_POST]:
                    sub = d[(d["control_experimental"] == grp) & (d["phase"] == ph)]
                    if sub.empty:
                        continue

                    pivot = sub.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")
                    mean = pivot.mean(axis=1)
                    sem = pivot.sem(axis=1)

                    mean_s = smooth_curve(mean.values, sigma=SMOOTH_SIGMA)
                    sem_s = smooth_curve(sem.values, sigma=SMOOTH_SIGMA)

                    ax.plot(
                        mean.index, mean_s,
                        linewidth=MEAN_LW,
                        linestyle=LS_BY_PHASE.get(ph, "-"),
                        color=COLOR_BY_GROUP.get(grp, None),
                        label=f"{grp} | {ph}",
                    )
                    ax.fill_between(mean.index, mean_s - sem_s, mean_s + sem_s,
                                    alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

            ax.axvline(0, linestyle="--", linewidth=1, color="black")
            ax.set_ylabel(ylab)

        axx.set_title(f"{part.upper()} (pellet-centered mm; aligned to pass_slit; t=0) — 4 curves (group × phase)")
        axy.set_xlabel("t (frames relative to pass_slit)")
        axx.legend(ncol=2)

        apply_display_zoom([axx, axy])
        plt.tight_layout()
        plt.show()

    def paired_delta_timeseries(d: pd.DataFrame, coord: str):
        out = {}
        for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
            dg = d[d["control_experimental"] == grp].copy()
            if dg.empty:
                continue

            pre = dg[dg["phase"] == PHASE_PRE]
            post = dg[dg["phase"] == PHASE_POST]
            if pre.empty or post.empty:
                continue

            pre_mat = pre.pivot_table(index="t", columns="subject_key", values=coord, aggfunc="mean")
            post_mat = post.pivot_table(index="t", columns="subject_key", values=coord, aggfunc="mean")

            common_subjects = sorted(set(pre_mat.columns).intersection(set(post_mat.columns)))
            if len(common_subjects) < 2:
                continue

            pre_mat = pre_mat[common_subjects]
            post_mat = post_mat[common_subjects]

            common_t = pre_mat.index.intersection(post_mat.index)
            pre_mat = pre_mat.loc[common_t]
            post_mat = post_mat.loc[common_t]

            delta_mat = post_mat - pre_mat

            mean_delta = delta_mat.mean(axis=1)
            sem_delta = delta_mat.sem(axis=1)

            mean_s = smooth_curve(mean_delta.values, sigma=SMOOTH_SIGMA)
            sem_s = smooth_curve(sem_delta.values, sigma=SMOOTH_SIGMA)

            out[grp] = (mean_delta.index, mean_s, sem_s, "paired")
        return out

    def unpaired_delta_timeseries(d: pd.DataFrame, coord: str):
        out = {}
        for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
            dg = d[d["control_experimental"] == grp].copy()
            if dg.empty:
                continue

            pre = dg[dg["phase"] == PHASE_PRE]
            post = dg[dg["phase"] == PHASE_POST]
            if pre.empty or post.empty:
                continue

            pre_piv = pre.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")
            post_piv = post.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")

            common_t = pre_piv.index.intersection(post_piv.index)
            pre_piv = pre_piv.loc[common_t]
            post_piv = post_piv.loc[common_t]

            pre_mean = pre_piv.mean(axis=1)
            post_mean = post_piv.mean(axis=1)
            pre_sem = pre_piv.sem(axis=1)
            post_sem = post_piv.sem(axis=1)

            delta_mean = post_mean - pre_mean
            delta_sem = np.sqrt(pre_sem**2 + post_sem**2)

            mean_s = smooth_curve(delta_mean.values, sigma=SMOOTH_SIGMA)
            sem_s = smooth_curve(delta_sem.values, sigma=SMOOTH_SIGMA)

            out[grp] = (delta_mean.index, mean_s, sem_s, "unpaired")
        return out

    def plot_delta_curves(aligned_df: pd.DataFrame, part: str):
        d = aligned_df[aligned_df["part"] == part].copy()
        if d.empty:
            print(f"[WARN] No data for part='{part}'.")
            return

        d = d[
            d["phase"].isin([PHASE_PRE, PHASE_POST]) &
            d["control_experimental"].isin([GROUP_CONTROL, GROUP_EXPERIMENTAL])
        ].copy()
        if d.empty:
            print(f"[WARN] No delta-eligible data for part='{part}'.")
            return

        if part == "elbow_angle":
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 4.5))
            coord = "angle_deg"

            paired = paired_delta_timeseries(d, coord=coord)
            unpaired = unpaired_delta_timeseries(d, coord=coord)

            for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
                if grp in paired:
                    t_idx, mean_s, sem_s, mode = paired[grp]
                elif grp in unpaired:
                    t_idx, mean_s, sem_s, mode = unpaired[grp]
                else:
                    continue

                ax.plot(
                    t_idx, mean_s,
                    linewidth=MEAN_LW,
                    color=COLOR_BY_GROUP.get(grp, None),
                    label=f"{grp} Δ(post-pre) [{mode}]",
                )
                ax.fill_between(t_idx, mean_s - sem_s, mean_s + sem_s,
                                alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

            ax.axhline(0, linestyle=":", linewidth=1, color="black")
            ax.axvline(0, linestyle="--", linewidth=1, color="black")
            ax.set_title("ELBOW ANGLE Δ(post - pre) (aligned to pass_slit; t=0)")
            ax.set_ylabel("ΔAngle (deg)")
            ax.set_xlabel("t (frames relative to pass_slit)")
            ax.legend()

            apply_display_zoom([ax])
            plt.tight_layout()
            plt.show()
            return

        # mcp/wrist: two axes (Δx_mm and Δy_mm)
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
        axx, axy = axes

        for ax, coord, ylab in [(axx, "x_mm", "ΔX (mm)"), (axy, "y_mm", "ΔY (mm)")]:
            paired = paired_delta_timeseries(d, coord=coord)
            unpaired = unpaired_delta_timeseries(d, coord=coord)

            for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
                if grp in paired:
                    t_idx, mean_s, sem_s, mode = paired[grp]
                elif grp in unpaired:
                    t_idx, mean_s, sem_s, mode = unpaired[grp]
                else:
                    continue

                ax.plot(
                    t_idx, mean_s,
                    linewidth=MEAN_LW,
                    color=COLOR_BY_GROUP.get(grp, None),
                    label=f"{grp} Δ(post-pre) [{mode}]",
                )
                ax.fill_between(t_idx, mean_s - sem_s, mean_s + sem_s,
                                alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

            ax.axhline(0, linestyle=":", linewidth=1, color="black")
            ax.axvline(0, linestyle="--", linewidth=1, color="black")
            ax.set_ylabel(ylab)

        axx.set_title(f"{part.upper()} Δ(post - pre) (pellet-centered mm; aligned to pass_slit; t=0)")
        axy.set_xlabel("t (frames relative to pass_slit)")
        axx.legend()

        apply_display_zoom([axx, axy])
        plt.tight_layout()
        plt.show()

    # =========================
    # RUN PLOTS
    # =========================
    for part in ["mcp", "wrist", "elbow_angle"]:
        plot_four_curves(aligned, part)

    for part in ["mcp", "wrist", "elbow_angle"]:
        plot_delta_curves(aligned, part)

    plt.close("all")


if __name__ == "__main__":
    main()