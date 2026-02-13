# ---- slit-aligned kinematics plotter (MCP + Wrist) ----
# 2x2 design:
#   phase: pre_ablate vs post_ablate
#   control_experimental: control vs experimental
#
# Outputs:
#  (1) "All four curves" plots (4 mean±SEM traces): control-pre, control-post, exp-pre, exp-post
#      - For each marker (mcp, wrist): 2 rows (X, Y), aligned to pass_slit (t=0)
#      - Optional faint individual trial traces
#      - Smoothed mean/SEM shading (visualization only)
#
#  (2) Delta plots (post - pre) for each group (control vs experimental)
#      - For each marker: 2 rows (ΔX, ΔY)
#      - Prefer *paired* delta across subjects (best), falling back to unpaired mean-difference delta
#
# Notes:
# - Reads metadata XLSX (must include: filename_no_ext, hand_lift_off, pass_slit, camera_label, phase, control_experimental)
# - Finds each *_cleaned.csv in CLEANED_DIR using filename_no_ext + CLEANED_SUFFIX
# - Mirrors LEFT camera (camcommerb4) into RIGHT coordinates: x' = 2048 - x
# - Window: start = hand_lift_off, end = pass_slit + 100
# - Time index: t = frame - pass_slit
# - Does NOT edit metadata or DLC files. Optional output writes only NEW summary CSV if enabled.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

# =========================
# USER CONFIG
# =========================
META_XLSX = "/Users/gavin/Documents/lateralreticularnucleus/compiled_metadata_sorted_cleaned.xlsx"

# All cleaned CSVs are in the same directory as this example file:
EXAMPLE_CLEANED_CSV = "/Users/gavin/Documents/lateralreticularnucleus/analyzed_videos/iter03/postablate_cam4_1_9_26_5B-rat6_2026-01-09_14_46_19_camcommerb4DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"
CLEANED_DIR = os.path.dirname(EXAMPLE_CLEANED_CSV)

# How your cleaned files are named:
CLEANED_SUFFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"

# Camera geometry (for mirroring)
IMG_W, IMG_H = 2048, 700

# Windowing rule
POST_FRAMES = 100  # end = pass_slit + 100

# Which camera labels correspond to LEFT vs RIGHT view
LEFT_CAMERA_TOKENS  = ["camcommerb4", "cam4"]   # left side view
RIGHT_CAMERA_TOKENS = ["camcommerb5", "cam5"]   # right side view

# Markers to plot
PARTS = ["mcp", "wrist"]
RIGHT_PREFIX = "right_"
LEFT_PREFIX  = "left_"

# Group labels in metadata
PHASE_PRE  = "pre_ablate"
PHASE_POST = "post_ablate"
GROUP_CONTROL = "control"
GROUP_EXPERIMENTAL = "experimental"

# Plot options
PLOT_INDIVIDUAL_TRIALS = True
INDIVIDUAL_ALPHA = 0.15
INDIVIDUAL_LW = 0.8

MEAN_LW = 2.5
SEM_ALPHA = 0.25

# Smoothing (visualization only)
SMOOTH_SIGMA = 2  # try 2–3 if still blocky

# Debug verbosity
DEBUG_FIRST_N = 3
PRINT_SKIPS = 20

# Optional output (writes a NEW CSV only)
SAVE_ALIGNED_TABLE = False
OUT_ALIGNED_CSV = os.path.join(CLEANED_DIR, "slit_aligned_mcp_wrist_xy_long_with_groups.csv")

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
        # linear fill for smoothing, then restore NaNs
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

def build_cleaned_path(filename_no_ext: str) -> str:
    return os.path.join(CLEANED_DIR, str(filename_no_ext) + CLEANED_SUFFIX)

def read_dlc_multiindex_csv(path: str) -> pd.DataFrame:
    # Standard DLC: 3 header rows (scorer/bodypart/coords) and first column is frame index
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)

def pick_scorer(df: pd.DataFrame) -> str:
    return pd.unique(df.columns.get_level_values(0))[0]

def extract_xy(df: pd.DataFrame, bodypart: str):
    scorer = pick_scorer(df)
    sub = df.loc[:, (scorer, bodypart, ["x", "y"])]
    x = pd.to_numeric(sub[(scorer, bodypart, "x")], errors="coerce")
    y = pd.to_numeric(sub[(scorer, bodypart, "y")], errors="coerce")
    return x, y

def slice_by_frame_iloc(x: pd.Series, y: pd.Series, start: int, end: int):
    """
    Robust slicing by *positional* frame number.
    Assumes frame numbers correspond to row positions (0-based), matching your pass_slit usage.
    """
    start = int(start); end = int(end)
    n = len(x)
    if n == 0:
        return x.iloc[0:0], y.iloc[0:0], start, end

    start2 = max(0, min(start, n - 1))
    end2   = max(0, min(end,   n - 1))

    if start2 > end2:
        return x.iloc[0:0], y.iloc[0:0], start2, end2

    return x.iloc[start2:end2 + 1], y.iloc[start2:end2 + 1], start2, end2

def normalize_subject_key(filename_no_ext: str) -> str:
    """
    Best-effort subject key from filename_no_ext.
    Removes leading pre/post ablate tokens so pre & post versions map to the same key.

    Example:
      postablate_cam4_... -> cam4_...
      pre_ablate_cam5_... -> cam5_...
    """
    s = str(filename_no_ext)

    # common prefixes users have used
    s = re.sub(r'^(pre[_-]?ablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(post[_-]?ablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(preablate[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(postablate[_-]?)', '', s, flags=re.IGNORECASE)

    # also handle "post-ablate" etc
    s = re.sub(r'^(pre[_-]?ablation[_-]?)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(post[_-]?ablation[_-]?)', '', s, flags=re.IGNORECASE)

    return s

# =========================
# LOAD + ALIGN
# =========================
meta = pd.read_excel(META_XLSX)
meta = meta.copy()
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

records = []
skipped = []
loaded_debug = 0

for _, row in meta.iterrows():
    i = int(row["meta_row"])
    fname_base = str(row["filename_no_ext"])
    cam_label  = str(row["camera_label"])
    phase      = str(row["phase"]).strip() if not pd.isna(row["phase"]) else ""
    group      = str(row["control_experimental"]).strip() if not pd.isna(row["control_experimental"]) else ""
    subject_key = normalize_subject_key(fname_base)

    if pd.isna(row["hand_lift_off"]) or pd.isna(row["pass_slit"]):
        skipped.append((i, fname_base, "NaN hand_lift_off/pass_slit", None))
        continue

    hand = int(row["hand_lift_off"])
    slit = int(row["pass_slit"])
    end  = int(slit + POST_FRAMES)

    # enforce sane ordering
    start = min(hand, slit)

    try:
        side = detect_side(cam_label, fname_base)
    except Exception as e:
        skipped.append((i, fname_base, f"side infer failed: {repr(e)}", None))
        continue

    cleaned_path = build_cleaned_path(fname_base)
    if not os.path.exists(cleaned_path):
        skipped.append((i, fname_base, "missing cleaned csv", cleaned_path))
        continue

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
            loaded_debug += 1

        for part in PARTS:
            if side == "right":
                bp = RIGHT_PREFIX + part
                if bp not in bodyparts:
                    skipped.append((i, fname_base, f"missing bodypart {bp}", cleaned_path))
                    continue
                x, y = extract_xy(df, bp)
            else:
                bp = LEFT_PREFIX + part
                if bp not in bodyparts:
                    skipped.append((i, fname_base, f"missing bodypart {bp}", cleaned_path))
                    continue
                x, y = extract_xy(df, bp)
                x = IMG_W - x  # mirror into right-facing coordinates

            # positional slice (0-based)
            xs, ys, start_used, end_used = slice_by_frame_iloc(
                x.reset_index(drop=True),
                y.reset_index(drop=True),
                start, end
            )

            if xs.empty:
                skipped.append((i, fname_base, f"empty slice for {bp} (range issue)", cleaned_path))
                continue

            frames = np.arange(start_used, end_used + 1, dtype=int)
            t = frames - slit  # align to pass_slit

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
                "x": xs.to_numpy(dtype=float),
                "y": ys.to_numpy(dtype=float),
                "hand_lift_off": hand,
                "pass_slit": slit,
                "end_frame": end,
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
# PLOTTING: ALL FOUR CURVES (control/experimental x pre/post)
# =========================
# Style mapping: color by group, linestyle by phase
COLOR_BY_GROUP = {
    GROUP_CONTROL: "royalblue",
    GROUP_EXPERIMENTAL: "firebrick",
}
LS_BY_PHASE = {
    PHASE_PRE: "-",
    PHASE_POST: "--",
}

def plot_four_curves(aligned_df: pd.DataFrame, part: str):
    d = aligned_df[aligned_df["part"] == part].copy()
    if d.empty:
        print(f"[WARN] No data for part='{part}'.")
        return

    # keep only expected labels
    d = d[d["phase"].isin([PHASE_PRE, PHASE_POST]) &
          d["control_experimental"].isin([GROUP_CONTROL, GROUP_EXPERIMENTAL])].copy()
    if d.empty:
        print(f"[WARN] No data for part='{part}' in required phase/group labels.")
        return

    d = d.sort_values(["trial", "t"])

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
    axx, axy = axes

    for ax, coord in [(axx, "x"), (axy, "y")]:

        # optional individual trials (faint)
        if PLOT_INDIVIDUAL_TRIALS:
            # draw trials but keep same color semantics by group (phase not emphasized for individuals)
            for (grp, ph), sub in d.groupby(["control_experimental", "phase"], sort=False):
                c = COLOR_BY_GROUP.get(grp, None)
                for _, g in sub.groupby("trial", sort=False):
                    ax.plot(g["t"], g[coord], alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW, color=c)

        # mean ± SEM for each of 4 cells
        for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
            for ph in [PHASE_PRE, PHASE_POST]:
                sub = d[(d["control_experimental"] == grp) & (d["phase"] == ph)]
                if sub.empty:
                    continue

                pivot = sub.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")

                mean = pivot.mean(axis=1)
                sem  = pivot.sem(axis=1)

                mean_s = smooth_curve(mean.values, sigma=SMOOTH_SIGMA)
                sem_s  = smooth_curve(sem.values,  sigma=SMOOTH_SIGMA)

                ax.plot(
                    mean.index,
                    mean_s,
                    linewidth=MEAN_LW,
                    linestyle=LS_BY_PHASE.get(ph, "-"),
                    color=COLOR_BY_GROUP.get(grp, None),
                    label=f"{grp} | {ph}",
                )
                ax.fill_between(mean.index, mean_s - sem_s, mean_s + sem_s,
                                alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_ylabel(f"{coord} (px)")

    axx.set_title(f"{part.upper()} (aligned to pass_slit; t=0) — 4 curves (group × phase)")
    axy.set_xlabel("t (frames relative to pass_slit)")
    axx.legend(ncol=2)
    plt.tight_layout()
    plt.show()

# =========================
# DELTA PLOTS: (post - pre) within each group
# Prefer paired-by-subject delta if we can pair pre/post for the same subject_key within each group.
# =========================
def paired_delta_timeseries(d: pd.DataFrame, coord: str):
    """
    Attempt paired delta (post - pre) across subject_key within each group.
    Returns: dict[group] -> (t_index, mean_delta, sem_delta, mode_str)
    mode_str is 'paired' if successful, otherwise None.
    """
    out = {}

    for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
        dg = d[d["control_experimental"] == grp].copy()
        if dg.empty:
            continue

        # We want a time x subject matrix for pre and for post.
        # Average multiple trials per subject at each t.
        pre = dg[dg["phase"] == PHASE_PRE]
        post = dg[dg["phase"] == PHASE_POST]
        if pre.empty or post.empty:
            continue

        pre_mat = pre.pivot_table(index="t", columns="subject_key", values=coord, aggfunc="mean")
        post_mat = post.pivot_table(index="t", columns="subject_key", values=coord, aggfunc="mean")

        # paired subjects = intersection
        common_subjects = sorted(set(pre_mat.columns).intersection(set(post_mat.columns)))
        if len(common_subjects) < 2:
            # not enough pairing
            continue

        pre_mat = pre_mat[common_subjects]
        post_mat = post_mat[common_subjects]

        # align on common t rows
        common_t = pre_mat.index.intersection(post_mat.index)
        pre_mat = pre_mat.loc[common_t]
        post_mat = post_mat.loc[common_t]

        delta_mat = post_mat - pre_mat  # (t x subject)

        mean_delta = delta_mat.mean(axis=1)
        sem_delta = delta_mat.sem(axis=1)

        mean_s = smooth_curve(mean_delta.values, sigma=SMOOTH_SIGMA)
        sem_s  = smooth_curve(sem_delta.values,  sigma=SMOOTH_SIGMA)

        out[grp] = (mean_delta.index, mean_s, sem_s, "paired")

    return out

def unpaired_delta_timeseries(d: pd.DataFrame, coord: str):
    """
    Unpaired delta (mean_post - mean_pre) with SEM propagated as sqrt(sem_pre^2 + sem_post^2).
    Returns: dict[group] -> (t_index, mean_delta, sem_delta, mode_str)
    """
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

        # common t
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
        sem_s  = smooth_curve(delta_sem.values,  sigma=SMOOTH_SIGMA)

        out[grp] = (delta_mean.index, mean_s, sem_s, "unpaired")

    return out

def plot_delta_curves(aligned_df: pd.DataFrame, part: str):
    d = aligned_df[aligned_df["part"] == part].copy()
    if d.empty:
        print(f"[WARN] No data for part='{part}'.")
        return

    d = d[d["phase"].isin([PHASE_PRE, PHASE_POST]) &
          d["control_experimental"].isin([GROUP_CONTROL, GROUP_EXPERIMENTAL])].copy()
    if d.empty:
        print(f"[WARN] No delta-eligible data for part='{part}'.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
    axx, axy = axes

    for ax, coord in [(axx, "x"), (axy, "y")]:
        paired = paired_delta_timeseries(d, coord=coord)

        # if paired failed for either group, fall back to unpaired for those groups
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
                label=f"{grp} Δ(post-pre) [{mode}]"
            )
            ax.fill_between(t_idx, mean_s - sem_s, mean_s + sem_s,
                            alpha=SEM_ALPHA, color=COLOR_BY_GROUP.get(grp, None))

        ax.axhline(0, linestyle=":", linewidth=1, color="black")
        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_ylabel(f"Δ{coord} (px)")

    axx.set_title(f"{part.upper()} Δ(post - pre) (aligned to pass_slit; t=0)")
    axy.set_xlabel("t (frames relative to pass_slit)")
    axx.legend()
    plt.tight_layout()
    plt.show()

# =========================
# RUN PLOTS
# =========================
if aligned.empty:
    print("\n[ERROR] aligned is empty -> nothing to plot. Check skipped reasons above.")
else:
    # Four-curve plots
    plot_four_curves(aligned, "mcp")
    plot_four_curves(aligned, "wrist")

    # Delta plots
    plot_delta_curves(aligned, "mcp")
    plot_delta_curves(aligned, "wrist")

plt.close("all")
