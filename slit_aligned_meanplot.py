# ---- slit-aligned kinematics plotter (MCP + Wrist) ----
# Baseline-vs-single-day comparisons using metadata['timestamp_date']
# AND split by metadata['control_experimental'] (control vs experimental)
#
# You asked for NEW graphs:
#   1) baseline pre-ablate on 11/26/25  vs post-ablate on 12/19/25
#   2) baseline pre-ablate on 11/26/25  vs post-ablate on 01/09/26
#   3) baseline pre-ablate on 11/26/25  vs post-ablate on 01/16/26
#
# For EACH comparison, you want BOTH groups shown:
#   - control baseline vs control post_day
#   - experimental baseline vs experimental post_day
#
# And an option to:
#   A) plot averaged across rats/trials (mean±SEM)  -> PLOT_MODE="mean"
#   B) plot rats individually on the SAME graph     -> PLOT_MODE="rats"
#      (one figure per comparison per marker; not one figure per rat)
#
# Figures produced (per marker MCP/Wrist):
#   - For each post day (3): one figure with 2 rows (X,Y)
#     containing 4 traces (control baseline, control post, exp baseline, exp post)
#     OR (rats mode) many traces: each rat baseline (solid) and post (dashed), colored by group.
#
# Notes:
# - Reads metadata XLSX (must include: filename_no_ext, hand_lift_off, pass_slit, camera_label, phase,
#   timestamp_date, control_experimental)
# - Loads corresponding cleaned DLC CSVs from one folder using filename_no_ext + CLEANED_SUFFIX
# - Mirrors LEFT camera (camcommerb4) into RIGHT coordinates: x' = 2048 - x
# - Window: start = hand_lift_off, end = pass_slit + POST_FRAMES
# - Time index: t = frame - pass_slit (t=0 at pass_slit)
# - DOES NOT edit metadata or DLC files. Optional output writes only NEW CSV if enabled.

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

EXAMPLE_CLEANED_CSV = "/Users/gavin/Documents/lateralreticularnucleus/analyzed_videos/iter03/postablate_cam4_1_9_26_5B-rat6_2026-01-09_14_46_19_camcommerb4DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"
CLEANED_DIR = os.path.dirname(EXAMPLE_CLEANED_CSV)
CLEANED_SUFFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000_cleaned.csv"

IMG_W, IMG_H = 2048, 700
POST_FRAMES = 100

LEFT_CAMERA_TOKENS  = ["camcommerb4", "cam4"]
RIGHT_CAMERA_TOKENS = ["camcommerb5", "cam5"]

PARTS = ["mcp", "wrist"]
RIGHT_PREFIX = "right_"
LEFT_PREFIX  = "left_"

PHASE_PRE  = "pre_ablate"
PHASE_POST = "post_ablate"

GROUP_CONTROL = "control"
GROUP_EXPERIMENTAL = "experimental"

# Plot mode:
#   "mean" -> mean±SEM across trials in each (group,condition)
#   "rats" -> each rat as a line; baseline solid and post dashed; group indicated in legend
PLOT_MODE = "rats"  # "mean" or "rats"

# Cosmetics
PLOT_INDIVIDUAL_TRIALS = False  # only used when PLOT_MODE="mean"
INDIVIDUAL_ALPHA = 0.12
INDIVIDUAL_LW = 0.8

MEAN_LW = 2.5
SEM_ALPHA = 0.25
SMOOTH_SIGMA = 2

BASELINE_DATE_STR = "11/26/25"
POST_DATES_STR = ["12/19/25", "01/09/26", "01/16/26"]

DEBUG_FIRST_N = 2
PRINT_SKIPS = 15

SAVE_ALIGNED_TABLE = False
OUT_ALIGNED_CSV = os.path.join(CLEANED_DIR, "slit_aligned_mcp_wrist_xy_long_with_dates_groups.csv")

# Colors for groups (used in rats-mode and mean-mode)
COLOR_BY_GROUP = {
    GROUP_CONTROL: "royalblue",
    GROUP_EXPERIMENTAL: "firebrick",
}

# =========================
# HELPERS
# =========================
def smooth_curve(y, sigma=2):
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

def build_cleaned_path(filename_no_ext: str) -> str:
    return os.path.join(CLEANED_DIR, str(filename_no_ext) + CLEANED_SUFFIX)

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

def slice_by_frame_iloc(x: pd.Series, y: pd.Series, start: int, end: int):
    start = int(start); end = int(end)
    n = len(x)
    if n == 0:
        return x.iloc[0:0], y.iloc[0:0], start, end
    start2 = max(0, min(start, n - 1))
    end2   = max(0, min(end,   n - 1))
    if start2 > end2:
        return x.iloc[0:0], y.iloc[0:0], start2, end2
    return x.iloc[start2:end2 + 1], y.iloc[start2:end2 + 1], start2, end2

def extract_rat_id(filename_no_ext: str) -> str:
    s = str(filename_no_ext).lower()
    m = re.search(r'rat(\d+)', s)
    if m:
        return f"rat{m.group(1)}"
    return "unknown"

def parse_timestamp_date(val) -> pd.Timestamp:
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val.normalize()
    s = str(val).strip()
    if not s:
        return pd.NaT
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if not pd.isna(dt):
        return dt.normalize()
    s2 = s.replace("_", "/").replace("-", "/")
    dt = pd.to_datetime(s2, errors="coerce", infer_datetime_format=True)
    if not pd.isna(dt):
        return dt.normalize()
    dt = pd.to_datetime(s2, errors="coerce", format="%m/%d/%y")
    if not pd.isna(dt):
        return dt.normalize()
    return pd.NaT

def datekey(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "NA"
    return ts.strftime("%m/%d/%y")

def normalize_group(val: str) -> str:
    s = str(val).strip().lower()
    if s in ["control", "ctrl", "c"]:
        return GROUP_CONTROL
    if s in ["experimental", "experiment", "exp", "e"]:
        return GROUP_EXPERIMENTAL
    return s  # leave as-is if weird; will likely be filtered out later

# =========================
# LOAD + ALIGN
# =========================
meta = pd.read_excel(META_XLSX).copy()
meta["meta_row"] = meta.index

required_cols = [
    "filename_no_ext",
    "hand_lift_off",
    "pass_slit",
    "camera_label",
    "phase",
    "timestamp_date",
    "control_experimental",
]
missing = [c for c in required_cols if c not in meta.columns]
if missing:
    raise KeyError(f"Metadata missing required columns: {missing}\nFound: {list(meta.columns)}")

meta["timestamp_dt"] = meta["timestamp_date"].apply(parse_timestamp_date)
meta["group_norm"] = meta["control_experimental"].apply(normalize_group)

records = []
skipped = []
dbg = 0

for _, row in meta.iterrows():
    i = int(row["meta_row"])
    fname_base = str(row["filename_no_ext"])
    cam_label  = str(row["camera_label"])
    phase      = str(row["phase"]).strip() if not pd.isna(row["phase"]) else ""
    group      = row["group_norm"]
    ts_dt      = row["timestamp_dt"]
    rat_id     = extract_rat_id(fname_base)

    if pd.isna(row["hand_lift_off"]) or pd.isna(row["pass_slit"]):
        skipped.append((i, fname_base, "NaN hand_lift_off/pass_slit", None))
        continue

    if pd.isna(ts_dt):
        skipped.append((i, fname_base, "NaN/unparsed timestamp_date", None))
        continue

    if group not in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
        skipped.append((i, fname_base, f"unknown group '{row['control_experimental']}'", None))
        continue

    hand = int(row["hand_lift_off"])
    slit = int(row["pass_slit"])
    end  = int(slit + POST_FRAMES)
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

        if dbg < DEBUG_FIRST_N:
            print(f"\n--- DEBUG row {i} ---")
            print("trial:", fname_base)
            print("rat_id:", rat_id, "| group:", group, "| phase:", phase, "| date:", datekey(ts_dt))
            print("camera_label:", cam_label, "| inferred side:", side)
            print("frames:", len(df), "| hand:", hand, "| slit:", slit, "| start:", start, "| end:", end)
            print("scorer:", scorer)
            dbg += 1

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
                x = IMG_W - x

            xs, ys, start_used, end_used = slice_by_frame_iloc(
                x.reset_index(drop=True),
                y.reset_index(drop=True),
                start, end
            )
            if xs.empty:
                skipped.append((i, fname_base, f"empty slice for {bp} (range issue)", cleaned_path))
                continue

            frames = np.arange(start_used, end_used + 1, dtype=int)
            t = frames - slit

            records.append(pd.DataFrame({
                "trial": fname_base,
                "rat_id": rat_id,
                "group": group,
                "phase": phase,
                "timestamp_dt": ts_dt,
                "part": part,
                "t": t,
                "x": xs.to_numpy(dtype=float),
                "y": ys.to_numpy(dtype=float),
            }))

    except Exception as e:
        skipped.append((i, fname_base, f"exception: {repr(e)}", cleaned_path))
        continue

aligned = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

print("\n=========================")
print("Aligned trials:", aligned["trial"].nunique() if not aligned.empty else 0)
print("Aligned rows:", len(aligned))
if not aligned.empty:
    print("\nGroups:", aligned["group"].value_counts())
    print("Phases:", aligned["phase"].value_counts())
    print("Dates (top):", aligned["timestamp_dt"].value_counts().head(10))
    print("Rats:", aligned["rat_id"].value_counts())
print("\nSkipped:", len(skipped))
if skipped:
    print("\nFirst skipped reasons:")
    for item in skipped[:PRINT_SKIPS]:
        print(item)

if SAVE_ALIGNED_TABLE and not aligned.empty:
    aligned.to_csv(OUT_ALIGNED_CSV, index=False)
    print("\nSaved aligned long table to:", OUT_ALIGNED_CSV)

# =========================
# SUBSETTING
# =========================
def subset_for_comparison(aligned_df, baseline_dt, post_dt, part):
    d = aligned_df[aligned_df["part"] == part].copy()
    base = d[(d["phase"] == PHASE_PRE) & (d["timestamp_dt"] == baseline_dt)].copy()
    post = d[(d["phase"] == PHASE_POST) & (d["timestamp_dt"] == post_dt)].copy()
    return base, post

# =========================
# PLOTTING
# =========================
def plot_mean_sem(ax, sub, coord, label, color, linestyle):
    if sub.empty:
        return False

    pivot = sub.pivot_table(index="t", columns="trial", values=coord, aggfunc="mean")
    mean = pivot.mean(axis=1)
    sem  = pivot.sem(axis=1)

    mean_s = smooth_curve(mean.values, sigma=SMOOTH_SIGMA)
    sem_s  = smooth_curve(sem.values,  sigma=SMOOTH_SIGMA)

    ax.plot(mean.index, mean_s, linewidth=MEAN_LW, label=label, color=color, linestyle=linestyle)
    ax.fill_between(mean.index, mean_s - sem_s, mean_s + sem_s, alpha=SEM_ALPHA, color=color)
    return True

def plot_rats(ax, sub, coord, label_prefix, color, linestyle):
    if sub.empty:
        return 0
    mat = sub.pivot_table(index="t", columns="rat_id", values=coord, aggfunc="mean")
    for rat in mat.columns:
        y = mat[rat].values.astype(float)
        y_s = smooth_curve(y, sigma=SMOOTH_SIGMA)
        ax.plot(mat.index, y_s, linewidth=MEAN_LW, linestyle=linestyle, color=color,
                label=f"{label_prefix} | {rat}")
    return len(mat.columns)

def plot_baseline_vs_post_split_groups(aligned_df, baseline_dt, post_dt, part, mode="mean"):
    base, post = subset_for_comparison(aligned_df, baseline_dt, post_dt, part)

    if base.empty:
        print(f"[WARN] No baseline data for {part} on {datekey(baseline_dt)}.")
        return
    if post.empty:
        print(f"[WARN] No post data for {part} on {datekey(post_dt)}.")
        return

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    axx, axy = axes

    for ax, coord in [(axx, "x"), (axy, "y")]:

        for grp in [GROUP_CONTROL, GROUP_EXPERIMENTAL]:
            c = COLOR_BY_GROUP[grp]

            base_g = base[base["group"] == grp]
            post_g = post[post["group"] == grp]

            if mode == "mean":
                # optional faint per-trial traces (within each group+condition)
                if PLOT_INDIVIDUAL_TRIALS:
                    for _, g in base_g.groupby("trial", sort=False):
                        ax.plot(g["t"], g[coord], alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW, color=c)
                    for _, g in post_g.groupby("trial", sort=False):
                        ax.plot(g["t"], g[coord], alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW, color=c)

                plot_mean_sem(
                    ax, base_g, coord,
                    label=f"{grp} baseline {datekey(baseline_dt)}",
                    color=c, linestyle="-"
                )
                plot_mean_sem(
                    ax, post_g, coord,
                    label=f"{grp} post {datekey(post_dt)}",
                    color=c, linestyle="--"
                )

            elif mode == "rats":
                # baseline solid, post dashed; each rat is a line (colored by group)
                plot_rats(ax, base_g, coord, f"{grp} baseline {datekey(baseline_dt)}", color=c, linestyle="-")
                plot_rats(ax, post_g, coord, f"{grp} post {datekey(post_dt)}", color=c, linestyle="--")

            else:
                raise ValueError("mode must be 'mean' or 'rats'")

        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_ylabel(f"{coord} (px)")

    axx.set_title(f"{part.upper()} baseline vs post ({datekey(baseline_dt)} vs {datekey(post_dt)}) | mode={mode}")
    axy.set_xlabel("t (frames relative to pass_slit)")
    axx.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

# =========================
# RUN
# =========================
if aligned.empty:
    print("\n[ERROR] aligned is empty -> nothing to plot.")
else:
    baseline_dt = parse_timestamp_date(BASELINE_DATE_STR)
    if pd.isna(baseline_dt):
        raise ValueError(f"Could not parse BASELINE_DATE_STR='{BASELINE_DATE_STR}'")

    post_dts = [parse_timestamp_date(s) for s in POST_DATES_STR]
    bad = [s for s, d in zip(POST_DATES_STR, post_dts) if pd.isna(d)]
    if bad:
        raise ValueError(f"Could not parse post date(s): {bad}")

    for post_dt in post_dts:
        for part in ["mcp", "wrist"]:
            plot_baseline_vs_post_split_groups(aligned, baseline_dt, post_dt, part, mode=PLOT_MODE)

plt.close("all")
