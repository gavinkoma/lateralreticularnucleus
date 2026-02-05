import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
META_XLSX = "/Users/gavin/Documents/lateralreticularnucleus/only_good.xlsx"
DLC_SUFFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000.csv"

BODYPART = "left_mcp"     # "left_mcp" or "right_mcp"
PRE_FRAMES = 50
POST_FRAMES = 50

# Optional likelihood filter (set to None to disable)
LIKELIHOOD_THRESH = None   # e.g., 0.9

# Plot controls
MAX_TRIALS_PER_GROUP = None   # e.g., 60 to subsample; None = plot all
ALPHA = 0.15                  # transparency for each trial line
LINEWIDTH = 1.0
# ==================================================


def fix_filename(name: str) -> str:
    """Remove trailing .csv (if present), then append DLC suffix."""
    name = str(name)
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return name + DLC_SUFFIX


def read_dlc_flat(csv_path: str) -> pd.DataFrame:
    """
    Reads DLC CSV (3 header rows) and flattens columns to:
      <bodypart>_<coord>, e.g. left_mcp_x, left_mcp_y, left_mcp_likelihood
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    df.columns = df.columns.droplevel(0)  # drop scorer
    df.columns = [f"{bp}_{coord}" for (bp, coord) in df.columns]
    return df


def load_and_align_xy(csv_path: str,
                      pass_slit: int,
                      bodypart: str,
                      pre: int,
                      post: int) -> pd.DataFrame | None:
    df = read_dlc_flat(csv_path)

    xcol = f"{bodypart}_x"
    ycol = f"{bodypart}_y"
    lcol = f"{bodypart}_likelihood"

    if xcol not in df.columns or ycol not in df.columns:
        mcp_like = [c for c in df.columns if "mcp" in c.lower()]
        raise KeyError(
            f"Missing {xcol} or {ycol} in:\n  {csv_path}\n"
            f"MCP-like columns present:\n  {mcp_like[:50]}"
        )

    start = pass_slit - pre
    end = pass_slit + post
    if start < 0 or end >= len(df):
        return None

    out = pd.DataFrame({
        "rel_frame": range(-pre, post + 1),
        "mcp_x": df.loc[start:end, xcol].to_numpy(),
        "mcp_y": df.loc[start:end, ycol].to_numpy(),
        "likelihood": df.loc[start:end, lcol].to_numpy() if lcol in df.columns else np.nan,
    })
    return out


# ===================== MAIN =====================

# Load metadata and fix filenames
meta = pd.read_excel(META_XLSX)
meta["filename"] = meta["filename"].apply(fix_filename)

# Build aligned dataset (X and Y)
all_trials = []
trial_id = 0
skipped_oob = 0

for _, row in meta.iterrows():
    trial_id += 1
    aligned = load_and_align_xy(
        csv_path=row["filename"],
        pass_slit=int(row["pass_slit"]),
        bodypart=BODYPART,
        pre=PRE_FRAMES,
        post=POST_FRAMES,
    )

    if aligned is None:
        skipped_oob += 1
        continue

    aligned["phase"] = row["phase"]
    aligned["trial_id"] = trial_id
    aligned["source_file"] = row["filename"]
    all_trials.append(aligned)

aligned_xy = pd.concat(all_trials, ignore_index=True)

print(f"Aligned rows: {len(aligned_xy)}")
print(f"Trials included: {aligned_xy[['trial_id']].drop_duplicates().shape[0]}")
print(f"Trials skipped (out-of-bounds window): {skipped_oob}")

# Optional likelihood filtering (row-wise)
if LIKELIHOOD_THRESH is not None:
    before = len(aligned_xy)
    aligned_xy = aligned_xy[aligned_xy["likelihood"].isna() | (aligned_xy["likelihood"] >= LIKELIHOOD_THRESH)].copy()
    after = len(aligned_xy)
    print(f"Likelihood filter: kept {after}/{before} rows (threshold={LIKELIHOOD_THRESH})")

# Optionally subsample trials per group to reduce clutter
def maybe_subsample(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    if MAX_TRIALS_PER_GROUP is None:
        return sub
    trial_ids = sub["trial_id"].drop_duplicates().to_numpy()
    if len(trial_ids) <= MAX_TRIALS_PER_GROUP:
        return sub
    keep = np.random.choice(trial_ids, size=MAX_TRIALS_PER_GROUP, replace=False)
    return sub[sub["trial_id"].isin(keep)]

pre_df = maybe_subsample(aligned_xy, "pre_ablate")
post_df = maybe_subsample(aligned_xy, "post_ablate")

# Plot: individual trials (X and Y panels)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# X panel
for tid, g in pre_df.groupby("trial_id"):
    axes[0].plot(g["rel_frame"], g["mcp_x"], alpha=ALPHA, linewidth=LINEWIDTH, label="pre_ablate" if tid == pre_df["trial_id"].min() else None)
for tid, g in post_df.groupby("trial_id"):
    axes[0].plot(g["rel_frame"], g["mcp_x"], alpha=ALPHA, linewidth=LINEWIDTH, label="post_ablate" if tid == post_df["trial_id"].min() else None)

# Y panel
for tid, g in pre_df.groupby("trial_id"):
    axes[1].plot(g["rel_frame"], g["mcp_y"], alpha=ALPHA, linewidth=LINEWIDTH)
for tid, g in post_df.groupby("trial_id"):
    axes[1].plot(g["rel_frame"], g["mcp_y"], alpha=ALPHA, linewidth=LINEWIDTH)

axes[0].axvline(0, linestyle="--")
axes[1].axvline(0, linestyle="--")

axes[0].set_ylabel(f"{BODYPART} X (pixels)")
axes[1].set_ylabel(f"{BODYPART} Y (pixels)")
axes[1].set_xlabel("Frames relative to pass_slit (0 = pass)")

axes[0].set_title(f"Individual trials (pre vs post), aligned to pass_slit — {BODYPART}")
axes[0].legend()

plt.tight_layout()
plt.show()
