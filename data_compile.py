import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ===================== CONFIG =====================
META_XLSX = "/Users/gavin/Documents/lateralreticularnucleus/only_good.xlsx"
DLC_SUFFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000.csv"

BODYPART = "left_mcp"     # change to "right_mcp" if desired
PRE_FRAMES = 50
POST_FRAMES = 50

SMOOTH_WINDOW = 7         # smoothing for plotting only (frames)

# Defines "MCP location in the slit" as mean over this window around rel_frame=0
SLIT_WINDOW = (-2, 2)     # frames around pass_slit for stats

# Optional likelihood filter (set to None to disable)
LIKELIHOOD_THRESH = None  # e.g., 0.9
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
    """
    Returns aligned window with rel_frame, mcp_x, mcp_y, likelihood (if present).
    """
    df = read_dlc_flat(csv_path)

    xcol = f"{bodypart}_x"
    ycol = f"{bodypart}_y"
    lcol = f"{bodypart}_likelihood"

    # Helpful debug if missing
    missing = [c for c in [xcol, ycol] if c not in df.columns]
    if missing:
        mcp_like = [c for c in df.columns if "mcp" in c.lower()]
        raise KeyError(
            f"Missing columns {missing} in:\n  {csv_path}\n"
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
    })

    if lcol in df.columns:
        out["likelihood"] = df.loc[start:end, lcol].to_numpy()
    else:
        out["likelihood"] = np.nan

    return out


def rolling_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)


# ===================== MAIN =====================

# 1) Load metadata and fix filenames
meta = pd.read_excel(META_XLSX)
meta["filename"] = meta["filename"].apply(fix_filename)

# 2) Build aligned dataset (X and Y)
all_trials = []
trial_id = 0
skipped_oob = 0

for _, row in meta.iterrows():
    trial_id += 1
    csv_path = row["filename"]
    pass_slit = int(row["pass_slit"])

    aligned = load_and_align_xy(
        csv_path=csv_path,
        pass_slit=pass_slit,
        bodypart=BODYPART,
        pre=PRE_FRAMES,
        post=POST_FRAMES,
    )

    if aligned is None:
        skipped_oob += 1
        continue

    aligned["phase"] = row["phase"]
    aligned["trial_id"] = trial_id
    aligned["source_file"] = csv_path
    all_trials.append(aligned)

aligned_xy = pd.concat(all_trials, ignore_index=True)

print(f"Aligned rows: {len(aligned_xy)}")
print(f"Trials included: {aligned_xy[['trial_id']].drop_duplicates().shape[0]}")
print(f"Trials skipped (out-of-bounds window): {skipped_oob}")

# 3) Optional likelihood filtering
if LIKELIHOOD_THRESH is not None:
    before = len(aligned_xy)
    aligned_xy = aligned_xy[aligned_xy["likelihood"].isna() | (aligned_xy["likelihood"] >= LIKELIHOOD_THRESH)].copy()
    after = len(aligned_xy)
    print(f"Likelihood filter: kept {after}/{before} rows (threshold={LIKELIHOOD_THRESH})")

# 4) Plot smoothed mean curves for X and Y by phase
summary = (
    aligned_xy
    .groupby(["phase", "rel_frame"])
    .agg(
        x_mean=("mcp_x", "mean"),
        y_mean=("mcp_y", "mean"),
        x_sem=("mcp_x", "sem"),
        y_sem=("mcp_y", "sem"),
    )
    .reset_index()
)

summary["x_smooth"] = summary.groupby("phase")["x_mean"].transform(lambda s: rolling_smooth(s, SMOOTH_WINDOW))
summary["y_smooth"] = summary.groupby("phase")["y_mean"].transform(lambda s: rolling_smooth(s, SMOOTH_WINDOW))

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for phase in ["pre_ablate", "post_ablate"]:
    sub = summary[summary["phase"] == phase]

    axes[0].plot(sub["rel_frame"], sub["x_smooth"], label=phase)
    axes[1].plot(sub["rel_frame"], sub["y_smooth"], label=phase)

axes[0].axvline(0, linestyle="--")
axes[1].axvline(0, linestyle="--")

axes[0].set_ylabel(f"{BODYPART} X (pixels)")
axes[1].set_ylabel(f"{BODYPART} Y (pixels)")
axes[1].set_xlabel("Frames relative to pass_slit (0 = pass)")

axes[0].set_title(f"Event-locked {BODYPART} position (smoothed window={SMOOTH_WINDOW})")
axes[0].legend()

plt.tight_layout()
plt.show()

# 5) Stats: per-trial MCP location "in slit" using SLIT_WINDOW
slit_df = aligned_xy[
    (aligned_xy["rel_frame"] >= SLIT_WINDOW[0]) &
    (aligned_xy["rel_frame"] <= SLIT_WINDOW[1])
].copy()

# Reduce each trial to one scalar for X and one for Y
trial_slit = (
    slit_df
    .groupby(["phase", "trial_id"])
    .agg(
        x_at_slit=("mcp_x", "mean"),
        y_at_slit=("mcp_y", "mean")
    )
    .reset_index()
)

# Split groups
pre = trial_slit[trial_slit["phase"] == "pre_ablate"]
post = trial_slit[trial_slit["phase"] == "post_ablate"]

pre_x = pre["x_at_slit"].to_numpy()
post_x = post["x_at_slit"].to_numpy()

pre_y = pre["y_at_slit"].to_numpy()
post_y = post["y_at_slit"].to_numpy()

# Welch t-tests (recommended)
tx, px = stats.ttest_ind(pre_x, post_x, equal_var=False, nan_policy="omit")
ty, py = stats.ttest_ind(pre_y, post_y, equal_var=False, nan_policy="omit")

dx = cohens_d(pre_x, post_x)
dy = cohens_d(pre_y, post_y)

print("\n===== SLIT-LOCKED STATS (per-trial scalar; Welch t-test) =====")
print(f"Slit window (frames): {SLIT_WINDOW[0]} to {SLIT_WINDOW[1]}")
print(f"N pre: {len(pre)}, N post: {len(post)}")

print("\n--- X at slit ---")
print(f"mean(pre)={np.nanmean(pre_x):.3f}, mean(post)={np.nanmean(post_x):.3f}")
print(f"t={tx:.4f}, p={px:.4e}, Cohen's d={dx:.3f}")

print("\n--- Y at slit ---")
print(f"mean(pre)={np.nanmean(pre_y):.3f}, mean(post)={np.nanmean(post_y):.3f}")
print(f"t={ty:.4f}, p={py:.4e}, Cohen's d={dy:.3f}")

# Optional: quick boxplots of slit position distributions
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot([pre_x, post_x], labels=["pre_ablate", "post_ablate"], showfliers=False)
plt.title("X at slit")
plt.ylabel("pixels")

plt.subplot(1, 2, 2)
plt.boxplot([pre_y, post_y], labels=["pre_ablate", "post_ablate"], showfliers=False)
plt.title("Y at slit")

plt.tight_layout()
plt.show()
