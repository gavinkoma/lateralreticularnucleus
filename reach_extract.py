import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
META_PATH = Path("/Users/gavin/Documents/lateralreticularnucleus/compiled_metadata.csv")
BASE_CSV_DIR = Path("/Users/gavin/Documents/lateralreticularnucleus/analyzed_videos/iter03")

FPS = 250.0  # Hz
CAM4_LABEL = "camcommerb4"
CAM5_LABEL = "camcommerb5"

# Base marker list (MCP will be auto-added depending on camera)
MARKERS_TO_PLOT = [
    "pellet",
    # add any others you always want:
    # "left_wrist", "right_wrist", ...
]

# DLC postfix appended to metadata filename stem
DLC_POSTFIX = "DLC_resnet50_josh_lrnDec5shuffle1_250000.csv"

# Output settings
OUT_SUBFOLDER_NAME = "reach_summary_pngs"
PLOT_FILE_APPENDIX = "__reachQC_v1"
FIG_DPI = 150

# -----------------------
# HELPERS
# -----------------------
def read_kin_csv(csv_path: Path) -> pd.DataFrame:
    """Read DLC-style (3-row header) or flat CSV; returns flattened columns like '{marker}_{coord}'."""
    csv_path = Path(csv_path)

    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
            new_cols = []
            for scorer, bodypart, coord in df.columns:
                if "Unnamed" in str(scorer) or "Unnamed" in str(bodypart) or "Unnamed" in str(coord):
                    new_cols.append("frame")
                else:
                    new_cols.append(f"{bodypart}_{coord}")
            df.columns = new_cols

            if df.columns.tolist().count("frame") > 1:
                first = df.columns.tolist().index("frame")
                keep = [True] * df.shape[1]
                for j in range(df.shape[1]):
                    if df.columns[j] == "frame" and j != first:
                        keep[j] = False
                df = df.loc[:, keep]

            if "frame" not in df.columns:
                df.insert(0, "frame", np.arange(len(df), dtype=int))
            return df
    except Exception:
        pass

    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        df.insert(0, "frame", np.arange(len(df), dtype=int))
    return df


def get_xy(df: pd.DataFrame, marker: str):
    xcol, ycol = f"{marker}_x", f"{marker}_y"
    if xcol in df.columns and ycol in df.columns:
        return df[xcol].to_numpy(dtype=float), df[ycol].to_numpy(dtype=float)
    return None, None


def finite_diff_speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    speed = np.full_like(x, np.nan, dtype=float)
    if valid.sum() < 3:
        return speed

    xv = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    yv = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    vx = np.gradient(xv, 1.0 / fps)
    vy = np.gradient(yv, 1.0 / fps)
    sp = np.sqrt(vx**2 + vy**2)

    speed[valid] = sp[valid]
    return speed


def direction_toward_pellet(mcp_x, mcp_y, pellet_x, pellet_y, fps: float) -> np.ndarray:
    valid = np.isfinite(mcp_x) & np.isfinite(mcp_y) & np.isfinite(pellet_x) & np.isfinite(pellet_y)
    out = np.full_like(mcp_x, np.nan, dtype=float)
    if valid.sum() < 3:
        return out

    mx = pd.Series(mcp_x).interpolate(limit_direction="both").to_numpy()
    my = pd.Series(mcp_y).interpolate(limit_direction="both").to_numpy()
    px = pd.Series(pellet_x).interpolate(limit_direction="both").to_numpy()
    py = pd.Series(pellet_y).interpolate(limit_direction="both").to_numpy()

    vx = np.gradient(mx, 1.0 / fps)
    vy = np.gradient(my, 1.0 / fps)

    tx = px - mx
    ty = py - my

    denom = np.sqrt(vx**2 + vy**2) * np.sqrt(tx**2 + ty**2)
    good = valid & (denom > 1e-9)

    out[good] = (vx[good] * tx[good] + vy[good] * ty[good]) / denom[good]
    return out


def build_csv_path_from_metadata_filename(filename_value: str) -> Path:
    base = Path(str(filename_value)).stem  # strips ".csv"
    return BASE_CSV_DIR / f"{base}{DLC_POSTFIX}"


def mcp_for_camera(camera_label: str) -> str | None:
    if camera_label == CAM4_LABEL:
        return "left_mcp"
    if camera_label == CAM5_LABEL:
        return "right_mcp"
    return None


def plot_summary(df: pd.DataFrame, csv_path: Path, camera_label: str,
                 markers_to_plot_base: list[str], plot_file_appendix: str) -> Path:
    n = len(df)
    t = np.arange(n) / FPS

    # Choose MCP per camera
    mcp_marker = mcp_for_camera(camera_label)
    if camera_label == CAM4_LABEL:
        view_name = "cam4 (left view) → left_mcp"
    elif camera_label == CAM5_LABEL:
        view_name = "cam5 (right view) → right_mcp"
    else:
        view_name = str(camera_label)

    # AUTO-INCLUDE MCP (no duplicates)
    markers_to_plot = list(dict.fromkeys(
        markers_to_plot_base + ([mcp_marker] if mcp_marker else [])
    ))

    # Fetch pellet + MCP for speed/direction
    pellet_x, pellet_y = get_xy(df, "pellet")
    mcp_x, mcp_y = (None, None)
    if mcp_marker:
        mcp_x, mcp_y = get_xy(df, mcp_marker)

    speed = None
    cos_dir = None
    if (mcp_x is not None) and (pellet_x is not None):
        speed = finite_diff_speed(mcp_x, mcp_y, FPS)
        cos_dir = direction_toward_pellet(mcp_x, mcp_y, pellet_x, pellet_y, FPS)

    # Build requested marker traces
    marker_traces = []
    missing_markers = []
    for m in markers_to_plot:
        x, y = get_xy(df, m)
        if x is None:
            missing_markers.append(m)
        else:
            marker_traces.append((m, x, y))

    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.15}
    )

    # 1) selected marker X vs time
    ax = axes[0]
    for m, x, y in marker_traces:
        ax.plot(t, x, linewidth=1.0, alpha=0.9, label=m)
    ax.set_ylabel("X (px)")
    ax.set_title(f"{csv_path.name} | {view_name}")
    if marker_traces:
        ax.legend(loc="upper right", ncol=min(4, len(marker_traces)), fontsize=8, frameon=False)

    if missing_markers:
        ax.text(
            0.01, 0.98,
            f"Missing markers: {', '.join(missing_markers)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9
        )

    # 2) selected marker Y vs time
    ax = axes[1]
    for m, x, y in marker_traces:
        ax.plot(t, y, linewidth=1.0, alpha=0.9, label=m)
    ax.set_ylabel("Y (px)")

    # 3) MCP speed
    ax = axes[2]
    if speed is not None:
        ax.plot(t, speed, linewidth=1.2)
        ax.set_ylabel("MCP speed (px/s)")
    else:
        ax.text(0.5, 0.5, "MCP speed unavailable (missing MCP or pellet)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_ylabel("MCP speed")

    # 4) direction toward pellet
    ax = axes[3]
    if cos_dir is not None:
        ax.plot(t, cos_dir, linewidth=1.2)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("cos(θ)")
    else:
        ax.text(0.5, 0.5, "Direction metric unavailable (missing MCP or pellet)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_ylabel("cos(θ)")

    axes[-1].set_xlabel("Time (s)")

    out_dir = csv_path.parent / OUT_SUBFOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / f"{csv_path.stem}{plot_file_appendix}__{camera_label}__summary.png"
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -----------------------
# MAIN
# -----------------------
meta = pd.read_csv(META_PATH)

required_cols = {"filename", "camera_label"}
missing = required_cols - set(meta.columns)
if missing:
    raise ValueError(f"compiled_metadata.csv missing required columns: {missing}")

meta_sub = meta[meta["camera_label"].isin([CAM4_LABEL, CAM5_LABEL])].copy()

print(f"Rows in metadata: {len(meta)}")
print(f"Rows for cam4/cam5: {len(meta_sub)}")
print(f"Base CSV dir: {BASE_CSV_DIR}")
print(f"Base markers to plot: {MARKERS_TO_PLOT} (+ auto MCP)")
print(f"Plot appendix: {PLOT_FILE_APPENDIX}")

saved = 0
missing_files = 0
errors = 0

for _, row in meta_sub.iterrows():
    csv_path = build_csv_path_from_metadata_filename(row["filename"])

    if not csv_path.exists():
        print(f"[MISSING] {csv_path}")
        missing_files += 1
        continue

    try:
        df = read_kin_csv(csv_path)
        out_png = plot_summary(
            df=df,
            csv_path=csv_path,
            camera_label=row["camera_label"],
            markers_to_plot_base=MARKERS_TO_PLOT,
            plot_file_appendix=PLOT_FILE_APPENDIX,
        )
        print(f"[OK] {out_png}")
        saved += 1
    except Exception as e:
        print(f"[ERROR] {csv_path} -> {e}")
        errors += 1

print("\nDone.")
print(f"Saved PNGs: {saved}")
print(f"Missing CSVs: {missing_files}")
print(f"Errors: {errors}")
