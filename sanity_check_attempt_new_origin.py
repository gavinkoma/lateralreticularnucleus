#!/usr/bin/env python3
"""
Spatial alignment + frame overlay + wrist trajectory plotting for rodent reach kinematics.

This version supports nested folder structures:
- Recursively indexes videos under --video_root
- Recursively indexes DLC CSVs under --csv_root
- Then matches metadata['filename_no_ext'] to:
    video: exact stem match (filename_no_ext)
    csv: file starting with filename_no_ext and containing "DLC" and ending with "_cleaned.csv" (preferred)

Mirroring robustness:
- Uses known IMG_W/IMG_H when video can't be read, so flipping still works deterministically.
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

LEFT_LABEL_HINTS = ["camcommerb4", "cam4", "left"]
RIGHT_LABEL_HINTS = ["camcommerb5", "cam5", "right"]


def is_left_view(camera_label: str) -> bool:
    s = str(camera_label).strip().lower()
    return any(h in s for h in LEFT_LABEL_HINTS) and not any(h in s for h in RIGHT_LABEL_HINTS)


def is_right_view(camera_label: str) -> bool:
    s = str(camera_label).strip().lower()
    return any(h in s for h in RIGHT_LABEL_HINTS)


def safe_int(v, default=None):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v, default=None):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def flip_x(x_px: np.ndarray, frame_width: int):
    return (frame_width - 1) - x_px


def px_to_mm(x_px: np.ndarray, y_px: np.ndarray, origin_x: float, origin_y: float, mm_per_px: float):
    # +Y up in mm
    x_mm = (x_px - origin_x) * mm_per_px
    y_mm = -(y_px - origin_y) * mm_per_px
    return x_mm, y_mm


def read_video_frame(video_path: str, frame_idx: int) -> tuple[np.ndarray, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Could not read frame {frame_idx} from video: {video_path}")

    h, w = frame.shape[:2]
    cap.release()
    return frame, w, h


def read_dlc_csv(csv_path: str) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Reads a cleaned DLC CSV with 3-row header: scorer, bodypart, coord.
    Returns:
      df: index=frame, columns MultiIndex (bodypart, coord) where coord in {x,y,likelihood}
    """
    raw = pd.read_csv(csv_path, header=[0, 1, 2])

    idx_cols = [c for c in raw.columns if str(c[0]).lower().startswith("unnamed")]
    if not idx_cols:
        idx_cols = [raw.columns[0]]

    idx_col = idx_cols[0]
    frame_index = pd.to_numeric(raw[idx_col], errors="coerce").astype("Int64")
    raw = raw.drop(columns=[idx_col])
    raw.index = frame_index
    raw = raw[~raw.index.isna()]
    raw.index = raw.index.astype(int)

    level0 = raw.columns.get_level_values(0)
    scorer_candidates = [s for s in sorted(set(level0)) if not str(s).lower().startswith("unnamed")]
    if not scorer_candidates:
        raise RuntimeError(f"Could not determine DLC scorer in: {csv_path}")
    scorer = scorer_candidates[0]

    cols = [c for c in raw.columns if c[0] == scorer]
    if not cols:
        raise RuntimeError(f"No columns for scorer '{scorer}' in: {csv_path}")

    df = raw[cols].copy()
    df.columns = pd.MultiIndex.from_tuples([(c[1], c[2]) for c in df.columns], names=["bodypart", "coord"])
    bodyparts = sorted(set(df.columns.get_level_values("bodypart")))
    return df, scorer, bodyparts


def select_bodyparts_for_side(bodyparts: list[str], side: str) -> list[str]:
    s = side.lower()
    if s not in ("left", "right"):
        return bodyparts
    preferred = [bp for bp in bodyparts if bp.lower().startswith(s + "_") or (s in bp.lower())]
    return preferred if preferred else bodyparts


def pick_wrist_bodypart(bodyparts: list[str], side: str) -> str:
    s = side.lower()
    cands = [bp for bp in bodyparts if "wrist" in bp.lower() and s in bp.lower()]
    if cands:
        return sorted(cands)[0]
    cands = [bp for bp in bodyparts if "wrist" in bp.lower()]
    if cands:
        return sorted(cands)[0]
    raise RuntimeError("Could not auto-detect a wrist bodypart (no bodypart contains 'wrist').")


# -----------------------------
# Recursive indexing
# -----------------------------

def index_videos(video_root: str) -> dict[str, str]:
    """
    Map: filename stem (no extension) -> full path
    If duplicates exist, keeps the shortest path (usually closer to root).
    """
    video_root = str(video_root)
    idx = {}
    for dirpath, _, filenames in os.walk(video_root):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in VIDEO_EXTS:
                stem = Path(fn).stem
                full = str(Path(dirpath) / fn)
                if stem not in idx:
                    idx[stem] = full
                else:
                    # prefer "closer" (shorter) path
                    if len(full) < len(idx[stem]):
                        idx[stem] = full
    return idx


def index_cleaned_csvs(csv_root: str) -> list[str]:
    """
    Return a list of candidate CSV paths under csv_root, recursively.
    """
    csv_root = str(csv_root)
    paths = []
    for dirpath, _, filenames in os.walk(csv_root):
        for fn in filenames:
            if fn.lower().endswith(".csv") and "dlc" in fn.lower() and "cleaned" in fn.lower():
                paths.append(str(Path(dirpath) / fn))
    return paths


def choose_csv_for_trial(trial: str, csv_paths: list[str]) -> str | None:
    """
    Prefer:
    - startswith trial
    - contains 'dlc'
    - endswith '_cleaned.csv' preferred
    """
    trial_low = trial.lower()

    starts = [p for p in csv_paths if Path(p).name.lower().startswith(trial_low)]
    if not starts:
        # fallback contains
        starts = [p for p in csv_paths if trial_low in Path(p).name.lower()]

    if not starts:
        return None

    # rank: cleaned suffix first, then shortest name
    def score(p):
        name = Path(p).name.lower()
        cleaned = 0 if name.endswith("_cleaned.csv") else 1
        return (cleaned, len(name), len(p))

    starts.sort(key=score)
    return starts[0]


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--metadata", required=True, help="Path to metadata.xlsx")
    ap.add_argument("--csv_root", required=True, help="ROOT folder containing cleaned DLC CSVs (can be nested).")
    ap.add_argument("--video_root", required=True, help="ROOT folder containing videos (can be nested).")
    ap.add_argument("--out_dir", required=True, help="Output folder for PNGs")

    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--only_filename_no_ext", default=None)

    ap.add_argument("--flip_left_view", action="store_true",
                    help="Flip left-view (camcommerb4/cam4) to match right-view orientation.")
    ap.add_argument("--p_cut", type=float, default=0.1)

    # Optional zoom (leave unset for full frame)
    ap.add_argument("--mm_limits", type=float, default=None,
                    help="Optional zoom: set axes to +/- this many mm around origin. Omit for full frame.")

    # Deterministic mirroring fallback
    ap.add_argument("--img_w", type=int, default=2048)
    ap.add_argument("--img_h", type=int, default=700)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_excel(args.metadata)

    required_cols = [
        "filename_no_ext", "pass_slit", "pellet_post_x", "pellet_post_y", "mm_per_px",
        "rat_id", "control_experimental", "phase", "camera_label",
        "hand_lift_off", "end_frame"
    ]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata is missing required columns: {missing}")

    if args.only_filename_no_ext:
        meta = meta[meta["filename_no_ext"].astype(str).str.contains(args.only_filename_no_ext, na=False)]
    if args.max_trials is not None:
        meta = meta.head(args.max_trials)

    # Build indices ONCE
    print(f"[INDEX] Walking video_root recursively: {args.video_root}")
    video_idx = index_videos(args.video_root)
    print(f"[INDEX] Found {len(video_idx)} video stems.")

    print(f"[INDEX] Walking csv_root recursively: {args.csv_root}")
    csv_paths = index_cleaned_csvs(args.csv_root)
    print(f"[INDEX] Found {len(csv_paths)} cleaned DLC CSVs.")

    n_done, n_fail = 0, 0

    for _, row in meta.iterrows():
        trial = str(row["filename_no_ext"]).strip()
        if not trial or trial.lower() == "nan":
            continue

        rat_id = str(row.get("rat_id", "")).strip()
        cond = str(row.get("control_experimental", "")).strip()
        phase = str(row.get("phase", "")).strip()
        camera_label = str(row.get("camera_label", "")).strip()

        pass_slit = safe_int(row.get("pass_slit"))
        hand_lift_off = safe_int(row.get("hand_lift_off"))
        end_frame = safe_int(row.get("end_frame"))

        pellet_x = safe_float(row.get("pellet_post_x"))
        pellet_y = safe_float(row.get("pellet_post_y"))
        mm_per_px = safe_float(row.get("mm_per_px"))

        if pass_slit is None or pellet_x is None or pellet_y is None or mm_per_px is None:
            print(f"[SKIP] Missing critical values for '{trial}'.")
            continue

        side = "right" if is_right_view(camera_label) else ("left" if is_left_view(camera_label) else "right")
        do_flip = bool(args.flip_left_view and side == "left")

        # Locate CSV
        csv_path = choose_csv_for_trial(trial, csv_paths)
        if not csv_path or not os.path.isfile(csv_path):
            print(f"[FAIL] No cleaned DLC CSV matched for '{trial}'.")
            n_fail += 1
            continue

        # Load DLC
        try:
            dlc, scorer, bodyparts = read_dlc_csv(csv_path)
        except Exception as e:
            print(f"[FAIL] Reading DLC CSV failed for '{trial}': {e}")
            n_fail += 1
            continue

        # Locate video by exact stem match
        video_path = video_idx.get(trial, None)

        frame_rgb = None
        frame_w, frame_h = None, None

        if video_path and os.path.isfile(video_path):
            try:
                frame_bgr, w, h = read_video_frame(video_path, pass_slit)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_w, frame_h = w, h
            except Exception as e:
                print(f"[WARN] Could not extract background frame for '{trial}': {e}")

        # fallback geometry when no frame read
        if frame_w is None or frame_h is None:
            frame_w, frame_h = args.img_w, args.img_h

        # origin in px (flip origin if needed)
        origin_x_px = pellet_x
        origin_y_px = pellet_y
        if do_flip:
            origin_x_px = float(flip_x(np.array(origin_x_px), frame_w))

        # choose markers based on view side
        use_bodyparts = select_bodyparts_for_side(bodyparts, side)

        # Pass-slit markers
        pass_points_mm = {}
        if pass_slit in dlc.index:
            row_ps = dlc.loc[pass_slit]
            for bp in use_bodyparts:
                if (bp, "x") not in dlc.columns or (bp, "y") not in dlc.columns:
                    continue
                x = row_ps[(bp, "x")]
                y = row_ps[(bp, "y")]
                lk = row_ps[(bp, "likelihood")] if (bp, "likelihood") in dlc.columns else 1.0
                if pd.isna(x) or pd.isna(y) or pd.isna(lk) or float(lk) < float(args.p_cut):
                    continue
                x = float(x); y = float(y)
                if do_flip:
                    x = float(flip_x(np.array(x), frame_w))
                x_mm, y_mm = px_to_mm(np.array(x), np.array(y), origin_x_px, origin_y_px, mm_per_px)
                pass_points_mm[bp] = (float(x_mm), float(y_mm))
        else:
            print(f"[WARN] pass_slit={pass_slit} not in DLC index for '{trial}'.")

        # Wrist trajectory
        traj_mm = None
        if hand_lift_off is not None and end_frame is not None:
            start_f = max(hand_lift_off, int(dlc.index.min()))
            stop_f = min(end_frame, int(dlc.index.max()))
            if start_f <= stop_f:
                try:
                    wrist_bp = pick_wrist_bodypart(bodyparts, side)
                    seg = dlc.loc[start_f:stop_f, [(wrist_bp, "x"), (wrist_bp, "y")]].copy()
                    if (wrist_bp, "likelihood") in dlc.columns:
                        seg_lk = dlc.loc[start_f:stop_f, (wrist_bp, "likelihood")]
                        seg = seg[seg_lk >= float(args.p_cut)]

                    xs = pd.to_numeric(seg[(wrist_bp, "x")], errors="coerce").to_numpy()
                    ys = pd.to_numeric(seg[(wrist_bp, "y")], errors="coerce").to_numpy()
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    xs, ys = xs[ok], ys[ok]
                    if xs.size > 0:
                        if do_flip:
                            xs = flip_x(xs, frame_w)
                        x_mm, y_mm = px_to_mm(xs, ys, origin_x_px, origin_y_px, mm_per_px)
                        traj_mm = (x_mm, y_mm)
                except Exception as e:
                    print(f"[WARN] Wrist trajectory skipped for '{trial}': {e}")

        # ---- Plot ----
        banner = f"trial={trial} | rat={rat_id} | condition={cond} | phase={phase}"
        out_png = os.path.join(args.out_dir, f"{trial}_passslit_overlay.png")

        fig = plt.figure(figsize=(12, 8), dpi=150)

        ax_banner = fig.add_axes([0.0, 0.92, 1.0, 0.08])
        ax_banner.set_facecolor("black")
        ax_banner.set_xticks([])
        ax_banner.set_yticks([])
        for sp in ax_banner.spines.values():
            sp.set_visible(False)
        ax_banner.text(0.01, 0.5, banner, color="white", fontsize=12,
                       va="center", ha="left", family="monospace")

        ax = fig.add_axes([0.06, 0.06, 0.90, 0.84])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.25)
        ax.axhline(0, linewidth=1, alpha=0.7)
        ax.axvline(0, linewidth=1, alpha=0.7)

        # Background with mm extent (full frame)
        if frame_rgb is not None:
            bg = frame_rgb[:, ::-1, :] if do_flip else frame_rgb

            x0_mm = (0 - origin_x_px) * mm_per_px
            x1_mm = ((frame_w - 1) - origin_x_px) * mm_per_px
            y0_mm = -((frame_h - 1) - origin_y_px) * mm_per_px
            y1_mm = -(0 - origin_y_px) * mm_per_px
            extent = [x0_mm, x1_mm, y0_mm, y1_mm]

            ax.imshow(bg, extent=extent, origin="upper")

        if traj_mm is not None:
            ax.plot(traj_mm[0], traj_mm[1], linewidth=2, alpha=0.9)

        if pass_points_mm:
            xs = [v[0] for v in pass_points_mm.values()]
            ys = [v[1] for v in pass_points_mm.values()]
            ax.scatter(xs, ys, s=18, alpha=0.95)

        # Optional zoom (omit flag for full frame)
        if args.mm_limits is not None and args.mm_limits > 0:
            ax.set_xlim(-args.mm_limits, args.mm_limits)
            ax.set_ylim(-args.mm_limits, args.mm_limits)

        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] {trial} | video={'yes' if video_path else 'no'} | csv={Path(csv_path).name} -> {out_png}")
        n_done += 1

    print(f"\nDone. Trials saved: {n_done} | failed: {n_fail}")


if __name__ == "__main__":
    main()
