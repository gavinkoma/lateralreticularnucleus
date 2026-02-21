#!/usr/bin/env python3
"""
Extract ONE frame per trial (the temporal alignment frame: metadata['pass_slit'])
and save it as a PNG so you can manually verify / fix origins later.

- Reads metadata.xlsx (must contain: filename_no_ext, pass_slit)
- Recursively searches for videos under --video_root (nested subfolders OK)
- Matches by exact video stem == filename_no_ext (most common)
  + fallback: any video filename that *contains* filename_no_ext
- Extracts that frame using OpenCV and writes to --out_dir

Optional:
- --banner : draw a simple black banner with trial + frame number on top
"""

import argparse
import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def safe_int(v, default=None):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def index_videos(video_root: str) -> dict[str, list[str]]:
    """
    Map: stem -> [fullpaths] (keep list to handle duplicates)
    """
    idx = {}
    for dirpath, _, filenames in os.walk(video_root):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in VIDEO_EXTS:
                stem = Path(fn).stem
                full = str(Path(dirpath) / fn)
                idx.setdefault(stem, []).append(full)
    return idx


def choose_best_path(paths: list[str]) -> str:
    # Prefer shortest path (often closer to root), then shortest filename
    paths = sorted(paths, key=lambda p: (len(p), len(Path(p).name)))
    return paths[0]


def find_video_path(trial_stem: str, video_idx: dict[str, list[str]], video_root: str) -> str | None:
    # 1) exact stem match
    if trial_stem in video_idx:
        return choose_best_path(video_idx[trial_stem])

    # 2) fallback: contains match (can be slower; we do a light scan)
    # We avoid full os.walk again by scanning the indexed stems first.
    trial_low = trial_stem.lower()
    candidates = []
    for stem, paths in video_idx.items():
        if trial_low in stem.lower():
            for p in paths:
                candidates.append(p)

    if candidates:
        return choose_best_path(candidates)

    return None


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from: {video_path}")

    return frame  # BGR uint8


def add_banner_bgr(img_bgr: np.ndarray, text: str, banner_h: int = 70) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    out = np.vstack([banner, img_bgr])

    # Put text (white)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    x = 12
    y = int(banner_h * 0.65)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to metadata .xlsx")
    ap.add_argument("--video_root", required=True, help="ROOT directory containing videos (nested OK)")
    ap.add_argument("--out_dir", required=True, help="Output directory for extracted PNGs")
    ap.add_argument("--max_trials", type=int, default=None, help="Optional cap for quick tests")
    ap.add_argument("--only_filename_no_ext", default=None, help="Process only trials whose filename_no_ext contains this substring")
    ap.add_argument("--banner", action="store_true", help="Add a black banner with trial + frame number")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_excel(args.metadata)

    required_cols = ["filename_no_ext", "pass_slit"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")

    if args.only_filename_no_ext:
        meta = meta[meta["filename_no_ext"].astype(str).str.contains(args.only_filename_no_ext, na=False)]

    if args.max_trials is not None:
        meta = meta.head(args.max_trials)

    print(f"[INDEX] Recursively scanning videos under: {args.video_root}")
    video_idx = index_videos(args.video_root)
    print(f"[INDEX] Found {sum(len(v) for v in video_idx.values())} video files ({len(video_idx)} unique stems).")

    n_ok, n_fail, n_skip = 0, 0, 0

    for _, row in meta.iterrows():
        trial = str(row["filename_no_ext"]).strip()
        if not trial or trial.lower() == "nan":
            n_skip += 1
            continue

        frame_idx = safe_int(row.get("pass_slit"), default=None)
        if frame_idx is None:
            print(f"[SKIP] {trial}: pass_slit is missing/invalid")
            n_skip += 1
            continue

        video_path = find_video_path(trial, video_idx, args.video_root)
        if not video_path or not os.path.isfile(video_path):
            print(f"[FAIL] {trial}: video not found under video_root")
            n_fail += 1
            continue

        try:
            frame_bgr = read_frame(video_path, frame_idx)
            if args.banner:
                frame_bgr = add_banner_bgr(frame_bgr, f"{trial} | pass_slit={frame_idx}")

            out_path = os.path.join(args.out_dir, f"{trial}_pass_slit_{frame_idx:06d}.png")
            ok = cv2.imwrite(out_path, frame_bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")

            print(f"[OK] {trial} -> {out_path}")
            n_ok += 1

        except Exception as e:
            print(f"[FAIL] {trial}: {e}")
            n_fail += 1

    print(f"\nDone. Saved={n_ok} | failed={n_fail} | skipped={n_skip}")


if __name__ == "__main__":
    main()
