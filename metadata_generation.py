#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


# -------- token regexes --------
RE_MDY = re.compile(r"^(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<y>\d{2,4})$")
RE_COHORT = re.compile(r"^Cohort[_-](?P<cohort>[A-Za-z0-9]+)$", re.IGNORECASE)
RE_CAMDIR = re.compile(r"^cam(?P<id>\d+)$", re.IGNORECASE)
RE_RATDIR = re.compile(r"^rat(?P<id>\d+)$", re.IGNORECASE)
RE_GROUP_RAT = re.compile(r"^(?P<group>[A-Za-z0-9]+)[-_]rat(?P<rat>\d+)$", re.IGNORECASE)   # 5A-rat5
RE_GROUP_SUBJECT = re.compile(r"^(?P<group>[A-Za-z0-9]+)[-_](?P<subject>\d+)$")               # 5B-2

RE_TIMESTAMP = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})(?:_(?P<rest>.+))?$"
)


def normalize_mdy(token: str) -> Optional[str]:
    """Convert M_D_YY or MM_DD_YY tokens to YYYY-MM-DD (assumes 2000+ for 2-digit years)."""
    m = RE_MDY.match(token)
    if not m:
        return None
    mm = int(m.group("m"))
    dd = int(m.group("d"))
    yy = int(m.group("y"))
    yyyy = (2000 + yy) if yy < 100 else yy
    return f"{yyyy:04d}-{mm:02d}-{dd:02d}"


def parse_timestamp_folder(token: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    m = RE_TIMESTAMP.match(token)
    if not m:
        return out
    out["timestamp_date"] = m.group("date")
    out["timestamp_time"] = f'{m.group("h")}:{m.group("m")}:{m.group("s")}'
    if m.group("rest"):
        out["timestamp_tag"] = m.group("rest")
        if m.group("rest").lower().startswith("cam"):
            out["camera_label"] = m.group("rest")
    return out


def parse_path_fields(mp4_path: Path, root: Path) -> Dict[str, object]:
    """
    Extract structured metadata from the mp4 path relative to root (experimental_videos).
    Handles your observed patterns:
      - /<MM_DD_YY>/camX/<group-subject>/<timestamp_folder>/<...>.mp4 (or no mp4 in example 1)
      - /<MM_DD_YY>/Cohort_5A/camX/<MM_DD_YY>/ratN/<timestamp_folder>/<file>.mp4
      - /post_ablate/camX/<M_D_YY>/<group-rat>/<timestamp_folder>/<file>.mp4
    """
    rel = mp4_path.relative_to(root)
    parts = rel.parts

    out: Dict[str, object] = {
        "root": str(root),
        "relative_path": str(rel),
        "filename": mp4_path.name,
        "mp4_path": str(mp4_path),
    }

    # phase if present
    for p in parts:
        if p in {"post_ablate", "pre_ablate", "baseline", "training"}:
            out["phase"] = p
            break

    # gather MDY tokens in order
    mdy_raw: List[str] = []
    mdy_norm: List[str] = []

    for p in parts:
        # cohort
        m = RE_COHORT.match(p)
        if m and "cohort" not in out:
            out["cohort"] = m.group("cohort")
            continue

        # cam directory
        m = RE_CAMDIR.match(p)
        if m and "cam_id" not in out:
            out["cam_id"] = int(m.group("id"))
            continue

        # rat directory
        m = RE_RATDIR.match(p)
        if m and "rat_id" not in out:
            out["rat_id"] = int(m.group("id"))
            continue

        # 5A-rat5
        m = RE_GROUP_RAT.match(p)
        if m:
            out.setdefault("group", m.group("group"))
            out.setdefault("rat_id", int(m.group("rat")))
            continue

        # 5B-2
        m = RE_GROUP_SUBJECT.match(p)
        if m and "group" not in out:
            out["group"] = m.group("group")
            out["subject_id"] = int(m.group("subject"))
            continue

        # date-ish folder (01_16_26 or 1_9_26)
        mdy = normalize_mdy(p)
        if mdy:
            mdy_raw.append(p)
            mdy_norm.append(mdy)
            continue

        # timestamp folder
        ts = parse_timestamp_folder(p)
        for k, v in ts.items():
            out.setdefault(k, v)

    # Assign MDY tokens heuristically
    if mdy_norm:
        out["mdy_tokens"] = "|".join(mdy_raw)
        out["mdy_dates"] = "|".join(mdy_norm)
        out["experiment_day_raw"] = mdy_raw[0]
        out["experiment_day"] = mdy_norm[0]
        if len(mdy_norm) >= 2:
            out["session_day_raw"] = mdy_raw[1]
            out["session_day"] = mdy_norm[1]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        type=str,
        help="Path to experimental_videos directory"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="compiled_metadata.csv",
        help="Output CSV filename saved inside root"
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="**/*.mp4",
        help="Glob pattern relative to root (default: **/*.mp4)"
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    mp4_files = sorted(root.glob(args.glob))
    if not mp4_files:
        raise SystemExit(f"No mp4 files matched pattern '{args.glob}' under {root}")

    rows: List[Dict[str, object]] = []
    for mp4 in mp4_files:
        try:
            rows.append(parse_path_fields(mp4, root))
        except Exception as e:
            rows.append({
                "root": str(root),
                "mp4_path": str(mp4),
                "relative_path": str(mp4.relative_to(root)),
                "parse_error": f"{type(e).__name__}: {e}",
            })

    df = pd.DataFrame(rows)

    # Nice-to-have: stable column order for key fields
    preferred = [
        "mp4_path", "relative_path", "filename",
        "phase", "cohort", "group", "rat_id", "subject_id",
        "cam_id",
        "experiment_day", "session_day",
        "timestamp_date", "timestamp_time", "camera_label", "timestamp_tag",
        "mdy_tokens", "mdy_dates",
        "parse_error",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_path = root / args.out
    df.to_csv(out_path, index=False)

    print(f"Root: {root}")
    print(f"MP4s found: {len(mp4_files)}")
    print(f"Wrote: {out_path}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
