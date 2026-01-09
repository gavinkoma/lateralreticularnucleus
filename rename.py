#!/usr/bin/env python3
"""
Rename LRN project .mp4 files based on their path and original datetime.

Expected structure (example):

lrn_project/Cohort_5B/11_26_25/cam1-5B/rat1/2025-11-26_12_36_48_camcommerb1/
    2025-11-26_12_36_48_camcommerb1_wbgam_h264.mp4

New filename pattern:
    Cohort_5B_cam1-5B_rat1_2025-11-26_12_36_48.mp4
"""

import argparse
import re
from pathlib import Path


DATETIME_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})")


def extract_full_datetime(filename: str) -> str | None:
    """
    Extract 'YYYY-MM-DD_HH_MM_SS' from the filename.

    Example:
        '2025-11-26_12_36_48_camcommerb1_wbgam_h264.mp4'
        -> '2025-11-26_12_36_48'
    """
    match = DATETIME_REGEX.search(filename)
    return match.group(1) if match else None


def build_new_name(mp4_path: Path) -> str | None:
    """
    Given the full path to an .mp4, build the new filename:
        <cohort>_<cam>_<rat>_<full_datetime>.mp4

    Using the example path:
        .../Cohort_5B/11_26_25/cam1-5B/rat1/<capture_dir>/<file>.mp4

    parents[0] = capture_dir
    parents[1] = rat1
    parents[2] = cam1-5B
    parents[3] = 11_26_25
    parents[4] = Cohort_5B
    """
    try:
        cohort = mp4_path.parents[4].name
        cam = mp4_path.parents[2].name
        rat = mp4_path.parents[1].name
    except IndexError:
        # Path is not deep enough / unexpected layout
        return None

    dt = extract_full_datetime(mp4_path.name)
    if dt is None:
        return None

    return f"{cohort}_{cam}_{rat}_{dt}{mp4_path.suffix}"


def rename_videos(root: Path, dry_run: bool = True) -> None:
    """
    Walk root, find all .mp4s, and either print (dry_run) or rename.
    """
    if not root.is_dir():
        print(f"❌ Root directory does not exist or is not a directory: {root}")
        return

    print(f"Scanning for .mp4 files under: {root}\n")
    if dry_run:
        print("🔎 TEST MODE (dry run): showing what *would* be renamed.\n"
              "    Use --apply to actually rename files.\n")
    else:
        print("⚠️ APPLY MODE: files will be renamed on disk.\n")

    count_total = 0
    count_renamed = 0
    count_skipped = 0

    for mp4_path in root.rglob("*.mp4"):
        count_total += 1
        rel = mp4_path.relative_to(root)

        new_name = build_new_name(mp4_path)
        if not new_name:
            print(f"  SKIP (cannot build new name) : {rel}")
            count_skipped += 1
            continue

        new_path = mp4_path.with_name(new_name)
        rel_new = new_path.relative_to(root)

        # No-op (already has desired name)
        if new_path == mp4_path:
            print(f"  NO CHANGE                        : {rel}")
            continue

        # Avoid overwriting existing files
        if new_path.exists():
            print(f"  SKIP (target exists already)     : {rel}  -->  {rel_new}")
            count_skipped += 1
            continue

        # Always print the mapping
        print(f"  RENAME: {rel}  -->  {rel_new}")

        if not dry_run:
            mp4_path.rename(new_path)
            count_renamed += 1

    print("\nSummary:")
    print(f"  Total .mp4 files found : {count_total}")
    print(f"  Renamed                : {count_renamed}")
    print(f"  Skipped                : {count_skipped}")

    if dry_run:
        print("\n✅ TEST MODE ONLY – no files were actually renamed.")
    else:
        print("\n✅ Done – files have been renamed.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename LRN project .mp4 videos using cohort, cam, rat, and datetime."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="lrn_project",
        help="Root folder to scan (default: ./lrn_project)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files (otherwise test mode / dry run).",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    ### important here ###

    dry_run = args.apply #put 'not' in front of args.apply if you want to run as test run. 
    # dry_run = not args.apply

    rename_videos(root, dry_run=dry_run)


if __name__ == "__main__":
    main()
