from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# ==========================
# EDIT THESE TWO LINES ONLY
# ==========================
METADATA_CSV = "./compiled_metadata.csv"   # <-- path to metadata csv
CALIBRATION_ROOT = "./calibration"  # <-- parent calibration folder
DATE_COLUMN = "experiment_day"  # or "timestamp_date"
# ==========================


# --- regex patterns ---
DATE_RE = re.compile(r"^(?P<date>\d{2}_\d{2}_\d{2})_")
CAM_RE  = re.compile(r"camcommerb(?P<cam>\d{2})", re.IGNORECASE)
KIND_RE = re.compile(r"(DLTcoefs|xypts)\.csv$", re.IGNORECASE)


def yyyy_mm_dd_to_mm_dd_yy(series):
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().any():
        raise ValueError("Unparseable dates found in metadata.")
    return dt.dt.strftime("%m_%d_%y")


# --- 1. load metadata ---
df = pd.read_csv(METADATA_CSV)
assert DATE_COLUMN in df.columns, f"{DATE_COLUMN} not found in metadata"

df["_date_key"] = yyyy_mm_dd_to_mm_dd_yy(df[DATE_COLUMN])


# --- 2. find all calculated_calibs folders ---
calibration_root = Path(CALIBRATION_ROOT)
calib_dirs = [p for p in calibration_root.rglob("calculated_calibs") if p.is_dir()]

if not calib_dirs:
    raise RuntimeError("No calculated_calibs folders found")

print(f"Found {len(calib_dirs)} calculated_calibs folder(s):")
for d in calib_dirs:
    print(" ", d)


# --- 3. index all calibration files ---
index = {}
collisions = defaultdict(list)

for d in calib_dirs:
    for f in d.glob("*.csv"):
        m_date = DATE_RE.search(f.name)
        m_cam  = CAM_RE.search(f.name)
        m_kind = KIND_RE.search(f.name)

        if not (m_date and m_cam and m_kind):
            continue

        date_key = m_date.group("date")
        cam = int(m_cam.group("cam"))     # 01..05 → 1..5
        kind = m_kind.group(1)            # DLTcoefs or xypts

        key = (date_key, cam, kind)
        if key in index:
            collisions[key].append(index[key])
            collisions[key].append(f)
        else:
            index[key] = f

if collisions:
    print("\n⚠️ COLLISIONS DETECTED:")
    for k, paths in collisions.items():
        print(f" {k}:")
        for p in set(paths):
            print("   ", p)
    raise RuntimeError("Ambiguous calibration files found")


print(f"Indexed {len(index)} calibration files")


# --- 4. append paths into metadata ---
missing = []

for cam in range(1, 6):
    df[f"cam{cam}_dltcoefs_path"] = df["_date_key"].map(
        lambda d: str(index.get((d, cam, "DLTcoefs"), ""))
    )

    df[f"cam{cam}_xypts_path"] = df["_date_key"].map(
        lambda d: str(index.get((d, cam, "xypts"), ""))
    )

    for d in df["_date_key"].unique():
        if (d, cam, "DLTcoefs") not in index:
            missing.append((d, cam, "DLTcoefs"))
        if (d, cam, "xypts") not in index:
            missing.append((d, cam, "xypts"))


df.drop(columns="_date_key", inplace=True)


# --- 5. save output ---
out_path = Path(METADATA_CSV).with_name(Path(METADATA_CSV).stem + "_with_calibs.csv")
df.to_csv(out_path, index=False)

print("\n✅ Wrote:", out_path)


# --- 6. report missing ---
missing = sorted(set(missing))
if missing:
    print("\n⚠️ Missing calibration mappings:")
    for d, cam, kind in missing:
        print(f"  date={d} cam{cam} {kind}")
else:
    print("All calibration files matched successfully 🎯")
