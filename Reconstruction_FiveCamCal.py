# 
# # ReconstructCSV — 5‑Camera DLT Reconstruction (Calibration Object)
# 
# This notebook reconstructs **3D calibration markers** from **five-camera** 2D xy CSVs and **DLT projection matrices**, then writes:
# - `3d_points.csv` — per-marker 3D coordinates, number of cameras used, and RMS reprojection error
# - `reprojections.csv` — per (marker, camera) row with measured vs projected pixels and error
# - `visibility.csv` — (optional) boolean matrix of which cameras saw which marker
# - `summary.txt` — overall and per-camera reprojection-error stats
# 
# **Note:** This notebook deliberately does **not** draw overlays. Use a separate `overlay.py` (or notebook) that reads `reprojections.csv` + images to make the PNG overlays.
# 

# ## PART 0 — Imports and Environment Setup

# --- Core library imports ---
import os        # file and path operations
import re        # regular expressions (for camera number extraction)
import glob      # pattern-based file search (e.g., *.csv)
import math      # basic math utilities (e.g., sqrt, hypot)
import argparse  # command-line argument parsing for script execution
from typing import Dict, List, Tuple  # type hints for better readability

# --- Scientific/data analysis libraries ---
import numpy as np  # numerical computing and array operations
import pandas as pd # data handling and CSV I/O


# Quick check to confirm we're using the standard-library 'glob' module
# Prints the module object, its file path, and current working directory
import sys

print("Glob object:", glob)
print("Loaded from:", getattr(glob, "__file__", "built-in"))
print("sys.path[0]:", sys.path[0])


# sanity check
print("glob module OK:", hasattr(glob, "glob"))


# 
# ## PART 1 — Configuration (paths & patterns)
# 
# - Define default folder paths for DLT coefficients, 2-D calibration points, and output results.  
# 
# - When the pipeline runs, it automatically detects cameras based on **two-digit IDs** embedded in filenames (e.g., `cal01`, `cal02`, … `cal05`).  
# 
# - We can modify these paths directly here or override them via command-line arguments if running the script outside the notebook.
# 

### --- Default directory paths for the 5-camera DLT reconstruction --- ###

# Folder containing DLT coefficient files for each camera
DEFAULT_DLT_DIR = "./DLTcoefs"                              # Example filenames: cal01_DLTcoefs.csv … cal05_DLTcoefs.csv
                                        
# Folder containing 2-D calibration marker coordinates (TyDLT xypts)
DEFAULT_XY_DIR  = "./xypts"                                 # Example filenames: cal01_xypts.csv … cal05_xypts.csv

DEFAULT_OUT_DIR = "./Multicam_Calib_Output"  # (3d_points.csv, reprojections.csv, visibility.csv, summary.txt)
                                        
### --- Regex pattern for camera ID extraction --- ###
# Detects two-digit camera numbers ('01'–'99') embedded in filenames
CAMNUM_REGEX = re.compile(r'(\d{2})')   # Used to automatically match DLT and xypts files by camera index
                                        

import numpy as np, re, pandas as pd

_NUM_PAT = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def read_dlt_matrix(path: str) -> np.ndarray:
    """
    Accepts DLT-11 or DLT-12 exports and returns a 3x4 P.
      • If 11 values: build P with L12 = 1.0 (standard DLT-11 scale choice)
      • If 12 values: reshape to (3,4)
      • Handles 11x2 or 11x4 CSVs with duplicated columns
    """
    # Try as a small CSV table first (handles 11x2, 11x4 nicely)
    try:
        df = pd.read_csv(path, header=None).dropna(axis=1, how='all')
        if df.shape[0] == 11 and df.shape[1] in (2, 3, 4):
            # collapse duplicated pairs like [col0≈col2, col1≈col3]
            if df.shape[1] >= 4:
                a0, a2 = df.iloc[:,0].astype(float).to_numpy(), df.iloc[:,2].astype(float).to_numpy()
                b1, b3 = df.iloc[:,1].astype(float).to_numpy(), df.iloc[:,3].astype(float).to_numpy()
                if np.allclose(a0, a2) and np.allclose(b1, b3):
                    df = df.iloc[:, :2]
            if df.shape[1] == 2:
                a = df.iloc[:,0].astype(float).to_numpy()
                b = df.iloc[:,1].astype(float).to_numpy()
                L = np.empty(2*len(a), float); L[0::2] = a; L[1::2] = b; L = L[:11]
                P = np.zeros((3,4), float)
                P[0,:]   = L[0:4]      # L1..L4
                P[1,:]   = L[4:8]      # L5..L8
                P[2,0:3] = L[8:11]     # L9..L11
                P[2,3]   = 1.0         # L12 implicit
                return P
    except Exception:
        pass

    # Fallback: regex all floats (handles headers/odd delimiters)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    nums = [float(x) for x in _NUM_PAT.findall(txt)]

    if len(nums) == 11:  # DLT-11
        L = np.array(nums, float)
        P = np.zeros((3,4), float)
        P[0,:]   = L[0:4]
        P[1,:]   = L[4:8]
        P[2,0:3] = L[8:11]
        P[2,3]   = 1.0
        return P

    if len(nums) >= 12:  # DLT-12
        return np.array(nums[:12], float).reshape(3,4)

    raise ValueError(f"{path} has {len(nums)} numeric values; expected 11 or 12.")


# from glob import glob
for f in sorted(glob.glob("./DLTcoefs/*")):
    try:
        P = read_dlt_matrix(f)
        print(os.path.basename(f), "->", P.shape)
    except Exception as e:
        print(os.path.basename(f), "ERROR:", e)


# 
# ## PART 2 — IO Helpers (DLT & XY readers, file discovery)
# 
# - `cam_num_from_path` extracts the two-digit camera ID from filenames.
# 
# - `read_dlt_matrix` reads 3×4 DLT matrices (comma or whitespace-delimited).
# 
# - `read_xy_csv` reads TyDLT xy CSVs and coerces to numeric.
# 
# - `discover_files` pairs cameras present in **both** DLT and XY folders.
# 

def cam_num_from_path(path: str) -> int:
    '''Extract a two-digit camera number like 01..99 from filename; return int or None.'''
    base = os.path.basename(path)
    m = CAMNUM_REGEX.search(base)
    return int(m.group(1)) if m else None


def read_xy_csv(path: str) -> pd.DataFrame:
    '''
    Read TyDLT xy CSV; expected columns x,y (header optional).
    Returns a DataFrame with columns ['x','y'] (floats). NaNs allowed.
    '''
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['x', 'y']
        else:
            df = pd.read_csv(path, header=None)
            df = df.iloc[:, :2]
            df.columns = ['x', 'y']
    except Exception:
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :2]
        df.columns = ['x', 'y']

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    return df


def discover_files(dlt_dir: str, xy_dir: str):
    '''
    Find DLT and XY files and map cam_num -> filepath for cameras present in BOTH sets.
    '''
    dlt_files = sorted(glob.glob(os.path.join(dlt_dir, "*")))
    xy_files  = sorted(glob.glob(os.path.join(xy_dir, "*")))

    dlt_map, xy_map = {}, {}
    for f in dlt_files:
        cam = cam_num_from_path(f)
        if cam is not None:
            dlt_map[cam] = f
    for f in xy_files:
        cam = cam_num_from_path(f)
        if cam is not None:
            xy_map[cam] = f

    cams = sorted(set(dlt_map.keys()) & set(xy_map.keys()))
    dlt_map = {c: dlt_map[c] for c in cams}
    xy_map  = {c: xy_map[c]  for c in cams}
    return dlt_map, xy_map


# 
# ## PART 3 — Geometry (Triangulation & Projection)
# 
# - `triangulate_homogeneous` uses SVD on stacked DLT equations to solve for X.
# 
# - `project_point` reprojects X back to each camera to compute pixel error.
# 

from typing import Tuple

def triangulate_homogeneous(points_uv, proj_mats, eps: float = 1e-12) -> Tuple[np.ndarray, bool]:
    '''
    Triangulate a 3D point from N>=2 camera observations using homogeneous DLT/SVD.
    points_uv: list of (u, v) floats for each camera
    proj_mats: list of 3x4 matrices P for each camera (same order as points_uv)
    Returns: (X, success) where X is shape (3,)
    '''
    A = []
    for (u, v), P in zip(points_uv, proj_mats):
        p1, p2, p3 = P[0, :], P[1, :], P[2, :]
        A.append(u * p3 - p1)
        A.append(v * p3 - p2)
    A = np.vstack(A)  # (2N, 4)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1, :]
    if abs(X_h[-1]) < eps:
        return np.array([np.nan, np.nan, np.nan]), False
    X = X_h[:3] / X_h[3]
    return X, True


def project_point(P: np.ndarray, X3: np.ndarray, eps: float = 1e-12):
    '''Project 3D point X3 (3,) into image using 3x4 P; return (u, v).'''
    X_h = np.hstack([X3, 1.0])
    uvw = P @ X_h
    if abs(uvw[2]) < eps:
        return np.nan, np.nan
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


# 
# ## PART 4 — Visibility Matrix & Data Assembly
# 
# We create a boolean visibility matrix of shape *(n_markers × n_cams)*, where visibility means finite x **and** y in the xy CSV.
# 

def build_visibility_and_arrays(XY: Dict[int, pd.DataFrame], cams: List[int]):
    '''
    Build a boolean visibility matrix (n_markers * n_cams). A marker is visible in cam
    if both x and y are finite for that row.
    Returns: (vis, n_markers)
    '''
    n_markers = max(len(XY[c]) for c in cams)
    vis = np.zeros((n_markers, len(cams)), dtype=bool)
    for j, cam in enumerate(cams):
        df = XY[cam]
        x = df['x'].reindex(range(n_markers)).to_numpy()
        y = df['y'].reindex(range(n_markers)).to_numpy()
        vis[:, j] = np.isfinite(x) & np.isfinite(y)
    return vis, n_markers


# 
# ## PART 5 — Reconstruction Loop (writes CSVs + summary) for Each Marker
# 
# 1. We gather all cameras with valid measurements for each marker.
# 
# 2. If ≥ 2 cameras, we triangulate 3D.
# 
# 3. We reproject to each *used* camera, and then compute per-camera pixel error.
# 
# 4. We write `3d_points.csv`, `reprojections.csv`, `visibility.csv` (optional), and `summary.txt`.
# 

def run_reconstruction(dlt_dir: str = DEFAULT_DLT_DIR,
                       xy_dir: str = DEFAULT_XY_DIR,
                       out_dir: str = DEFAULT_OUT_DIR,
                       write_visibility: bool = True):
    '''
    Returns: paths to (3d_points.csv, reprojections.csv, visibility.csv|None, summary.txt)
    '''
    os.makedirs(out_dir, exist_ok=True)

    dlt_map, xy_map = discover_files(dlt_dir, xy_dir)
    if not dlt_map or not xy_map:
        raise RuntimeError("No overlapping cameras found between DLT and XY folders. "
                           "Ensure filenames include two-digit cam numbers like 01..05.")

    cams = sorted(dlt_map.keys())
    P   = {c: read_dlt_matrix(dlt_map[c]) for c in cams}
    XY  = {c: read_xy_csv(xy_map[c]) for c in cams}

    vis, n_markers = build_visibility_and_arrays(XY, cams)

    rows_3d, rows_rep = [], []

    for m in range(n_markers):
        obs_uv, obs_P, used_cams = [], [], []
        for cam in cams:
            df = XY[cam]
            if m < len(df):
                um = df.iloc[m, 0]
                vm = df.iloc[m, 1]
                if np.isfinite(um) and np.isfinite(vm):
                    obs_uv.append((float(um), float(vm)))
                    obs_P.append(P[cam])
                    used_cams.append(cam)

        if len(obs_uv) < 2:
            rows_3d.append({"marker_id": m + 1, "X": np.nan, "Y": np.nan, "Z": np.nan,
                            "n_cams": len(obs_uv), "RMS_px": np.nan})
            continue

        X3, ok = triangulate_homogeneous(obs_uv, obs_P)
        if (not ok) or (not np.isfinite(X3).all()):
            rows_3d.append({"marker_id": m + 1, "X": np.nan, "Y": np.nan, "Z": np.nan,
                            "n_cams": len(obs_uv), "RMS_px": np.nan})
            continue

        errs = []
        for (um, vm), cam in zip(obs_uv, used_cams):
            up, vp = project_point(P[cam], X3)
            err = math.hypot(um - up, vm - vp)
            errs.append(err)
            rows_rep.append({
                "marker_id": m + 1,
                "cam": cam,
                "u_meas": um, "v_meas": vm,
                "u_proj": up, "v_proj": vp,
                "err_px": err
            })

        rms = float(np.sqrt(np.mean(np.asarray(errs) ** 2))) if errs else np.nan
        rows_3d.append({"marker_id": m + 1, "X": X3[0], "Y": X3[1], "Z": X3[2],
                        "n_cams": len(obs_uv), "RMS_px": rms})

    df_3d = pd.DataFrame(rows_3d)
    df_rep = pd.DataFrame(rows_rep)

    path_3d = os.path.join(out_dir, "3d_points.csv")
    path_rep = os.path.join(out_dir, "reprojections.csv")
    df_3d.to_csv(path_3d, index=False)
    df_rep.to_csv(path_rep, index=False)

    path_vis = None
    if write_visibility:
        vis_df = pd.DataFrame(vis, columns=[f"cam{c:02d}" for c in cams])
        path_vis = os.path.join(out_dir, "visibility.csv")
        vis_df.to_csv(path_vis, index=False)

    lines = []
    lines.append(f"Total cameras: {len(cams)} — {cams}")
    lines.append(f"Total markers: {n_markers}")
    lines.append(f"Markers with 3D: {int(df_3d['X'].notna().sum())}")
    if not df_rep.empty:
        overall = df_rep['err_px'].mean()
        lines.append(f"Overall mean reprojection error (px): {overall:.4f}")
        for cam in cams:
            mu = df_rep.loc[df_rep['cam'] == cam, 'err_px'].mean()
            n  = df_rep.loc[df_rep['cam'] == cam, 'err_px'].size
            lines.append(f"Cam {cam:02d}: mean={mu:.4f} px  (n={n})")
    else:
        lines.append("No reprojection rows (no markers had >=2 visible cameras).")

    path_sum = os.path.join(out_dir, "summary.txt")
    with open(path_sum, "w") as f:
        f.write("\n".join(lines))

    return path_3d, path_rep, (path_vis or ""), path_sum


# 
# ## PART 6 — (Optional) CLI Entrypoint
# 
# If we want to export this notebook as a script, we can run it from the command line.
# 

def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera DLT reconstruction to CSVs")
    p.add_argument("--dlt_dir", type=str, default=DEFAULT_DLT_DIR,
                   help="Folder with 3x4 DLT matrices (filenames contain two-digit cam numbers)")
    p.add_argument("--xy_dir", type=str, default=DEFAULT_XY_DIR,
                   help="Folder with TyDLT xy CSVs (filenames contain two-digit cam numbers)")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Output folder for CSVs and summary")
    p.add_argument("--no_visibility", action="store_true",
                   help="Do not write visibility.csv")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    path_3d, path_rep, path_vis, path_sum = run_reconstruction(
        dlt_dir=args.dlt_dir,
        xy_dir=args.xy_dir,
        out_dir=args.out_dir,
        write_visibility=not args.no_visibility
    )

    print("Wrote:")
    print("  3D:", path_3d)
    print("  Reprojections:", path_rep)
    if path_vis:
        print("  Visibility:", path_vis)
    print("  Summary:", path_sum)


# 
# ## Execute Pipeline
# 
# Set our folder paths and run the full reconstruction pipeline end-to-end.
# 
# This executes all steps:
# 
# 1. We discover DLT and xypts files (by camera number)
# 
# 2. We reconstruct 3D calibration points
# 
# 3. We compute reprojection errors
# 
# 4. We write all output CSVs and summary text into `Multicam_Calib_Output/`
# 

# Adjust these if our data is already uploaded/mounted:
dlt_dir = DEFAULT_DLT_DIR
xy_dir  = DEFAULT_XY_DIR
out_dir = DEFAULT_OUT_DIR

# Run:
try:
    p3d, prep, pvis, psum = run_reconstruction(dlt_dir, xy_dir, out_dir, write_visibility=True)
    print("3D points CSV:", p3d)
    print("Reprojections CSV:", prep)
    print("Visibility CSV:", pvis)
    print("Summary TXT:", psum)
except Exception as e:
    print("⚠️ Setup hint — make sure our folders exist and filenames include two‑digit cam numbers like 01..05")
    print("dlt_dir:", dlt_dir)
    print("xy_dir :", xy_dir)
    print("out_dir:", out_dir)
    print("Error:", e)




# # qwe47890-=-
# 

# ## If entire pipeline executes well, output should display as:
# 
# 3D points CSV: ./Multicam_Calib_Output/3d_points.csv
# 
# Reprojections CSV: ./Multicam_Calib_Output/reprojections.csv
# 
# Visibility CSV: ./Multicam_Calib_Output/visibility.csv
# 
# Summary TXT: ./Multicam_Calib_Output/summary.txt

