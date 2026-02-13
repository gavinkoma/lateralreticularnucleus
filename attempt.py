#!/usr/bin/env python3
"""
kclean.py — manual kinematics cleaner for DeepLabCut CSVs.

Keys:
  i     : identity swap (swap selected frames between source marker(s) and chosen target marker)
  d     : delete selected frames (sets likelihood=0 for those frames; x/y kept)
  l     : fill low-likelihood segment for a chosen marker in current x-range (then click a line)
  s     : save as *_cleaned.csv
  esc   : clear selection + cancel fill

Likelihood cutoff:
  Slider (0..1) controls which points are considered "valid" for display/selection/fill.
  - Points with likelihood < cutoff are hidden and ignored by lasso selection.
  - Fill interpolates x/y across frames where likelihood < cutoff (within current view), then sets
    likelihood on those frames to cutoff.

NEW:
  Optional metadata lookup from an XLSX containing reach windows:
    hand_lift_off (start), pass_slit (mid), end_frame (end)

  Usage:
    python kclean.py file.csv
    python kclean.py file.csv --meta compiled_metadata_sorted.xlsx
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.widgets import LassoSelector, Slider


# ---------------- utilities ---------------- #

def read_dlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if df.columns.nlevels != 3:
        raise ValueError("Expected DLC CSV with 3 header levels: scorer/bodypart/coord.")
    return df


def bodyparts_in_file_order(df: pd.DataFrame) -> List[str]:
    return list(pd.unique(df.columns.get_level_values(1)))


def choose_from_list(title: str, items: List[str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        out = simpledialog.askstring(title, "Type one of:\n" + ", ".join(items))
        root.destroy()
        if out is None:
            return None
        out = out.strip()
        return out if out in items else None
    except Exception:
        print("\n" + title)
        for i, it in enumerate(items):
            print(f"  [{i}] {it}")
        s = input("Enter index (blank to cancel): ").strip()
        if not s:
            return None
        try:
            j = int(s)
            return items[j] if 0 <= j < len(items) else None
        except Exception:
            return None


def _scorer(df: pd.DataFrame) -> str:
    return str(df.columns.get_level_values(0)[0])


def _colkey(df: pd.DataFrame, bp: str, coord: str):
    return (_scorer(df), bp, coord)


def _colpos(df: pd.DataFrame, colkey) -> int:
    cols = df.columns
    hits = np.where(cols == colkey)[0]
    if hits.size == 0:
        raise KeyError(f"Column not found: {colkey}")
    return int(hits[0])


def _read_col(df: pd.DataFrame, j: int) -> np.ndarray:
    return df.iloc[:, j].to_numpy(dtype=float)


# ---------------- metadata lookup ---------------- #

def _norm_name(s: str) -> str:
    """Normalize filenames/paths for matching."""
    s = str(s).strip().replace("\\", "/")
    s = os.path.basename(s)
    if s.lower().endswith(".csv"):
        s = s[:-4]
    return s.lower()


def _strip_dlc_suffix(name_no_ext: str) -> str:
    """
    DLC file example:
      postablate_..._camcommerb1DLC_resnet50_..._250000
    Metadata often has:
      postablate_..._camcommerb1
    This strips everything from 'dlc' onward.
    """
    s = name_no_ext
    idx = s.lower().find("dlc")
    if idx != -1:
        s = s[:idx]
    # also strip trailing underscores
    return s.rstrip("_").lower()


def _find_existing_meta_path(csv_path: str, meta_arg: Optional[str]) -> Optional[str]:
    if meta_arg:
        return meta_arg if os.path.exists(meta_arg) else None

    # Try a few common locations
    candidates = [
        os.path.join(os.path.dirname(csv_path), "compiled_metadata_sorted.xlsx"),
        os.path.join(os.getcwd(), "compiled_metadata_sorted.xlsx"),
        os.path.join(os.path.dirname(os.getcwd()), "compiled_metadata_sorted.xlsx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _best_filename_column(df_meta: pd.DataFrame) -> Optional[str]:
    """
    Heuristic: pick a column whose name suggests filenames and whose values look like .csv-ish strings.
    """
    cols = list(df_meta.columns)
    if not cols:
        return None

    # Prefer likely name columns
    preferred = []
    for c in cols:
        cl = str(c).lower()
        score = 0
        if "file" in cl or "name" in cl or "csv" in cl or "video" in cl or "path" in cl:
            score += 2
        if "dlc" in cl:
            score += 1
        preferred.append((score, c))
    preferred.sort(reverse=True, key=lambda x: x[0])

    # Validate by sampling values
    for _score, c in preferred:
        ser = df_meta[c].dropna().astype(str)
        if ser.empty:
            continue
        sample = ser.head(50).str.lower()
        looks_like = sample.str.contains("camcommerb").mean() if len(sample) else 0.0
        has_csv = sample.str.contains(".csv").mean() if len(sample) else 0.0
        has_dlc = sample.str.contains("dlc").mean() if len(sample) else 0.0
        if (looks_like >= 0.10) or (has_csv >= 0.10) or (has_dlc >= 0.10):
            return c

    # Fallback: first object-like column
    for c in cols:
        if df_meta[c].dtype == object:
            return c
    return None


def lookup_reach_window(
    csv_path: str,
    meta_path: Optional[str],
    verbose: bool = True,
) -> Optional[Tuple[int, int, int]]:
    """
    Returns (hand_lift_off, pass_slit, end_frame) if found, else None.
    """
    if not meta_path or not os.path.exists(meta_path):
        if verbose:
            print("[meta] no metadata file found; skipping lookup")
        return None

    try:
        dfm = pd.read_excel(meta_path)
    except Exception as e:
        print(f"[meta] failed to read metadata xlsx: {meta_path}\n  {e}")
        return None

    # Find required frame columns (case-insensitive)
    colmap = {str(c).lower(): c for c in dfm.columns}
    def pick_col(name: str) -> Optional[str]:
        # exact or contains fallback
        if name in colmap:
            return colmap[name]
        # mild fuzzy
        for k, orig in colmap.items():
            if name in k:
                return orig
        return None

    c_start = pick_col("hand_lift_off")
    c_mid   = pick_col("pass_slit")
    c_end   = pick_col("end_frame")
    if not (c_start and c_mid and c_end):
        print("[meta] missing required columns. Need hand_lift_off, pass_slit, end_frame (or close names).")
        print("       columns present:", list(dfm.columns))
        return None

    c_name = _best_filename_column(dfm)
    if not c_name:
        print("[meta] could not infer filename column in metadata.")
        print("       columns present:", list(dfm.columns))
        return None

    # Build key for opened CSV
    csv_key = _strip_dlc_suffix(_norm_name(csv_path))

    # Build keys for metadata rows
    names = dfm[c_name].fillna("").astype(str).tolist()
    meta_keys = [_strip_dlc_suffix(_norm_name(x)) for x in names]

    # Exact match first
    hits = [i for i, k in enumerate(meta_keys) if k == csv_key]

    # If no exact hits, try substring containment (helps if metadata has extra prefixes)
    if not hits:
        hits = [i for i, k in enumerate(meta_keys) if (csv_key in k) or (k in csv_key)]

    if not hits:
        if verbose:
            print(f"[meta] no match for key='{csv_key}' using column '{c_name}' in {os.path.basename(meta_path)}")
        return None

    # If multiple hits, pick the first non-null window if possible
    for idx in hits:
        row = dfm.iloc[idx]
        try:
            s = int(round(float(row[c_start])))
            m = int(round(float(row[c_mid])))
            e = int(round(float(row[c_end])))
            if verbose:
                print(f"[meta] matched row {idx} using '{c_name}' -> window: start={s}, mid={m}, end={e}")
            return (s, m, e)
        except Exception:
            continue

    if verbose:
        print("[meta] found matching row(s) but could not parse frame columns as integers.")
    return None


# ---------------- data classes ---------------- #

@dataclass(frozen=True)
class SelectedPoint:
    bp: str
    coord: str  # "x" or "y"
    pos: int    # positional row index


# ---------------- main GUI ---------------- #

class DLCCleanGUI:
    def __init__(self, csv_path: str, meta_path: Optional[str] = None):
        self.csv_path = csv_path
        self.df = read_dlc_csv(csv_path)

        # reach window (optional)
        meta_path_found = _find_existing_meta_path(csv_path, meta_path)
        self.meta_path = meta_path_found
        win = lookup_reach_window(csv_path, meta_path_found, verbose=True)
        self.reach_start: Optional[int] = None
        self.reach_mid: Optional[int] = None
        self.reach_end: Optional[int] = None
        if win is not None:
            self.reach_start, self.reach_mid, self.reach_end = win

        # order: last bodypart in file at top -> first at bottom
        self.bodyparts_file_order = bodyparts_in_file_order(self.df)
        self.bodyparts = list(reversed(self.bodyparts_file_order))

        self.n = len(self.df)
        self.t = np.arange(self.n)

        self.pcutoff = 0.60
        self.selected: List[SelectedPoint] = []
        self.await_fill_click = False
        self._did_autoscale = False

        # --- layout ---
        self.fig = plt.figure(figsize=(14, 6))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[4.6, 1.4, 1.0], wspace=0.25)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_hm = self.fig.add_subplot(gs[0, 1])
        self.ax_leg = self.fig.add_subplot(gs[0, 2])
        self.ax_leg.axis("off")

        # slider
        slider_ax = self.fig.add_axes([0.72, 0.03, 0.23, 0.03])
        self.slider = Slider(slider_ax, "p cutoff", 0.0, 1.0, valinit=self.pcutoff)
        self.slider.on_changed(self._on_pcutoff)

        # colors: red top -> purple bottom
        cmap = plt.cm.rainbow
        vals = np.linspace(1.0, 0.0, len(self.bodyparts))
        self.bp_color: Dict[str, Tuple[float, float, float, float]] = {
            bp: cmap(v) for bp, v in zip(self.bodyparts, vals)
        }

        # artists
        self.lines: Dict[Tuple[str, str], Line2D] = {}
        self.scatter = None
        self.im = None
        self.cbar = None

        # reach window artists
        self._reach_artists = []

        # status
        self.status = self.ax.text(
            0.01, 0.99, "",
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.15)
        )

        self._init_plots()
        self._connect()

    # ---------- plotting ---------- #

    def _set_status(self, msg: str):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _likelihood_matrix(self) -> np.ndarray:
        rows = []
        for bp in self.bodyparts:
            jp = _colpos(self.df, _colkey(self.df, bp, "likelihood"))
            rows.append(_read_col(self.df, jp))
        return np.vstack(rows)

    def _clear_reach_guides(self):
        for a in self._reach_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._reach_artists = []

    def _draw_reach_guides(self):
        self._clear_reach_guides()
        if self.reach_start is None or self.reach_end is None:
            return

        s = int(np.clip(self.reach_start, 0, self.n - 1))
        e = int(np.clip(self.reach_end,   0, self.n - 1))
        m = None if self.reach_mid is None else int(np.clip(self.reach_mid, 0, self.n - 1))

        # shaded window
        span = self.ax.axvspan(s, e, alpha=0.08)
        self._reach_artists.append(span)

        # vertical lines
        ls = self.ax.axvline(s, linestyle="--", lw=1.2)
        le = self.ax.axvline(e, linestyle="--", lw=1.2)
        self._reach_artists.extend([ls, le])

        if m is not None:
            lm = self.ax.axvline(m, linestyle=":", lw=1.2)
            self._reach_artists.append(lm)

        # annotate
        txt = f"reach window: start={s}, mid={m if m is not None else 'NA'}, end={e}"
        self._set_status(txt)

    def _init_plots(self):
        self.ax.set_title("DLC trajectories (x dashed, y solid) | lasso select | keys: i d l s esc")
        self.ax.set_xlabel("frame (row index)")
        self.ax.set_ylabel("pixels")

        # Create lines once
        for bp in self.bodyparts:
            col = self.bp_color[bp]
            lx, = self.ax.plot([], [], linestyle="--", lw=1.0, alpha=0.85, color=col)
            ly, = self.ax.plot([], [], linestyle="-",  lw=1.0, alpha=0.85, color=col)
            lx.set_picker(5)
            ly.set_picker(5)
            self.lines[(bp, "x")] = lx
            self.lines[(bp, "y")] = ly

        # Heatmap once
        P = self._likelihood_matrix()
        self.im = self.ax_hm.imshow(P, aspect="auto", origin="lower",
                                    interpolation="nearest", vmin=0.0, vmax=1.0)
        self.ax_hm.set_title("likelihood")
        self.ax_hm.set_yticks(range(len(self.bodyparts)))
        self.ax_hm.set_yticklabels(self.bodyparts, fontsize=8)
        self.ax_hm.set_xticks([])

        self.cbar = self.fig.colorbar(self.im, ax=self.ax_hm, fraction=0.046, pad=0.04)
        self.cbar.set_label("p")

        self._draw_legend()
        self._refresh_all(force_autoscale=True)

        # Apply reach window zoom after initial autoscale (no behavioral changes elsewhere)
        if self.reach_start is not None and self.reach_end is not None:
            s = int(np.clip(self.reach_start, 0, self.n - 1))
            e = int(np.clip(self.reach_end,   0, self.n - 1))
            if e > s:
                self.ax.set_xlim(s, e)
            self._draw_reach_guides()

    def _draw_legend(self):
        handles = [Line2D([0], [0], color=self.bp_color[bp], lw=2, label=bp) for bp in self.bodyparts]
        self.ax_leg.legend(handles=handles, loc="upper left", frameon=False,
                           title="Markers", fontsize=9, title_fontsize=10)

    def _refresh_all(self, force_autoscale: bool = False):
        ymins, ymaxs = [], []

        for bp in self.bodyparts:
            jx = _colpos(self.df, _colkey(self.df, bp, "x"))
            jy = _colpos(self.df, _colkey(self.df, bp, "y"))
            jp = _colpos(self.df, _colkey(self.df, bp, "likelihood"))

            x = _read_col(self.df, jx)
            y = _read_col(self.df, jy)
            p = _read_col(self.df, jp)

            keep = np.isfinite(p) & (p >= self.pcutoff)

            x_plot = x.copy()
            y_plot = y.copy()
            x_plot[~keep] = np.nan
            y_plot[~keep] = np.nan

            self.lines[(bp, "x")].set_data(self.t, x_plot)
            self.lines[(bp, "y")].set_data(self.t, y_plot)

            if np.any(keep):
                ymins.extend([np.nanmin(x[keep]), np.nanmin(y[keep])])
                ymaxs.extend([np.nanmax(x[keep]), np.nanmax(y[keep])])

        if self.im is not None:
            self.im.set_data(self._likelihood_matrix())
            self.im.set_clim(0.0, 1.0)

        if force_autoscale or not self._did_autoscale:
            self.ax.set_xlim(0, self.n - 1)
            if ymins and ymaxs:
                self.ax.set_ylim(min(ymins), max(ymaxs))
            self._did_autoscale = True

        self._draw_selection()
        # re-draw reach guides so they stay on top after refresh
        if self.reach_start is not None and self.reach_end is not None:
            self._draw_reach_guides()
        self.fig.canvas.draw_idle()

    def _draw_selection(self):
        if self.scatter is not None:
            try:
                self.scatter.remove()
            except Exception:
                pass
            self.scatter = None

        if not self.selected:
            return

        xs = np.array([p.pos for p in self.selected], dtype=float)
        ys = []
        for p in self.selected:
            jv = _colpos(self.df, _colkey(self.df, p.bp, p.coord))
            v = _read_col(self.df, jv)
            ys.append(v[p.pos])
        ys = np.array(ys, dtype=float)

        self.scatter = self.ax.scatter(xs, ys, s=18, edgecolors="k", zorder=10)

    # ---------- interaction ---------- #

    def _connect(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.lasso = LassoSelector(self.ax, self._on_lasso)

    def _on_pcutoff(self, val):
        self.pcutoff = float(val)
        self._set_status(f"p cutoff = {self.pcutoff:.2f}")
        self._refresh_all(force_autoscale=False)

    def _on_lasso(self, verts):
        poly = Path(verts)
        self.selected.clear()

        xlim = self.ax.get_xlim()
        pos_view = np.where((self.t >= xlim[0]) & (self.t <= xlim[1]))[0]

        for bp in self.bodyparts:
            jp = _colpos(self.df, _colkey(self.df, bp, "likelihood"))
            p = _read_col(self.df, jp)[pos_view]
            pkeep = np.isfinite(p) & (p >= self.pcutoff)

            for coord in ("x", "y"):
                jv = _colpos(self.df, _colkey(self.df, bp, coord))
                v = _read_col(self.df, jv)[pos_view]

                good = np.isfinite(v) & pkeep
                if not np.any(good):
                    continue

                pts = np.column_stack([pos_view[good], v[good]])
                inside = poly.contains_points(pts)
                for pos in pos_view[good][inside]:
                    self.selected.append(SelectedPoint(bp, coord, int(pos)))

        self.selected = list({(p.bp, p.coord, p.pos): p for p in self.selected}.values())
        self._draw_selection()
        self.fig.canvas.draw_idle()

        note = " | (FILL ARMED: click a line)" if self.await_fill_click else ""
        print(f"[select] {len(self.selected)} points{note}")

    def _on_key(self, ev):
        if not ev.key:
            return
        k = ev.key.lower()

        if k == "escape":
            self.selected.clear()
            self.await_fill_click = False
            self._set_status("")
            self._draw_selection()
            self.fig.canvas.draw_idle()
            print("[select] cleared")
            return

        if k == "d":
            self._delete_selected()
            return

        if k == "i":
            self._identity_swap()
            return

        if k == "l":
            self.await_fill_click = True
            self._set_status("FILL ARMED: click an x/y line for the marker you want to fill (current view)")
            print("[fill] armed. Now single-click the marker trajectory line (x or y).")
            return

        if k == "s":
            self._save()
            return

    def _on_pick(self, ev):
        if not self.await_fill_click:
            return
        artist = ev.artist
        for (bp, _coord), ln in self.lines.items():
            if ln is artist:
                self._fill_marker_in_view(bp)
                self.await_fill_click = False
                self._set_status("")
                return

    # ---------- operations ---------- #

    def _group_positions(self) -> Dict[str, np.ndarray]:
        by_bp: Dict[str, set] = {}
        for p in self.selected:
            by_bp.setdefault(p.bp, set()).add(int(p.pos))
        return {bp: np.array(sorted(pos), dtype=int) for bp, pos in by_bp.items()}

    def _delete_selected(self):
        groups = self._group_positions()
        if not groups:
            print("[delete] nothing selected")
            return

        for bp, pos in groups.items():
            jp = _colpos(self.df, _colkey(self.df, bp, "likelihood"))
            self.df.iloc[pos, jp] = 0.0

        print(f"[delete] set likelihood=0 for {sum(len(v) for v in groups.values())} frames")
        self._refresh_all(force_autoscale=False)

    def _identity_swap(self):
        groups = self._group_positions()
        if not groups:
            print("[identity] nothing selected")
            return

        target = choose_from_list("Assign identity (swap frames into this marker)", self.bodyparts)
        if not target:
            print("[identity] cancelled")
            return

        for src, pos in groups.items():
            if src == target:
                continue

            for coord in ("x", "y", "likelihood"):
                j_src = _colpos(self.df, _colkey(self.df, src, coord))
                j_tgt = _colpos(self.df, _colkey(self.df, target, coord))

                tmp = self.df.iloc[pos, j_src].to_numpy(copy=True)
                self.df.iloc[pos, j_src] = self.df.iloc[pos, j_tgt].to_numpy(copy=True)
                self.df.iloc[pos, j_tgt] = tmp

        print(f"[identity] swapped selected frames into '{target}' (and swapped back into sources)")
        self._refresh_all(force_autoscale=False)

    def _fill_marker_in_view(self, bp: str):
        xlim = self.ax.get_xlim()
        pos_view = np.where((self.t >= xlim[0]) & (self.t <= xlim[1]))[0]
        if pos_view.size < 3:
            print("[fill] zoom to a meaningful range first")
            return

        jp = _colpos(self.df, _colkey(self.df, bp, "likelihood"))
        p = _read_col(self.df, jp)[pos_view]

        bad = ~np.isfinite(p) | (p < self.pcutoff)
        good = ~bad

        if bad.sum() == 0:
            print("[fill] nothing below cutoff in view for this marker")
            return
        if good.sum() < 2:
            print("[fill] not enough above-cutoff points in view to interpolate")
            return

        for coord in ("x", "y"):
            jv = _colpos(self.df, _colkey(self.df, bp, coord))
            v = _read_col(self.df, jv)[pos_view]

            filled = np.interp(
                pos_view[bad].astype(float),
                pos_view[good].astype(float),
                v[good].astype(float),
            )
            v2 = v.copy()
            v2[bad] = filled
            self.df.iloc[pos_view, jv] = v2

        self.df.iloc[pos_view[bad], jp] = self.pcutoff

        print(f"[fill] filled '{bp}' in view (rows {pos_view.min()}..{pos_view.max()}, cutoff={self.pcutoff:.2f})")
        self._refresh_all(force_autoscale=False)

    def _save(self):
        base, ext = os.path.splitext(self.csv_path)
        out = base + "_cleaned" + ext
        self.df.to_csv(out)
        print("[save]", out)

    def run(self):
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("usage: python kclean.py file.csv [--meta compiled_metadata_sorted.xlsx]")
        sys.exit(1)

    csv_path = sys.argv[1]
    meta_path = None
    if "--meta" in sys.argv:
        i = sys.argv.index("--meta")
        if i + 1 < len(sys.argv):
            meta_path = sys.argv[i + 1]

    DLCCleanGUI(csv_path, meta_path=meta_path).run()


if __name__ == "__main__":
    main()
