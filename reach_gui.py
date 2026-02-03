import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Reach Selector (DLC CSV)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def read_dlc_csv(uploaded_file) -> pd.DataFrame:
    """
    Reads a DeepLabCut-style CSV with 3 header rows:
    (scorer, bodyparts, coords) => MultiIndex columns.
    """
    return pd.read_csv(uploaded_file, header=[0, 1, 2])

def get_available_bodyparts(df: pd.DataFrame):
    # df.columns is MultiIndex: (scorer, bodyparts, coords)
    return sorted({c[1] for c in df.columns if isinstance(c, tuple) and len(c) == 3})

def extract_series(df: pd.DataFrame, bodypart: str, coord: str, scorer=None) -> pd.Series:
    """
    Extract x/y/likelihood for a given bodypart.
    If scorer is None, use the first non-'scorer' scorer level found.
    """
    # find scorer candidates
    scorers = [c[0] for c in df.columns if isinstance(c, tuple) and len(c) == 3 and c[0] != "scorer"]
    if not scorers:
        raise ValueError("Could not find DLC scorer level in CSV columns.")
    if scorer is None:
        scorer = scorers[0]
    col = (scorer, bodypart, coord)
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    return df[col]

def parse_cam_from_name(name: str):
    m = re.search(r"(?:^|_)cam(\d)(?:_|$)", name)
    return int(m.group(1)) if m else None

def default_side_from_cam(cam_id: int | None):
    """
    Placeholder heuristic: you can change this mapping to match your rig.
    Return 'left_mcp' or 'right_mcp' (or None to force user choice).
    """
    if cam_id is None:
        return None
    # Example heuristic (EDIT THIS):
    # cams 1-2 -> right, cams 3-5 -> left
    return "right_mcp" if cam_id in (1, 2) else "left_mcp"

def compute_metrics(mcp_x, mcp_y, pellet_x, pellet_y, fps=None):
    """
    Per-frame:
      dist = ||mcp - pellet||
      speed = ||d(mcp)/dt||  (pixels/frame or pixels/sec if fps given)
      cos_theta = cos(angle between velocity vector and (pellet - mcp))
    """
    mcp_x = np.asarray(mcp_x, dtype=float)
    mcp_y = np.asarray(mcp_y, dtype=float)
    pellet_x = np.asarray(pellet_x, dtype=float)
    pellet_y = np.asarray(pellet_y, dtype=float)

    dxp = pellet_x - mcp_x
    dyp = pellet_y - mcp_y
    dist = np.sqrt(dxp**2 + dyp**2)

    vx = np.gradient(mcp_x)  # pixels/frame
    vy = np.gradient(mcp_y)

    speed = np.sqrt(vx**2 + vy**2)  # pixels/frame
    if fps is not None and fps > 0:
        speed = speed * fps  # pixels/sec

    # cos(theta) between v and vector-to-pellet
    vnorm = np.sqrt(vx**2 + vy**2)
    pnorm = np.sqrt(dxp**2 + dyp**2)
    denom = vnorm * pnorm
    cos_theta = np.full_like(dist, np.nan, dtype=float)
    ok = denom > 1e-12
    cos_theta[ok] = (vx[ok] * dxp[ok] + vy[ok] * dyp[ok]) / denom[ok]

    return dist, speed, cos_theta

def segment_slope(y, frames):
    """
    Linear slope of y vs frames: units y_per_frame (or y_per_sec if frames are time).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(frames, dtype=float)
    if len(y) < 2:
        return np.nan
    # polyfit slope
    m = np.polyfit(x, y, 1)[0]
    return float(m)

def vline(fig, x, name):
    fig.add_vline(x=x, line_width=2, line_dash="dash", annotation_text=name, annotation_position="top")

# -----------------------------
# App state
# -----------------------------
if "selections" not in st.session_state:
    st.session_state["selections"] = []  # list of dicts: {file, start, stop, side, ...}
if "per_frame_rows" not in st.session_state:
    st.session_state["per_frame_rows"] = []  # output rows (one per selected frame)

st.title("Reach Segment Selector + Metrics Export (DLC CSVs)")

with st.sidebar:
    st.header("1) Upload DLC CSVs")
    uploads = st.file_uploader(
        "Upload one or more DLC CSVs",
        type=["csv"],
        accept_multiple_files=True
    )

    st.header("2) Options")
    fps = st.number_input("FPS (optional). If blank/0 -> speed is pixels/frame.", min_value=0.0, value=0.0, step=10.0)
    use_fps = fps if fps and fps > 0 else None

    st.caption("Tip: edit the camera→side mapping inside `default_side_from_cam()` if you want auto-side selection.")

if not uploads:
    st.info("Upload one or more DLC CSVs to begin.")
    st.stop()

# Build a name->file map
file_map = {u.name: u for u in uploads}
file_names = list(file_map.keys())

colA, colB = st.columns([1, 1])

with colA:
    st.header("Review a file")
    chosen_name = st.selectbox("Choose a CSV", file_names)
    uploaded_file = file_map[chosen_name]
    cam_id = parse_cam_from_name(chosen_name)
    st.write(f"Detected camera: **{cam_id if cam_id is not None else 'N/A'}**")

# Read CSV
df = read_dlc_csv(uploaded_file)
bodyparts = get_available_bodyparts(df)

# Decide side
auto_side = default_side_from_cam(cam_id)
available_mcp = [bp for bp in ("left_mcp", "right_mcp") if bp in bodyparts]
if not available_mcp:
    st.error("Could not find left_mcp or right_mcp in this CSV.")
    st.stop()

with colA:
    st.subheader("MCP side")
    side = st.radio(
        "Select which MCP to use",
        options=available_mcp,
        index=(available_mcp.index(auto_side) if auto_side in available_mcp else 0),
        horizontal=True
    )

# Extract series
scorer_candidates = [c[0] for c in df.columns if isinstance(c, tuple) and len(c) == 3 and c[0] != "scorer"]
scorer = scorer_candidates[0]

pellet_x = extract_series(df, "pellet", "x", scorer=scorer)
pellet_y = extract_series(df, "pellet", "y", scorer=scorer)

mcp_x = extract_series(df, side, "x", scorer=scorer)
mcp_y = extract_series(df, side, "y", scorer=scorer)

n_frames = len(df)
frames = np.arange(n_frames)

# Compute per-frame metrics
dist, speed, cos_theta = compute_metrics(mcp_x, mcp_y, pellet_x, pellet_y, fps=use_fps)

with colB:
    st.header("Select reach frames")
    start, stop = st.slider(
        "Frame range (inclusive)",
        min_value=0,
        max_value=n_frames - 1,
        value=(0, min(n_frames - 1, 100)),
        step=1
    )
    if stop < start:
        start, stop = stop, start

    seg_idx = np.arange(start, stop + 1)
    seg_slope = segment_slope(dist[seg_idx], seg_idx)

    st.markdown(
        f"""
        **Selected:** frames **{start} → {stop}**  (N={len(seg_idx)})  
        **Segment slope (dist→pellet vs frame):** `{seg_slope:.6g}` (pixels/frame)  
        """
    )

# Plot distance and speed with selection
fig_dist = go.Figure()
fig_dist.add_trace(go.Scatter(x=frames, y=dist, mode="lines", name="Distance MCP→Pellet"))
vline(fig_dist, start, "start")
vline(fig_dist, stop, "stop")
fig_dist.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Frame", yaxis_title="Pixels")

fig_speed = go.Figure()
fig_speed.add_trace(go.Scatter(x=frames, y=speed, mode="lines", name="MCP speed"))
vline(fig_speed, start, "start")
vline(fig_speed, stop, "stop")
fig_speed.update_layout(
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Frame",
    yaxis_title=("Pixels/sec" if use_fps else "Pixels/frame"),
)

fig_cos = go.Figure()
fig_cos.add_trace(go.Scatter(x=frames, y=cos_theta, mode="lines", name="cos(theta) toward pellet"))
vline(fig_cos, start, "start")
vline(fig_cos, stop, "stop")
fig_cos.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Frame", yaxis_title="Cosine")

st.subheader("Plots")
st.plotly_chart(fig_dist, use_container_width=True)
st.plotly_chart(fig_speed, use_container_width=True)
st.plotly_chart(fig_cos, use_container_width=True)

# Save selection
st.subheader("Save selection")
save_col1, save_col2 = st.columns([1, 2])

with save_col1:
    if st.button("Add this selection to output", type="primary"):
        # Build per-frame rows for this selection
        for f in seg_idx:
            st.session_state["per_frame_rows"].append({
                "source_csv": chosen_name,
                "mcp_side": side,
                "start_frame": int(start),
                "stop_frame": int(stop),
                "frame": int(f),
                "dist_to_pellet_px": float(dist[f]),
                "mcp_speed": float(speed[f]),
                "cos_theta_to_pellet": float(cos_theta[f]) if np.isfinite(cos_theta[f]) else np.nan,
                "segment_slope_dist_px_per_frame": float(seg_slope) if np.isfinite(seg_slope) else np.nan
            })

        st.session_state["selections"].append({
            "source_csv": chosen_name,
            "mcp_side": side,
            "start_frame": int(start),
            "stop_frame": int(stop),
            "segment_slope_dist_px_per_frame": float(seg_slope) if np.isfinite(seg_slope) else np.nan,
            "n_frames": int(len(seg_idx)),
        })
        st.success("Selection added.")

with save_col2:
    st.write("Selections saved this session:")
    if st.session_state["selections"]:
        st.dataframe(pd.DataFrame(st.session_state["selections"]), use_container_width=True)
    else:
        st.caption("None yet.")

# Export
st.subheader("Export")
if st.session_state["per_frame_rows"]:
    out_df = pd.DataFrame(st.session_state["per_frame_rows"])
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download per-frame metrics CSV",
        data=csv_bytes,
        file_name="reach_segments_metrics_per_frame.csv",
        mime="text/csv",
    )
else:
    st.caption("Add at least one selection to enable export.")

# Convenience: clear
if st.button("Clear all selections"):
    st.session_state["selections"] = []
    st.session_state["per_frame_rows"] = []
    st.rerun()
