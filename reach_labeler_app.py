import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Reach Labeler", layout="wide")

st.title("Reach CSV Labeler (Good Reach Segments)")

# -----------------------------
# Helpers
# -----------------------------
def find_numeric_cols(df):
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        # keep column if it has at least some numeric values
        if s.notna().sum() > 0:
            cols.append(c)
    return cols

def safe_diff(a):
    # np.diff but keeps same length by padding first value
    da = np.diff(a, prepend=a[0])
    return da

def compute_features(df, frame_col, paw_x, paw_y, pellet_x, pellet_y, fps=None):
    out = df.copy()

    # time step
    if fps is not None and fps > 0:
        dt = 1.0 / fps
    else:
        dt = 1.0  # "per frame" units

    px = out[paw_x].to_numpy(dtype=float)
    py = out[paw_y].to_numpy(dtype=float)
    gx = out[pellet_x].to_numpy(dtype=float)
    gy = out[pellet_y].to_numpy(dtype=float)

    # distance paw -> pellet
    dx = gx - px
    dy = gy - py
    dist = np.sqrt(dx * dx + dy * dy)
    out["dist_to_pellet"] = dist

    # velocity (per frame or per second if fps provided)
    vx = safe_diff(px) / dt
    vy = safe_diff(py) / dt
    out["paw_vx"] = vx
    out["paw_vy"] = vy
    speed = np.sqrt(vx * vx + vy * vy)
    out["speed"] = speed

    # cos(theta) between velocity vector and vector pointing to pellet
    # v · d / (||v|| ||d||)
    vnorm = np.maximum(np.sqrt(vx * vx + vy * vy), 1e-9)
    dnorm = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-9)
    cos_theta = (vx * dx + vy * dy) / (vnorm * dnorm)
    out["cos_theta_to_pellet"] = np.clip(cos_theta, -1.0, 1.0)

    return out

def make_plot(df, x_col, y_cols, title):
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation="h"),
    )
    return fig

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload a reach CSV", type=["csv"])
if uploaded is None:
    st.stop()

# --- Robust CSV loader: handles DeepLabCut multi-row headers ---
raw_bytes = uploaded.getvalue()
text_head = raw_bytes[:4000].decode("utf-8", errors="ignore").lower()

def load_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    # Heuristic: DLC CSV usually contains these tokens in the first lines
    dlc_tokens = ["scorer", "bodyparts", "coords"]
    looks_like_dlc = all(tok in text_head for tok in dlc_tokens)

    if looks_like_dlc:
        df0 = pd.read_csv(io.BytesIO(file_bytes), header=[0, 1, 2])
        # Flatten MultiIndex columns: (scorer, bodypart, coord) -> bodypart_coord
        df0.columns = [
            f"{c[1]}_{c[2]}".strip().replace(" ", "_") if isinstance(c, tuple) else str(c)
            for c in df0.columns
        ]
        return df0
    else:
        return pd.read_csv(io.BytesIO(file_bytes))

df = load_csv_safely(raw_bytes)


# Try to coerce object columns into numeric when possible
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = pd.to_numeric(df[c], errors="ignore")

st.caption(f"Loaded: {uploaded.name} | rows={len(df):,} cols={len(df.columns):,}")

numeric_cols = find_numeric_cols(df)

if len(numeric_cols) == 0:
    st.error(
        "I couldn't find any numeric columns in this CSV. "
        "Your x/y columns may have been read as strings. "
        "Check the CSV formatting (no extra text in numeric columns)."
    )
    st.stop()


# -----------------------------
# Column selection
# -----------------------------
with st.sidebar:
    st.header("Columns")
    frame_col = st.selectbox(
        "Frame column",
        options=(["frame"] if "frame" in df.columns else []) + df.columns.tolist(),
        index=0 if "frame" in df.columns else 0,
    )

    st.subheader("Paw marker (x/y)")
    # Smart-ish defaults: prefer columns ending in _x/_y if present
    x_candidates = [c for c in numeric_cols if str(c).lower().endswith(("_x", "x"))]
    y_candidates = [c for c in numeric_cols if str(c).lower().endswith(("_y", "y"))]

    default_paw_x = x_candidates[0] if x_candidates else numeric_cols[0]
    default_paw_y = y_candidates[0] if y_candidates else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])

    paw_x = st.selectbox("paw_x", options=numeric_cols, index=numeric_cols.index(default_paw_x))
    paw_y = st.selectbox("paw_y", options=numeric_cols, index=numeric_cols.index(default_paw_y))

    st.subheader("Pellet location")
    pellet_mode = st.radio("Pellet source", ["Use columns", "Use constant (x,y)"], horizontal=False)

    if pellet_mode == "Use columns":
        pellet_x = st.selectbox("pellet_x", options=numeric_cols, index=min(2, len(numeric_cols)-1))
        pellet_y = st.selectbox("pellet_y", options=numeric_cols, index=min(3, len(numeric_cols)-1))
    else:
        const_x = st.number_input("pellet_x constant", value=0.0)
        const_y = st.number_input("pellet_y constant", value=0.0)
        # create constant columns
        pellet_x = "__pellet_x_const__"
        pellet_y = "__pellet_y_const__"
        df[pellet_x] = const_x
        df[pellet_y] = const_y

    st.subheader("Sampling")
    fps = st.number_input("FPS (optional, improves speed units)", value=0.0, min_value=0.0)
    fps_val = float(fps) if fps and fps > 0 else None

# Ensure frame exists
if frame_col not in df.columns:
    st.error(f"Frame column '{frame_col}' not found.")
    st.stop()

# Sort by frame just in case
df = df.sort_values(frame_col).reset_index(drop=True)

# -----------------------------
# Compute features
# -----------------------------
required = [frame_col, paw_x, paw_y, pellet_x, pellet_y]
missing = [c for c in required if c is None or c not in df.columns]
if missing:
    st.error(f"Missing/invalid selections: {missing}. Please choose valid columns.")
    st.stop()

feat_df = compute_features(df, frame_col, paw_x, paw_y, pellet_x, pellet_y, fps=fps_val)

# -----------------------------
# Segment selection UI
# -----------------------------
st.subheader("Select a 'good reach' segment")

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.6])
min_frame = int(feat_df[frame_col].min())
max_frame = int(feat_df[frame_col].max())

with c1:
    start_frame = st.number_input("Start frame", value=min_frame, min_value=min_frame, max_value=max_frame)
with c2:
    end_frame = st.number_input("End frame", value=min(min_frame + 50, max_frame), min_value=min_frame, max_value=max_frame)
with c3:
    label = st.selectbox("Label", ["good_reach", "bad_reach", "uncertain"], index=0)

if end_frame < start_frame:
    st.warning("End frame is before start frame — swapping them.")
    start_frame, end_frame = end_frame, start_frame

# Persist labels in session
if "segments" not in st.session_state:
    st.session_state["segments"] = []

add = st.button("Add segment ✅", type="primary")
if add:
    st.session_state["segments"].append(
        {"start_frame": int(start_frame), "end_frame": int(end_frame), "label": label}
    )

# Show current segments
st.markdown("### Labeled segments")
seg_df = pd.DataFrame(st.session_state["segments"])
st.dataframe(seg_df, use_container_width=True, height=200)

# -----------------------------
# Plots
# -----------------------------
plot_left, plot_right = st.columns([1, 1])

with plot_left:
    fig1 = make_plot(
        feat_df,
        frame_col,
        ["dist_to_pellet", "speed"],
        title="Distance-to-pellet + Speed",
    )
    # overlay segments
    for seg in st.session_state["segments"]:
        color = "rgba(0,200,0,0.15)" if seg["label"] == "good_reach" else "rgba(200,0,0,0.12)" if seg["label"] == "bad_reach" else "rgba(150,150,0,0.12)"
        fig1.add_vrect(x0=seg["start_frame"], x1=seg["end_frame"], fillcolor=color, line_width=0)
    st.plotly_chart(fig1, use_container_width=True)

with plot_right:
    fig2 = make_plot(
        feat_df,
        frame_col,
        ["cos_theta_to_pellet"],
        title="cos(theta) toward pellet (velocity vs pellet vector)",
    )
    for seg in st.session_state["segments"]:
        color = "rgba(0,200,0,0.15)" if seg["label"] == "good_reach" else "rgba(200,0,0,0.12)" if seg["label"] == "bad_reach" else "rgba(150,150,0,0.12)"
        fig2.add_vrect(x0=seg["start_frame"], x1=seg["end_frame"], fillcolor=color, line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.markdown("### Export")

base_name = uploaded.name.rsplit(".", 1)[0]
labels_csv = seg_df.to_csv(index=False).encode("utf-8")
labels_json = json.dumps(st.session_state["segments"], indent=2).encode("utf-8")

cA, cB = st.columns(2)
with cA:
    st.download_button(
        "Download labels CSV",
        data=labels_csv,
        file_name=f"{base_name}_labels.csv",
        mime="text/csv",
        disabled=seg_df.empty,
    )
with cB:
    st.download_button(
        "Download labels JSON",
        data=labels_json,
        file_name=f"{base_name}_labels.json",
        mime="application/json",
        disabled=seg_df.empty,
    )

st.info(
    "Next step: once you label ~20–50 examples, we can learn/tune a weighted score that finds similar segments automatically."
)
