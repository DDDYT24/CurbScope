"""Label each track as parked, moving or unsure.

Run with `make label` (streamlit run apps/label_tracks.py -- --kitti-dir ~/data/kitti).
Shows crops of the track's box over time and its world trajectory next to the ego path.
It never shows what the rules decided, so the labels stay independent of them.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curbscope import io_kitti, labeling  # noqa: E402
from curbscope.config import all_drives, load_config  # noqa: E402
from curbscope.geo_plots import DEV, GRID, MUTED, YOLO_BOX  # noqa: E402

THUMBS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-dir")
    parser.add_argument("--labels-dir", default=str(ROOT / "labels"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    return parser.parse_args(sys.argv[1:])


@st.cache_resource
def open_drive(kitti_dir: str, drive: str):
    return io_kitti.open_drive(kitti_dir, drive)


@st.cache_data(max_entries=400)
def frame_image(kitti_dir: str, drive: str, frame: int):
    return io_kitti.cam2_image(open_drive(kitti_dir, drive), frame)


@st.cache_data
def drive_tables(results_dir: str, drive: str):
    tracks = pd.read_csv(Path(results_dir) / f"tracks_{drive}.csv")
    located = pd.read_csv(Path(results_dir) / f"geo_{drive}.csv")
    return tracks, located


@st.cache_data
def ego_path(kitti_dir: str, drive: str):
    return io_kitti.poses(open_drive(kitti_dir, drive))[:, :2, 3]


def trajectory_figure(path, track_xy, frames):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    ax.plot(path[:, 0], path[:, 1], color=MUTED, linewidth=2, label="our car")
    if len(track_xy):
        ax.scatter(track_xy[:, 0], track_xy[:, 1], c=frames, cmap="Blues", vmin=frames.min() - 5,
                   s=28, edgecolors=DEV, linewidths=0.6, label="this track (darker = later)")
        cx, cy = track_xy.mean(axis=0)
        span = max(25.0, float(abs(track_xy - [cx, cy]).max()) + 10)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("east (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("north (m)", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    cfg = load_config()
    kitti_dir = args.kitti_dir or cfg["kitti_dir"]
    st.set_page_config(page_title="Label tracks", layout="wide")

    with st.sidebar:
        drive = st.selectbox("Drive", all_drives(cfg))
        tracks, located = drive_tables(args.results_dir, drive)
        lengths = tracks.groupby("track_id").size()
        min_frames = st.slider("Minimum track length (frames)", 1, int(lengths.max()), 1)
        only_open = st.toggle("Only unlabelled", value=True)

    path = labeling.label_path(args.labels_dir, drive)
    labels = labeling.read_labels(path)
    done = dict(zip(labels["track_id"].astype(int), labels["label"], strict=True))
    notes = dict(zip(labels["track_id"].astype(int), labels["note"], strict=True))
    candidates = [int(t) for t in lengths.sort_values(ascending=False).index
                  if lengths[t] >= min_frames]
    queue = [t for t in candidates if not (only_open and t in done)]
    with st.sidebar:
        st.progress(len(done) / max(len(lengths), 1),
                    text=f"{len(done)} of {len(lengths)} tracks labelled")

    if not queue:
        st.success("Every track in this view is labelled.")
        return
    key = f"pos_{drive}"
    st.session_state[key] = min(st.session_state.get(key, 0), len(queue) - 1)
    track_id = queue[st.session_state[key]]
    boxes = tracks[tracks["track_id"] == track_id].sort_values("frame")
    own = located[(located["track_id"] == track_id) & (located["status"] == "known")]

    current = done.get(track_id)
    st.subheader(f"Track {track_id} · {boxes['class'].mode()[0]} · {len(boxes)} frames"
                 + (f" · labelled {current}" if current else ""))

    columns = st.columns(THUMBS)
    for col, frame in zip(columns, labeling.pick_frames(boxes["frame"].to_numpy(), THUMBS),
                          strict=False):
        row = boxes[boxes["frame"] == frame].iloc[0]
        crop = labeling.crop_box(frame_image(kitti_dir, drive, int(frame)),
                                 (row.x1, row.y1, row.x2, row.y2))
        col.image(crop, caption=f"frame {frame}", width="stretch")

    left, right = st.columns([3, 2])
    middle = boxes.iloc[len(boxes) // 2]
    context = frame_image(kitti_dir, drive, int(middle.frame)).copy()
    rgb = tuple(int(YOLO_BOX[i:i + 2], 16) for i in (1, 3, 5))
    cv2.rectangle(context, (int(middle.x1), int(middle.y1)), (int(middle.x2), int(middle.y2)),
                  rgb, 3)
    left.image(context, caption=f"frame {int(middle.frame)}", width="stretch")
    right.pyplot(trajectory_figure(ego_path(kitti_dir, drive), own[["x", "y"]].to_numpy(),
                                   own["frame"].to_numpy()))
    if own.empty:
        right.caption("No LiDAR position for this track.")

    note = st.text_input("Note (for example: ID switch with track 52)",
                         value=notes.get(track_id, ""), key=f"note_{drive}_{track_id}")
    b1, b2, b3, _, back, skip = st.columns([1, 1, 1, 2, 1, 1])
    choice = None
    if b1.button("Parked", type="primary", width="stretch"):
        choice = "parked"
    if b2.button("Moving", type="primary", width="stretch"):
        choice = "moving"
    if b3.button("Unsure", width="stretch"):
        choice = "unsure"
    if back.button("Back", width="stretch", disabled=st.session_state[key] == 0):
        st.session_state[key] -= 1
        st.rerun()
    if skip.button("Skip", width="stretch"):
        st.session_state[key] = (st.session_state[key] + 1) % len(queue)
        st.rerun()
    if choice:
        labeling.save_label(path, track_id, choice, note)
        if not only_open:  # the saved track stays in the list, so step past it
            st.session_state[key] = (st.session_state[key] + 1) % len(queue)
        st.rerun()


main()
