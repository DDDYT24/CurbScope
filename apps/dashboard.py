"""CurbScope dashboard: curb occupancy on a map, tracks frame by frame, sign OCR, evaluation.

Run with `make dashboard`. Reads only files in results/ (and KITTI images for the frame viewer,
sign photos for the sign page); every number comes from the pipeline's own outputs.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curbscope import geo, io_kitti, motion, report  # noqa: E402
from curbscope.config import all_drives, load_config  # noqa: E402

BLUE, INK, MUTED, GRID, PALE = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3de", "#c9c7bf"
FROZEN = ROOT / "configs" / "frozen_thresholds.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-dir")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    return parser.parse_args(sys.argv[1:])


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def open_drive(kitti_dir: str, drive: str):
    return io_kitti.open_drive(kitti_dir, drive)


def frozen_thresholds():
    try:
        return motion.load_frozen(FROZEN)[0]
    except (FileNotFoundError, ValueError):
        return None


def occupancy_colour(value) -> str:
    """Pale to full blue with occupancy; grey where nothing can be parked."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return PALE
    shade = np.array([205, 226, 251]) + (np.array([16, 66, 129]) - [205, 226, 251]) * min(value, 1)
    return "#" + "".join(f"{int(c):02x}" for c in shade)


# --- overview -----------------------------------------------------------------------------


def overview(results: Path, drive: str, cfg: dict) -> None:
    info = read_json(results / f"info_{drive}.json")
    features = read_csv(results / f"motion_features_{drive}.csv")
    summary = (read_json(results / "segments_summary.json") or {}).get("drives", {}).get(drive)
    segs = read_csv(results / f"segments_{drive}.csv")
    geojson = read_json(results / f"segments_{drive}.geojson")

    if summary:
        st.markdown(f"**{report._pct(summary['overall_occupancy'])}** of "
                    f"{summary['usable_m']:.0f} m usable curb is taken by "
                    f"{summary['parked_at_curb']} parked vehicles.")
    else:
        st.markdown("Curb occupancy appears once the parked/moving thresholds are frozen "
                    "(`make tune`, then `make segments`). The map shows every located track.")

    cols = st.columns(4)
    cols[0].metric("Distance driven", f"{info['distance_m']:.0f} m" if info else "-")
    cols[1].metric("Tracks", len(features) if features is not None else "-")
    cols[2].metric("Tracks with a position",
                   int(features["median_x"].notna().sum()) if features is not None else "-")
    cols[3].metric("Parked at the curb", summary["parked_at_curb"] if summary else "-")

    left, right = st.columns([3, 2])
    with left:
        if info is None:
            st.info("Run `make info` for this drive to place it on the map.")
        else:
            st_folium(build_map(info, features, geojson), height=460, use_container_width=True,
                      returned_objects=[])
    with right:
        if segs is not None:
            st.pyplot(occupancy_chart(segs))
        elif features is not None:
            counts = features["class"].value_counts().rename_axis("class").reset_index(
                name="tracks")
            st.dataframe(counts, hide_index=True, width="stretch")


def build_map(info: dict, features, geojson) -> folium.Map:
    lat0, lon0 = info["start_lat_lon"]
    fmap = folium.Map(location=[lat0, lon0], zoom_start=17, tiles="OpenStreetMap",
                      control_scale=True)
    if geojson:
        for f in geojson["features"]:
            p, g = f["properties"], f["geometry"]
            if g["type"] == "LineString":
                folium.PolyLine([[lat, lon] for lon, lat in g["coordinates"]], weight=7,
                                color=occupancy_colour(p.get("occupancy")), opacity=0.95,
                                tooltip=f"{p['side']} {p['start_m']:.0f}-{p['end_m']:.0f} m: "
                                        f"occupancy {report._pct(p.get('occupancy'))}, "
                                        f"{p['vehicles']} vehicles").add_to(fmap)
            else:
                lon, lat = g["coordinates"]
                parked = p.get("parked")
                state = {True: "parked", False: "moving"}.get(parked, "undecided")
                folium.CircleMarker([lat, lon], radius=5, weight=1.5, color=INK,
                                    fill=True, fill_opacity=1.0 if parked else 0.0,
                                    fill_color=BLUE if parked else "#ffffff",
                                    tooltip=f"track {p['track_id']} {p['class']}: {state}"
                                    ).add_to(fmap)
    elif features is not None:
        placed = features.dropna(subset=["median_x"])
        latlon = geo.world_to_latlon(placed[["median_x", "median_y"]].to_numpy(), lat0, lon0)
        for (lat, lon), (_, r) in zip(latlon, placed.iterrows(), strict=True):
            folium.CircleMarker([lat, lon], radius=4, weight=1.5, color=BLUE, fill=True,
                                fill_opacity=0.6, tooltip=f"track {int(r.track_id)} {r['class']}"
                                ).add_to(fmap)
    bounds = fmap.get_bounds()
    if bounds and None not in bounds[0]:
        fmap.fit_bounds(bounds, padding=(20, 20))
    return fmap


def occupancy_chart(segs: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(5, 4.6), sharex=True, dpi=110)
    for ax, side in zip(axes, ["left", "right"], strict=True):
        part = segs[segs["side"] == side]
        values = part["occupancy"].fillna(0).to_numpy()
        ax.bar(part["start_m"] + part["segment_m"] / 2, values, width=part["segment_m"] * 0.85,
               color=BLUE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(f"{side.capitalize()} side", color=INK, fontsize=10, loc="left")
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[1].set_xlabel("Distance along our path (m)", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


# --- tracks -------------------------------------------------------------------------------


def tracks_view(results: Path, drive: str, kitti_dir: str) -> None:
    tracks = read_csv(results / f"tracks_{drive}.csv")
    if tracks is None:
        st.info("Run `make track` first.")
        return
    try:
        raw = open_drive(kitti_dir, drive)
    except FileNotFoundError:
        st.info(f"The frame viewer needs the KITTI images in {kitti_dir} "
                "(see docs/download_kitti.md).")
        return
    frozen = frozen_thresholds()
    status = {}
    features = read_csv(results / f"motion_features_{drive}.csv")
    if frozen and features is not None:
        status = motion.parked_by_track(features, frozen)

    frame = st.slider("Frame", 0, len(raw.timestamps) - 1, 0)
    image = io_kitti.cam2_image(raw, frame).copy()
    rows = tracks[tracks["frame"] == frame]
    for r in rows.itertuples():
        state = status.get(r.track_id)
        colour = (42, 120, 214) if state is True else (31, 31, 30) if state is False else (
            120, 120, 112)
        cv2.rectangle(image, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, 3)
        cv2.putText(image, str(r.track_id), (int(r.x1), max(int(r.y1) - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    st.image(image, width="stretch")
    legend = ("Blue: parked, black: moving, grey: undecided (rule B, frozen thresholds)."
              if status else "Parked or moving shows here once the thresholds are frozen.")
    st.caption(f"{len(rows)} tracked boxes · {legend}")


# --- signs --------------------------------------------------------------------------------


def signs_view(results: Path, cfg: dict) -> None:
    outputs = read_csv(results / "ocr_outputs.csv")
    if outputs is None:
        st.info(report.PENDING["ocr"])
        return
    photo = st.selectbox("Photo", sorted(outputs["file"].unique()))
    rows = outputs[outputs["file"] == photo]
    left, right = st.columns([2, 3])
    path = ROOT / cfg["ocr"]["signs_dir"] / photo
    if path.exists():
        left.image(str(path), width="stretch")
    else:
        left.info(f"{path.name} is not in {cfg['ocr']['signs_dir']}/ on this machine.")
    table = rows[["recognizer", "text", "cer", "parse_ok"]].rename(
        columns={"recognizer": "Recogniser", "text": "Text read", "cer": "CER",
                 "parse_ok": "Rule right"})
    right.dataframe(table, hide_index=True, width="stretch",
                    column_config={"CER": st.column_config.NumberColumn(format="%.3f")})
    choice = right.radio("Parsed rule from", rows["recognizer"].tolist(), horizontal=True)
    right.json(json.loads(rows[rows["recognizer"] == choice]["parsed"].iloc[0]))


# --- evaluation ---------------------------------------------------------------------------


def evaluation_view(results: Path, cfg: dict) -> None:
    for title, section in report.SECTIONS[1:]:
        st.subheader(title)
        st.markdown(section(results, cfg))
        if section is report.geo_section and (results / "geo_error.png").exists():
            st.image(str(results / "geo_error.png"), width="stretch")
        if section is report.motion_section and (results / "tune_dev.png").exists():
            st.image(str(results / "tune_dev.png"), caption="Threshold search on dev drives",
                     width="stretch")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    results = Path(args.results_dir)
    kitti_dir = args.kitti_dir or cfg["kitti_dir"]
    st.set_page_config(page_title="CurbScope", layout="wide")

    with st.sidebar:
        st.title("CurbScope")
        drives = all_drives(cfg)
        drive = st.radio("Drive", drives, format_func=lambda d: f"{report._short(d)} · "
                                                              f"{report._split(cfg, d)}")
    title = f"Drive {report._short(drive)} · {report._split(cfg, drive)}"
    tab_overview, tab_tracks, tab_signs, tab_eval = st.tabs(
        ["Overview", "Tracks", "Signs", "Evaluation"])
    with tab_overview:
        st.subheader(title)
        overview(results, drive, cfg)
    with tab_tracks:
        st.subheader(title)
        tracks_view(results, drive, kitti_dir)
    with tab_signs:
        signs_view(results, cfg)
    with tab_eval:
        evaluation_view(results, cfg)


main()
