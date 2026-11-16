"""Figures for M2: error versus range, and a visual check of the calibration on one frame."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from curbscope import geo  # noqa: E402

DEV, TEST = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated pair)
YOLO_BOX, LABEL_BOX = "#1baf7a", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_errors(pairs: pd.DataFrame, path, edges: list[float]) -> None:
    """Two panels (center, footprint error) against range; dev and test drives coloured."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=120)
    for ax, column, title in zip(axes, ["center_error_m", "footprint_error_m"],
                                 ["Error to label centre", "Error to label footprint"],
                                 strict=True):
        for split, color in (("dev", DEV), ("test", TEST)):
            part = pairs[pairs["split"] == split]
            ax.scatter(part["range_m"], part[column], s=9, color=color, alpha=0.35,
                       linewidths=0, label=f"{split} drives (each dot: one box)")
            mids, medians = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                inside = part[(part["range_m"] >= lo) & (part["range_m"] < hi)][column]
                if len(inside):
                    mids.append((lo + hi) / 2)
                    medians.append(inside.median())
            ax.plot(mids, medians, color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor="white", markeredgewidth=2, label=f"{split} median per bin")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Range of labelled vehicle (m)", color=MUTED, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("Horizontal error (m)", color=MUTED, fontsize=9)
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, ncol=4,
               loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_check(image, velo, calib, rows: pd.DataFrame, label_boxes: list[np.ndarray],
               path, title: str) -> None:
    """One frame: LiDAR points coloured by distance, YOLO boxes, projected label boxes,
    and the located position of each box."""
    K = calib["P_rect_20"][:3, :3]
    cam = geo.apply_transform(calib["T_cam2_velo"], velo[:, :3])
    cam = cam[cam[:, 2] > 0.5]
    uv = geo.project(cam, K)
    h, w = image.shape[:2]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    fig, ax = plt.subplots(figsize=(12.42, 4.6), dpi=100)
    ax.imshow(image)
    ax.scatter(uv[keep, 0], uv[keep, 1], c=np.linalg.norm(cam[keep], axis=1), cmap="Blues_r",
               vmin=0, vmax=40, s=1, linewidths=0)
    for corners in label_boxes:
        c_uv = geo.project(geo.apply_transform(calib["T_cam2_velo"], corners), K)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot(c_uv[[a, b], 0], c_uv[[a, b], 1], color=LABEL_BOX, linewidth=1.2)
    for _, r in rows.iterrows():
        ax.add_patch(plt.Rectangle((r["x1"], r["y1"]), r["x2"] - r["x1"], r["y2"] - r["y1"],
                                   fill=False, edgecolor=YOLO_BOX, linewidth=2))
        if r["status"] == "known":
            p = geo.project(np.array([[r["cam_x"], r["cam_y"], r["cam_z"]]]), K)[0]
            ax.plot(*p, marker="o", markersize=8, color="white", markeredgecolor=INK)
    ax.plot([], [], color=YOLO_BOX, linewidth=2, label="YOLO box")
    ax.plot([], [], color=LABEL_BOX, linewidth=1.2, label="KITTI label box")
    ax.plot([], [], marker="o", linestyle="", color="white", markeredgecolor=INK,
            label="located position")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
              fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
