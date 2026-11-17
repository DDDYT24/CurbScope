"""Accuracy of both parked/moving rules over a grid of thresholds, on the dev drives only.

Usage: python scripts/tune.py --split dev
Read the table and plot, choose one threshold per rule with a selection rule you write down,
and put them in configs/frozen_thresholds.yaml. The test drive is never used here.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curbscope import motion  # noqa: E402
from curbscope.config import DEFAULT, load_config  # noqa: E402
from curbscope.geo_plots import DEV, GRID, INK, MUTED, _style  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--labels-dir", default="labels")
    args = parser.parse_args()
    if args.split == "test":
        sys.exit("tune.py never looks at the test drive; thresholds are chosen on dev only.")

    cfg = load_config(args.config)
    features, labels = motion.load_split(cfg["drives"]["dev"], args.results_dir, args.labels_dir)
    # Same tracks for every threshold and both rules: those both rules can decide.
    common = features.dropna(subset=["box_shift_over_height", "spread_m"]).reset_index(drop=True)
    grid = cfg["motion"]["tune_grid"]

    rule_a = []
    for t in grid["rule_a_max_shift_over_height"]:
        s = motion.score(labels, {"a": motion.rule_a(common, t)}, common)
        rule_a.append({"max_shift_over_height": t, **_summary(s["rules"]["a"])})
    rule_b = []
    for v in grid["rule_b_max_speed_mps"]:
        for t in grid["rule_b_max_spread_m"]:
            s = motion.score(labels, {"b": motion.rule_b(common, t, v)}, common)
            rule_b.append({"max_spread_m": t, "max_speed_mps": v, **_summary(s["rules"]["b"])})

    base = motion.score(labels, {"a": motion.rule_a(common, 1.0)}, common)
    out = {
        "split": "dev",
        "drives": cfg["drives"]["dev"],
        "compared_tracks": base["compared_tracks"],
        "label_counts": base["label_counts"],
        "unsure": base["unsure"],
        "rule_a": rule_a,
        "rule_b": rule_b,
    }
    results = Path(args.results_dir)
    (results / "tune_dev.json").write_text(json.dumps(out, indent=2) + "\n")
    _plot(out, results / "tune_dev.png")

    print(f"dev drives, {out['compared_tracks']} tracks {out['label_counts']}, "
          f"{out['unsure']} unsure excluded")
    print("\nrule A  max_shift_over_height  accuracy  precision  recall")
    for r in rule_a:
        print(f"        {r['max_shift_over_height']:>21}  {_f(r['accuracy'])}  "
              f"{_f(r['precision_parked'])}  {_f(r['recall_parked'])}")
    print("\nrule B  max_speed_mps  max_spread_m  accuracy  precision  recall")
    for r in rule_b:
        print(f"        {r['max_speed_mps']:>13}  {r['max_spread_m']:>12}  {_f(r['accuracy'])}  "
              f"{_f(r['precision_parked'])}  {_f(r['recall_parked'])}")
    print(f"\nwrote {results / 'tune_dev.json'} and {results / 'tune_dev.png'}")


def _summary(rule: dict) -> dict:
    return {k: rule[k] for k in ("accuracy", "precision_parked", "recall_parked")}


def _f(value) -> str:
    return "     n/a" if value is None else f"{value:8.3f}"


def _plot(out: dict, path: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=120)
    a = out["rule_a"]
    ax_a.plot([r["max_shift_over_height"] for r in a], [r["accuracy"] for r in a],
              color=DEV, linewidth=2, marker="o", markersize=8, markeredgecolor="white",
              markeredgewidth=2)
    ax_a.set_xscale("log")
    ax_a.set_title("Rule A (image)", color=INK, fontsize=11, loc="left")
    ax_a.set_xlabel("max box shift / box height", color=MUTED, fontsize=9)
    ax_a.set_ylabel("Accuracy on dev drives", color=MUTED, fontsize=9)
    speeds = sorted({r["max_speed_mps"] for r in out["rule_b"]})
    shades = plt.cm.Blues([0.45 + 0.5 * i / max(len(speeds) - 1, 1) for i in range(len(speeds))])
    for v, color in zip(speeds, shades, strict=True):
        rows = [r for r in out["rule_b"] if r["max_speed_mps"] == v]
        ax_b.plot([r["max_spread_m"] for r in rows], [r["accuracy"] for r in rows], color=color,
                  linewidth=2, marker="o", markersize=8, markeredgecolor="white",
                  markeredgewidth=2, label=f"speed <= {v:g} m/s")
    ax_b.set_title("Rule B (world)", color=INK, fontsize=11, loc="left")
    ax_b.set_xlabel("max spread (m)", color=MUTED, fontsize=9)
    ax_b.legend(frameon=False, fontsize=8, labelcolor=INK)
    for ax in (ax_a, ax_b):
        _style(ax)
        ax.grid(True, color=GRID)
    ax_a.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
