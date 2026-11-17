"""Score both parked/moving rules with the owner's frozen thresholds.

Usage: python scripts/evaluate.py --split dev|test
The test drive is refused unless configs/frozen_thresholds.yaml exists, is committed and has
no uncommitted change, so the thresholds provably came before the test result. The file's
SHA-256 is stored in results/motion_eval.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from curbscope import motion
from curbscope.config import DEFAULT, load_config

FROZEN = Path("configs/frozen_thresholds.yaml")


def committed_unchanged(path: Path) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0
    return tracked and clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--config", default=str(DEFAULT))
    parser.add_argument("--frozen", default=str(FROZEN))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--labels-dir", default="labels")
    args = parser.parse_args()

    try:
        frozen, frozen_sha = motion.load_frozen(args.frozen)
    except (FileNotFoundError, ValueError) as err:
        sys.exit(str(err))
    if args.split == "test" and not committed_unchanged(Path(args.frozen)):
        sys.exit(f"{args.frozen} must be committed, with no uncommitted change, before the "
                 "test drive is evaluated.")

    cfg = load_config(args.config)
    drives = cfg["drives"][args.split]
    try:
        features, labels = motion.load_split(drives, args.results_dir, args.labels_dir)
    except FileNotFoundError as err:
        sys.exit(str(err))
    result = motion.score(labels, motion.predict(features, frozen), features)
    result["drives"] = drives

    path = Path(args.results_dir) / "motion_eval.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    splits = previous.get("splits", {}) if previous.get("frozen_sha256") == frozen_sha else {}
    splits[args.split] = result
    out = {"frozen_sha256": frozen_sha,
           "frozen": {k: frozen[k] for k in motion.FROZEN_KEYS},
           "splits": dict(sorted(splits.items()))}
    path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{args.split}: {result['compared_tracks']} tracks {result['label_counts']}, "
          f"{result['unsure']} unsure and {result['not_decided_by_every_rule']} undecided "
          "excluded")
    for name, r in result["rules"].items():
        print(f"  {name}: accuracy {r['accuracy']}, precision {r['precision_parked']}, "
              f"recall {r['recall_parked']}, {r['confusion']}")
    print(f"wrote {path} (frozen thresholds SHA-256 {frozen_sha[:12]}...)")


if __name__ == "__main__":
    main()
