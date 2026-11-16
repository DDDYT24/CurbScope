"""Download the YOLO weights named in the config and check their SHA-256."""

import urllib.request
from pathlib import Path

from curbscope.config import load_config
from curbscope.detect_track import check_weights


def main() -> None:
    cfg = load_config()["detect"]
    path = Path(cfg["weights"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {cfg['weights_url']}")
        urllib.request.urlretrieve(cfg["weights_url"], path)
    check_weights(path, cfg["weights_sha256"])
    print(f"{path} OK (SHA-256 matches the config)")


if __name__ == "__main__":
    main()
