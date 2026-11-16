# CurbScope

How much curb can you park on, how much of it is taken, and when did each car arrive?
CurbScope answers this per 20 m curb segment from public KITTI driving data (camera, LiDAR,
GPS/IMU), and reads Edmonton parking signs into machine-readable time rules. Every number is
checked against hand-made labels.

## Run it

Needs WSL2 Ubuntu (or Linux), `make` and [uv](https://docs.astral.sh/uv/).
uv downloads Python 3.11 itself.

```bash
make setup                                  # pinned dependencies into ~/.venvs/curbscope
make test                                   # unit tests, no KITTI data needed
make info DRIVE=2011_09_26_drive_0019       # needs a downloaded drive, see below
```

KITTI needs a free login, so the data is not downloaded by a script:
see [docs/download_kitti.md](docs/download_kitti.md). Data goes to `~/data/kitti` by default;
use `make info KITTI_DIR=/other/path DRIVE=...` to point elsewhere.

## How it works

- [M0: loading a KITTI drive](docs/how_it_works_m0.md)

## Data and licences

| Source | Used for | Licence |
| --- | --- | --- |
| [KITTI raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) | camera, LiDAR, GPS/IMU, 3D tracklets | CC BY-NC-SA 3.0, non-commercial |
| [pykitti](https://github.com/utiasSTARS/pykitti) | reading KITTI raw files | MIT |

No KITTI files, photos or model weights are stored in this repository.

## Limitations

- KITTI was recorded in Karlsruhe, Germany. Edmonton parking rules are applied only as a
  demonstration configuration, not as a legal judgement about German streets.
