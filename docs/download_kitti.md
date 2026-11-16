# Downloading KITTI raw

KITTI raw needs a free account and is licensed for non-commercial use only
(CC BY-NC-SA 3.0). Read the terms when you register.

## Drives

Three Residential drives, all recorded on 2011-09-26, all with `tracklet_labels.xml`:

| Drive | Split | Frames | Why |
| --- | --- | --- | --- |
| 2011_09_26_drive_0019 | dev | 481 | residential street lined with parked cars on both sides |
| 2011_09_26_drive_0022 | dev | 800 | longest residential drive with tracklets; parked cars and some traffic |
| 2011_09_26_drive_0023 | test | 474 | many parked cars, held out for the final evaluation only |

Why these: Residential matches the curbside project's setting; tracklets give 3D ground truth for
the M2 position error; one date means one calibration download. The frame counts and scene
descriptions come from the KITTI listing as remembered, not yet confirmed on disk:
after download, `make info` prints frames, distance, start point and how many labelled cars stay
still (a "static" tracklet moves less than 1 m in the world). If a drive has few static cars, or
the test drive starts on the same street as a dev drive, swap it before any tuning.

Each synced+rectified zip is roughly 4 to 5 MB per frame (four cameras plus Velodyne), so about
2 GB, 3.5 GB and 2 GB. Check the sizes on the download page.

## What to download

On https://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential, log in, then for each drive:

- `[synced+rectified data]`, e.g. `2011_09_26_drive_0019_sync.zip`
- `[tracklets]`, e.g. `2011_09_26_drive_0019_tracklets.zip`

and once, from the Calibration category: `2011_09_26_calib.zip`.

## Where to put it

Unzip inside WSL, not under `/mnt/c` (reading through `/mnt/c` is much slower):

```bash
mkdir -p ~/data/kitti
cd ~/data/kitti
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_calib.zip
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_drive_0019_sync.zip
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_drive_0019_tracklets.zip
```

The zips already contain the `2011_09_26/` folder. The result must look like:

```
~/data/kitti/2011_09_26/
    calib_cam_to_cam.txt
    calib_imu_to_velo.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0019_sync/
        image_00/ image_01/ image_02/ image_03/
        oxts/
        velodyne_points/
        tracklet_labels.xml
```

If `tracklet_labels.xml` lands somewhere else, move it into the `_sync` folder.

## Check

```bash
make info DRIVE=2011_09_26_drive_0019
```

prints frames, duration, distance driven and tracklet counts, and writes
`results/info_2011_09_26_drive_0019.json`.
