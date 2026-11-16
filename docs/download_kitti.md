# Downloading KITTI raw

KITTI raw needs a free account and is licensed for non-commercial use only
(CC BY-NC-SA 3.0). Read the terms when you register.

## Drives

Three Residential drives, all recorded on 2011-09-26, all with `tracklet_labels.xml`:

| Drive | Split | Frames | Labelled cars | Size |
| --- | --- | --- | --- | --- |
| 2011_09_26_drive_0022 | dev | 806 | 53 | 3.1 GB |
| 2011_09_26_drive_0036 | dev | 809 | 80 | 3.1 GB |
| 2011_09_26_drive_0023 | test | 480 | 150 | 1.9 GB |

Frames, labelled cars and sizes are from the KITTI Residential listing.

Why these: Residential matches the curbside project's setting; tracklets give 3D ground truth for
the M2 position error; one date means one calibration download. Among the Residential drives with
tracklets these have the most labelled cars, so they likely have the most parked cars.
The drive with the most cars is held out as the test drive, so the final number rests on the
largest sample. `make info` also counts how many labelled cars stay still (a "static" tracklet
moves less than 1 m in the world) and prints each drive's start point.

## What to download

On https://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential, log in, then for each drive:

- `[synced+rectified data]`, e.g. `2011_09_26_drive_0022_sync.zip`
- `[tracklets]`, e.g. `2011_09_26_drive_0022_tracklets.zip`

and once, from the Calibration category: `2011_09_26_calib.zip`.

## Where to put it

Unzip inside WSL, not under `/mnt/c` (reading through `/mnt/c` is much slower):

```bash
mkdir -p ~/data/kitti
cd ~/data/kitti
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_calib.zip
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_drive_0022_sync.zip
unzip /mnt/c/Users/<you>/Downloads/2011_09_26_drive_0022_tracklets.zip
```

The zips already contain the `2011_09_26/` folder. The result must look like:

```
~/data/kitti/2011_09_26/
    calib_cam_to_cam.txt
    calib_imu_to_velo.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0022_sync/
        image_00/ image_01/ image_02/ image_03/
        oxts/
        velodyne_points/
        tracklet_labels.xml
```

If `tracklet_labels.xml` lands somewhere else, move it into the `_sync` folder.

## Check

```bash
make info DRIVE=2011_09_26_drive_0022
```

prints frames, duration, distance driven and tracklet counts, and writes
`results/info_2011_09_26_drive_0022.json`.
