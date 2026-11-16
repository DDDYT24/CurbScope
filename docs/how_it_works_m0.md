# How it works: M0, loading a KITTI drive

## What a drive is

A KITTI raw drive is one recording of the car, 10 frames per second. "Synced+rectified" means
frame *i* of every sensor was taken at the same moment, and the camera images are already
undistorted. Per frame there are four camera images, one Velodyne scan (about 100,000 points)
and one OXTS line (GPS position plus IMU orientation).

## Coordinate frames

Every transform is a 4x4 matrix named `T_<to>_<from>`. `T_w_imu` turns a point in the IMU frame
into the world frame: `p_world = T_w_imu @ p_imu`. Reading right to left, names chain:
`T_w_cam2 = T_w_imu @ T_imu_cam2`, and `T_imu_cam2` is the inverse of the calibration
`T_cam2_imu`. That is the only formula M2 needs to put a camera point in the world.

- **velo**: Velodyne LiDAR. x forward, y left, z up.
- **cam2**: left colour camera. x right, y down, z forward.
- **imu**: the OXTS unit. x forward, y left, z up.
- **world**: pykitti builds it from GPS: a flat metric map (Mercator, scaled to the latitude of
  the first frame), with the first frame's position as the origin. Units are metres.

`geo.invert_transform` uses the fact that a rotation's inverse is its transpose, which is exact
and cheaper than a general matrix inverse. The tests check it against `numpy.linalg.inv`, check
that transforming and transforming back returns the same points, and check that
velo -> cam2 -> world gives the same answer as velo -> world.

## Loading

`io_kitti.open_drive` takes a drive id such as `2011_09_26_drive_0019`, splits it into date and
number, and hands it to pykitti. pykitti reads the calibration and the OXTS file at once and
reads images and scans lazily, one frame at a time, so opening a drive is fast.

`io_kitti.load_tracklets` reads KITTI's 3D labels. Each tracklet is one object with a fixed box
size (h, w, l), a first frame, and one pose per frame in the Velodyne frame. Those poses become the
ground truth for the M2 position error.

## The info command

`make info DRIVE=<id>` prints:

- **frames** and **duration** from the timestamps;
- **distance** driven: the sum of the horizontal steps between consecutive OXTS positions
  (height changes are ignored, because GPS height is noisy);
- **start**: latitude and longitude of the first frame;
- **tracklets** per class, and how many of them are **static**: their labelled positions, moved
  into the world frame, spread less than 1 m. This is only a quick way to see whether a drive has
  many parked cars before choosing it. The parked or moving evaluation in M3 uses hand labels.

## Tests without KITTI

`tests/conftest.py` writes a 3-frame fake drive (calibration files, OXTS lines and timestamps)
into a temporary folder. The real pykitti code reads it, so CI checks the whole loading path
without downloading anything.
