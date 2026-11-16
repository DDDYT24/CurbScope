# How it works: M2, putting every box on the map

From a moving car, a parked car moves across the image. In world coordinates it stays put.
M2 gives every tracked box a world position (metres) and a latitude and longitude, so M3 can
tell parked from moving by looking at where a track is, not where its box is.

## From a box to a 3D point

For each frame, once (`geo.frame_points`):

1. Take the Velodyne scan and drop points less than 0.3 m above the road. The Velodyne sits
   1.73 m above the road, so this keeps points with Velodyne z above -1.43 m. This assumes a
   flat road near the car.
2. Move the remaining points into the cam2 frame with `T_cam2_velo` and keep those in front of
   the camera.
3. Project them to pixels with the intrinsics `K = P_rect_20[:3, :3]`:
   `u = fx * x / z + cx`, `v = fy * y / z + cy`. pykitti's `T_cam2_velo` already contains the
   rectifying rotation and cam2's sideways offset, so the intrinsics finish the job.

Then for each box (`geo.box_points`, `geo.locate`):

4. Keep the points whose pixels fall in the central 60% of the box. The edges of a box often
   contain road, other cars or background.
5. If fewer than 10 points remain, the position is **unknown**: usually a distant or heavily
   occluded car. These are counted, not guessed.
6. Sort the points by distance from the camera and split wherever two neighbours are more than
   0.5 m apart. The first group with at least 5 points is the object; smaller groups in front
   are treated as noise (a branch, a pole, a mirror).
7. The median of that group is the object's position in the cam2 frame. The median ignores a
   few stray points that the mean would not.

All these numbers are in `configs/default.yaml` and were set before looking at any result.

## From cam2 to world and to latitude, longitude

`T_w_cam2 = T_w_imu @ inv(T_cam2_imu)` for that frame's pose, as in M0. The world frame is
pykitti's: x east, y north, z up, in metres, with the first GPS fix as the origin. pykitti
builds it with a spherical Mercator projection scaled by the cosine of the first latitude.
`geo.world_to_latlon` runs the same projection backwards with pyproj, so the conversion is exact
rather than approximate. Converting every GPS fix of a drive to world and comparing with
pykitti's poses gives a largest difference that rounds to 0 at a micrometre
(`latlon_roundtrip_max_m` in the results).

## Checking against KITTI's labels

KITTI's tracklets give each labelled vehicle's box in every frame. `geo_eval` turns them into
world positions and, frame by frame, pairs them with our located cars, trucks and buses:

- **Matching:** the Hungarian algorithm finds the pairing with the smallest total distance.
  Pairs more than 4 m apart do not count. The 4 m gate is about one car length and was fixed
  before looking at the results.
- **Only labels in view:** a label counts only if its box centre projects inside the image.

Two errors per pair, both horizontal:

- **Center error:** distance to the label's box centre. LiDAR only sees the side of a car that
  faces us, while the label's position is the middle of the car. So even a perfect measurement
  is off by up to half a car length (about 1 to 2.5 m). This bias is the main part of the
  number.
- **Footprint error:** distance to the outline of the label's box on the ground, 0 when the point
  is inside. It answers "is the point on the right car?" without the bias.

Results are reported per range bin (0-10, 10-20, 20-40, 40+ m) for each drive in
`results/geo_error_<drive>.json`, plotted in `results/geo_error.png`, and the counts of
unknown boxes, unmatched boxes and unmatched labels are reported next to them.
`results/geo_check_<drive>.png` draws one frame with the LiDAR points, our boxes, the labels'
3D boxes and our located points, to check by eye that the calibration is used correctly.

## Outputs

`results/geo_<drive>.csv`, one row per tracked box: `frame, timestamp, track_id, class,
status (known/unknown), n_points, range_m, x, y, z (world, m), lat, lon, cam_x, cam_y, cam_z`.
M3 reads the world positions; M4 reads them for vehicles, hydrants and stop signs.

## Limits

- Flat road assumed when dropping ground points.
- KITTI's Velodyne scans are not motion-compensated: points swept at different moments of the
  rotation are mixed, worst at the sides of a moving car.
- A box's nearest cluster can belong to something in front of the object (another car, a
  hedge). The footprint error shows how often this happens.
