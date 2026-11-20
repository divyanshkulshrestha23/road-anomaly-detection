"""Monocular distance model shared by calibration (Python) and the app (Kotlin mirror in
app/rider/src/main/java/dtu/rider/distance/DistanceEstimator.kt). Keep the two in sync.

Two estimates from one bounding box, combined:
  width:  Z = W_real * f / w_px                 (needs the vehicle's real width)
  ground: Z = f * h_cam / (y_bottom - y_horizon)  (needs camera height and horizon row; flat road)
f is stored as a fraction of image width (f_norm = f_px / img_w) and the horizon as a fraction of
image height, so calibration carries over to any frame resolution with the same field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Typical vehicle widths in metres (approximate; replace with measured values where possible).
DEFAULT_WIDTHS = {"car": 1.75, "motorcycle": 0.80, "bicycle": 0.60, "bus": 2.55, "truck": 2.50}


@dataclass
class Calibration:
    f_norm: float = 0.5 / math.tan(math.radians(66 / 2))  # placeholder: ~66 deg horizontal FOV
    horizon_norm: float = 0.5                              # placeholder: camera level with road
    camera_height_m: float = 1.0                           # placeholder: handlebar height
    widths: dict = field(default_factory=lambda: dict(DEFAULT_WIDTHS))
    calibrated: bool = False


def width_distance(cal: Calibration, cls: str, box_w_px: float, img_w: int) -> float | None:
    w = cal.widths.get(cls)
    if not w or box_w_px <= 1:
        return None
    return w * cal.f_norm * img_w / box_w_px


def ground_distance(cal: Calibration, y_bottom_px: float, img_w: int, img_h: int) -> float | None:
    dy = y_bottom_px - cal.horizon_norm * img_h
    if dy <= 2:
        return None  # at or above the horizon: no ground-plane estimate
    return cal.f_norm * img_w * cal.camera_height_m / dy


def fused_distance(cal: Calibration, cls: str, box: tuple[float, float, float, float], img_w: int, img_h: int) -> float | None:
    """box = (x1, y1, x2, y2) in pixels. Averages the two estimates when they roughly agree."""
    zw = width_distance(cal, cls, box[2] - box[0], img_w)
    zg = ground_distance(cal, box[3], img_w, img_h)
    if zw is None:
        return zg
    if zg is None or not (0.5 * zw <= zg <= 2.0 * zw):
        return zw
    return 0.5 * (zw + zg)


def _self_test() -> None:
    # Project synthetic vehicles through a pinhole camera and check the estimates recover distance.
    cal = Calibration(f_norm=0.8, horizon_norm=0.45, camera_height_m=1.05)
    W, H = 640, 480
    f_px = cal.f_norm * W
    for cls, Z in [("car", 5.0), ("car", 12.0), ("motorcycle", 20.0), ("bus", 30.0)]:
        w_px = cal.widths[cls] * f_px / Z
        y_bottom = cal.horizon_norm * H + f_px * cal.camera_height_m / Z
        box = (W / 2 - w_px / 2, y_bottom - w_px, W / 2 + w_px / 2, y_bottom)
        zf = fused_distance(cal, cls, box, W, H)
        assert abs(zf - Z) / Z < 1e-6, (cls, Z, zf)
    print("distance_model self-test passed")


if __name__ == "__main__":
    _self_test()
