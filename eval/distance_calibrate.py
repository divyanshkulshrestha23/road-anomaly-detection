"""Calibrate monocular distance from photos taken at measured distances, and report the error.

Photos: frames saved with the app's Snap button (preferred, same crop as the app), one vehicle
centred in front. Name each file with the vehicle class and distance, for example:
    car_5m_1.jpg   car_10m.jpg   motorcycle_7.5m.jpg   bus_20m_a.jpg
Classes: car, motorcycle, bus, truck, bicycle.

Fits, from those photos (valid for portrait and landscape frames of the same camera):
  f_long          focal length relative to the image's long side (from the width method)
  horizon_offset  horizon row below the image centre, relative to the long side
                  (from the ground method, given --height; depends on how the phone is tilted)
Then reports per-photo error for width, ground and fused estimates, using leave-one-out
(each photo is predicted with a calibration fitted on the others), and writes distance_calib.json
for app/rider/src/main/assets/.

Usage:
  python eval/distance_calibrate.py --photos calib_photos/ --height 1.05 \
      --width car=1.70 motorcycle=0.78 --weights yolo11n.pt --out distance_calib.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from distance_model import DEFAULT_WIDTHS, Calibration, fused_distance, ground_distance, width_distance  # noqa: E402

NAME_RX = re.compile(r"(car|motorcycle|bus|truck|bicycle)[_\-]?(\d+(?:\.\d+)?)\s*m", re.I)


def detect(weights: str, photos: list[Path]) -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(weights)
    rows = []
    for p in photos:
        m = NAME_RX.search(p.stem)
        if not m:
            print(f"skip (name has no class/distance): {p.name}")
            continue
        cls, z = m.group(1).lower(), float(m.group(2))
        r = model.predict(str(p), imgsz=640, conf=0.25, verbose=False)[0]
        H, W = r.orig_shape
        best = None
        for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            if model.names[int(c)] != cls:
                continue
            # Prefer the most central, largest box of the named class.
            cx = (b[0] + b[2]) / 2
            score = (b[2] - b[0]) * (b[3] - b[1]) * (1 - abs(cx - W / 2) / W)
            if best is None or score > best[0]:
                best = (score, b)
        if best is None:
            print(f"skip (no {cls} detected): {p.name}")
            continue
        rows.append({"file": p.name, "cls": cls, "z": z, "box": best[1], "W": W, "H": H})
    return rows


def fit(rows: list[dict], height: float, widths: dict) -> Calibration:
    f_vals = [(r["box"][2] - r["box"][0]) * r["z"] / (widths[r["cls"]] * max(r["W"], r["H"])) for r in rows if r["cls"] in widths]
    f_long = statistics.median(f_vals)
    hz = [(r["box"][3] - f_long * max(r["W"], r["H"]) * height / r["z"] - r["H"] / 2) / max(r["W"], r["H"]) for r in rows]
    return Calibration(f_long=f_long, horizon_offset=statistics.median(hz), camera_height_m=height,
                       widths=dict(widths), calibrated=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--height", type=float, required=True, help="Camera lens height above the road, metres")
    ap.add_argument("--width", nargs="*", default=[], help="Measured widths, e.g. car=1.70 motorcycle=0.78")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--out", type=Path, default=Path("distance_calib.json"))
    a = ap.parse_args()

    widths = dict(DEFAULT_WIDTHS)
    for kv in a.width:
        k, v = kv.split("=")
        widths[k.strip().lower()] = float(v)
    photos = sorted(p for p in a.photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = detect(a.weights, photos)
    if len(rows) < 3:
        print(f"Need at least 3 usable photos, got {len(rows)}", file=sys.stderr)
        return 1

    cal = fit(rows, a.height, widths)
    print(f"Fitted on {len(rows)} photos: f_long={cal.f_long:.4f} "
          f"(FOV across long side ~{2 * __import__('math').degrees(__import__('math').atan(0.5 / cal.f_long)):.1f} deg), "
          f"horizon_offset={cal.horizon_offset:.4f}")

    # Leave-one-out error.
    report = []
    for i, r in enumerate(rows):
        others = rows[:i] + rows[i + 1:]
        c = fit(others, a.height, widths) if len(others) >= 2 else cal
        box = tuple(r["box"])
        zw = width_distance(c, r["cls"], box[2] - box[0], r["W"], r["H"])
        zg = ground_distance(c, box[3], r["W"], r["H"])
        zf = fused_distance(c, r["cls"], box, r["W"], r["H"])
        e = lambda z: None if z is None else round(100 * (z - r["z"]) / r["z"], 1)
        report.append({"file": r["file"], "class": r["cls"], "true_m": r["z"],
                       "width_m": None if zw is None else round(zw, 2), "width_err_pct": e(zw),
                       "ground_m": None if zg is None else round(zg, 2), "ground_err_pct": e(zg),
                       "fused_m": None if zf is None else round(zf, 2), "fused_err_pct": e(zf)})

    def mae_pct(key):
        v = [abs(x[key]) for x in report if x[key] is not None]
        return round(sum(v) / len(v), 1) if v else None

    summary = {"photos": len(report), "mean_abs_err_pct": {k: mae_pct(f"{k}_err_pct") for k in ("width", "ground", "fused")}}
    print(f"{'file':28s} {'true':>6s} {'width':>7s} {'ground':>7s} {'fused':>7s} {'err%':>6s}")
    for x in report:
        print(f"{x['file'][:28]:28s} {x['true_m']:6.1f} {str(x['width_m']):>7s} {str(x['ground_m']):>7s} {str(x['fused_m']):>7s} {str(x['fused_err_pct']):>6s}")
    print("Leave-one-out mean absolute error (%):", summary["mean_abs_err_pct"])

    out = {"f_long": cal.f_long, "horizon_offset": cal.horizon_offset, "camera_height_m": cal.camera_height_m,
           "widths": cal.widths, "calibrated": True, "fitted_on_photos": len(rows), "loo_error": summary}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)
    print(f"Wrote {a.out} (copy to app/rider/src/main/assets/) and {a.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
