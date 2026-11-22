"""Analyse one recorded ride from the app (Android/data/dtu.rider/files/recordings/ride_<time>/).

Reports:
  * IMU: real sampling rate, jitter and gaps (Android treats 50 Hz as a hint)
  * GPS: fix rate, gaps, speed, accuracy
  * Detector: frames analysed, FPS, inference time, pothole detections
  * Vehicles: tracks, closest distance, lowest time-to-collision
  * Crash rule: replays the app's multi-signal rule (Python port of accident/CrashDetector.kt)
    and lists every impact with which conditions held, so thresholds can be tuned
Writes report.md and ride_plot.png into the ride folder.

Usage:
  python eval/analyze_ride.py RIDE_DIR [--impact-g 2.0 --tilt-deg 50 ...]
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import deque
from pathlib import Path

import pandas as pd

G = 9.81


class CrashRule:
    """Port of dtu.rider.accident.CrashDetector. Keep thresholds and logic in sync."""

    def __init__(self, impact_g=2.0, tilt_deg=50.0, tilt_hold_ms=1000, min_speed_before=4.2,
                 max_speed_after=1.5, still_std=0.6, still_ms=3000, eval_window_ms=10_000, confirm_ms=20_000):
        self.p = dict(impact_g=impact_g, tilt_deg=tilt_deg, tilt_hold_ms=tilt_hold_ms, min_speed_before=min_speed_before,
                      max_speed_after=max_speed_after, still_std=still_std, still_ms=still_ms,
                      eval_window_ms=eval_window_ms, confirm_ms=confirm_ms)
        self.state = "IDLE"
        self.g = None
        self.upright = None
        self.calib: list = []
        self.speeds: deque = deque()
        self.mags: deque = deque()
        self.events: list[dict] = []
        self.cur: dict | None = None

    def on_gps(self, t_ms: int, speed: float | None):
        if speed is None or math.isnan(speed):
            return
        self.speeds.append((t_ms, speed))
        while self.speeds and t_ms - self.speeds[0][0] > 15_000:
            self.speeds.popleft()

    def tilt(self) -> float:
        u, g = self.upright, self.g
        if u is None:
            return 0.0
        dot = sum(a * b for a, b in zip(g, u))
        n = math.sqrt(sum(a * a for a in g)) * math.sqrt(sum(b * b for b in u))
        return 0.0 if n == 0 else math.degrees(math.acos(max(-1.0, min(1.0, dot / n))))

    def on_imu(self, t: int, ax: float, ay: float, az: float):
        P = self.p
        if self.g is None:
            self.g = [ax, ay, az]
        self.g = [0.9 * gv + 0.1 * a for gv, a in zip(self.g, (ax, ay, az))]
        if self.upright is None:
            self.calib.append((ax, ay, az))
            if len(self.calib) >= 150:
                self.upright = [sum(c[i] for c in self.calib) / len(self.calib) for i in range(3)]
            return
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        self.mags.append((t, mag))
        while self.mags and t - self.mags[0][0] > P["still_ms"]:
            self.mags.popleft()

        if self.state == "IDLE":
            if mag / G > P["impact_g"]:
                before = [s for ts, s in self.speeds if t - ts <= 5000]
                self.state = "IMPACT"
                self.cur = {"t_ms": t, "peak_g": mag / G, "speed_before": max(before) if before else 0.0,
                            "tilt_ok": False, "tilt_since": None, "max_tilt": 0.0, "speed_ok": False, "still_ok": False}
        elif self.state == "IMPACT":
            c = self.cur
            c["peak_g"] = max(c["peak_g"], mag / G)
            tl = self.tilt()
            c["max_tilt"] = max(c["max_tilt"], tl)
            if tl > P["tilt_deg"]:
                c["tilt_since"] = c["tilt_since"] or t
                if t - c["tilt_since"] >= P["tilt_hold_ms"]:
                    c["tilt_ok"] = True
            else:
                c["tilt_since"] = None
            speed_now = self.speeds[-1][1] if self.speeds else None
            c["speed_ok"] = c["speed_before"] >= P["min_speed_before"] and speed_now is not None and speed_now <= P["max_speed_after"]
            vals = [m for _, m in self.mags]
            c["still_ok"] = t - c["t_ms"] >= P["still_ms"] and len(vals) > 1 and statistics.stdev(vals) < P["still_std"]
            if c["tilt_ok"] and c["speed_ok"] and c["still_ok"]:
                self.state = "CONFIRMING"
                c["outcome"] = "CRASH (countdown started)"
                c["confirm_at"] = t
            elif t - c["t_ms"] > P["eval_window_ms"]:
                self.state = "IDLE"
                c["outcome"] = "no alert"
                self.events.append(c); self.cur = None
        elif self.state == "CONFIRMING":
            if t - self.cur["confirm_at"] >= P["confirm_ms"]:
                self.state = "IDLE"   # in the app: ALERT + SMS (unless cancelled)
                self.events.append(self.cur); self.cur = None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride", type=Path)
    ap.add_argument("--impact-g", type=float, default=2.0)
    ap.add_argument("--tilt-deg", type=float, default=50.0)
    ap.add_argument("--max-speed-after", type=float, default=1.5)
    ap.add_argument("--still-std", type=float, default=0.6)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    d = a.ride
    lines = [f"# Ride analysis: {d.name}", ""]
    dev = d / "device.txt"
    if dev.exists():
        lines += ["Device: " + ", ".join(l.strip() for l in dev.read_text().splitlines() if l.strip()), ""]

    imu = read_csv(d / "imu.csv")
    gps = read_csv(d / "gps.csv")
    det = read_csv(d / "detections.csv")
    veh = read_csv(d / "vehicles.csv")

    if imu is not None:
        dt = imu["t_ns"].diff().dropna() / 1e6
        dur = (imu["t_ns"].iloc[-1] - imu["t_ns"].iloc[0]) / 1e9
        mag_g = (imu[["ax", "ay", "az"]] ** 2).sum(axis=1) ** 0.5 / G
        lines += ["## IMU", f"- Duration {dur:.0f} s, {len(imu)} samples, **mean rate {len(imu) / max(dur, 1e-6):.1f} Hz** (requested 50 Hz)",
                  f"- Interval median {dt.median():.1f} ms, 95th pct {dt.quantile(0.95):.1f} ms, gaps > 100 ms: {(dt > 100).sum()}",
                  f"- Acceleration magnitude: max {mag_g.max():.2f} g, samples above 2 g: {(mag_g > 2).sum()}", ""]
    if gps is not None:
        gdt = gps["t_wall_ms"].diff().dropna() / 1000 if "t_wall_ms" in gps else gps["t_ms"].diff().dropna() / 1000
        sp = gps["speed_mps"].dropna()
        lines += ["## GPS", f"- {len(gps)} fixes, interval median {gdt.median():.1f} s, gaps > 3 s: {(gdt > 3).sum()}",
                  f"- Speed: max {sp.max() * 3.6:.0f} km/h, mean {sp.mean() * 3.6:.0f} km/h" if len(sp) else "- No speed values",
                  f"- Accuracy median {gps['accuracy_m'].median():.1f} m" if "accuracy_m" in gps and gps["accuracy_m"].notna().any() else "", ""]
    if det is not None:
        frames = det.drop_duplicates("frame")
        span = (frames["t_ms"].iloc[-1] - frames["t_ms"].iloc[0]) / 1000
        inf = frames["inference_ms"]
        pot = det[det["label"] == "pothole"]
        lines += ["## Road detector", f"- {len(frames)} frames, **{len(frames) / max(span, 1e-6):.1f} FPS** overall",
                  f"- Inference median {inf.median():.0f} ms, 95th pct {inf.quantile(0.95):.0f} ms",
                  f"- Pothole boxes: {len(pot)} in {pot['frame'].nunique()} frames", ""]
    if veh is not None:
        ttc = pd.to_numeric(veh["ttc_s"], errors="coerce")
        lines += ["## Vehicles", f"- {veh['track_id'].nunique()} tracks, closest {veh['distance_m'].min():.1f} m, "
                  f"lowest TTC {ttc.min():.1f} s" if ttc.notna().any() else f"- {veh['track_id'].nunique()} tracks, closest {veh['distance_m'].min():.1f} m", ""]

    if imu is not None and "t_wall_ms" in imu:
        lines += ["## Crash rule replay"]
        rule = CrashRule(impact_g=a.impact_g, tilt_deg=a.tilt_deg, max_speed_after=a.max_speed_after, still_std=a.still_std)
        # The app keeps ~50 Hz (>= 18 ms apart) even if the phone delivers more; replay the same stream.
        keep, last = [], None
        for t_ns in imu["t_ns"]:
            ok = last is None or t_ns - last >= 18_000_000
            keep.append(ok)
            if ok:
                last = t_ns
        imu50 = imu[keep]
        lines.append(f"Replay uses {len(imu50)} of {len(imu)} IMU samples (~50 Hz, as the app does).")
        ev = [("imu", int(r.t_wall_ms), r.ax, r.ay, r.az) for r in imu50.itertuples()]
        if gps is not None and "t_wall_ms" in gps:
            ev += [("gps", int(r.t_wall_ms), r.speed_mps) for r in gps.itertuples()]
        ev.sort(key=lambda e: e[1])
        for e in ev:
            if e[0] == "gps":
                rule.on_gps(e[1], e[2])
            else:
                rule.on_imu(e[1], e[2], e[3], e[4])
        if rule.cur is not None:
            rule.cur.setdefault("outcome", "ride ended during evaluation"); rule.events.append(rule.cur)
        t0 = int(imu["t_wall_ms"].iloc[0])
        lines += [f"Thresholds: {rule.p}", ""]
        if rule.events:
            lines += ["| t (s) | peak g | speed before (km/h) | max tilt | tilt ok | speed ok | still ok | outcome |", "|---|---|---|---|---|---|---|---|"]
            for c in rule.events:
                lines.append(f"| {(c['t_ms'] - t0) / 1000:.1f} | {c['peak_g']:.2f} | {c['speed_before'] * 3.6:.0f} | {c['max_tilt']:.0f} | "
                             f"{c['tilt_ok']} | {c['speed_ok']} | {c['still_ok']} | {c.get('outcome', '')} |")
        else:
            lines.append("No impacts above the threshold.")
        lines.append("")
    elif imu is not None:
        lines += ["## Crash rule replay", "Skipped: imu.csv has no t_wall_ms column (recorded with an older app build).", ""]

    if not a.no_plot and imu is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = (imu["t_ns"] - imu["t_ns"].iloc[0]) / 1e9
        fig, axs = plt.subplots(2 if gps is None else 3, 1, figsize=(11, 7), sharex=gps is not None and "t_wall_ms" in gps and "t_wall_ms" in imu)
        axs[0].plot(t, mag_g, lw=0.6); axs[0].axhline(a.impact_g, color="r", ls="--", lw=0.8); axs[0].set_ylabel("|a| (g)")
        axs[1].plot(t[1:], dt.values, lw=0.6); axs[1].set_ylabel("IMU interval (ms)")
        if gps is not None and "t_wall_ms" in gps and "t_wall_ms" in imu:
            tg = (gps["t_wall_ms"] - imu["t_wall_ms"].iloc[0]) / 1000
            axs[2].plot(tg, gps["speed_mps"] * 3.6, marker=".", lw=0.8); axs[2].set_ylabel("GPS speed (km/h)")
        axs[-1].set_xlabel("time (s)")
        fig.tight_layout(); fig.savefig(d / "ride_plot.png", dpi=110); plt.close(fig)
        lines.append("![ride plot](ride_plot.png)")

    report = "\n".join(l for l in lines if l is not None)
    (d / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
