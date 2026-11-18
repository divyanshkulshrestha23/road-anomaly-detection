"""Copy-paste augmentation: place sign crops into real road scenes at realistic sizes.

The sign dataset is mostly close-ups (median box ~60% of the image), while a sign seen from a
moving two-wheeler is usually well under 1% of the frame. This script cuts each sign out, scales
it to 2.5-12% of the background width, and pastes it beside the road (mostly on the left, since
India drives on the left) in an RDD road image from the same split. The background's own
road-damage labels are kept, so composites carry both kinds of label.

Outputs (in --signs dir, next to signs_prepare.py output):
  images/train_paste, val_paste, test_paste, test_paste_occ25, test_paste_occ50 (+ labels)
  data_signs_paste*.yaml  (realistic-size sign evaluation sets)
  data_combined.yaml      (rewritten: adds train_paste to train and val_paste to val)

Usage:
  python detector/signs_paste.py --signs SIGN_OUT --rdd RDD_OUT [--train-copies 2]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_SIDE = 1280


def list_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    idir, ldir = root / "images" / split, root / "labels" / split
    if not idir.exists():
        return []
    out = []
    for p in sorted(idir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            lab = ldir / f"{p.stem}.txt"
            if lab.exists():
                out.append((p, lab))
    return out


def open_scaled(path: Path, max_side: int = MAX_SIDE):
    from PIL import Image
    im = Image.open(path)
    try:
        im.draft("RGB", (max_side, max_side))  # fast JPEG decode at reduced size
    except Exception:
        pass
    im = im.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    return im


def occlude_crop(crop, frac: float, donor, rng: random.Random):
    """Cover `frac` of the crop from one side with a patch from the donor (background) image."""
    w, h = crop.size
    side = rng.choice(["left", "right", "top", "bottom"])
    pw, ph = (max(1, int(w * frac)), h) if side in ("left", "right") else (w, max(1, int(h * frac)))
    dw, dh = donor.size
    sx, sy = rng.randint(0, max(0, dw - pw)), rng.randint(0, max(0, dh - ph))
    patch = donor.crop((sx, sy, sx + pw, sy + ph)).resize((pw, ph))
    out = crop.copy()
    out.paste(patch, (0 if side != "right" else w - pw, 0 if side != "bottom" else h - ph))
    return out


def make_one(job: tuple) -> int:
    from PIL import ImageFilter, ImageStat
    sign_img, sign_lab, bg_img, bg_lab, out_img, out_lab, seed, occ = job
    rng = random.Random(seed)
    rows = [l.split() for l in Path(sign_lab).read_text().splitlines() if len(l.split()) >= 5]
    if not rows:
        return 0
    gid, cx, cy, bw, bh = int(rows[0][0]), *map(float, rows[0][1:5])
    sign = open_scaled(Path(sign_img), 1024)
    SW, SH = sign.size
    m = 0.08
    x1 = max(0, int((cx - bw / 2 - bw * m) * SW)); y1 = max(0, int((cy - bh / 2 - bh * m) * SH))
    x2 = min(SW, int((cx + bw / 2 + bw * m) * SW)); y2 = min(SH, int((cy + bh / 2 + bh * m) * SH))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0
    crop = sign.crop((x1, y1, x2, y2))
    # Sign box inside the crop, as fractions of the crop.
    ix1 = ((cx - bw / 2) * SW - x1) / (x2 - x1); iy1 = ((cy - bh / 2) * SH - y1) / (y2 - y1)
    ix2 = ((cx + bw / 2) * SW - x1) / (x2 - x1); iy2 = ((cy + bh / 2) * SH - y1) / (y2 - y1)

    bg = open_scaled(Path(bg_img))
    W, H = bg.size
    if occ:
        # Separate RNG so size and position match the un-occluded variant exactly.
        crop = occlude_crop(crop, occ, bg, random.Random(seed + 1))

    # Size: sign width 2.5-12% of frame width, log-uniform (far signs are more common).
    wf = math.exp(rng.uniform(math.log(0.025), math.log(0.12)))
    tw = max(6, int(W * wf))
    th = max(6, int(tw * crop.height / crop.width))
    if th > 0.3 * H:
        th = int(0.3 * H); tw = max(6, int(th * crop.width / crop.height))
    crop = crop.resize((tw, th))

    # Position: roadside, above the road surface. Left side 70% of the time.
    if rng.random() < 0.7:
        px = int(rng.uniform(0.02, 0.30) * W)
    else:
        px = int(rng.uniform(0.70, 0.98) * W) - tw
    py = int(rng.uniform(0.15, 0.50) * H)
    px = max(0, min(W - tw, px)); py = max(0, min(H - th, py))

    # Rough brightness match to the local background, plus slight blur for small signs.
    region = bg.crop((px, py, px + tw, py + th))
    bm = sum(ImageStat.Stat(region).mean) / 3 + 1e-3
    cm = sum(ImageStat.Stat(crop).mean) / 3 + 1e-3
    f = max(0.6, min(1.4, 0.5 + 0.5 * bm / cm))
    crop = crop.point(lambda v: max(0, min(255, int(v * f))))
    if tw < 48:
        crop = crop.filter(ImageFilter.GaussianBlur(0.6))
    bg.paste(crop, (px, py))
    bg.save(out_img, quality=90)

    sx1, sy1 = px + ix1 * tw, py + iy1 * th
    sx2, sy2 = px + ix2 * tw, py + iy2 * th
    lines = [l for l in Path(bg_lab).read_text().splitlines() if l.strip()]
    lines.append(f"{gid} {(sx1 + sx2) / 2 / W:.6f} {(sy1 + sy2) / 2 / H:.6f} {(sx2 - sx1) / W:.6f} {(sy2 - sy1) / H:.6f}")
    Path(out_lab).write_text("\n".join(lines) + "\n")
    return 1


def run_jobs(jobs: list, name: str, workers: int) -> int:
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(make_one, jobs, chunksize=16), 1):
            done += r
            if i % 1000 == 0 or i == len(jobs):
                print(f"  {name}: {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", type=Path, required=True, help="Output dir of signs_prepare.py")
    ap.add_argument("--rdd", type=Path, required=True, help="Output dir of rdd_prepare.py")
    ap.add_argument("--train-copies", type=int, default=2)
    ap.add_argument("--rare-copies", type=int, default=6,
                    help="Train composites per image for rare classes (--rare-classes)")
    ap.add_argument("--rare-classes", nargs="*", default=["hospital", "school"])
    ap.add_argument("--occlude-fracs", type=float, nargs="*", default=[0.25, 0.5])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = a.signs
    with open(out / "data_signs.yaml", "r", encoding="utf-8") as f:
        names_by_id = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}
    rare_ids = {i for i, n in names_by_id.items() if n in set(a.rare_classes)}

    def sign_class(lab: Path) -> int | None:
        for line in lab.read_text().splitlines():
            if line.split():
                return int(line.split()[0])
        return None

    sets: dict[str, list] = {}
    for split in ("train", "val", "test"):
        signs = list_pairs(out, split)
        bgs = list_pairs(a.rdd, split)
        if not signs or not bgs:
            print(f"Skipping {split}: {len(signs)} sign images, {len(bgs)} backgrounds")
            continue
        copies = a.train_copies if split == "train" else 1
        variants = [(f"{split}_paste", None)]
        if split == "test":
            variants += [(f"test_paste_occ{int(f * 100)}", f) for f in a.occlude_fracs]
        for name, occ in variants:
            (out / "images" / name).mkdir(parents=True, exist_ok=True)
            (out / "labels" / name).mkdir(parents=True, exist_ok=True)
            jobs = []
            # Same background and seed across occlusion variants, so they differ only by occlusion.
            vr = random.Random(a.seed * 10 + {"train": 1, "val": 2, "test": 3}[split])
            for si, (simg, slab) in enumerate(signs):
                n_copies = copies
                if split == "train" and sign_class(slab) in rare_ids:
                    n_copies = max(copies, a.rare_copies)
                for k in range(n_copies):
                    bimg, blab = vr.choice(bgs)
                    stem = f"paste{k}__{simg.stem}"
                    jobs.append((str(simg), str(slab), str(bimg), str(blab),
                                 str(out / "images" / name / f"{stem}.jpg"),
                                 str(out / "labels" / name / f"{stem}.txt"),
                                 vr.randrange(1 << 30), occ))
            n = run_jobs(jobs, name, a.workers)
            sets[name] = n
            print(f"Wrote {n} composites to {name}", flush=True)

    def load(fn):
        with open(out / fn, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def dump(fn, d):
        with open(out / fn, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    base = load("data_signs.yaml")
    names = base["names"]
    img = lambda n: str((out / "images" / n).resolve())
    for name in [n for n in sets if n.startswith("test_paste")]:
        tag = name.replace("test_", "")  # paste, paste_occ25, paste_occ50
        dump(f"data_signs_{tag}.yaml", {"train": base["train"], "val": img(name), "test": img(name), "names": names})

    if (out / "data_combined.yaml").exists():
        comb = load("data_combined.yaml")
        if "train_paste" in sets and img("train_paste") not in comb["train"]:
            comb["train"].append(img("train_paste"))
        if "val_paste" in sets:
            # Realistic-size sign val replaces the close-up sign val for checkpoint selection.
            comb["val"] = [v for v in comb["val"] if v != img("val")] + [img("val_paste")]
        dump("data_combined.yaml", comb)
        print(f"Updated {out / 'data_combined.yaml'}")
    print({k: v for k, v in sets.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
