"""Prepare a YOLO-format sign dataset (e.g. a Roboflow export) for the shared detector.

* Maps source classes to our sign classes via detector/configs/signs_map.yaml (exact names, then regex rules).
* Writes labels with global class ids: pothole classes first (0..3), then sign classes,
  so the output can be merged with the RDD output in one data.yaml.
* Builds occluded copies of val/test (signs partly covered at given fractions) to measure
  clear vs occluded accuracy, and optionally adds occluded copies to train.
* Writes data_signs.yaml (signs only) and, with --rdd, data_combined.yaml (potholes + signs).

Usage:
  python detector/signs_prepare.py --src SIGNS_EXPORT --out OUT --inspect
  python detector/signs_prepare.py --src SIGNS_EXPORT --out OUT [--rdd RDD_OUT] [--occlude-train-frac 0.3]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from rdd_prepare import IMAGE_EXTS, detect_split, normalise, read_texts  # noqa: E402

CONFIGS = Path(__file__).parent / "configs"


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def target_classes(mode: str, smap: dict, mapper: dict[str, str | None] | None = None) -> tuple[list[str], list[str]]:
    """Returns (all class names in global order, sign class names)."""
    cls = load_yaml(CONFIGS / "classes.yaml")
    pothole = list(cls["pothole_classes"])
    if mode == "grouped":
        signs = list(cls["sign_classes"])
    else:
        signs = sorted({v for v in (mapper or {}).values() if v})
    return pothole + signs, signs


def build_mapper(mode: str, smap: dict, names: list[str]) -> dict[str, str | None]:
    """Source name (normalised) -> our sign class name, or None to drop. Unmatched names are absent."""
    import re
    m: dict[str, str | None] = {}
    for group, members in (smap.get("groups") or {}).items():
        for n in members:
            m[normalise(n)] = group
    for n in smap.get("drop") or []:
        m[normalise(n)] = None
    rules = [(g, re.compile(pat)) for g, pat in (smap.get("rules") or [])]
    for raw in names:
        n = normalise(raw).replace("-", "_")
        if n in m:
            continue
        for g, rx in rules:
            if rx.search(n):
                m[n] = None if g == "drop" else g
                break
    if mode == "fine":
        m = {k: (k if v is not None else None) for k, v in m.items()}
    return m


def find_source_names(src: Path) -> list[str]:
    for y in sorted(src.rglob("*.yaml")) + sorted(src.rglob("*.yml")):
        d = load_yaml(y)
        names = d.get("names") if isinstance(d, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    raise SystemExit("No data.yaml with 'names' found in the sign export")


def scan_pairs(src: Path) -> list[tuple[Path, Path, str]]:
    """(image, label, split) for a Roboflow-style layout: <split>/images, <split>/labels."""
    labels: dict[tuple[str, str], Path] = {}
    images: list[tuple[Path, str]] = []
    for root, _d, files in os.walk(src):
        for fn in files:
            p = Path(root) / fn
            sp = detect_split(p, src)
            if sp is None:
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                images.append((p, sp))
            elif p.suffix.lower() == ".txt":
                labels[(sp, p.stem)] = p
    pairs = []
    for img, sp in sorted(images):
        lab = labels.get((sp, img.stem))
        if lab is not None:
            pairs.append((img, lab, sp))
    return pairs


def occlude(img, boxes_px: list[tuple[float, float, float, float]], frac: float, rng: random.Random, donors: list):
    """Cover about `frac` of each box's area with a patch from another image, entering from one side.

    Mimics a tree branch or hoarding hiding part of a sign. Returns a new image.
    """
    from PIL import Image
    out = img.copy()
    W, H = out.size
    for (x1, y1, x2, y2) in boxes_px:
        bw, bh = max(1, int(x2 - x1)), max(1, int(y2 - y1))
        side = rng.choice(["left", "right", "top", "bottom"])
        if side in ("left", "right"):
            pw, ph = max(1, int(bw * frac)), bh
        else:
            pw, ph = bw, max(1, int(bh * frac))
        donor = rng.choice(donors) if donors else img
        dw, dh = donor.size
        sx = rng.randint(0, max(0, dw - pw)); sy = rng.randint(0, max(0, dh - ph))
        patch = donor.crop((sx, sy, sx + pw, sy + ph))
        if patch.size != (pw, ph):
            patch = patch.resize((pw, ph))
        px = int(x1) if side != "right" else int(x2) - pw
        py = int(y1) if side != "bottom" else int(y2) - ph
        out.paste(patch, (max(0, min(W - pw, px)), max(0, min(H - ph, py))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--rdd", type=Path, help="RDD output dir from rdd_prepare.py, to write data_combined.yaml")
    ap.add_argument("--occlude-fracs", type=float, nargs="*", default=[0.25, 0.5])
    ap.add_argument("--occlude-train-frac", type=float, default=0.3,
                    help="Share of train images that also get an occluded copy (0 to disable)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    smap = load_yaml(CONFIGS / "signs_map.yaml")
    mode = smap.get("mode", "grouped")
    src_names = find_source_names(a.src)
    mapper = build_mapper(mode, smap, src_names)
    all_names, sign_names = target_classes(mode, smap, mapper)
    gid = {n: i for i, n in enumerate(all_names)}

    unmapped = [n for n in src_names if normalise(n) not in mapper]
    pairs = scan_pairs(a.src)
    print(f"Mode: {mode}. Source classes ({len(src_names)}): {src_names}")
    print(f"Image/label pairs: {len(pairs)}  by split: {dict(Counter(sp for _i, _l, sp in pairs))}")
    if unmapped:
        print(f"WARNING: source classes not in signs_map.yaml (will be dropped): {unmapped}")
    texts = read_texts([l for _i, l, _s in pairs])
    src_counts: Counter = Counter()
    for _i, lab, _s in pairs:
        for line in texts[lab].splitlines():
            if line.strip():
                c = int(float(line.split()[0]))
                src_counts[src_names[c] if c < len(src_names) else f"<id {c}>"] += 1
    print("Boxes per source class:")
    for k, v in src_counts.most_common():
        print(f"  {k:28s} {v:6d} -> {mapper.get(normalise(k), 'UNMAPPED')}")
    if a.inspect:
        return 0
    if a.out is None:
        ap.error("--out is required unless --inspect")

    from PIL import Image
    rng = random.Random(a.seed)
    out = a.out
    stats = defaultdict(Counter)
    kept: dict[str, list[tuple[Path, list]]] = defaultdict(list)  # split -> (img, [(cls, cx, cy, w, h)])
    for img, lab, sp in pairs:
        rows = []
        for line in texts[lab].splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            name = src_names[c] if c < len(src_names) else None
            tgt = mapper.get(normalise(name)) if name else None
            if tgt is None:
                stats[sp]["_dropped"] += 1
                continue
            rows.append((tgt, *map(float, parts[1:5])))
            stats[sp][tgt] += 1
        if rows:
            kept[sp].append((img, rows))
        stats[sp]["_images"] += 1 if rows else 0

    def write_set(name: str, items, transform=None):
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for i, (img, rows) in enumerate(items):
            stem = f"sign__{img.stem}" if transform is None else f"sign_occ__{img.stem}"
            dst = out / "images" / name / f"{stem}{img.suffix.lower() if transform is None else '.jpg'}"
            if transform is None:
                if not dst.exists():
                    try:
                        os.symlink(img.resolve(), dst)
                    except OSError:
                        import shutil
                        shutil.copy2(img, dst)
            else:
                transform(img, rows).save(dst, quality=92)
            with open(out / "labels" / name / f"{stem}.txt", "w", encoding="utf-8") as f:
                for tgt, cx, cy, w, h in rows:
                    f.write(f"{gid[tgt]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{len(items)}", flush=True)

    donor_paths = [p for p, _r in rng.sample(kept["train"], min(200, len(kept["train"])))] if kept["train"] else []
    donors = []
    for p in donor_paths:
        with Image.open(p) as im:
            donors.append(im.convert("RGB").copy())

    def make_occ(frac):
        def f(img_path, rows):
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                boxes = [((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H) for _t, cx, cy, w, h in rows]
                return occlude(im, boxes, frac, rng, [d for d in donors if d is not None])
        return f

    for sp in ("train", "val", "test"):
        print(f"Writing {sp}: {len(kept[sp])} images", flush=True)
        write_set(sp, kept[sp])

    if a.occlude_train_frac > 0 and kept["train"]:
        n = int(len(kept["train"]) * a.occlude_train_frac)
        sel = rng.sample(kept["train"], n)
        print(f"Adding {n} occluded train copies (fraction 0.25-0.5)", flush=True)
        def mixed(img_path, rows):
            return make_occ(rng.uniform(0.25, 0.5))(img_path, rows)
        write_set("train_occ", sel, mixed)

    eval_split = "test" if kept["test"] else "val"
    occ_sets = {}
    for frac in a.occlude_fracs:
        name = f"{eval_split}_occ{int(frac * 100)}"
        print(f"Writing {name}", flush=True)
        write_set(name, kept[eval_split], make_occ(frac))
        occ_sets[name] = frac

    names_map = {i: n for i, n in enumerate(all_names)}
    train_dirs = [str((out / "images" / "train").resolve())]
    if (out / "images" / "train_occ").exists():
        train_dirs.append(str((out / "images" / "train_occ").resolve()))

    def dump(fn, d):
        with open(out / fn, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    val_dir = str((out / "images" / ("val" if kept["val"] else eval_split)).resolve())
    dump("data_signs.yaml", {"train": train_dirs, "val": val_dir,
                             "test": str((out / "images" / eval_split).resolve()), "names": names_map})
    for name in occ_sets:
        dump(f"data_signs_{name}.yaml", {"train": train_dirs, "val": str((out / "images" / name).resolve()),
                                          "test": str((out / "images" / name).resolve()), "names": names_map})
    if a.rdd:
        r = load_yaml(a.rdd / "data.yaml")
        rp = Path(r["path"])
        dump("data_combined.yaml", {
            "train": [str(rp / r["train"])] + train_dirs,
            "val": [str(rp / r["val"]), val_dir],
            "test": [str(rp / r["test"]), str((out / "images" / eval_split).resolve())],
            "names": names_map,
        })
        print(f"Wrote {out / 'data_combined.yaml'}")

    summary = {"mode": mode, "source_classes": src_names, "classes": all_names,
               "unmapped_source_classes": unmapped, "occluded_eval_sets": occ_sets,
               "per_split": {k: dict(v) for k, v in stats.items()}}
    with open(out / "signs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["per_split"], indent=2))
    print(f"Wrote {out / 'data_signs.yaml'} and occluded eval sets: {list(occ_sets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
