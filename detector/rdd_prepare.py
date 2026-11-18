"""Prepare RDD2022 for YOLO training.

Handles two layouts, because the Kaggle re-upload's exact structure is not
verified yet:
  * Original release: images + Pascal VOC XML (class codes D00, D10, D20, D40, ...)
  * YOLO layout: images/ + labels/ *.txt, with class names in a data.yaml

Keeps only the four classes in detector/configs/classes.yaml (pothole_classes).
Other RDD classes (e.g. D43, D44, D50, repair, block crack) are dropped.
Images with labels but no kept objects are kept as background images.
Images with no label file at all (e.g. the unlabelled test split) are skipped.

Usage:
  python detector/rdd_prepare.py --src /content/rdd_raw --inspect
  python detector/rdd_prepare.py --src /content/rdd_raw --out /content/rdd_yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES_YAML = Path(__file__).parent / "configs" / "classes.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# RDD2022 class codes -> our class names.
RDD_CODE_MAP = {
    "d00": "longitudinal_crack",
    "d10": "transverse_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
}

# Country / subset detection from path. Order matters: China subsets first.
COUNTRY_PATTERNS = [
    ("China_Drone", re.compile(r"china[_\- ]?drone", re.I)),
    ("China_MotorBike", re.compile(r"china[_\- ]?motor[_\- ]?bike", re.I)),
    ("Czech", re.compile(r"czech", re.I)),
    ("India", re.compile(r"india", re.I)),
    ("Japan", re.compile(r"japan", re.I)),
    ("Norway", re.compile(r"norway", re.I)),
    ("United_States", re.compile(r"united[_\- ]?states|\busa?\b", re.I)),
]

# Aerial drone images do not match a handlebar-mounted phone view.
DEFAULT_EXCLUDE = {"China_Drone"}


def load_target_classes() -> list[str]:
    with open(CLASSES_YAML, "r", encoding="utf-8") as f:
        return list(yaml.safe_load(f)["pothole_classes"])


def normalise(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def map_name(raw: str, targets: set[str]) -> str | None:
    """Map a source class name or code to a target class, or None to drop it."""
    n = normalise(raw)
    if n in RDD_CODE_MAP:
        return RDD_CODE_MAP[n]
    if n in targets:
        return n
    # Common spelled-out variants, e.g. "Longitudinal Crack", "potholes".
    for t in targets:
        if n.rstrip("s") == t or n.replace("_", "") == t.replace("_", ""):
            return t
    return None


def detect_country(path: Path) -> str:
    s = str(path)
    for name, pat in COUNTRY_PATTERNS:
        if pat.search(s):
            return name
    return "Unknown"


def scan(src: Path) -> tuple[list[Path], dict[str, list[Path]], dict[str, list[Path]], list[Path]]:
    images: list[Path] = []
    xml_by_stem: dict[str, list[Path]] = defaultdict(list)
    txt_by_stem: dict[str, list[Path]] = defaultdict(list)
    yamls: list[Path] = []
    for root, _dirs, files in os.walk(src):
        for fn in files:
            p = Path(root) / fn
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(p)
            elif ext == ".xml":
                xml_by_stem[p.stem].append(p)
            elif ext == ".txt" and p.name not in {"classes.txt", "README.txt"}:
                txt_by_stem[p.stem].append(p)
            elif ext in {".yaml", ".yml"}:
                yamls.append(p)
    images.sort()
    return images, xml_by_stem, txt_by_stem, yamls


def pick_label(img: Path, candidates: list[Path]) -> Path | None:
    """Choose the label file belonging to this image when stems collide."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the label whose path shares the longest common prefix with the image.
    img_parts = img.parts
    def score(p: Path) -> int:
        n = 0
        for a, b in zip(img_parts, p.parts):
            if a != b:
                break
            n += 1
        return n
    return max(candidates, key=score)


def yolo_source_names(yamls: list[Path]) -> list[str] | None:
    for y in yamls:
        try:
            with open(y, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            names = d["names"]
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=int)]
            if isinstance(names, list):
                return [str(n) for n in names]
    return None


def image_size(img: Path) -> tuple[int, int]:
    from PIL import Image  # imported lazily; only needed for XML without <size>
    with Image.open(img) as im:
        return im.size


def parse_voc(xml_path: Path, img: Path, targets: set[str], drop: Counter) -> list[tuple[str, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = h = 0
    if size is not None:
        w = int(float(size.findtext("width", "0") or 0))
        h = int(float(size.findtext("height", "0") or 0))
    if w <= 0 or h <= 0:
        w, h = image_size(img)
    boxes = []
    for obj in root.findall("object"):
        raw = obj.findtext("name", "")
        cls = map_name(raw, targets)
        if cls is None:
            drop[raw] += 1
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            drop["<degenerate box>"] += 1
            continue
        boxes.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return boxes


def parse_yolo(txt: Path, src_names: list[str], targets: set[str], drop: Counter) -> list[tuple[str, float, float, float, float]]:
    boxes = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        raw = src_names[cid] if 0 <= cid < len(src_names) else f"<id {cid}>"
        cls = map_name(raw, targets)
        if cls is None:
            drop[raw] += 1
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((cls, cx, cy, bw, bh))
    return boxes


def inspect(src: Path) -> int:
    images, xmls, txts, yamls = scan(src)
    print(f"Source: {src}")
    print(f"Images: {len(images)}   XML files: {sum(map(len, xmls.values()))}   TXT files: {sum(map(len, txts.values()))}   YAML: {len(yamls)}")
    by_country = Counter(detect_country(p) for p in images)
    print("Images by country/subset:")
    for k, v in sorted(by_country.items()):
        print(f"  {k:16s} {v}")
    names = yolo_source_names(yamls)
    if names:
        print(f"YOLO class names found in YAML: {names}")
    raw_counts: Counter = Counter()
    for paths in list(xmls.values())[:3000]:
        for obj in ET.parse(paths[0]).getroot().findall("object"):
            raw_counts[obj.findtext("name", "")] += 1
    if raw_counts:
        print("XML class names (sample of up to 3000 files):")
        for k, v in raw_counts.most_common():
            print(f"  {k:20s} {v}")
    top = sorted({str(p.relative_to(src)).split(os.sep)[0] for p in images})[:20]
    print(f"Top-level folders containing images: {top}")
    return 0


def prepare(src: Path, out: Path, seed: int, val_frac: float, test_frac: float,
            include: set[str] | None, exclude: set[str], max_background_frac: float) -> int:
    targets = load_target_classes()
    tset = set(targets)
    cid = {n: i for i, n in enumerate(targets)}
    images, xmls, txts, yamls = scan(src)
    src_names = yolo_source_names(yamls)
    mode = "xml" if xmls else ("yolo" if txts and src_names else None)
    if mode is None:
        print("ERROR: found neither VOC XML labels nor YOLO labels with a names YAML. Run --inspect.", file=sys.stderr)
        return 1
    print(f"Label format detected: {mode}")

    drop: Counter = Counter()
    records = []  # (img, country, boxes)
    no_label = 0
    for img in images:
        country = detect_country(img)
        if country in exclude or (include and country not in include):
            continue
        if mode == "xml":
            lab = pick_label(img, xmls.get(img.stem, []))
            if lab is None:
                no_label += 1
                continue
            boxes = parse_voc(lab, img, tset, drop)
        else:
            lab = pick_label(img, txts.get(img.stem, []))
            if lab is None:
                no_label += 1
                continue
            boxes = parse_yolo(lab, src_names, tset, drop)
        records.append((img, country, boxes))

    # Cap background (no-object) images so they do not dominate training.
    rng = random.Random(seed)
    pos = [r for r in records if r[2]]
    bg = [r for r in records if not r[2]]
    max_bg = int(len(pos) * max_background_frac / max(1e-9, 1 - max_background_frac))
    rng.shuffle(bg)
    bg_dropped = max(0, len(bg) - max_bg)
    records = pos + bg[:max_bg]

    # Stratified split by country.
    by_c: dict[str, list] = defaultdict(list)
    for r in records:
        by_c[r[1]].append(r)
    split_of = {}
    for country, rs in by_c.items():
        rs.sort(key=lambda r: str(r[0]))
        rng.shuffle(rs)
        n = len(rs)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, r in enumerate(rs):
            split_of[r[0]] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(Counter)
    manifest_rows = []
    india_lists = defaultdict(list)
    used_names: set[str] = set()
    for img, country, boxes in records:
        s = split_of[img]
        name = f"{country}__{img.stem}"
        i = 1
        while name in used_names:
            i += 1
            name = f"{country}__{img.stem}_{i}"
        used_names.add(name)
        dst_img = out / "images" / s / f"{name}{img.suffix.lower()}"
        if not dst_img.exists():
            try:
                os.symlink(img.resolve(), dst_img)
            except OSError:
                import shutil
                shutil.copy2(img, dst_img)
        with open(out / "labels" / s / f"{name}.txt", "w", encoding="utf-8") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write(f"{cid[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                stats[f"{s}/{country}"][cls] += 1
        stats[f"{s}/{country}"]["_images"] += 1
        manifest_rows.append((name, s, country, len(boxes), str(img.relative_to(src))))
        if country == "India":
            india_lists[s].append(str(dst_img))

    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {i: n for i, n in enumerate(targets)}}
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # India-only evaluation sets, for the report.
    for s in ("val", "test"):
        if india_lists[s]:
            (out / f"india_{s}.txt").write_text("\n".join(india_lists[s]) + "\n", encoding="utf-8")
    if india_lists["val"]:
        india_yaml = dict(data_yaml)
        india_yaml["val"] = str(out / "india_val.txt")
        india_yaml["test"] = str(out / "india_test.txt") if india_lists["test"] else india_yaml["test"]
        with open(out / "data_india.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(india_yaml, f, sort_keys=False)

    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "country", "n_boxes", "source_path"])
        w.writerows(manifest_rows)

    summary = {
        "label_format": mode,
        "classes": targets,
        "excluded_subsets": sorted(exclude),
        "images_used": len(records),
        "images_skipped_no_label": no_label,
        "background_images_dropped": bg_dropped,
        "dropped_objects_by_source_class": dict(drop.most_common()),
        "per_split_country": {k: dict(v) for k, v in sorted(stats.items())},
    }
    with open(out / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    totals = Counter()
    for k, v in stats.items():
        totals[k.split("/")[0]] += v["_images"]
    print(f"Images used: {len(records)}  (train {totals['train']}, val {totals['val']}, test {totals['test']})")
    print(f"Skipped (no label file): {no_label}   Background images dropped by cap: {bg_dropped}")
    print(f"Dropped objects by source class: {dict(drop.most_common())}")
    print(f"Wrote {out / 'data.yaml'}")
    if india_lists["val"]:
        print(f"Wrote {out / 'data_india.yaml'} (India val: {len(india_lists['val'])} images)")
    else:
        print("WARNING: no India images found in val. Check country detection with --inspect.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--include", nargs="*", help="Only these subsets, e.g. India Japan")
    ap.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    ap.add_argument("--max-background-frac", type=float, default=0.2,
                    help="Max share of no-object images in the final set")
    a = ap.parse_args()
    if a.inspect:
        return inspect(a.src)
    if a.out is None:
        ap.error("--out is required unless --inspect")
    return prepare(a.src, a.out, a.seed, a.val_frac, a.test_frac,
                   set(a.include) if a.include else None, set(a.exclude or []),
                   a.max_background_frac)


if __name__ == "__main__":
    sys.exit(main())
