"""Export trained YOLO weights to TFLite for the Android app, and describe the result.

Does FP16 only by default: it keeps a float32 input, which is what the app expects, and avoids
the INT8 calibration step that ran out of memory on Kaggle. INT8 is optional (--int8), with a
small calibration fraction.

Writes to --out:
  detector.tflite   (copy into app/rider/src/main/assets/)
  labels.txt        (class names in model order; must match the app's assets/labels.txt)
  model_info.json   (input/output shapes and dtypes, to check against the app's decoder)

Usage:
  python detector/export_tflite.py --weights best.pt --out exports/ [--imgsz 640] [--int8 --data data.yaml]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path


def describe(tflite: Path) -> dict:
    """Load the model with whichever TFLite interpreter is installed and report its tensors."""
    Interpreter = None
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except Exception:
        try:
            import tensorflow as tf  # type: ignore
            Interpreter = tf.lite.Interpreter
        except Exception:
            pass
    if Interpreter is None:
        return {"note": "no TFLite interpreter installed; shapes not checked"}
    it = Interpreter(model_path=str(tflite))
    it.allocate_tensors()
    info = {
        "inputs": [{"shape": [int(x) for x in d["shape"]], "dtype": str(d["dtype"].__name__)} for d in it.get_input_details()],
        "outputs": [{"shape": [int(x) for x in d["shape"]], "dtype": str(d["dtype"].__name__)} for d in it.get_output_details()],
    }
    # One dummy run, timed, as a rough CPU sanity check (not a phone benchmark).
    import numpy as np
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], np.random.rand(*d["shape"]).astype(d["dtype"]))
    t0 = time.time(); it.invoke(); info["cpu_invoke_ms_on_this_machine"] = round((time.time() - t0) * 1000, 1)
    out = it.get_tensor(it.get_output_details()[0]["index"])
    info["output_value_range"] = [float(out.min()), float(out.max())]
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true", help="Also try an INT8 export (needs --data)")
    ap.add_argument("--data", type=Path, help="data.yaml for INT8 calibration")
    ap.add_argument("--fraction", type=float, default=0.05, help="Share of the dataset used for INT8 calibration")
    a = ap.parse_args()

    from ultralytics import YOLO
    a.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(a.weights))
    names = [model.names[i] for i in sorted(model.names)]
    (a.out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Classes ({len(names)}): {names}", flush=True)

    before = set(glob.glob(str(a.weights.parent / "**" / "*.tflite"), recursive=True))
    print("Exporting FP16 TFLite...", flush=True)
    res = model.export(format="tflite", imgsz=a.imgsz, half=True)
    print(f"Export returned: {res}", flush=True)
    found = sorted(set(glob.glob(str(a.weights.parent / "**" / "*.tflite"), recursive=True)) - before)
    if res and str(res).endswith(".tflite") and os.path.exists(str(res)):
        found = [str(res)] + [f for f in found if f != str(res)]
    if not found:
        found = sorted(glob.glob(str(a.weights.parent / "**" / "*.tflite"), recursive=True))
    if not found:
        print("ERROR: no .tflite produced", file=sys.stderr)
        return 1
    fp16 = Path(next((f for f in found if "16" in Path(f).name), found[0]))
    shutil.copy2(fp16, a.out / "detector.tflite")
    print(f"Copied {fp16.name} -> {a.out / 'detector.tflite'}", flush=True)

    info = {"source_weights": str(a.weights), "imgsz": a.imgsz, "classes": names,
            "fp16_file": fp16.name, "fp16_size_mb": round(fp16.stat().st_size / 1e6, 2)}
    try:
        info["fp16"] = describe(a.out / "detector.tflite")
    except Exception as e:
        info["fp16"] = {"error": repr(e)}

    if a.int8:
        if not a.data:
            print("Skipping INT8: --data not given")
        else:
            try:
                print(f"Exporting INT8 TFLite (calibration fraction {a.fraction})...", flush=True)
                r8 = model.export(format="tflite", imgsz=a.imgsz, int8=True, data=str(a.data), fraction=a.fraction)
                if r8 and os.path.exists(str(r8)):
                    shutil.copy2(r8, a.out / "detector_int8.tflite")
                    info["int8_file"] = Path(str(r8)).name
            except Exception as e:
                info["int8_error"] = repr(e)
                print(f"INT8 export failed: {e}", flush=True)

    with open(a.out / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
