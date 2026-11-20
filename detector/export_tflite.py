"""Export trained YOLO weights to TFLite for the Android app, and describe the result.

Exports FP32 by default: float32 input/output, which is what the app expects, works with the
Android GPU delegate, and needs no INT8 calibration (that step ran out of memory on Kaggle).
Recent Ultralytics (>= 8.4.83) replaced format='tflite' with 'litert' and dropped FP16; on older
versions the script falls back to format='tflite', half=True.
Optional: --dynamic-int8 also writes a smaller model (INT8 weights, float activations, no
calibration) for CPU-only phones; --int8 tries full INT8 with a small calibration fraction.

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
    ap.add_argument("--dynamic-int8", action="store_true", help="Also export INT8-weight / float-activation model")
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

    def run_export(**kw):
        before = set(glob.glob(str(a.weights.parent / "**" / "*.tflite"), recursive=True))
        res = model.export(imgsz=a.imgsz, **kw)
        print(f"Export returned: {res}", flush=True)
        if res and str(res).endswith(".tflite") and os.path.exists(str(res)):
            return Path(str(res))
        new = sorted(set(glob.glob(str(a.weights.parent / "**" / "*.tflite"), recursive=True)) - before)
        if res and os.path.isdir(str(res)):
            new = sorted(glob.glob(str(Path(str(res)) / "*.tflite"))) + new
        return Path(new[0]) if new else None

    print("Exporting float TFLite (LiteRT FP32)...", flush=True)
    fp16 = None
    try:
        fp16 = run_export(format="litert")
    except Exception as e:
        print(f"LiteRT export unavailable ({e}); trying legacy format='tflite', half=True", flush=True)
        fp16 = run_export(format="tflite", half=True)
    if fp16 is None:
        print("ERROR: no .tflite produced", file=sys.stderr)
        return 1
    shutil.copy2(fp16, a.out / "detector.tflite")
    print(f"Copied {fp16.name} -> {a.out / 'detector.tflite'}", flush=True)

    info = {"source_weights": str(a.weights), "imgsz": a.imgsz, "classes": names,
            "float_file": fp16.name, "float_size_mb": round(fp16.stat().st_size / 1e6, 2)}
    try:
        info["float"] = describe(a.out / "detector.tflite")
    except Exception as e:
        info["float"] = {"error": repr(e)}

    if a.dynamic_int8:
        try:
            print("Exporting dynamic INT8 (w8a32)...", flush=True)
            p8 = run_export(format="litert", quantize="w8a32")
            if p8:
                shutil.copy2(p8, a.out / "detector_w8a32.tflite")
                info["w8a32_size_mb"] = round(p8.stat().st_size / 1e6, 2)
                info["w8a32"] = describe(a.out / "detector_w8a32.tflite")
        except Exception as e:
            info["w8a32_error"] = repr(e)
            print(f"w8a32 export failed: {e}", flush=True)

    if a.int8:
        if not a.data:
            print("Skipping INT8: --data not given")
        else:
            try:
                print(f"Exporting INT8 TFLite (calibration fraction {a.fraction})...", flush=True)
                r8 = run_export(format="litert", quantize=8, data=str(a.data), fraction=a.fraction)
                if r8:
                    shutil.copy2(r8, a.out / "detector_int8.tflite")
                    info["int8_file"] = r8.name
            except Exception as e:
                info["int8_error"] = repr(e)
                print(f"INT8 export failed: {e}", flush=True)

    with open(a.out / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
