# Shared detector: potholes + road signs (run `shared_v1`)

- Model: YOLO11n, 640 px, batch 32, Kaggle T4, 5 h cap -> 51 epochs (~5.6 min each)
- Classes (9): pothole, longitudinal_crack, transverse_crack, alligator_crack, regulatory, warning, hospital, school, speed_limit
- Data: RDD2022 (as in pothole_v1) + Indian Traffic SignBoards (Roboflow, MIT; 75 classes grouped into 5),
  close-up signs plus signs pasted into RDD road scenes at 2.5-12% of frame width; hospital/school oversampled 6x in pasted train
- Export: FP32 LiteRT, 10.7 MB, input [1, 3, 640, 640], output [1, 13, 8400] (see `shared_v1_model_info.json`)
- All numbers are on held-out **test** sets. Full output: `shared_v1_summary.json`.

## Road damage: shared model vs pothole-only model (mAP@0.5, test)

| Test set | pothole_v1 (4 classes, 138 epochs) | shared_v1 (9 classes, 51 epochs) | Change |
|---|---|---|---|
| All countries | 0.555 | 0.529 | -0.026 |
| India only | 0.364 | 0.313 | -0.051 |

Per class, shared_v1, mAP@0.5: all countries / India
pothole 0.440 / 0.385, longitudinal 0.539 / 0.255, transverse 0.483 / 0.017, alligator 0.653 / 0.594.

Adding signs cost some road-damage accuracy, most on Indian roads. Likely causes: about a third of
the epochs (time cap), model capacity shared with 5 more classes, and unlabelled signs in RDD images
(and unlabelled damage in close-up sign images).

## Signs: clear vs occluded (sign classes only, mAP@0.5, test)

| Sign size | Clear | 25% covered | 50% covered |
|---|---|---|---|
| Close-up (original dataset) | 0.937 | 0.913 | 0.837 |
| **Realistic size (pasted into road scenes)** | **0.906** | **0.802** | **0.443** |

Per class, realistic size, mAP@0.5 (clear / 25% / 50%):

| Class | Clear | 25% | 50% | Test boxes |
|---|---|---|---|---|
| regulatory | 0.975 | 0.960 | 0.607 | 358 |
| warning | 0.951 | 0.932 | 0.559 | 335 |
| speed_limit | 0.955 | 0.921 | 0.558 | 120 |
| hospital | 0.982 | 0.648 | 0.352 | 8 |
| school | 0.666 | 0.550 | 0.142 | 20 |

## Reading

- Realistic-size signs are detected well when clear (0.906) and stay usable at 25% occlusion (0.802),
  but accuracy roughly halves at 50% (0.443). Small signs lose more from occlusion than close-ups do
  (close-up 50%: 0.837), because less of the sign remains in pixels.
- School is the weakest class in every condition. Hospital and school scores rest on 8 and 20 test
  boxes, so they are noisy (e.g. close-up hospital scores higher at 50% occlusion than at 25%).
- Synthetic occlusion (patches from other road images) is a proxy for real occluders; a small real
  occluded test set would make this claim stronger.

## Decision for the app

Use the shared model (one model covers Modules 1 and 2), accepting the road-damage drop for the
mid-term; note it as a limitation. Options after the review: longer training, fine-tuning from
pothole_v1 weights, or India-weighted training.
