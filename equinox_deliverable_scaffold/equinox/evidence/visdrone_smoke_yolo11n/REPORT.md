# VisDrone smoke-run evidence: direct YOLO11n

## Scope

This package evaluates the recorded direct-inference CSV against the prepared 50-image VisDrone2019-DET validation smoke subset. It evaluates object detection only. Recorded inference time is **11.057 s / 50 images** (0.221 s/image); it is provenance from the existing run, not a timing claim produced by this evaluator.

At confidence >= 0.25 and IoU >= 0.50, micro precision is **0.773**, recall **0.196**, F1 **0.313**, and mAP@0.5 across supported classes **0.181**. Counts: 461 TP, 135 FP, 1893 FN, 2354 supported ground-truth objects, 596 supported predictions.

## Compatible taxonomy

| VisDrone label | Evaluation class | COCO YOLO11n prediction |
| --- | --- | --- |
| pedestrian, people | person | person |
| bicycle | bicycle | bicycle |
| car | car | car |
| truck | truck | truck |
| bus | bus | bus |
| motor | motorcycle | motorcycle |

Excluded rather than guessed: **van, tricycle, awning-tricycle**, every unsupported COCO prediction, and VisDrone ignore/other annotations. The counts excluded from ground truth are in `metrics.json`; excluded COCO predictions are likewise recorded there.

## Per-class results

| Class | Support | Pred. | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 54 | 3 | 1 | 2 | 53 | 0.333 | 0.019 | 0.035 | 0.019 |
| bus | 0 | 3 | 0 | 3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| car | 428 | 218 | 170 | 48 | 258 | 0.780 | 0.397 | 0.526 | 0.342 |
| motorcycle | 412 | 6 | 4 | 2 | 408 | 0.667 | 0.010 | 0.019 | 0.009 |
| person | 1451 | 358 | 282 | 76 | 1169 | 0.788 | 0.194 | 0.312 | 0.181 |
| truck | 9 | 8 | 4 | 4 | 5 | 0.500 | 0.444 | 0.471 | 0.356 |

AP@0.5 uses confidence-ranked, one-to-one greedy matching and all-points interpolated precision. It is not COCO mAP@[.5:.95].

## Error categories

| Category | Count |
| --- | ---: |
| false_negative_localization | 188 |
| false_negative_misclassification | 34 |
| false_negative_missed | 1671 |
| false_positive_localization | 72 |
| false_positive_misclassification | 17 |
| false_positive_no_overlap | 46 |

## Annotated evidence

Green boxes are compatible mapped VisDrone ground truth. Amber boxes are retained compatible YOLO predictions. These are actual, downscaled inputs, selected as three high-confidence true-positive examples and two failure or edge examples.

- [success: 0000021_00000_d_0000001](annotated/01_success_0000021_00000_d_0000001.jpg)
- [success: 0000069_01878_d_0000005](annotated/02_success_0000069_01878_d_0000005.jpg)
- [success: 0000022_00500_d_0000005](annotated/03_success_0000022_00500_d_0000005.jpg)
- [failure: 0000026_02500_d_0000029](annotated/04_failure_0000026_02500_d_0000029.jpg)
- [failure: 0000076_02142_d_0000009](annotated/05_failure_0000076_02142_d_0000009.jpg)

## Reproduce

```bash
cd equinox_deliverable_scaffold/equinox
python -m pipeline.evaluation.visdrone_smoke \
  --dataset data/visdrone_smoke \
  --predictions artifacts/detections/visdrone_smoke_yolo11n_direct.csv \
  --output evidence/visdrone_smoke_yolo11n
```

No raw dataset, archive, model weights, or full prediction CSV is copied into this evidence bundle. Dataset source and archive provenance are preserved in `metrics.json`. Review dataset terms at [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset).

## Limitations

- This is a 50-image smoke subset, not the official VisDrone evaluation protocol or a held-out Equinox validation set.
- The COCO-pretrained YOLO11n model was not fine-tuned on VisDrone; domain shift, small objects, occlusion, and class definitions materially limit performance.
- Only the conservative compatible taxonomy intersection is evaluated. Unsupported VisDrone classes and unsupported COCO predictions are excluded, not treated as correct or incorrect.
- AP@0.5 is implemented as all-points interpolated AP over the recorded predictions. It is not COCO mAP@[.5:.95].
- No severity, road blockage, lane-clearance, emergency-accessibility, tracking, or temporal validation is established by this detector smoke run.
