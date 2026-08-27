"""Evaluate the recorded COCO YOLO11n VisDrone smoke run.

The evaluator deliberately assesses only a documented intersection of the
COCO and VisDrone taxonomies.  It is not an official VisDrone benchmark.
Run from the ``equinox`` directory:

    python -m pipeline.evaluation.visdrone_smoke \
      --dataset data/visdrone_smoke \
      --predictions artifacts/detections/visdrone_smoke_yolo11n_direct.csv \
      --output evidence/visdrone_smoke_yolo11n
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# VisDrone YOLO label IDs, after prepare_visdrone.py conversion.
VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
)
# This is intentionally conservative.  In particular, van and the two
# tricycle classes do not have an unambiguous COCO class and are excluded.
VISDRONE_TO_EVAL = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
}
COCO_TO_EVAL = {
    "person": "person", "bicycle": "bicycle", "car": "car",
    "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
}
EVAL_CLASSES = tuple(sorted(set(VISDRONE_TO_EVAL.values())))
UNSUPPORTED_VISDRONE = tuple(name for name in VISDRONE_CLASSES if name not in VISDRONE_TO_EVAL)


@dataclass(frozen=True)
class Box:
    image_id: str
    cls: str
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Intersection over union for xyxy boxes; invalid boxes have zero area."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(predictions: list[Box], ground_truth: list[Box], iou_threshold: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Confidence-ordered, one-to-one class-specific IoU matching."""
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for prediction_index, prediction in sorted(enumerate(predictions), key=lambda item: (-item[1].confidence, item[0])):
        candidates = [
            (iou(prediction.xyxy, target.xyxy), target_index)
            for target_index, target in enumerate(ground_truth)
            if target_index not in matched_gt and target.image_id == prediction.image_id and target.cls == prediction.cls
        ]
        if not candidates:
            continue
        overlap, target_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if overlap >= iou_threshold:
            matched_gt.add(target_index)
            matches.append((prediction_index, target_index, overlap))
    matched_predictions = {prediction_index for prediction_index, _, _ in matches}
    return matches, matched_predictions, matched_gt


def average_precision_50(predictions: list[Box], ground_truth: list[Box], iou_threshold: float = 0.5) -> float | None:
    """All-points interpolated AP at one IoU threshold for one class.

    Predictions are globally confidence-ranked and matched once per image.  A
    class with no ground truth has undefined AP and returns ``None``.
    """
    if not ground_truth:
        return None
    matches, matched_predictions, _ = greedy_match(predictions, ground_truth, iou_threshold)
    match_order = {prediction_index for prediction_index, _, _ in matches}
    ranked = sorted(enumerate(predictions), key=lambda item: (-item[1].confidence, item[0]))
    true_positive = false_positive = 0
    precision_recall: list[tuple[float, float]] = []
    for prediction_index, _ in ranked:
        if prediction_index in match_order:
            true_positive += 1
        else:
            false_positive += 1
        precision_recall.append((true_positive / (true_positive + false_positive), true_positive / len(ground_truth)))
    # Integral of the monotonically non-increasing precision envelope over
    # observed recall changes, equivalent to all-points VOC/COCO-style AP.
    envelope: list[tuple[float, float]] = []
    best_precision = 0.0
    for precision, recall in reversed(precision_recall):
        best_precision = max(best_precision, precision)
        envelope.append((best_precision, recall))
    envelope.reverse()
    ap, previous_recall = 0.0, 0.0
    for precision, recall in envelope:
        if recall > previous_recall:
            ap += precision * (recall - previous_recall)
            previous_recall = recall
    return ap


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def load_ground_truth(dataset: Path) -> tuple[list[Box], dict[str, Path], Counter]:
    images_dir, labels_dir = dataset / "val" / "images", dataset / "val" / "labels"
    image_paths = sorted(images_dir.glob("*"))
    if len(image_paths) == 0 or not labels_dir.is_dir():
        raise FileNotFoundError("Expected data/visdrone_smoke/val/{images,labels}")
    paths_by_id = {path.stem: path for path in image_paths if path.is_file()}
    ground_truth: list[Box] = []
    excluded = Counter()
    for image_id, image_path in paths_by_id.items():
        width, height = _image_size(image_path)
        label_path = labels_dir / f"{image_id}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label in {label_path}: {line!r}")
            class_index = int(fields[0])
            source_class = VISDRONE_CLASSES[class_index]
            mapped = VISDRONE_TO_EVAL.get(source_class)
            if mapped is None:
                excluded[source_class] += 1
                continue
            cx, cy, box_w, box_h = (float(value) for value in fields[1:])
            x1, y1 = (cx - box_w / 2) * width, (cy - box_h / 2) * height
            x2, y2 = (cx + box_w / 2) * width, (cy + box_h / 2) * height
            ground_truth.append(Box(image_id, mapped, (x1, y1, x2, y2)))
    return ground_truth, paths_by_id, excluded


def load_predictions(path: Path, image_ids_in_order: list[str], confidence_threshold: float) -> tuple[list[Box], Counter, int]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    predictions: list[Box] = []
    excluded = Counter()
    below_threshold = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                # ``run_yolo.discover_frames`` groups this flat image directory
                # under its directory name and assigns lexical sorted positions.
                # Refuse a different contract rather than silently pairing frame
                # indices to unrelated source images.
                if row["clip_id"] != "images":
                    raise ValueError(f"Expected clip_id 'images', got {row['clip_id']!r}")
                image_id = image_ids_in_order[int(row["frame_idx"])]
                confidence = float(row["conf"])
                xyxy = tuple(float(value) for value in json.loads(row["bbox"]))
            except (KeyError, ValueError, TypeError, IndexError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid prediction row: {row!r}") from error
            if len(xyxy) != 4:
                raise ValueError(f"Prediction bbox must contain four values: {row!r}")
            mapped = COCO_TO_EVAL.get(row["cls"])
            if mapped is None:
                excluded[row["cls"]] += 1
            elif confidence < confidence_threshold:
                below_threshold += 1
            else:
                predictions.append(Box(image_id, mapped, xyxy, confidence))
    return predictions, excluded, below_threshold


def _best_overlap(box: Box, candidates: Iterable[Box], same_class: bool | None = None) -> float:
    values = [
        iou(box.xyxy, candidate.xyxy) for candidate in candidates
        if candidate.image_id == box.image_id and (same_class is None or (candidate.cls == box.cls) == same_class)
    ]
    return max(values, default=0.0)


def evaluate(predictions: list[Box], ground_truth: list[Box], iou_threshold: float = 0.5) -> tuple[dict, list[dict], Counter, dict[str, list[int]]]:
    """Return summary, class rows, error labels, and indices for annotation."""
    matches, matched_predictions, matched_ground_truth = greedy_match(predictions, ground_truth, iou_threshold)
    match_by_prediction = {index: (target, overlap) for index, target, overlap in matches}
    errors = Counter()
    annotations: dict[str, list[int]] = {"success": [], "failure": []}
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_predictions:
            annotations["success"].append(prediction_index)
            continue
        other_overlap = _best_overlap(prediction, ground_truth, same_class=False)
        same_overlap = _best_overlap(prediction, ground_truth, same_class=True)
        errors["false_positive_misclassification" if other_overlap >= iou_threshold else "false_positive_localization" if same_overlap > 0 else "false_positive_no_overlap"] += 1
        annotations["failure"].append(prediction_index)
    for target_index, target in enumerate(ground_truth):
        if target_index in matched_ground_truth:
            continue
        other_overlap = _best_overlap(target, predictions, same_class=False)
        same_overlap = _best_overlap(target, predictions, same_class=True)
        errors["false_negative_misclassification" if other_overlap >= iou_threshold else "false_negative_localization" if same_overlap > 0 else "false_negative_missed"] += 1
    class_rows: list[dict] = []
    for cls in EVAL_CLASSES:
        pred_indices = [index for index, box in enumerate(predictions) if box.cls == cls]
        gt_indices = [index for index, box in enumerate(ground_truth) if box.cls == cls]
        true_positive = sum(1 for index in pred_indices if index in matched_predictions)
        false_positive = len(pred_indices) - true_positive
        false_negative = len(gt_indices) - sum(1 for index in gt_indices if index in matched_ground_truth)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        class_rows.append({"class": cls, "support": len(gt_indices), "predictions": len(pred_indices), "tp": true_positive, "fp": false_positive, "fn": false_negative, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "ap50": average_precision_50([predictions[i] for i in pred_indices], [ground_truth[i] for i in gt_indices], iou_threshold)})
    total_tp = len(matches)
    total_fp = len(predictions) - total_tp
    total_fn = len(ground_truth) - total_tp
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    valid_ap = [row["ap50"] for row in class_rows if row["ap50"] is not None]
    summary = {"support": len(ground_truth), "predictions": len(predictions), "tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "map50": sum(valid_ap) / len(valid_ap) if valid_ap else None}
    return summary, class_rows, errors, annotations


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _annotate(image_path: Path, output: Path, predictions: list[Box], ground_truth: list[Box], title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for box in ground_truth:
            draw.rectangle(box.xyxy, outline="#42d392", width=3)
            draw.text((box.xyxy[0], max(0, box.xyxy[1] - 12)), f"GT {box.cls}", fill="#42d392", font=font, stroke_width=1, stroke_fill="black")
        for box in predictions:
            draw.rectangle(box.xyxy, outline="#ffb020", width=3)
            draw.text((box.xyxy[0], min(image.height - 12, box.xyxy[3])), f"P {box.cls} {box.confidence:.2f}", fill="#ffb020", font=font, stroke_width=1, stroke_fill="black")
        draw.rectangle((0, 0, image.width, 24), fill="black")
        draw.text((5, 5), title, fill="white", font=font)
        image.thumbnail((1000, 700))
        image.save(output, quality=84, optimize=True)


def _select_images(predictions: list[Box], ground_truth: list[Box], annotations: dict[str, list[int]]) -> list[tuple[str, str]]:
    # Three highest-confidence TPs and two meaningful edge cases.  A repeated
    # source image is avoided to give reviewers five distinct examples.
    successful = sorted(annotations["success"], key=lambda index: predictions[index].confidence, reverse=True)
    failures = sorted(annotations["failure"], key=lambda index: predictions[index].confidence, reverse=True)
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index in successful:
        image_id = predictions[index].image_id
        if image_id not in seen:
            selected.append((image_id, "success"))
            seen.add(image_id)
        if len(selected) == 3:
            break
    for index in failures:
        image_id = predictions[index].image_id
        if image_id not in seen:
            selected.append((image_id, "failure"))
            seen.add(image_id)
        if len(selected) == 5:
            break
    # If there are fewer than two FP images, demonstrate a missed-GT edge case.
    for target in ground_truth:
        if len(selected) == 5:
            break
        if target.image_id not in seen:
            selected.append((target.image_id, "failure"))
            seen.add(target.image_id)
    return selected


def build_bundle(dataset: Path, prediction_csv: Path, output: Path, confidence_threshold: float = 0.25, iou_threshold: float = 0.5) -> dict:
    """Evaluate recorded predictions and write a compact, reviewable bundle."""
    ground_truth, image_paths, excluded_gt = load_ground_truth(dataset)
    image_ids = sorted(image_paths)
    predictions, excluded_predictions, below_threshold = load_predictions(prediction_csv, image_ids, confidence_threshold)
    summary, class_rows, errors, annotations = evaluate(predictions, ground_truth, iou_threshold)
    if output.exists():
        shutil.rmtree(output)
    (output / "annotated").mkdir(parents=True)
    selected = _select_images(predictions, ground_truth, annotations)
    asset_rows = []
    for position, (image_id, kind) in enumerate(selected, 1):
        filename = f"{position:02d}_{kind}_{image_id}.jpg"
        _annotate(image_paths[image_id], output / "annotated" / filename, [box for box in predictions if box.image_id == image_id], [box for box in ground_truth if box.image_id == image_id], f"{kind.upper()} | green GT | amber prediction")
        asset_rows.append({"file": f"annotated/{filename}", "source_image_id": image_id, "category": kind, "description": "Green boxes are mapped VisDrone ground truth; amber boxes are retained mapped YOLO predictions."})
    provenance = json.loads((dataset / "provenance.json").read_text(encoding="utf-8"))
    result = {"schema_version": "1.0", "evaluation": {"dataset": "VisDrone2019-DET validation smoke subset", "images": len(image_paths), "prediction_artifact": prediction_csv.name, "model": "YOLO11n COCO pretrained (direct inference)", "confidence_threshold": confidence_threshold, "iou_threshold": iou_threshold, "matching": "confidence-ordered greedy one-to-one matching within image and mapped class", "frame_mapping": "CSV clip_id must be 'images'; frame_idx is paired to the lexical sort of val/images, matching run_yolo.discover_frames for the flat images directory.", "recorded_inference_seconds": 11.057, "timing_provenance": "Recorded smoke-run wall time supplied with the existing 50-image artifact; this evaluator does not re-run inference."}, "mapping": {"visdrone_to_evaluation": VISDRONE_TO_EVAL, "coco_to_evaluation": COCO_TO_EVAL, "excluded_visdrone_classes": list(UNSUPPORTED_VISDRONE), "excluded_ground_truth_counts": dict(sorted(excluded_gt.items())), "excluded_prediction_counts": dict(sorted(excluded_predictions.items())), "predictions_below_confidence_threshold": below_threshold}, "micro": summary, "per_class": class_rows, "error_categories": dict(sorted(errors.items())), "assets": asset_rows, "source_provenance": {key: provenance[key] for key in ("source", "source_url", "splits", "images_copied", "limit_per_split", "classes", "class_counts", "annotation_format", "ignored_categories", "download") if key in provenance}, "limitations": ["This is a 50-image smoke subset, not the official VisDrone evaluation protocol or a held-out Equinox validation set.", "The COCO-pretrained YOLO11n model was not fine-tuned on VisDrone; domain shift, small objects, occlusion, and class definitions materially limit performance.", "Only the conservative compatible taxonomy intersection is evaluated. Unsupported VisDrone classes and unsupported COCO predictions are excluded, not treated as correct or incorrect.", "AP@0.5 is implemented as all-points interpolated AP over the recorded predictions. It is not COCO mAP@[.5:.95].", "No severity, road blockage, lane-clearance, emergency-accessibility, tracking, or temporal validation is established by this detector smoke run."]}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "per_class_metrics.csv", class_rows, ["class", "support", "predictions", "tp", "fp", "fn", "precision", "recall", "f1", "ap50"])
    _write_csv(output / "error_summary.csv", [{"category": category, "count": count} for category, count in sorted(errors.items())], ["category", "count"])
    (output / "evidence_manifest.json").write_text(json.dumps({"assets": asset_rows, "license_note": "Annotated assets are derivative, downscaled visual evidence from the VisDrone2019-DET validation subset. They are included only for review; see source_provenance in metrics.json and the upstream dataset repository for dataset terms."}, indent=2) + "\n", encoding="utf-8")
    report = _render_report(result)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_report(result: dict) -> str:
    micro = result["micro"]
    rows = "\n".join(f"| {row['class']} | {row['support']} | {row['predictions']} | {row['tp']} | {row['fp']} | {row['fn']} | {_format(row['precision'])} | {_format(row['recall'])} | {_format(row['f1'])} | {_format(row['ap50'])} |" for row in result["per_class"])
    error_rows = "\n".join(f"| {category} | {count} |" for category, count in result["error_categories"].items())
    assets = "\n".join(f"- [{asset['category']}: {asset['source_image_id']}]({asset['file']})" for asset in result["assets"])
    limitations = "\n".join(f"- {item}" for item in result["limitations"])
    return f"""# VisDrone smoke-run evidence: direct YOLO11n

## Scope

This package evaluates the recorded direct-inference CSV against the prepared 50-image VisDrone2019-DET validation smoke subset. It evaluates object detection only. Recorded inference time is **11.057 s / 50 images** (0.221 s/image); it is provenance from the existing run, not a timing claim produced by this evaluator.

At confidence >= {result['evaluation']['confidence_threshold']:.2f} and IoU >= {result['evaluation']['iou_threshold']:.2f}, micro precision is **{_format(micro['precision'])}**, recall **{_format(micro['recall'])}**, F1 **{_format(micro['f1'])}**, and mAP@0.5 across supported classes **{_format(micro['map50'])}**. Counts: {micro['tp']} TP, {micro['fp']} FP, {micro['fn']} FN, {micro['support']} supported ground-truth objects, {micro['predictions']} supported predictions.

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
{rows}

AP@0.5 uses confidence-ranked, one-to-one greedy matching and all-points interpolated precision. It is not COCO mAP@[.5:.95].

## Error categories

| Category | Count |
| --- | ---: |
{error_rows}

## Annotated evidence

Green boxes are compatible mapped VisDrone ground truth. Amber boxes are retained compatible YOLO predictions. These are actual, downscaled inputs, selected as three high-confidence true-positive examples and two failure or edge examples.

{assets}

## Reproduce

```bash
cd equinox_deliverable_scaffold/equinox
python -m pipeline.evaluation.visdrone_smoke \\
  --dataset data/visdrone_smoke \\
  --predictions artifacts/detections/visdrone_smoke_yolo11n_direct.csv \\
  --output evidence/visdrone_smoke_yolo11n
```

No raw dataset, archive, model weights, or full prediction CSV is copied into this evidence bundle. Dataset source and archive provenance are preserved in `metrics.json`. Review dataset terms at [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset).

## Limitations

{limitations}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = build_bundle(args.dataset, args.predictions, args.output, args.confidence, args.iou)
    print(json.dumps(result["micro"], sort_keys=True))


if __name__ == "__main__":
    main()
