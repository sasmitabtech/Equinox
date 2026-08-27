"""Run a trained Ultralytics YOLO detector/tracker over extracted frames.

The CSV written by this module is the contract consumed by
``pipeline.features.feature_engineer``: ``clip_id, frame_idx, track_id, cls,
bbox, conf``.  ``bbox`` is JSON (never a Python expression) in xyxy pixel
coordinates.  By default ByteTrack is used through ``YOLO.track`` so dwell
time can be computed downstream.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CSV_COLUMNS = ("clip_id", "frame_idx", "track_id", "cls", "bbox", "conf")


def _numeric_frame_index(path: Path) -> int | None:
    """Return a preprocessing frame index, or ``None`` for arbitrary images."""
    try:
        return int(path.stem)
    except ValueError:
        return None


def _frame_sort_key(path: Path) -> tuple[int, int | str, str]:
    """Naturally order extracted numeric frames, then arbitrary image names."""
    numeric_index = _numeric_frame_index(path)
    if numeric_index is not None:
        return (0, numeric_index, path.name.casefold())
    return (1, path.name.casefold(), path.name)


def frame_indexes(frame_paths: list[Path]) -> list[int]:
    """Map an ordered clip's paths to deterministic contract frame indices.

    Preprocessing emits numeric stems (``00042.jpg``), whose value is retained.
    Public image datasets such as VisDrone have descriptive stems instead, so
    they receive their zero-based position in the deterministic sorted order.
    """
    numeric_indexes = [_numeric_frame_index(path) for path in frame_paths]
    if all(index is not None for index in numeric_indexes):
        return [int(index) for index in numeric_indexes]
    return list(range(len(frame_paths)))


def discover_frames(source: str | Path) -> dict[str, list[Path]]:
    """Find images and return deterministic, clip-grouped frame paths."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Frame source does not exist: {source_path}")

    if source_path.is_file():
        candidates = [source_path]
    else:
        candidates = [
            path for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    if not candidates:
        raise ValueError(
            f"No supported image frames found under {source_path}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )

    grouped: dict[str, list[Path]] = {}
    for path in candidates:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported frame image extension: {path}")
        grouped.setdefault(path.parent.name, []).append(path)

    return {
        clip_id: sorted(paths, key=_frame_sort_key)
        for clip_id, paths in sorted(grouped.items(), key=lambda item: item[0].casefold())
    }


def _tolist(value: Any) -> list[Any]:
    """Convert Torch/NumPy-like result values without importing either library."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _finite_bbox(raw_box: Any, frame_path: Path) -> list[float]:
    values = _tolist(raw_box)
    if len(values) != 4:
        raise ValueError(f"YOLO returned an invalid bbox for {frame_path}: {values!r}")
    try:
        box = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"YOLO returned a non-numeric bbox for {frame_path}: {values!r}") from error
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"YOLO returned a non-finite bbox for {frame_path}: {values!r}")
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _result_detections(result: Any, frame_path: Path) -> list[tuple[str, list[float], float, str | None]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tolist(getattr(boxes, "xyxy", []))
    classes = _tolist(getattr(boxes, "cls", []))
    confidences = _tolist(getattr(boxes, "conf", []))
    raw_track_ids = getattr(boxes, "id", None)
    track_ids = _tolist(raw_track_ids) if raw_track_ids is not None else [None] * len(xyxy)
    if not (len(xyxy) == len(classes) == len(confidences) == len(track_ids)):
        raise ValueError(f"YOLO returned inconsistent detection arrays for {frame_path}")

    names = getattr(result, "names", {})
    detections = []
    for box, cls, confidence, track_id in zip(xyxy, classes, confidences, track_ids):
        try:
            class_id, score = int(cls), float(confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(f"YOLO returned invalid class/confidence for {frame_path}") from error
        if not math.isfinite(score):
            raise ValueError(f"YOLO returned a non-finite confidence for {frame_path}")
        track_value = None if track_id is None else str(int(track_id))
        detections.append((_class_name(names, class_id), _finite_bbox(box, frame_path), score, track_value))
    # Ultralytics does not promise an ordering of boxes; make CSV output stable.
    return sorted(detections, key=lambda row: (row[0], *row[1], -row[2], row[3] or ""))


def _load_yolo(weights: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is required for detector inference. Run `pip install -r requirements.txt`."
        ) from error
    try:
        return YOLO(weights)
    except Exception as error:
        raise RuntimeError(f"Could not load YOLO weights '{weights}': {error}") from error


def run(
    source: str | Path,
    out_path: str | Path,
    weights: str = "models/detector.pt",
    conf: float = 0.25,
    device: str | None = None,
    track: bool = True,
    model_factory: Callable[[str], Any] | None = None,
) -> Path:
    """Infer detections over frames and write a deterministic contract CSV.

    ``model_factory`` is an injection seam for tests and offline integrations;
    production callers should leave it as ``None``.
    """
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be between 0 and 1, got {conf}")
    grouped_frames = discover_frames(source)
    model = (model_factory or _load_yolo)(weights)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    invoke_kwargs: dict[str, Any] = {"conf": conf, "verbose": False}
    if device is not None:
        invoke_kwargs["device"] = device

    for clip_id, frame_paths in grouped_frames.items():
        try:
            # A call per clip deliberately resets ByteTrack between videos.
            infer = model.track if track else model.predict
            request_kwargs = {"source": [str(path) for path in frame_paths], **invoke_kwargs}
            if track:
                request_kwargs["persist"] = False
            results = list(infer(**request_kwargs))
        except Exception as error:
            action = "tracking" if track else "detection"
            raise RuntimeError(f"YOLO {action} failed for clip '{clip_id}': {error}") from error
        if len(results) != len(frame_paths):
            raise RuntimeError(
                f"YOLO returned {len(results)} result(s) for {len(frame_paths)} frame(s) in clip '{clip_id}'"
            )
        for frame_path, frame_idx, result in zip(frame_paths, frame_indexes(frame_paths), results):
            for rank, (cls, bbox, score, track_id) in enumerate(_result_detections(result, frame_path)):
                rows.append({
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    # Detection-only runs still honour the contract, but those IDs
                    # are frame-local and cannot be used for cross-frame dwell time.
                    "track_id": track_id or f"untracked-{frame_idx}-{rank}",
                    "cls": cls,
                    "bbox": json.dumps(bbox, separators=(",", ":"), allow_nan=False),
                    "conf": f"{score:.6f}",
                })

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} detections from {sum(map(len, grouped_frames.values()))} frame(s) -> {output}")
    return output


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO over extracted image frames.")
    parser.add_argument("--source", required=True, help="A frame file or directory of clip/frame images")
    parser.add_argument("--out", default="data/detections/detections.csv", dest="out_path", help="CSV path to write")
    parser.add_argument("--model", default="models/detector.pt", dest="weights", help="YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence from 0 to 1")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, mps, or 0")
    parser.add_argument("--no-track", action="store_false", dest="track", help="Use detection only (frame-local IDs)")
    args = parser.parse_args(argv)
    run(**vars(args))


if __name__ == "__main__":
    main()
