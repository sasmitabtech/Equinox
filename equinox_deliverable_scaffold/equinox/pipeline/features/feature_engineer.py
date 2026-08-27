"""
Stage 3: Feature Engineering
Input : per-frame detections (from the YOLO detector + tracker, trained separately
        via Ultralytics CLI — see plan §4) as a list of dicts:
        {clip_id, frame_idx, track_id, cls, bbox: [x1,y1,x2,y2], conf}
        + the road/lane mask for the clip's location.
Output: features/{clip_id}.parquet — one row per incident-window with the
        features the severity model (Stage 4) actually consumes.

This module intentionally does NOT run the detector itself — it consumes
already-produced detections so it can be developed/tested independently
(e.g. against hand-labeled or mocked detections) while the detector is
still being trained.
"""
import argparse
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


def occupied_fraction(boxes: list, mask_area: float) -> float:
    """Fraction of the usable road/lane area covered by obstruction boxes.
    TODO: replace the naive area-sum with proper polygon overlap against
    the corridor mask (shapely) to avoid double-counting overlapping boxes."""
    if mask_area <= 0:
        return 0.0
    total = sum(
        max(0.0, x2 - x1) * max(0.0, y2 - y1)
        for x1, y1, x2, y2 in boxes
    )
    return min(total / mask_area, 1.0)


def clear_lane_width(boxes: list, corridor_width_px: float) -> float:
    """Rough proxy: corridor width minus the widest single obstruction's
    horizontal span. TODO: replace with a proper free-space scan across
    the corridor cross-section using the mask geometry."""
    if not boxes:
        return corridor_width_px
    widest = max(max(0.0, x2 - x1) for x1, y1, x2, y2 in boxes)
    return max(corridor_width_px - widest, 0.0)


def dwell_time(track_frames: dict, fps: float) -> dict:
    """frames a given track_id persists / fps = seconds present.
    A short dwell (e.g. a car briefly stopped at a signal) should NOT
    trigger an incident — see plan §1.6 false-positive handling."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return {tid: len(frames) / fps for tid, frames in track_frames.items()}


def build_incident_window(detections: pd.DataFrame, corridor_width_px: float,
                           mask_area: float, fps: float) -> dict:
    """Roll a burst of frame-level detections into one incident-level
    feature row. This is the unit the severity model (Stage 4) is trained on."""
    if detections.empty:
        raise ValueError("Cannot build an incident window from empty detections")
    required = {"clip_id", "frame_idx", "track_id", "cls", "bbox"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Detections missing required column(s): {sorted(missing)}")
    if corridor_width_px < 0 or mask_area < 0:
        raise ValueError("corridor_width_px and mask_area must be non-negative")
    boxes = detections["bbox"].tolist()
    track_frames = detections.groupby("track_id")["frame_idx"].apply(list).to_dict()
    dwell = dwell_time(track_frames, fps)

    return {
        "clip_id": detections["clip_id"].iloc[0],
        "start_frame": detections["frame_idx"].min(),
        "end_frame": detections["frame_idx"].max(),
        "n_obstructions": detections["track_id"].nunique(),
        "occupied_fraction": occupied_fraction(boxes, mask_area),
        "clear_lane_width_px": clear_lane_width(boxes, corridor_width_px),
        "max_dwell_s": max(dwell.values()) if dwell else 0.0,
        "obstruction_classes": ",".join(sorted(detections["cls"].unique())),
        # TODO: add queue_length (stationary-vehicle count in a defined zone)
        # TODO: add is_known_corridor flag from the location/mask metadata
    }


def run(detections_path: str, out_dir: str):
    # TODO: replace with real detection loading (json/csv from the tracker)
    detections = pd.read_csv(detections_path)
    required = {"clip_id", "frame_idx", "track_id", "cls", "bbox"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Detections missing required column(s): {sorted(missing)}")
    if detections.empty:
        raise ValueError("Detections input is empty; no feature rows can be built")

    def parse_bbox(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            parsed = list(value)
        elif isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid bbox literal: {value!r}") from exc
        else:
            raise ValueError(f"Unsupported bbox value: {value!r}")
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
            raise ValueError(f"Bounding box must contain four coordinates: {value!r}")
        try:
            coords = tuple(float(coord) for coord in parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bounding box coordinates must be numeric: {value!r}") from exc
        if not all(math.isfinite(coord) for coord in coords):
            raise ValueError(f"Bounding box coordinates must be finite: {value!r}")
        return coords

    detections["bbox"] = detections["bbox"].apply(parse_bbox)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip_id, group in detections.groupby("clip_id"):
        # NOTE: corridor_width_px / mask_area should come from per-location
        # calibration data, not hardcoded — placeholder values below.
        row = build_incident_window(group, corridor_width_px=400.0,
                                     mask_area=120_000.0, fps=3.0)
        rows.append(row)

    if not rows:
        raise ValueError("No incident windows could be built from detections")
    out_df = pd.DataFrame(rows)
    out_path = out_dir / "incident_windows.parquet"
    out_df.to_parquet(out_path)
    print(f"Wrote {len(out_df)} incident-window feature rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.detections, args.out)
