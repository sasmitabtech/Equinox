"""
Stage 2: Preprocessing
Input : manifest.csv (from ingestion)
Output: data/frames/{clip_id}/{frame_idx:05d}.jpg sampled at --fps,
        plus a per-clip check that a road/lane mask exists for its location.

Success criteria (see plan §1.2): sampling rate within 5% of target,
every unique location has a verified mask before frames are used downstream.
"""
import argparse
import csv
import os
from pathlib import Path

import cv2


def extract_frames(clip_path: str, out_dir: Path, target_fps: float):
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video clip: {clip_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    stride = max(int(round(src_fps / target_fps)), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            # TODO: apply stabilization here if drone footage has camera jitter
            output_path = out_dir / f"{saved:05d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                cap.release()
                raise IOError(f"Could not write extracted frame: {output_path}")
            saved += 1
        idx += 1
    cap.release()
    return saved


def check_mask(location: str, masks_dir: Path) -> bool:
    """A mask is required per unique vantage point before frames are usable
    for feature engineering (see plan §1.2, §1.3)."""
    if not location or Path(location).name != location:
        return False
    candidate = masks_dir / f"{location}.png"
    try:
        candidate.resolve().relative_to(masks_dir.resolve())
    except ValueError:
        return False
    return candidate.is_file()


def resolve_manifest_path(filepath: str, manifest_path: Path) -> Path:
    """Resolve both POSIX and Windows paths in a portable manifest.

    New manifests contain paths relative to the manifest directory. Existing
    manifests in the wild may contain backslashes or paths relative to the
    repository working directory, so try both locations before failing.
    """
    if not filepath or not filepath.strip():
        raise ValueError("Manifest row has an empty filepath")
    normalised = filepath.replace("\\", os.sep)
    candidate = Path(normalised)
    if candidate.is_absolute():
        return candidate
    candidates = [manifest_path.parent / candidate, Path.cwd() / candidate]
    for path in candidates:
        if path.is_file():
            return path
    # Return the contextual path so the error from OpenCV is actionable.
    return candidates[0]


def preprocess(manifest_csv: str, frames_out: str, masks_dir: str, target_fps: float):
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")
    manifest_path = Path(manifest_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    frames_out = Path(frames_out)
    masks_dir = Path(masks_dir)
    missing_masks = set()

    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip_id = (row.get("clip_id") or "").strip()
            location = (row.get("location") or "").strip()
            if not clip_id:
                raise ValueError("Manifest contains a row without clip_id")
            if Path(clip_id).name != clip_id:
                raise ValueError(f"Unsafe clip_id in manifest: {clip_id!r}")
            if not check_mask(location, masks_dir):
                missing_masks.add(location)
                continue  # don't process frames for a location with no mask yet
            clip_path = resolve_manifest_path(row.get("filepath", ""), manifest_path)
            if not clip_path.is_file():
                raise FileNotFoundError(
                    f"Clip for {clip_id!r} not found at {clip_path} "
                    f"(manifest: {manifest_path})"
                )
            n = extract_frames(str(clip_path), frames_out / clip_id, target_fps)
            print(f"{clip_id}: {n} frames extracted -> {frames_out / clip_id}")

    if missing_masks:
        print(f"\nSkipped clips for {len(missing_masks)} location(s) with no mask "
              f"yet: {sorted(missing_masks)}. Draw masks in {masks_dir} and re-run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, dest="frames_out")
    parser.add_argument("--masks_dir", default="data/masks")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    preprocess(args.manifest, args.frames_out, args.masks_dir, args.fps)
