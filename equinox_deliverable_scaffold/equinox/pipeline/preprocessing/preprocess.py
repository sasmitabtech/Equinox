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
from pathlib import Path

import cv2


def extract_frames(clip_path: str, out_dir: Path, target_fps: float):
    cap = cv2.VideoCapture(clip_path)
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
            cv2.imwrite(str(out_dir / f"{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def check_mask(location: str, masks_dir: Path) -> bool:
    """A mask is required per unique vantage point before frames are usable
    for feature engineering (see plan §1.2, §1.3)."""
    return (masks_dir / f"{location}.png").exists()


def preprocess(manifest_csv: str, frames_out: str, masks_dir: str, target_fps: float):
    frames_out = Path(frames_out)
    masks_dir = Path(masks_dir)
    missing_masks = set()

    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            clip_id, location = row["clip_id"], row["location"]
            if not check_mask(location, masks_dir):
                missing_masks.add(location)
                continue  # don't process frames for a location with no mask yet
            n = extract_frames(row["filepath"], frames_out / clip_id, target_fps)
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
