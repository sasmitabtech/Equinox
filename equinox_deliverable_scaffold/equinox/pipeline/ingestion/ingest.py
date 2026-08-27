"""
Stage 1: Ingestion
Input : raw drone clips in --raw_dir + a sidecar .json per clip with
        {location, lat, lon, timestamp, source}
Output: a manifest.csv registering every clip (idempotent — safe to re-run)

Contract (do not change without updating downstream stages):
    clip_id, filepath, location, lat, lon, timestamp, fps, duration_s, source
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2


def probe_clip(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    cap.release()
    return fps, duration


def load_sidecar(clip_path: Path) -> dict:
    sidecar = clip_path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar) as f:
            return json.load(f)
    # No metadata sidecar — flag it rather than silently guessing.
    return {"location": "UNKNOWN", "lat": None, "lon": None,
            "timestamp": None, "source": "unspecified"}


def ingest(raw_dir: str, out_csv: str):
    raw_dir = Path(raw_dir)
    existing_ids = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            existing_ids = {row["clip_id"] for row in csv.DictReader(f)}

    rows = []
    for clip_path in sorted(raw_dir.glob("*.mp4")) + sorted(raw_dir.glob("*.mov")):
        clip_id = clip_path.stem
        if clip_id in existing_ids:
            continue  # idempotent: skip already-registered clips
        fps, duration = probe_clip(str(clip_path))
        meta = load_sidecar(clip_path)
        rows.append({
            "clip_id": clip_id,
            "filepath": str(clip_path),
            "location": meta.get("location"),
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "timestamp": meta.get("timestamp"),
            "fps": round(fps, 2),
            "duration_s": round(duration, 2),
            "source": meta.get("source"),
        })

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "filepath", "location", "lat", "lon",
            "timestamp", "fps", "duration_s", "source"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Ingested {len(rows)} new clip(s); {len(existing_ids)} already registered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ingest(args.raw_dir, args.out)
