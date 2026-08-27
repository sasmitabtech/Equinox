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
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video clip: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    cap.release()
    return fps, duration


def load_sidecar(clip_path: Path) -> dict:
    sidecar = clip_path.with_suffix(".json")
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as f:
            return json.load(f)
    # No metadata sidecar — flag it rather than silently guessing.
    return {"location": "UNKNOWN", "lat": None, "lon": None,
            "timestamp": None, "source": "unspecified"}


def ingest(raw_dir: str, out_csv: str):
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw media directory does not exist: {raw_dir}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            existing_ids = {
                row["clip_id"] for row in csv.DictReader(f)
                if row.get("clip_id")
            }

    rows = []
    # suffix.lower() also picks up clips copied from Windows/macOS systems.
    clip_paths = sorted(
        (path for path in raw_dir.iterdir()
         if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}),
        key=lambda path: path.name.lower(),
    )
    clip_ids: dict[str, list[Path]] = {}
    for clip_path in clip_paths:
        clip_ids.setdefault(clip_path.stem, []).append(clip_path)
    duplicate_ids = {
        clip_id: paths for clip_id, paths in clip_ids.items() if len(paths) > 1
    }
    if duplicate_ids:
        details = "; ".join(
            f"{clip_id!r}: {', '.join(path.name for path in paths)}"
            for clip_id, paths in sorted(duplicate_ids.items())
        )
        raise ValueError(
            "Multiple media files map to the same clip_id; rename or remove "
            f"one before ingesting ({details})"
        )
    for clip_path in clip_paths:
        clip_id = clip_path.stem
        if clip_id in existing_ids:
            continue  # idempotent: skip already-registered clips
        fps, duration = probe_clip(str(clip_path))
        meta = load_sidecar(clip_path)
        # Store a portable path relative to the manifest, rather than an
        # absolute path or a platform-specific backslash path.
        filepath = Path(
            os.path.relpath(clip_path, out_path.parent.resolve())
        ).as_posix()
        rows.append({
            "clip_id": clip_id,
            "filepath": filepath,
            "location": meta.get("location"),
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "timestamp": meta.get("timestamp"),
            "fps": round(fps, 2),
            "duration_s": round(duration, 2),
            "source": meta.get("source"),
        })

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
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
