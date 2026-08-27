"""Prepare a small, reproducible VisDrone2019-DET dataset for YOLO.

This module never downloads data implicitly. Point ``--source`` at an archive
that has already been downloaded and extracted (or at the extracted dataset
root), or explicitly opt in to the documented validation download with
``--download-val``. To convert a local source, use this command::

    python -m pipeline.ingestion.prepare_visdrone --source /data/VisDrone \
        --output data/visdrone --limit 100

The official layout is ``VisDrone2019-DET-{train,val}/images`` and
``.../annotations``.  A split may also be passed directly when its images and
annotations directories are children of ``--source``.  Annotation rows are
``x,y,w,h,score,category,truncation,occlusion``; ignored/invalid boxes are
discarded and coordinates are clipped to image bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import struct
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]
SOURCE_NAME = "VisDrone2019-DET"
SOURCE_URL = "https://github.com/VisDrone/VisDrone-Dataset"
ULTRALYTICS_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _image_size(path: Path) -> tuple[int, int]:
    """Read width/height without requiring Pillow or OpenCV."""
    with path.open("rb") as fh:
        head = fh.read(32)
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            return struct.unpack(">II", head[16:24])
        if head.startswith(b"BM") and len(head) >= 26:
            return struct.unpack("<II", head[18:26])
        if head.startswith(b"\xff\xd8"):
            fh.seek(2)
            while True:
                marker = fh.read(1)
                if not marker:
                    break
                if marker != b"\xff":
                    continue
                while marker == b"\xff":
                    marker = fh.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                raw_len = fh.read(2)
                if len(raw_len) != 2:
                    break
                length = struct.unpack(">H", raw_len)[0]
                if marker[0] in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                    data = fh.read(5)
                    if len(data) == 5:
                        return struct.unpack(">HH", data[1:5])[::-1]
                fh.seek(max(0, length - 2), 1)
    raise ValueError(f"Cannot determine image dimensions: {path}")


def _find_split(source: Path, split: str) -> tuple[Path, Path]:
    candidates = [source / f"VisDrone2019-DET-{split}", source / split, source]
    for root in candidates:
        images, annotations = root / "images", root / "annotations"
        if images.is_dir() and annotations.is_dir():
            return images, annotations
    raise FileNotFoundError(f"Could not find images/annotations for split {split!r} below {source}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a zip while rejecting absolute paths and ``..`` traversal."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename!r}")
        handle.extractall(destination)


def download_validation_archive(cache_dir: str | Path, url: str = ULTRALYTICS_VAL_URL, expected_sha256: str | None = None) -> dict:
    """Download/cache and safely extract the validation archive (explicit opt-in)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "VisDrone2019-DET-val.zip"
    if not archive.exists():
        try:
            # macOS Python installations sometimes lack the system CA path;
            # use certifi when present, while retaining certificate validation.
            context = ssl.create_default_context()
            try:
                import certifi  # type: ignore
                context.load_verify_locations(certifi.where())
            except (ImportError, OSError):
                # Standard macOS/Homebrew CA bundle fallback.
                try:
                    context.load_verify_locations("/etc/ssl/cert.pem")
                except OSError:
                    pass
            with urllib.request.urlopen(url, timeout=60, context=context) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (OSError, urllib.error.URLError) as exc:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"Unable to download VisDrone validation archive from {url}: {exc}") from exc
    digest = _sha256(archive)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {archive}: expected {expected_sha256}, got {digest}")
    extracted = cache_dir / "extracted"
    marker = extracted / ".extracted-sha256"
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != digest:
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True)
        try:
            _safe_extract(archive, extracted)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            shutil.rmtree(extracted, ignore_errors=True)
            raise RuntimeError(f"Unable to safely extract {archive}: {exc}") from exc
        marker.write_text(digest + "\n", encoding="utf-8")
    return {"url": url, "archive": str(archive.resolve()), "sha256": digest, "extracted": str(extracted.resolve())}


def _parse_annotation(path: Path, width: int, height: int) -> tuple[list[str], Counter]:
    labels: list[str] = []
    counts: Counter = Counter()
    if not path.exists():
        return labels, counts
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        try:
            x, y, box_w, box_h = (float(value) for value in fields[:4])
            score, category = float(fields[4]), int(fields[5])
        except ValueError:
            continue
        # category 0 is ignored; 11 is "others" and is not a YOLO class here.
        if not 1 <= category <= len(CLASS_NAMES) or score <= 0 or box_w <= 0 or box_h <= 0:
            continue
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + box_w), min(float(height), y + box_h)
        if x2 <= x1 or y2 <= y1:
            continue
        labels.append(f"{category - 1} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        counts[CLASS_NAMES[category - 1]] += 1
    return labels, counts


def prepare(source: str | Path, output: str | Path, limit: int | None = None, splits: tuple[str, ...] = ("train", "val")) -> dict:
    """Convert VisDrone splits and return the provenance dictionary."""
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    output.mkdir(parents=True, exist_ok=True)
    class_counts: Counter = Counter()
    split_counts: dict[str, int] = {}
    copied = 0
    for split in splits:
        images_dir, annotations_dir = _find_split(source, split)
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if limit is not None:
            image_paths = image_paths[:limit]
        split_root = output / split
        (split_root / "images").mkdir(parents=True, exist_ok=True)
        (split_root / "labels").mkdir(parents=True, exist_ok=True)
        for image in image_paths:
            width, height = _image_size(image)
            labels, counts = _parse_annotation(annotations_dir / f"{image.stem}.txt", width, height)
            shutil.copy2(image, split_root / "images" / image.name)
            (split_root / "labels" / f"{image.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            class_counts.update(counts)
            copied += 1
        split_counts[split] = len(image_paths)
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: train/images\nval: val/images\nnc: %d\nnames: %s\n" % (len(CLASS_NAMES), json.dumps(CLASS_NAMES)), encoding="utf-8")
    provenance = {"source": SOURCE_NAME, "source_url": SOURCE_URL, "source_root": str(source.resolve()), "splits": split_counts, "images_copied": copied, "limit_per_split": limit, "classes": CLASS_NAMES, "class_counts": dict(sorted(class_counts.items())), "annotation_format": "x,y,w,h,score,category,truncation,occlusion", "ignored_categories": [0, 11]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Already-extracted VisDrone root")
    parser.add_argument("--output", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--limit", type=int, help="Maximum images per split (opt-in smoke subset)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--download-val", action="store_true", help="Explicitly download/cache the Ultralytics validation archive")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/visdrone_cache"))
    parser.add_argument("--sha256", help="Optional expected SHA-256 for the archive")
    args = parser.parse_args()
    if args.download_val:
        download = download_validation_archive(args.cache_dir, expected_sha256=args.sha256)
        result = prepare(download["extracted"], args.output, args.limit, ("val",))
        result["download"] = download
        (args.output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        if args.source is None:
            parser.error("--source is required unless --download-val is supplied")
        result = prepare(args.source, args.output, args.limit, tuple(args.splits))
    print(f"Prepared {result['images_copied']} image(s) in {args.output}")


if __name__ == "__main__":
    main()
