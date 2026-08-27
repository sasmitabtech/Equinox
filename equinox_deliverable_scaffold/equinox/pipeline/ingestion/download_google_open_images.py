"""Download a small, reproducible Google Open Images subset for vehicle pretraining.

The full Open Images release is very large. This utility downloads only the first
N images for selected classes from Google's official metadata and records provenance.
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
ANNOTATIONS = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"


def download_subset(output: Path, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    descriptions = output / "class-descriptions.csv"
    annotations = output / "train-annotations-bbox.csv"
    urllib.request.urlretrieve(CLASS_DESCRIPTIONS, descriptions)
    urllib.request.urlretrieve(ANNOTATIONS, annotations)

    wanted = {"Car", "Bus", "Truck", "Motorcycle"}
    class_ids = {}
    with descriptions.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) == 2 and row[1] in wanted:
                class_ids[row[0]] = row[1]

    selected = []
    seen = set()
    with annotations.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = class_ids.get(row["ClassName"])
            image_id = row["ImageID"]
            if label and image_id not in seen and len(selected) < limit:
                selected.append({"image_id": image_id, "class": label, "source": "Google Open Images V7"})
                seen.add(image_id)

    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"source": "Google Research Open Images V7", "license": "CC BY 2.0", "images": selected}, handle, indent=2)
    print(f"Recorded {len(selected)} Google Open Images samples in {output / 'manifest.json'}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/google_open_images"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    download_subset(args.out, args.limit)