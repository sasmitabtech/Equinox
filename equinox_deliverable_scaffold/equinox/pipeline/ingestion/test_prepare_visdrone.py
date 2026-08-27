"""Small stdlib tests for the VisDrone converter (run with pytest or unittest)."""

import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_visdrone import _safe_extract, prepare


def png(width=100, height=80):
    # The converter only needs the PNG signature and IHDR dimensions.
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


class PrepareVisDroneTest(unittest.TestCase):
    def test_converts_annotations_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "out"
            for split in ("train", "val"):
                (root / f"VisDrone2019-DET-{split}" / "images").mkdir(parents=True)
                (root / f"VisDrone2019-DET-{split}" / "annotations").mkdir()
                image = root / f"VisDrone2019-DET-{split}" / "images" / "000001.jpg.png"
                image.write_bytes(png())
                (root / f"VisDrone2019-DET-{split}" / "annotations" / "000001.jpg.txt").write_text("10,20,30,20,1,4,0,0\n0,0,2,2,1,0,0,0\n")
            result = prepare(root, output, limit=1)
            label = (output / "train/labels/000001.jpg.txt").read_text().strip()
            self.assertEqual(label, "3 0.250000 0.375000 0.300000 0.250000")
            self.assertEqual(result["images_copied"], 2)
            self.assertEqual(result["class_counts"], {"car": 2})
            self.assertIn("names:", (output / "dataset.yaml").read_text())

    def test_limit_is_per_split_and_no_download_occurs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "source", Path(temp) / "out"
            for split in ("train", "val"):
                base = root / f"VisDrone2019-DET-{split}"
                (base / "images").mkdir(parents=True)
                (base / "annotations").mkdir()
                for index in range(2):
                    (base / "images" / f"{index}.png").write_bytes(png())
            result = prepare(root, output, limit=1)
            self.assertEqual(result["splits"], {"train": 1, "val": 1})

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, destination = Path(temp) / "bad.zip", Path(temp) / "out"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "nope")
            with self.assertRaises(ValueError):
                _safe_extract(archive, destination)


if __name__ == "__main__":
    unittest.main()
