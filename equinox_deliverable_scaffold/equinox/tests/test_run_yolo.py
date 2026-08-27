import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.detection.run_yolo import discover_frames, frame_indexes, main, run


class TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, cls, conf, ids=None):
        self.xyxy = TensorLike(xyxy)
        self.cls = TensorLike(cls)
        self.conf = TensorLike(conf)
        self.id = TensorLike(ids) if ids is not None else None


class FakeResult:
    names = {0: "person", 2: "car"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self):
        self.calls = []

    def track(self, source, persist, **kwargs):
        self.calls.append((source, persist, kwargs))
        return [
            FakeResult(FakeBoxes([[9, 8, 2, 1], [4, 5, 6, 7]], [2, 0], [.9, .8], [7, 3]))
            for _ in source
        ]

    def predict(self, source, **kwargs):
        self.calls.append((source, None, kwargs))
        return [FakeResult(FakeBoxes([[1, 2, 3, 4]], [2], [.7])) for _ in source]


class RunYoloTests(unittest.TestCase):
    def test_writes_sorted_contract_csv_with_json_bboxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "frames"
            clip_b = root / "clip_b"
            clip_a = root / "clip_a"
            clip_b.mkdir(parents=True)
            clip_a.mkdir(parents=True)
            for path in (clip_b / "00010.jpg", clip_a / "00002.jpg", clip_a / "00001.jpg"):
                path.touch()
            model = FakeModel()
            output = Path(temporary) / "detections.csv"

            run(root, output, weights="fake.pt", conf=.4, device="cpu", model_factory=lambda _: model)

            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([(row["clip_id"], row["frame_idx"]) for row in rows], [
                ("clip_a", "1"), ("clip_a", "1"), ("clip_a", "2"), ("clip_a", "2"),
                ("clip_b", "10"), ("clip_b", "10"),
            ])
            self.assertEqual(json.loads(rows[0]["bbox"]), [2.0, 1.0, 9.0, 8.0])
            self.assertEqual(rows[0]["cls"], "car")
            self.assertEqual(rows[1]["track_id"], "3")
            self.assertEqual(model.calls[0][0], [str(clip_a / "00001.jpg"), str(clip_a / "00002.jpg")])
            self.assertEqual(model.calls[0][2], {"conf": .4, "verbose": False, "device": "cpu"})

    def test_arbitrary_image_names_have_deterministic_sorted_frame_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_dir = Path(temporary) / "visdrone" / "images"
            image_dir.mkdir(parents=True)
            for name in ("0000023_01233_d_0000011.jpg", "0000001_05999_d_0000011.jpg", "0000023_00000_d_0000008.jpg"):
                (image_dir / name).touch()

            grouped = discover_frames(image_dir)
            paths = grouped["images"]

            self.assertEqual([path.name for path in paths], [
                "0000001_05999_d_0000011.jpg", "0000023_00000_d_0000008.jpg", "0000023_01233_d_0000011.jpg",
            ])
            self.assertEqual(frame_indexes(paths), [0, 1, 2])

    def test_numeric_frame_names_preserve_numeric_order_and_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame_dir = Path(temporary) / "clip"
            frame_dir.mkdir()
            for name in ("00010.jpg", "00002.jpg", "00001.jpg"):
                (frame_dir / name).touch()

            paths = discover_frames(frame_dir.parent)["clip"]

            self.assertEqual([path.name for path in paths], ["00001.jpg", "00002.jpg", "00010.jpg"])
            self.assertEqual(frame_indexes(paths), [1, 2, 10])

    def test_detection_only_uses_frame_local_contract_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "clip" / "00000.jpg"
            frame.parent.mkdir()
            frame.touch()
            model = FakeModel()
            output = Path(temporary) / "detections.csv"

            run(frame.parent.parent, output, track=False, model_factory=lambda _: model)

            with output.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["track_id"], "untracked-0-0")
            self.assertEqual(model.calls[0], ([str(frame)], None, {"conf": .25, "verbose": False}))

    def test_rejects_invalid_confidence_without_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "clip" / "00000.jpg"
            frame.parent.mkdir()
            frame.touch()
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                run(frame.parent.parent, Path(temporary) / "out.csv", conf=1.1,
                    model_factory=lambda _: self.fail("model should not load"))

    def test_cli_maps_out_flag_to_run_parameter(self):
        with patch("pipeline.detection.run_yolo.run") as mocked_run:
            main(["--source", "frames", "--out", "result.csv", "--model", "model.pt"])
        mocked_run.assert_called_once_with(
            source="frames", out_path="result.csv", weights="model.pt", conf=.25,
            device=None, track=True,
        )


if __name__ == "__main__":
    unittest.main()
