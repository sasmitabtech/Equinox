"""Focused regression tests for pipeline boundary validation."""
import csv
import importlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


preprocess = optional_import("pipeline.preprocessing.preprocess")
ingestion = optional_import("pipeline.ingestion.ingest")
features = optional_import("pipeline.features.feature_engineer")
training = optional_import("pipeline.training.train_severity")
api = optional_import("pipeline.deployment.api")


class PipelineHardeningTests(unittest.TestCase):
    @unittest.skipUnless(ingestion, "OpenCV is not installed")
    def test_ingest_rejects_same_stem_media_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "foo.mp4").touch()
            (raw / "foo.mov").touch()
            with self.assertRaisesRegex(ValueError, "same clip_id"):
                ingestion.ingest(str(raw), str(root / "manifest.csv"))

    @unittest.skipUnless(preprocess, "OpenCV is not installed")
    def test_resolve_manifest_path_accepts_windows_separators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "data" / "manifest.csv"
            clip = root / "data" / "raw" / "clip.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"test")
            manifest.touch()
            self.assertEqual(
                preprocess.resolve_manifest_path(r"raw\clip.mp4", manifest), clip
            )

    @unittest.skipUnless(features, "Pandas/NumPy are not installed")
    def test_bbox_parser_does_not_execute_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "pwned"
            detections = root / "detections.csv"
            with detections.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["clip_id", "frame_idx", "track_id", "cls", "bbox", "conf"])
                writer.writerow(["clip", 0, 1, "vehicle", f"__import__('os').system('touch {marker}')", .9])
            with self.assertRaisesRegex(ValueError, "Invalid bbox literal"):
                features.run(str(detections), str(root / "out"))
            self.assertFalse(marker.exists())

    @unittest.skipUnless(features, "Pandas/NumPy are not installed")
    def test_geometry_ignores_inverted_box_area(self):
        self.assertEqual(features.occupied_fraction([(10, 10, 0, 20)], 100), 0.0)
        self.assertEqual(features.clear_lane_width([(10, 10, 0, 20)], 100), 100)

    @unittest.skipUnless(training, "Pandas/scikit-learn are not installed")
    def test_split_by_clip_rejects_meaningless_dataset(self):
        frame = training.pd.DataFrame({
            "clip_id": ["a", "b"], "severity": ["Normal", "Critical"],
        })
        with self.assertRaisesRegex(ValueError, "At least 3 distinct clips"):
            training.split_by_clip(frame)

    @unittest.skipUnless(training, "Pandas/scikit-learn are not installed")
    def test_split_by_clip_has_disjoint_groups(self):
        frame = training.pd.DataFrame({
            "clip_id": ["a", "a", "b", "c", "d"],
            "severity": ["Normal", "Normal", "Critical", "Severe", "Moderate"],
        })
        train, validation = training.split_by_clip(frame, test_size=0.25, seed=1)
        self.assertTrue(set(train.clip_id).isdisjoint(set(validation.clip_id)))

    @unittest.skipUnless(api, "FastAPI is not installed")
    def test_api_defaults_are_not_shared(self):
        kwargs = dict(
            incident_class="vehicle", location="x", lat=0, lon=0,
            timestamp="now", severity="Normal", accessibility_score=100,
            recommended_action="none", status="Open",
        )
        first = api.Incident(incident_id="TEST-1", **kwargs)
        second = api.Incident(incident_id="TEST-2", **kwargs)
        first.contributing_factors.append("only first")
        self.assertEqual(second.contributing_factors, [])

    @unittest.skipUnless(api, "FastAPI is not installed")
    def test_api_rejects_video_traversal(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as context:
            api.serve_video("../models/severity_model.pkl")
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
