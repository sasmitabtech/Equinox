"""Regression coverage for conservative VisDrone smoke evaluation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.evaluation.visdrone_smoke import (  # noqa: E402
    Box, COCO_TO_EVAL, UNSUPPORTED_VISDRONE, VISDRONE_TO_EVAL,
    average_precision_50, evaluate, greedy_match, iou,
)


class VisDroneSmokeEvaluationTests(unittest.TestCase):
    def test_iou_handles_overlap_and_invalid_boxes(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(iou((10, 10, 0, 0), (0, 0, 5, 5)), 0.0)

    def test_greedy_matching_uses_confidence_and_one_target_once(self):
        targets = [Box("a", "car", (0, 0, 10, 10))]
        predictions = [Box("a", "car", (0, 0, 10, 10), .2), Box("a", "car", (0, 0, 10, 10), .9)]
        matches, matched_predictions, matched_targets = greedy_match(predictions, targets, .5)
        self.assertEqual(matches[0][0], 1)
        self.assertEqual(matched_predictions, {1})
        self.assertEqual(matched_targets, {0})

    def test_ap50_is_confidence_ranked(self):
        targets = [Box("a", "car", (0, 0, 10, 10)), Box("b", "car", (0, 0, 10, 10))]
        predictions = [Box("a", "car", (30, 30, 40, 40), .9), Box("a", "car", (0, 0, 10, 10), .8), Box("b", "car", (0, 0, 10, 10), .7)]
        self.assertAlmostEqual(average_precision_50(predictions, targets), 2 / 3)

    def test_mapping_is_conservative(self):
        self.assertEqual(VISDRONE_TO_EVAL["pedestrian"], "person")
        self.assertEqual(VISDRONE_TO_EVAL["people"], "person")
        self.assertEqual(VISDRONE_TO_EVAL["motor"], "motorcycle")
        self.assertIn("van", UNSUPPORTED_VISDRONE)
        self.assertNotIn("van", COCO_TO_EVAL)

    def test_empty_predictions_are_scored_as_false_negatives(self):
        summary, rows, errors, _ = evaluate([], [Box("a", "car", (0, 0, 10, 10))])
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (0, 0, 1))
        self.assertEqual(errors["false_negative_missed"], 1)
        self.assertEqual(next(row for row in rows if row["class"] == "car")["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
