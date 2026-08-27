"""
Stage 5: Validation
Input : trained severity/accessibility models + a held-out feature set
        (clips never touched during training/tuning).
Output: docs/validation_report.md — metrics + a handful of qualitative
        example cards (correct AND incorrect) per plan §1.5 / §7.

Run this against a truly untouched split, not the same val split used
during training iteration, to get a number you can defend to judges.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.training.train_severity import FEATURE_COLS


def validate(model_path: str, features_path: str, labels_path: str, report_out: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = pd.read_parquet(features_path)
    labels = pd.read_csv(labels_path)
    df = features.merge(labels, on=["clip_id", "start_frame", "end_frame"])

    X = df[FEATURE_COLS]
    y_true = df["severity"]
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, output_dict=False)
    cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Moderate", "Severe", "Critical"])

    # Flag the failure mode judges will ask about first.
    critical_as_normal = ((y_true == "Critical") & (y_pred == "Normal")).sum()

    lines = [
        "# Validation Report\n",
        f"Held-out clips: {df['clip_id'].nunique()} | windows: {len(df)}\n",
        "## Classification report\n```\n" + report + "\n```\n",
        "## Confusion matrix (rows=true, cols=pred; Normal/Moderate/Severe/Critical)\n",
        "```\n" + str(cm) + "\n```\n",
        f"**Critical incidents misclassified as Normal: {critical_as_normal}** "
        "— this is the failure mode with real safety cost; investigate these rows first.\n",
        "## TODO before submission\n",
        "- [ ] Attach 5-10 qualitative example cards (frame + prediction + ground truth)\n",
        "- [ ] Break down performance by location/time-of-day (plan §3.6 bias check)\n",
        "- [ ] Report alert latency separately (needs frame-timestamped runs, not just this static eval)\n",
    ]

    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text("\n".join(lines))
    print(f"Wrote validation report -> {report_out}")
    if critical_as_normal > 0:
        print(f"⚠ {critical_as_normal} Critical→Normal miss(es) — review before demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--report_out", default="docs/validation_report.md")
    args = parser.parse_args()
    validate(args.model, args.features, args.labels, args.report_out)
