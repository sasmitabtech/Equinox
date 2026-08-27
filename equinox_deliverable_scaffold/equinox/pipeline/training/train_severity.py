"""
Stage 4b: Severity / Accessibility Model
Input : incident_windows.parquet (from feature_engineer.py) + a labels.csv
        with {clip_id, start_frame, end_frame, severity, accessibility_score}
        produced during manual labeling per the rubric in
        dataset/labeling_rubric_template.md.
Output: models/severity_model.pkl + models/accessibility_model.pkl
        + a training log printed to stdout (capture into
        docs/validation_report.md manually or via --log_out).

Deliberately a small, interpretable model (GradientBoosting) rather than a
neural net — see plan §4 rationale: easier to defend to judges, and
appropriate for the realistic dataset size (§3.1).

IMPORTANT: split by clip_id, not by row, so frames/windows from the same
clip never appear in both train and validation (leakage — see plan §3.7).
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, mean_absolute_error

FEATURE_COLS = ["n_obstructions", "occupied_fraction", "clear_lane_width_px", "max_dwell_s"]


def load_dataset(features_path: str, labels_path: str) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    labels = pd.read_csv(labels_path)
    df = features.merge(labels, on=["clip_id", "start_frame", "end_frame"], how="inner")
    if len(df) < len(labels):
        print(f"WARNING: {len(labels) - len(df)} labeled window(s) had no matching "
              f"feature row — check clip_id/frame alignment before trusting metrics.")
    return df


def split_by_clip(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["clip_id"]))
    return df.iloc[train_idx], df.iloc[val_idx]


def train(features_path: str, labels_path: str, out_dir: str):
    df = load_dataset(features_path, labels_path)
    if df["clip_id"].nunique() < 3:
        print("WARNING: fewer than 3 distinct clips in the labeled set — a "
              "clip-level split isn't meaningful yet. Label more clips before "
              "trusting held-out metrics.")

    train_df, val_df = split_by_clip(df)
    X_train, X_val = train_df[FEATURE_COLS], val_df[FEATURE_COLS]

    # --- Severity classifier (Normal/Moderate/Severe/Critical) ---
    sev_clf = GradientBoostingClassifier(random_state=42)
    sev_clf.fit(X_train, train_df["severity"])
    print("\n== Severity classification report (held-out clips) ==")
    print(classification_report(val_df["severity"], sev_clf.predict(X_val)))

    # --- Accessibility score regressor (0-100) ---
    acc_reg = GradientBoostingRegressor(random_state=42)
    acc_reg.fit(X_train, train_df["accessibility_score"])
    mae = mean_absolute_error(val_df["accessibility_score"], acc_reg.predict(X_val))
    print(f"\n== Accessibility score MAE (held-out clips): {mae:.2f} ==")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "severity_model.pkl", "wb") as f:
        pickle.dump(sev_clf, f)
    with open(out_dir / "accessibility_model.pkl", "wb") as f:
        pickle.dump(acc_reg, f)
    print(f"\nSaved models to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args.features, args.labels, args.out)
