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
import math
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
    feature_required = {"clip_id", "start_frame", "end_frame", *FEATURE_COLS}
    label_required = {
        "clip_id", "start_frame", "end_frame", "severity", "accessibility_score"
    }
    missing_features = feature_required.difference(features.columns)
    missing_labels = label_required.difference(labels.columns)
    if missing_features:
        raise ValueError(f"Features missing required column(s): {sorted(missing_features)}")
    if missing_labels:
        raise ValueError(f"Labels missing required column(s): {sorted(missing_labels)}")
    if features.empty or labels.empty:
        raise ValueError("Features and labels must both contain at least one row")
    keys = ["clip_id", "start_frame", "end_frame"]
    if features.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError("Feature and label keys must be unique per incident window")
    df = features.merge(labels, on=keys, how="inner", validate="one_to_one")
    if len(df) < len(labels):
        print(f"WARNING: {len(labels) - len(df)} labeled window(s) had no matching "
              f"feature row — check clip_id/frame alignment before trusting metrics.")
    if df.empty:
        raise ValueError("No labeled feature rows matched on clip_id/frame range")
    if df["clip_id"].isna().any() or (df["clip_id"].astype(str).str.strip() == "").any():
        raise ValueError("clip_id must be non-empty for every labeled row")
    if df["severity"].isna().any() or (df["severity"].astype(str).str.strip() == "").any():
        raise ValueError("Severity labels must be non-empty")
    numeric = df[FEATURE_COLS + ["accessibility_score"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.map(math.isfinite).all().all():
        raise ValueError("Feature and accessibility values must be finite numbers")
    if not numeric["accessibility_score"].between(0, 100).all():
        raise ValueError("accessibility_score must be between 0 and 100")
    df[FEATURE_COLS + ["accessibility_score"]] = numeric
    return df


def split_by_clip(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    if "clip_id" not in df.columns:
        raise ValueError("Dataset must contain a clip_id column for grouped validation")
    if df["clip_id"].isna().any() or (df["clip_id"].astype(str).str.strip() == "").any():
        raise ValueError("clip_id must be non-empty for every row")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n_clips = df["clip_id"].nunique()
    if n_clips < 3:
        raise ValueError(
            f"At least 3 distinct clips are required for clip-level validation; found {n_clips}"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["clip_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    if train_df.empty or val_df.empty:
        raise ValueError("Clip-level split produced an empty train or validation set")
    return train_df, val_df


def train(features_path: str, labels_path: str, out_dir: str):
    df = load_dataset(features_path, labels_path)
    train_df, val_df = split_by_clip(df)
    X_train, X_val = train_df[FEATURE_COLS], val_df[FEATURE_COLS]

    if train_df["severity"].nunique() < 2:
        raise ValueError(
            "Training split contains fewer than two severity classes; "
            "add labeled clips or choose a different split seed"
        )
    if val_df["severity"].nunique() < 2:
        print("WARNING: validation split contains only one severity class; "
              "classification metrics are limited until more clips are labeled.")

    # --- Severity classifier (Normal/Moderate/Severe/Critical) ---
    sev_clf = GradientBoostingClassifier(random_state=42)
    sev_clf.fit(X_train, train_df["severity"])
    print("\n== Severity classification report (held-out clips) ==")
    print(classification_report(val_df["severity"], sev_clf.predict(X_val), zero_division=0))

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
