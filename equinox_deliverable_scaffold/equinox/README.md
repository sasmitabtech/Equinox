# Equinox — Pipeline Scaffold

Companion code to `docs/EQUINOX_Deliverable_Plan.md`. This is a **working skeleton**, not a finished model — each module has real, runnable I/O contracts (so two people can build against each other's stubs in parallel) with `# TODO` markers where the actual model/logic goes.

## Structure
```
equinox/
├── docs/
│   └── EQUINOX_Deliverable_Plan.md   ← full plan (read this first)
├── dashboard/
│   └── dashboard_mockup.html         ← standalone operator UI mockup, open directly in a browser
├── dataset/
│   └── labeling_rubric_template.md   ← fill this in before labeling anything
└── pipeline/
    ├── ingestion/ingest.py           ← §1.1 raw clip -> manifest.csv
    ├── preprocessing/preprocess.py   ← §1.2 clip -> sampled frames + mask check
    ├── features/feature_engineer.py  ← §1.3 detections -> feature vector
    ├── training/train_severity.py    ← §1.4 features -> severity/accessibility model
    ├── validation/validate.py        ← §1.5 held-out eval -> validation_report.md
    └── deployment/api.py             ← §1.6 FastAPI service the dashboard calls
```

## Quickstart (once you have real data)
```bash
pip install ultralytics opencv-python scikit-learn fastapi uvicorn pandas pyarrow --break-system-packages

python pipeline/ingestion/ingest.py --raw_dir data/raw --out data/manifest.csv
python pipeline/preprocessing/preprocess.py --manifest data/manifest.csv --out data/frames --fps 3
# label data/frames/ in Roboflow/CVAT, export YOLO format to data/labels/
# train detector separately with `yolo train` (Ultralytics CLI) — see plan §4
python pipeline/features/feature_engineer.py --detections data/detections --out data/features
python pipeline/training/train_severity.py --features data/features --out models/severity_model.pkl
python pipeline/validation/validate.py --model models/severity_model.pkl --features data/features_val
uvicorn pipeline.deployment.api:app --reload
```

Open `dashboard/dashboard_mockup.html` in a browser to see the target UI while the API/model are still being built — the dashboard is deliberately decoupled so UI work isn't blocked on model readiness.
