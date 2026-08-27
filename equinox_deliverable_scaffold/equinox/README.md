# Equinox Corridor Watch

Equinox is a demo-scale pipeline for turning drone or aerial road footage into
road-blockage features, severity/accessibility estimates, and an operator API.
It is an engineering scaffold with a real detector integration seam; it is not
yet a production or safety-certified system.

## Architecture

```text
raw clips + sidecars
        │
        ▼
ingestion/ingest.py ──► data/manifest.csv
        │
        ▼
preprocessing/preprocess.py ──► data/frames/{clip_id}/*.jpg + mask checks
        │
        ▼
detection/run_yolo.py ──► detections.csv (YOLO + optional ByteTrack IDs)
        │
        ▼
features/feature_engineer.py ──► incident_windows.parquet
        │
        ▼
training/train_severity.py ──► severity/accessibility .pkl models
        │
        ├── validation/validate.py ──► validation report
        └── deployment/api.py ──► FastAPI + dashboard
```

The stable hand-off files are `manifest.csv`, the detector CSV contract
(`clip_id, frame_idx, track_id, cls, bbox, conf`),
`incident_windows.parquet`, and the model files under `models/`.

## What is in this checkout

- `data/raw/` contains four small **synthetic generated fixtures**. Their
  sidecar JSON files, masks, detections, labels, and sampled frames are for
  smoke testing and dashboard demonstration; they are not a real-world
  evaluation set.
- `pipeline/ingestion/prepare_visdrone.py` converts a locally available
  VisDrone2019-DET archive/extraction into YOLO layout and writes provenance.
  A real VisDrone validation smoke subset can be downloaded on demand, but no
  downloaded VisDrone data is committed to this repository.
- `models/` contains the scaffold's existing severity/accessibility artifacts.
  They were produced from the tiny synthetic fixture set and must not be
  presented as generalization evidence.
- `dashboard/dashboard_mockup.html` is the operator UI. The FastAPI root route
  serves it when the API is running.

See [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) for provenance, labels,
split policy, reproducibility, and limitations. The longer design plan is in
[`docs/EQUINOX_Deliverable_Plan.md`](docs/EQUINOX_Deliverable_Plan.md).

## Setup

Run these commands from this directory (`equinox_deliverable_scaffold/equinox`):

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No network download is required for the checked-in synthetic fixtures or for
the unit tests that do not exercise optional model inference. A detector
weights file and a network connection are required for the YOLO/VisDrone path.

## Reproduce the synthetic smoke data

The checkout already includes the fixtures. To regenerate them deliberately
(this overwrites the generated files under `data/`), run:

```bash
python pipeline/generate_demo_data.py
```

For an existing or newly supplied raw-clip directory:

```bash
python pipeline/ingestion/ingest.py \
  --raw_dir data/raw \
  --out data/manifest.csv

python pipeline/preprocessing/preprocess.py \
  --manifest data/manifest.csv \
  --out data/frames \
  --masks_dir data/masks \
  --fps 3
```

The ingestion manifest stores portable paths relative to the manifest. The
preprocessor also accepts older manifests containing Windows backslashes.

## Prepare a real VisDrone smoke subset

This is an explicit opt-in download. It downloads and caches the VisDrone
validation archive, converts up to 50 images into YOLO labels, and writes
`data/visdrone_smoke/provenance.json`:

```bash
python pipeline/ingestion/prepare_visdrone.py \
  --download-val \
  --limit 50 \
  --output data/visdrone_smoke \
  --cache-dir data/visdrone_cache
```

The downloaded archive and generated subset are gitignored. For an archive
downloaded separately (including a train split), use:

```bash
python pipeline/ingestion/prepare_visdrone.py \
  --source /path/to/VisDrone \
  --output data/visdrone \
  --limit 50 \
  --splits train val
```

The converter records the source, URL, split counts, class counts, annotation
format, ignored categories, and source-root metadata. Pass `--sha256` when an
expected archive digest is available; the observed digest is recorded in the
download metadata. Do not commit the downloaded images or archive.

## Run detector, features, and models

The detector requires a compatible Ultralytics `.pt` weights file. The
repository does not claim that `models/detector.pt` exists; provide your own
trained/fine-tuned weights or change `--model`:

```bash
python pipeline/detection/run_yolo.py \
  --source data/visdrone_smoke/val/images \
  --out data/detections/detections_yolo.csv \
  --model models/detector.pt \
  --no-track \
  --conf 0.25
```

Use `--no-track` for frame-local detections. Without tracking, dwell-time
features are not meaningful across frames.

Build the incident-window features from one detector CSV:

```bash
python pipeline/features/feature_engineer.py \
  --detections data/detections/detections_yolo.csv \
  --out data/features
```

Train the downstream severity and accessibility models. This joins feature
windows to labels by `clip_id/start_frame/end_frame` and splits by whole clip
to prevent frame leakage:

```bash
python pipeline/training/train_severity.py \
  --features data/features/incident_windows.parquet \
  --labels data/labels/labels.csv \
  --out models
```

Validate a model against a feature/label set that was not used for fitting:

```bash
python pipeline/validation/validate.py \
  --model models/severity_model.pkl \
  --features data/features/incident_windows.parquet \
  --labels data/labels/labels.csv \
  --report_out docs/validation_report.md
```

Do not treat the checked-in four-row fixture report as a meaningful benchmark.
No new real-data metrics are claimed by this repository until a genuinely
held-out set has been labeled and evaluated.

## Run the API and dashboard

```bash
uvicorn pipeline.deployment.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Then open <http://127.0.0.1:8000/>. Useful API routes include `/health`,
`/incidents`, `/incidents/{incident_id}`, status updates under
`/incidents/{incident_id}/status`, clearance re-checks, and evidence videos
under `/api/video/{filename}`.

### Dashboard frontend development

The dashboard is a React/Vite application in `dashboard/frontend`. For a live
frontend development session, run the FastAPI service above and then:

```bash
cd dashboard/frontend
npm install
npm run dev
```

Vite proxies `/health`, `/incidents`, and `/api` requests to the local FastAPI
service. For the production dashboard that FastAPI serves at `/`, run:

```bash
cd dashboard/frontend
npm run build
```

Use `VITE_API_BASE_URL` to point the built frontend at a future remote API
(for example a Cloudflare Worker); leave it unset for same-origin local serving.

## Acceptance checks

Run the checks from the project directory after installing requirements:

```bash
python -m compileall -q pipeline tests
python -m unittest discover -s tests -v
python -m unittest discover -s pipeline/ingestion -p 'test_*.py' -v
```

The tests cover detector CSV behavior, VisDrone conversion, portable paths,
safe bounding-box parsing, grouped splits, API defaults, and video path
safety. In a minimal environment missing optional dependencies, relevant
hardening tests are reported as skips; install `requirements.txt` for the
full suite.

## Current limitations

The four included clips are synthetic and too small for credible accuracy
claims. The detector stage consumes weights; it does not train them. Current
feature geometry uses placeholder corridor calibration and an area-sum
occupancy approximation rather than polygon intersection. Severity labels are
incident-window labels, so real deployment needs substantially more clips,
locations, lighting conditions, and independent held-out evaluation. The API
is an unauthenticated local demo service and should not be exposed directly to
the public internet.
