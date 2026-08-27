# Equinox Dataset Card

Last updated: 2026-08-27

This card describes the data currently shipped with Equinox and the optional
public-data preparation path. It is intentionally explicit about what is a
fixture, what is real public data, and what has not been evaluated.

## Intended use

Equinox consumes aerial/drone road footage to prototype vehicle/obstruction
detection, corridor occupancy features, incident severity, and emergency-route
accessibility estimates. The data and models in this checkout are suitable for
software smoke tests, interface development, and a local demonstration. They
are not suitable for traffic enforcement, emergency dispatch automation,
public-safety decisions, or claims of field performance.

## Data inventory

### Checked-in synthetic fixtures

`data/raw/` contains four generated MP4 clips, each 640×480, 15 fps, and 4
seconds long. The generator is `pipeline/generate_demo_data.py`. The scenarios
are:

| Clip | Scenario | Severity label | Accessibility label |
|---|---|---:|---:|
| `clip_01_hosur_j4` | stalled truck + debris | Critical | 28 |
| `clip_02_service_rd` | parked van | Severe | 45 |
| `clip_03_neeladri_rd` | slow vehicle | Moderate | 68 |
| `clip_04_wipro_jct` | clear traffic | Normal | 95 |

Each fixture has a sidecar metadata JSON file, a corridor mask, 12 sampled
frames, detector-like CSV rows, and one incident-window label. The checked-in
`data/manifest.csv`, `data/detections/detections.csv`,
`data/features/incident_windows.parquet`, and `data/labels/labels.csv` are
derived artifacts from these fixtures. They are not a random sample of real
traffic and contain only one labeled incident window per clip.

The sidecar timestamps are metadata embedded by the generator, not evidence of
real collection at those locations. The fixtures contain rendered labels and
simple shapes, so they do not represent real camera motion, weather, occlusion,
vehicle appearance, or traffic behavior.

### Optional real public data: VisDrone2019-DET

`pipeline/ingestion/prepare_visdrone.py` converts an extracted VisDrone2019-DET
directory into YOLO image/label layout. The `--download-val` option downloads
the validation archive from the URL pinned in that script, then can limit the
conversion to a small smoke subset. The default smoke workflow is:

```bash
python pipeline/ingestion/prepare_visdrone.py \
  --download-val \
  --limit 50 \
  --output data/visdrone_smoke \
  --cache-dir data/visdrone_cache
```

This is the only real public image set currently wired into the repository. It
is not present in the checkout until the command is run, and the archive,
cache, extracted data, and smoke subset are gitignored. An externally obtained
train/validation extraction can be converted with `--source`; see the project
README for the exact command.

The converter preserves the ten source classes in YOLO order:

`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`,
`awning-tricycle`, `bus`, `motor`.

Source annotation rows are interpreted as
`x,y,width,height,score,category,truncation,occlusion`. Category 0 and category
11 (`others`) are ignored by the converter. Boxes are clipped to image bounds,
invalid/empty boxes are skipped, and class counts plus split counts are written
to `provenance.json`.

VisDrone labels road users and vehicles; it does **not** label Equinox's
incident classes, blocked lanes, stalled-vs-moving state, severity, or
accessibility. Therefore VisDrone can support detector pretraining/smoke
testing, but it cannot by itself train or validate the downstream severity
model.

The upstream dataset's terms and attribution requirements apply. Review the
official VisDrone repository and release terms before redistribution or
commercial use. The converter records the upstream URL and local source root;
it does not grant a new license to the data.

### Legacy metadata helper

`data/dataset_sources.json` and
`pipeline/ingestion/download_google_open_images.py` describe an older Open
Images metadata workflow. That helper downloads class/annotation metadata and
records image IDs; it does not download the image pixels or produce a complete
training dataset. Open Images is not used by the current smoke workflow.

## Labels and features

The detector CSV contract is:

| Field | Meaning |
|---|---|
| `clip_id` | Source clip/group identifier |
| `frame_idx` | Numeric frame index |
| `track_id` | ByteTrack ID, or frame-local fallback when tracking is disabled |
| `cls` | Detector class name |
| `bbox` | JSON/Python-literal-compatible `[x1, y1, x2, y2]` coordinates; feature parsing is literal-only |
| `conf` | Detector confidence |

Feature engineering rolls each clip's detections into an incident-window row
with `n_obstructions`, `occupied_fraction`, `clear_lane_width_px`,
`max_dwell_s`, and the frame range. The current implementation uses fixed
placeholder corridor dimensions (`400 px` and `120,000 px²`) and an area-sum
occupancy approximation. These values are not calibrated measurements.

Downstream labels in `data/labels/labels.csv` are `severity` (Normal,
Moderate, Severe, Critical) and `accessibility_score` (0–100). The rubric
template is in `dataset/labeling_rubric_template.md`; the four shipped labels
should be treated as demonstrations, not a validated annotation protocol.

## Splits and evaluation

The severity trainer joins features and labels on
`clip_id/start_frame/end_frame`, rejects duplicate keys, and uses
`GroupShuffleSplit` by `clip_id` so frames/windows from one clip cannot cross
the train/validation boundary. It requires at least three distinct clips and
at least two severity classes in the training partition. A validation split
with only one class is allowed with an explicit warning, but is not strong
evidence.

The four synthetic clips are not an independent test set. Existing model files
and `docs/validation_report.md` are scaffold artifacts based on tiny synthetic
data; do not quote them as real-world metrics. A credible evaluation still
needs independent clips, multiple locations/vantage points, lighting and
weather variation, clip-level train/validation/test separation, and labeled
incident windows that were not used during fitting.

## Reproducibility and provenance

Synthetic fixtures can be regenerated with:

```bash
python pipeline/generate_demo_data.py
```

VisDrone conversion is deterministic for a fixed source tree, split selection,
and limit. The download helper caches the archive, computes SHA-256, and writes
the observed digest and download metadata. Supply `--sha256` when an expected
digest is available to fail closed on a mismatch. Generated public-data paths
are gitignored so large or licensed data is not accidentally committed; keep
the command, source URL, split/limit, and provenance JSON with any experiment
record.

## Known gaps and risks

- Four synthetic clips cannot establish detector, severity, accessibility, or
  latency performance.
- VisDrone is a vehicle/object dataset, not an incident-severity dataset.
- The repository has no detector-training command or committed detector
  weights; `run_yolo.py` performs inference using a supplied Ultralytics model.
- A detector-only run creates frame-local IDs, so dwell time is not valid across
  frames unless tracking is enabled and IDs are stable.
- Masks are checked for presence, but the feature stage currently does not
  rasterize boxes against the mask.
- Location, camera, weather, time-of-day, road geometry, and class balance are
  not sufficient for deployment generalization.
- Synthetic rendered text/shapes may make the task unrealistically easy.
- The API is an unauthenticated local demo and must not be exposed directly to
  the public internet or used as an emergency control system.

## Acceptance checks

From the project directory, after installing `requirements.txt`:

```bash
python -m compileall -q pipeline tests
python -m unittest discover -s tests -v
python -m unittest discover -s pipeline/ingestion -p 'test_*.py' -v
```

These checks verify the detector CSV contract, safe VisDrone conversion,
portable manifest paths, literal-only box parsing, grouped validation splits,
API defaults, and video path safety. In an environment missing declared
dependencies, dependency-specific hardening tests may be reported as skips;
that is an environment limitation, not a model result.
