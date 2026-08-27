# ELCIA Smart City Drone-AI Challenge 2026 — form draft

Use only after replacing every `[VERIFY]`/`[PLACEHOLDER]`. Current repository main is `1ea4b4774492466902bb2bfc1dfc77c501033717`; this is a stale pre-submission fact and will change after the requested history cleanup and submission-package work. No tag currently exists.

## Email

`joshiny.maria@btech.christuniversity.in` (record this email in the response).

## Section 1 — Team details

**Team Name:** `Equinox` [VERIFY official spelling]

**Project Title:** `Emergency Corridor & Road Blockage Intelligence` [VERIFY; the implementation README also uses “Equinox Corridor Watch”]

**In one sentence, what exact problem does your working prototype solve?**

> Emergency operators need to know whether a road blockage leaves enough clear space for an ambulance or other emergency vehicle to pass, rather than merely seeing that traffic is busy.

## Section 2 — Technical submission

**GitHub Repository URL:** `https://github.com/sasmitabtech/Equinox`

**Final GitHub Commit Hash or Release Tag:** `[PLACEHOLDER: final post-cleanup full hash or immutable release tag]`  
Current main for reference only: `1ea4b4774492466902bb2bfc1dfc77c501033717`.

**Final 5-Minute Demonstration Video URL:** `[PLACEHOLDER: tested public video URL]`

**Dashboard/Working Application URL:** `[PLACEHOLDER: tested public Cloudflare/app URL]`  
Local-only URL currently available: `http://127.0.0.1:8000/` (not a public submission URL).

## Section 3 — Current build status

**What is working TODAY? (116 words; maximum 150)**

> Local demo-scale pipeline and operator dashboard are runnable from the repository. Checked-in synthetic fixtures provide four MP4 clips, sidecars, masks, 12 sampled frames per clip, detector CSV, feature parquet, labels, and severity/accessibility models. FastAPI serves health, incident listing/detail, evidence MP4s, status transitions Open → Assigned → In Action → Cleared → Verified, and a clearance re-check endpoint. The dashboard shows four incident cards, video evidence, bounding-box overlays, severity, accessibility score, contributing factors, and recommended actions. Detector code is wired to an externally supplied Ultralytics model and can emit a detector CSV. Core tests pass (14) and VisDrone converter tests pass (3). This is a local fixture/demo workflow; no hosted dashboard or live drone feed is evidenced.

**What is simulated, mocked or manually configured? (80 words; maximum 100)**

> Four MP4s are generated synthetic scenes (stalled truck/debris, parked van, slow vehicle, clear traffic), not ELCIA footage. The dashboard’s four incident records, classes, scores, recommendations, coordinates, timestamps, factors, bounding-box overlay positions/confidences, and initial statuses are seeded; overlays are not produced by the API at runtime. Clearance/re-check changes values by deterministic rules. Corridor width/mask area are fixed placeholders; occupancy and lane width are approximations. The included models are trained on the four synthetic windows; no detector weights are tracked in Git.

**What remains future scope and is NOT implemented? (48 words; maximum 100)**

> Field-valid detector/severity evaluation on independently labeled real footage; training/fine-tuning and committed detector weights; calibrated polygon/mask geometry and reliable lane/free-space estimation; real-time/live drone ingestion; hosted/authenticated service; multi-location, weather, and time-of-day testing; robust tracking, latency, and false-alarm measurements; production safety validation and human/operational integration.

**What can the jury observe today that did not exist in your original proposal? (60 words; maximum 100)**

> Compared with the proposal, the repository now contains a runnable local FastAPI operator service and HTML dashboard, generated video fixtures with evidence playback, incident API/status workflow and clearance re-check endpoint, ingestion/preprocessing/feature/detection/training/validation modules, reproducible dataset documentation, a detector CSV contract, a VisDrone conversion path, hardening tests, and persisted severity/accessibility artifacts. These are prototype engineering additions; they do not establish field performance.

## Section 4 — Testing and evidence

**Best measurable result:**

> On the checked-in synthetic evaluation of 4 clips/4 incident windows, severity accuracy was 0.75, macro F1 was 0.67, and weighted F1 was 0.67. Per-class F1 was Normal 1.00, Moderate 1.00, Severe 0.00, and Critical 0.67. The Severe sample was predicted as Critical; Critical→Normal misses were 0. These are smoke-test results, not a credible real-world benchmark. No defensible detector precision/recall/mAP, false-alert rate, or end-to-end alert latency is currently recorded.

**GitHub link to results/test/evidence folder:** `[PLACEHOLDER: final committed evidence folder URL]`  
Currently available paths are `equinox_deliverable_scaffold/equinox/docs/validation_report.md`, `.../tests/`, `.../data/raw/`, and `.../data/detections/detections.csv`; no dedicated evidence folder currently exists.

**Three successful cases and two failure/edge cases:**

1. `clip_01_hosur_j4`: generated stalled vehicle plus debris persists across 12 sampled frames; dashboard shows Critical and accessibility 28.
2. `clip_02_service_rd`: generated illegal-parking case; dashboard shows Severe and accessibility 45.
3. `clip_04_wipro_jct`: generated clear-traffic case; dashboard shows Normal and accessibility 95.
4. Observed evaluation error: the Severe sample is predicted as Critical, one error out of four; see `docs/validation_report.md`.
5. Tracking edge case: `--no-track` creates frame-local IDs, so dwell time cannot be interpreted across frames; see `pipeline/detection/run_yolo.py` and `docs/DATASET_CARD.md`.

**Biggest limitation:**

> The prototype is driven by four tiny generated fixtures and seeded dashboard records. It has no independently labeled real ELCIA footage, no committed detector weights, no defensible field metrics, and its occupancy/lane geometry uses placeholder approximations.

## Section 5 — Dataset

**What data did you use?**

- [ ] ELCIA/ELCITA challenge footage — not evidenced.
- [ ] Self-recorded footage — not evidenced.
- [ ] Public video/images — do not select for Open Images metadata alone.
- [x] Synthetic/generated data — checked-in fixtures.
- [ ] Staged test cases — only select if the team can substantiate staging.
- [ ] Other.
- `[VERIFY] Public dataset: select only if reporting the locally run VisDrone smoke subset.`

**Dataset names/source links:**

> Synthetic fixtures are generated by `pipeline/generate_demo_data.py` and stored under `data/raw/`. Optional local public data is VisDrone2019-DET: https://github.com/VisDrone/VisDrone-Dataset. The local VisDrone smoke subset is gitignored and is not available to a clean GitHub clone. Open Images is only a legacy metadata helper and should not be claimed as used image data.

**Approximate amount of data used:**

> Four generated 640×480, 15-fps, 4-second clips: 240 video frames total; 48 sampled frames; 60 synthetic detection rows; 4 labeled incident windows. Optional local VisDrone smoke run: 50 validation images and 2,466 source boxes.

**What data did your team personally annotate/label?**

> None. Current labels are generated fixture labels, not personally annotated real footage.

## Section 6 — Team ownership

**Team Member 1 — name and personal implementation:**

> Joshiny Maria — repository metadata confirms authorship of the current detector/VisDrone integration commit, but the exact personally implemented scope is `[VERIFY WITH TEAM MEMBER]`.

**Team Member 2 — name and personal implementation:**

> Sasmita S — listed in the proposal and repository history; exact personally implemented scope is `[VERIFY WITH TEAM MEMBER]`.

**Which part are you most confident demonstrating LIVE?**

> The local dashboard/API workflow on seeded fixtures: select an incident, play evidence video, inspect severity/accessibility/recommendation, move through status states, and run the clearance re-check. Do not claim true end-to-end YOLO-from-video confidence until detector weights and that path are reproducibly packaged.

## Section 7 — Declaration

Tick these only after the final hash/tag, public links, and spoken disclosures are tested:

- [ ] GitHub represents the current working implementation.
- [ ] Working, simulated, and future features are clearly separated.
- [ ] Results are measured, not estimated or fabricated.
- [ ] Team can run code, change parameters, show failures, and explain files.
- [ ] Video is ≤5 minutes and team member(s) speak while demonstrating the implementation.
- [ ] Passwords are not included.
