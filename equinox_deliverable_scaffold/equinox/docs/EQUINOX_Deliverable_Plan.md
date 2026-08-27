# Equinox — Deliverable Execution Plan
### Emergency Corridor & Road Blockage Intelligence — Track 1: Smart Mobility & Road Incident Intelligence

This plan turns the Track 1 proposal into three concrete, gradeable deliverables: **(1) an AI/ML pipeline, (2) a working prototype, (3) a trained model + dataset**. It is written to be handed straight to the two of you as a shared execution reference.

---

## 1. AI/ML Pipeline — Scope, Stages, and Contracts

**Purpose of the pipeline:** turn recorded drone footage into a per-incident record (class, severity, emergency accessibility score, recommended action) with evidence attached, reliably enough to demo live and defend under judge questioning.

Each stage below is defined as a **contract**: what goes in, what comes out, and how you know it worked — so any stage can be built, tested, and swapped independently.

### 1.1 Data Ingestion
- **Input:** Raw drone MP4/MOV clips + a manifest (location name/GPS, timestamp, area, weather note).
- **Process:** Standardize filenames, extract metadata, register each clip in a lightweight index (SQLite or CSV) keyed by `clip_id`.
- **Output:** `data/raw/{clip_id}.mp4` + `manifest.csv` (clip_id, location, lat, lon, timestamp, fps, duration, source).
- **Success criteria:** Every clip has complete metadata; no orphaned files; ingestion script re-runnable without duplicating entries (idempotent).

### 1.2 Preprocessing
- **Input:** Raw clips from ingestion.
- **Process:** Frame extraction at a fixed sampling rate (e.g., 2–5 fps — enough for tracking, light enough to label), resolution normalization, stabilization/de-jitter if the drone footage has camera motion, road/lane mask overlay (manual polygon per fixed location, or homography if drone angle varies).
- **Output:** `data/frames/{clip_id}/{frame_id}.jpg` + `masks/{location_id}.png`.
- **Success criteria:** Frame extraction rate matches spec (±5%); masks visually verified for each unique vantage point; no corrupted frames.

### 1.3 Feature Engineering
- **Input:** Extracted frames + masks + object detections (from the model, see 1.4).
- **Process:** Derive the features severity logic actually needs — not raw pixels: per-frame occupied-road-fraction, clear-lane-width estimate, obstruction dwell time (frames persisted), obstruction position relative to corridor mask, queue length proxy (count of stationary vehicles in a zone).
- **Output:** `features/{clip_id}.parquet` — one row per tracked-object-per-frame, plus a rolled-up per-incident feature vector.
- **Success criteria:** Feature values are stable across consecutive frames for a static scene (no noisy flicker); features correlate visibly with your own manual severity labels on a held-out sample.

### 1.4 Model Training
- **Input:** Labeled frames (bounding boxes + class) for detection; labeled incident windows (severity, accessibility) for the downstream classifier.
- **Process:** Two-model split (recommended, see §4):
  1. **Detector/tracker** — fine-tune a YOLO-family model on your vehicle/obstruction/debris classes; track IDs across frames (ByteTrack or similar).
  2. **Severity/accessibility model** — a small classifier (gradient-boosted trees or a shallow MLP) on the engineered features from 1.3, mapped to Normal/Moderate/Severe/Critical and a 0–100 accessibility score. This is deliberately simple and interpretable — you can defend it to judges as "rule-informed ML," not a black box.
- **Output:** `models/detector.pt`, `models/severity_model.pkl`, a training log with metrics per epoch/fold.
- **Success criteria:** Detector meets a minimum mAP@0.5 threshold you set (see §7) on a held-out val set; severity model's confusion matrix shows no catastrophic Critical↔Normal confusion.

### 1.5 Validation
- **Input:** Held-out clips never seen during training (ideally from a different time-of-day or location than training data).
- **Process:** Run the full pipeline end-to-end, compare predicted incident/severity/accessibility against your ground-truth labels; compute precision/recall/F1 for detection, accuracy/F1 for severity, and alert latency (time from incident onset in footage to alert generation).
- **Output:** `reports/validation_report.md` with metrics, confusion matrices, and 5–10 qualitative example cards (frame + prediction + ground truth).
- **Success criteria:** Meets thresholds defined in §7; failure cases are categorized (occlusion, lighting, shadow-as-vehicle, etc.) — this categorization is itself judge-visible evidence of rigor.

### 1.6 Deployment (Demo-Scale)
- **Input:** Trained models + validated pipeline.
- **Process:** Wrap inference in a simple service (FastAPI or Flask) that the dashboard calls; batch-process a recorded clip and stream results to the UI as if live (simulate real-time by replaying frames on a timer).
- **Output:** A running local (or lightly hosted) service + dashboard reachable by URL for the demo.
- **Success criteria:** End-to-end demo runs without manual intervention from "load video" to "incident verified" in under the judging session's time budget; survives a live restart.

### 1.7 How the Stages Connect

```
Ingestion → Preprocessing → [Detector/Tracker] → Feature Engineering
    → [Severity/Accessibility Model] → Validation → Deployment (API + Dashboard)
```

Treat `manifest.csv`, `features/*.parquet`, and `models/*` as the stable interfaces between stages — this lets one person work on the detector while the other builds the dashboard against mocked feature output, in parallel.

---

## 2. Prototype Requirements

### 2.1 What it must demonstrate
1. Loading a recorded drone clip and running it through the pipeline "live" (simulated real-time).
2. Detecting and classifying at least 3 of the 5 incident types from the proposal (stalled vehicle, illegal parking, debris, congestion, lane/junction blockage) — pick the 3 most reliably detectable given your data.
3. Producing severity + emergency accessibility score with visual evidence (bounding boxes on the frame).
4. Operator workflow: alert appears → operator assigns/acts → status changes → system re-checks → marks Verified.
5. A dashboard showing incident cards with all fields from the proposal's output table.

### 2.2 Interaction Method
A **web dashboard** (not a CLI or notebook) — judges relate to something that looks like an operations tool. Two panels:
- **Map/feed view:** location pins colored by severity, click to see live evidence snapshot.
- **Incident queue:** sortable/filterable list of incidents with status pipeline (Open → Assigned → In Action → Cleared → Verified), one-click "Assign to me" / "Mark cleared" buttons.

### 2.3 MVP vs. Optional Enhancements

| MVP (must-have for demo) | Optional (strengthens competitive position) |
|---|---|
| Load a recorded clip, run detection + severity end-to-end | Live/streaming ingestion from a second camera angle |
| Incident card with class, severity, accessibility score, snapshot | Heatmap of historical blockage hotspots over time |
| Manual status transitions (Open→Assigned→...→Verified) | Auto re-check + auto-verify on next frame pass |
| One working location with a hand-drawn road mask | Multi-location support with per-location masks |
| Static accessibility score (0–100) with one-line rationale | Route-suggestion for diversion (shortest clear alternate) |
| Basic auth-free dashboard | Role-based operator login, audit log of actions |
| Confusion-matrix-backed metrics slide | Jetson edge-deployment demo (even a mocked one) |

Build the left column first, completely, before touching the right column. A polished MVP beats a broken stretch feature every time in judging.

---

## 3. Dataset Requirements

### 3.1 Sizing (realistic for a hackathon-scale build)
- **Detection/tracking:** 1,500–3,000 labeled frames minimum, drawn from at least 6–10 distinct clips across 2–3 locations/vantage points and 2+ lighting conditions (midday, overcast/dusk). More frames from fewer clips overfits to one background — diversity of *scene*, not just frame count, is what a judge will probe.
- **Severity/accessibility model:** 150–300 labeled incident windows (a window = a short span of frames representing one incident instance) is enough for a simple tabular classifier if your features (§1.3) are well-designed. This is the more feasible dataset given time constraints — lean on it.
- If you cannot source enough real drone footage, supplement with public traffic-camera or dashcam footage reframed to an aerial-like crop, clearly labeled as **synthetic/supplementary** in your documentation — judges respect honesty about data provenance far more than an inflated "all real" claim.

### 3.2 Data Quality Standards
- Minimum resolution: 720p source (so downsampled frames still resolve a car vs. a shadow).
- No duplicate frames (dedupe near-identical consecutive frames if sampling rate is high).
- Every frame used for training has an unambiguous mask/context (you know which road/lane it belongs to).
- Class balance check: don't let 90% of your frames be "Normal" — actively oversample or specifically shoot/collect blockage scenarios.

### 3.3 Labeling Methodology
- **Detection labels:** bounding box + class (vehicle-stalled, vehicle-parked-illegal, debris, pedestrian-if-relevant) using CVAT, Roboflow, or Label Studio (all free-tier friendly, exportable to YOLO format).
- **Severity/incident labels:** a rubric, not vibes. Write down explicit criteria per severity level (e.g., "Critical = <1.5m clear lane width across full corridor for >10 consecutive seconds") *before* labeling, so both team members label consistently. Store the rubric in `docs/labeling_rubric.md`.
- **Inter-labeler agreement:** since there are two of you, double-label a 10% sample independently and compute agreement (simple % match, or Cohen's kappa if you want to show rigor). Low agreement = fix the rubric before labeling the rest.

### 3.4 Data Collection Strategy
- Prioritize 2–3 fixed drone vantage points you can re-shoot/re-use across multiple scenario types (stalled vehicle staged, illegal parking staged, natural congestion footage) — controlled restaging is legitimate and lets you guarantee positive examples exist.
- If actual drone flights are restricted (permissions), use existing public aerial traffic footage (YouTube traffic cams, open datasets like UAVDT, VisDrone) for the detector's general vehicle-detection pretraining, then fine-tune/calibrate severity logic on your own smaller staged/local set. Document this clearly — it's a legitimate, common strategy, not a shortcut to hide.

### 3.5 Augmentation
- Standard: horizontal flip (careful — check lane-direction semantics still make sense), brightness/contrast jitter (simulate lighting change), slight rotation (drone angle variance), motion blur (simulate camera shake), synthetic occlusion patches (simulate one vehicle blocking another).
- Avoid augmentations that would break your road-mask alignment (e.g., don't randomly crop without also transforming the mask).

### 3.6 Bias Detection & Mitigation
- Check performance split by: time-of-day, vantage point/location, vehicle type/size (two-wheelers are common in Indian traffic and easy for a detector to miss or misclassify — explicitly test this).
- If one location dominates your training data, expect and disclose that generalization to new locations is unproven — this is a real limitation, state it in your risks section rather than let a judge catch it.

### 3.7 Documentation for Reproducibility
Maintain `docs/DATASET_CARD.md` with: source of each clip, collection date/method, labeling rubric version, class distribution, known gaps, train/val/test split logic (split by *clip*, not by frame, to avoid leakage — frames from the same clip are highly correlated).

---

## 4. Technical Stack

| Component | Primary choice | Alternative | Trade-off |
|---|---|---|---|
| Detection model | YOLOv8/YOLOv11 (Ultralytics) | YOLO-NAS, RT-DETR | Ultralytics has the best docs/community support for a time-boxed build; RT-DETR can be more accurate but heavier to tune |
| Tracking | ByteTrack (built into Ultralytics) | DeepSORT | ByteTrack is simpler to wire up and fast; DeepSORT gives more robust re-ID but adds a dependency and complexity |
| Severity/accessibility model | scikit-learn (GradientBoosting/RandomForest) on engineered features | Small PyTorch MLP | sklearn is faster to train/debug/explain to judges with limited data; a NN needs more data than you'll realistically have |
| Frame/video processing | OpenCV | FFmpeg + PyAV | OpenCV is the default, well documented; FFmpeg is faster for bulk extraction if you have many long clips |
| Labeling tool | Roboflow (free tier, exports YOLO format directly) | CVAT (self-hosted), Label Studio | Roboflow is fastest to get going with zero setup; CVAT gives more control but costs setup time you may not have |
| Backend/API | FastAPI | Flask | FastAPI gives free request validation + auto docs (nice for judge Q&A on "how does the API work"); Flask is marginally simpler if team is more familiar with it |
| Dashboard | React + Tailwind (or plain HTML/JS if time-constrained) | Streamlit | React gives you a genuinely operator-tool-looking UI, which matters for judging "prototype quality"; Streamlit is dramatically faster to build but reads as "a script with a UI," which can undersell the work |
| Model serving | Local FastAPI service, no cloud needed for demo | AWS/GCP hosted endpoint | Local keeps costs at zero and avoids network dependency during judging (critical — never demo over unreliable venue wifi if avoidable); cloud is only worth it if you need to show "production readiness" and have budget/time |
| Storage | Local filesystem + SQLite | Cloud storage (S3) + Postgres | Local is zero-cost, zero-latency for a demo; cloud only matters if judges specifically weight "scalability architecture," in which case *describe* the cloud path in docs without necessarily building it |
| Edge target (future) | NVIDIA Jetson (as proposed) | Coral TPU, mobile inference (TensorRT/ONNX) | Jetson matches your proposal's stated direction; only worth prototyping if you have hardware access — otherwise keep it as a documented "future deployment target," not a build item |

**Team-expertise note:** pick the row in each trade-off that matches what you already know, not what looks most impressive on paper — a working sklearn model beats a half-finished neural net every time in a judged demo.

---

## 5. Timeline & Milestones

Assume a **3-week sprint** to a submission deadline (adjust dates to your actual deadline; the dependency structure is what matters).

| Week | Focus | Key deliverables | Dependencies |
|---|---|---|---|
| **Week 1 — Foundation** | Data collection/sourcing, labeling rubric, pipeline skeleton | Manifest + raw clips ingested; labeling rubric finalized; 30% of frames labeled; ingestion+preprocessing code working end-to-end on dummy data | None — can start immediately |
| **Week 1–2 overlap** | Detector training v1 | First YOLO fine-tune run, rough mAP baseline; dashboard UI skeleton (static, no live data) built in parallel by teammate 2 | Needs ~50%+ frames labeled |
| **Week 2 — Core build** | Severity model + feature engineering; API wiring; finish labeling | Feature pipeline producing stable features; severity classifier v1 trained; FastAPI serving detector+severity end-to-end; dashboard connected to live API with mock/partial data | Needs detector v1 + labeled incident windows |
| **Week 2–3 overlap** | Validation + iteration | Run held-out validation, fix worst failure modes (re-label, re-augment, adjust thresholds); polish dashboard interactions (status transitions, evidence display) | Needs full pipeline connected |
| **Week 3 — Integration & polish** | Full end-to-end demo rehearsal, documentation, deck | Full demo script rehearsed 3+ times end-to-end; validation report + dataset card finalized; slide deck + README done | Needs everything above working |
| **Final 2–3 days — Buffer** | Bug fixes only, no new features | Freeze features; only fix crashes/regressions; record a backup demo video in case of live-demo failure | — |

**Hard rule:** no new features after the "Integration & polish" milestone begins — that time is protected for making the existing MVP unbreakable, since a crash during judging costs more than a missing feature.

---

## 6. Risks & Mitigations

| Risk | Deliverable affected | Mitigation |
|---|---|---|
| Not enough real drone footage / flight permission delays | Dataset, Model | Start public-dataset pretraining (VisDrone/UAVDT) Week 1 so you're not blocked; treat your own footage as fine-tuning/validation data, not the sole source |
| Labeling takes longer than planned (it always does) | Dataset, Model | Time-box labeling sessions; prioritize labeling the severity/incident windows (smaller n, higher leverage) over exhaustive frame-level boxes |
| Detector accuracy is mediocre on your specific scenes | Model, Prototype | Have a fallback: hand-tuned heuristic thresholds (occupancy %, dwell time) can substitute for a weak classifier in the demo while you keep improving the model — be transparent about this in docs, don't hide it |
| Two-person team, both blocked on the same critical path | All three | Split by interface contracts (§1.7) so one person can build/test against mocked outputs while the other works upstream; agree on file formats (`manifest.csv`, `features/*.parquet`) on Day 1 so no one waits on the other |
| Dashboard looks unfinished / generic | Prototype | Budget explicit UI polish time in Week 3, not as an afterthought; a clean, purpose-built operator UI is one of the highest-visibility judging signals |
| Live demo fails (wifi, crash, camera) | Prototype | Always have a pre-recorded backup demo video plus a fully offline local version that needs no network |
| Overclaiming results ("95% accuracy") without real held-out validation | Model, credibility with judges | Report metrics only from a genuinely held-out split; be explicit about sample size — small-n metrics with honest caveats read as more credible than suspiciously perfect numbers |
| Scope creep toward optional enhancements before MVP is solid | All three | Enforce the MVP/optional split in §2.3; review weekly against it |

---

## 7. Evaluation Metrics & Validation Approach

Match judging criteria to concrete, computed numbers you can show on a slide:

- **Detection:** Precision, Recall, F1, mAP@0.5 (and mAP@0.5:0.95 if time allows) — target a self-set bar like mAP@0.5 ≥ 0.6 for a hackathon-scale dataset (state it as *your* bar, don't compare to production benchmarks judges won't expect you to hit).
- **Tracking:** ID-switch rate (qualitative is fine if you don't have time for MOTA) — show a short clip with consistent track IDs across frames as evidence.
- **Severity/accessibility classification:** Accuracy, per-class F1, confusion matrix (this is where you can show real rigor with a modest dataset).
- **System-level:** False-positive rate on "Normal traffic" clips (per your own testing plan table), alert latency (seconds from incident visibility to alert), end-to-end demo success rate across repeated runs (rehearse this — "runs cleanly 5/5 times" is a credible, judge-legible claim).
- **Robustness/generalization:** Report performance separately on a held-out *location* not seen in training, if you have one — even a small held-out set demonstrating the model isn't purely memorizing one background is disproportionately convincing to judges.
- **Real-world applicability framing:** Tie every metric back to the operator workflow — "at this false-positive rate, an operator sees roughly X alerts/hour, of which Y need action" is more persuasive than a bare F1 score.

---

## 8. Documentation & Presentation Strategy

### 8.1 Written documentation (leave-behind for judges)
- `README.md` — one-page: problem, approach, how to run the demo locally, architecture diagram.
- `docs/DATASET_CARD.md` — per §3.7.
- `docs/validation_report.md` — metrics, confusion matrices, and 5–10 example cards (correct + failure cases side by side — showing failure cases *builds* credibility, it doesn't undercut it).
- `docs/ARCHITECTURE.md` — the pipeline diagram from §1.7, plus the tech-stack table from §4 with your actual final choices and *why*.

### 8.2 Live presentation structure (for mixed technical/non-technical judges)
1. **Open with the human stakes, not the tech** (15–30 sec): one sentence on why "the road looks clear but isn't" costs lives — this is already in your proposal, use it verbatim in spirit.
2. **Live demo first, architecture second.** Judges remember what they saw work. Run the dashboard end-to-end (load clip → alert → operator action → verified) before you show a single architecture slide.
3. **One architecture slide**, using the §1.7 pipeline diagram — keep it to boxes and arrows, not a wall of tool names.
4. **One metrics slide**, framed per §7's "real-world applicability" note — a judge without an ML background should still get the "so what."
5. **One "what we'd build next" slide** — mention 2–3 items from your Optional Enhancements list (§2.3) to show you know the difference between demo-scope and production-scope, which itself signals maturity.
6. **Close on the operator, not the algorithm** — end by reiterating who uses this and what decision it changes for them.

### 8.3 UI/UX presentation notes
- Keep the incident-severity color coding consistent everywhere (dashboard, evidence snapshots, slide deck) — Normal/Moderate/Severe/Critical should always map to the same 4 colors, so judges pattern-match instantly without you explaining the legend twice.
- Show the evidence snapshot *inside* the incident card, not in a separate tab — judges shouldn't have to hunt for "proof" the detection is real.
- If using the provided dashboard mockup (see accompanying HTML file) as your visual reference, adapt the copy to whatever specific location/scenario you actually demo, so the UI text matches your real demo data rather than placeholder text.

---

## Immediate Next Actions (This Week)
1. Finalize the labeling rubric (§3.3) — do this together, in one sitting, before either of you labels a single frame.
2. Agree on the file-format contracts in §1.7 so you can split work in parallel starting tomorrow.
3. Start sourcing/shooting footage and simultaneously kick off a pretraining run on a public aerial-vehicle dataset so the detector isn't blocked on your own data collection.
4. Stand up the dashboard skeleton against mocked JSON so UI work isn't blocked on the model being ready.
