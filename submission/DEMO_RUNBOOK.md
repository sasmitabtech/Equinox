# Equinox finalist demonstration runbook

Target recording length: **4:45 maximum**. A team member must speak while showing the working implementation. Record the local run offline; replace each bracketed public placeholder only after it has been tested from a clean browser/session.

## Before recording

From `equinox_deliverable_scaffold/equinox`, install `requirements.txt`, then run:

```bash
uvicorn pipeline.deployment.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Keep a second terminal ready for `curl http://127.0.0.1:8000/health`. Have the GitHub repository at `[FINAL_GITHUB_URL]` and the final evidence folder at `[FINAL_EVIDENCE_URL]` ready in a separate tab. Do not present localhost as a public application URL.

## Timed spoken sequence

| Time | Screen action | Spoken explanation |
|---|---|---|
| 0:00–0:25 | Show the dashboard and repository README. | “Equinox helps an emergency-corridor operator determine whether a blockage leaves an emergency vehicle enough clear passage. This submission is a local demo-scale prototype using recorded footage.” |
| 0:25–0:55 | Run/show `/health`; return to the dashboard. | “The FastAPI service is running locally. The service exposes incident data, evidence video, status updates, and a clearance re-check.” |
| 0:55–1:45 | Select `INC-001`, play the evidence video, point to the stalled-vehicle/debris overlays, Critical severity, score 28, and contributing factors. | “This synthetic fixture represents a stalled vehicle and debris. The card displays the evidence clip, severity, accessibility score, and recommended action.” |
| 1:45–2:25 | Click `Assign to Me`, then `Suggest Diversion`/`In Action`; show the pipeline steps and changed status. | “The operator workflow records assignment and action status. These are API-backed state transitions in the local demo.” |
| 2:25–3:00 | Click `Mark Cleared`, then `Verify Clearance Scan`; show Normal/98 and Verified. | “The clearance scan completes the demonstrated workflow. The re-check behavior is deterministic simulation, not a new camera inference.” |
| 3:00–3:35 | Select `INC-002` and `INC-004`; show the Severe and Normal cases and their videos. | “The other fixtures demonstrate illegal parking/severe and clear traffic/normal cases. All four supplied clips are generated synthetic fixtures, not ELCIA/ELCITA footage.” |
| 3:35–4:10 | Show `docs/validation_report.md`, then `data/detections/detections.csv` and the tests folder in GitHub. | “The checked-in smoke evaluation has four windows: accuracy 0.75 and macro F1 0.67. It is not a field benchmark. The report records the Severe-to-Critical error, and the tests cover pipeline hardening.” |
| 4:10–4:45 | Show `pipeline/detection/run_yolo.py` and, only if actually prepared, run the tested detector command on prepared image frames with supplied weights. | “The real YOLO integration seam consumes image frames and writes the detector CSV contract. The checked-in dashboard records and overlays remain seeded; detector weights and live drone ingestion are not claimed unless the exact run is reproducible here.” |

## Required disclosures

Say these plainly: the four videos and labels are generated; dashboard incidents and overlay coordinates/confidences are seeded; clearance re-check changes values by deterministic rules; corridor geometry is placeholder/approximate; no live drone feed, hosted service, or safety-certified deployment is claimed; no detector precision/recall/mAP or alert-latency result is claimed.

## Failure case to show

Open `docs/validation_report.md` and point out that the Severe sample was predicted as Critical (one error in four synthetic windows). Also explain that `--no-track` creates frame-local IDs, so dwell time is not valid across frames. Do not describe the four-window result as real-world validation.

## Recording contingency checklist

- [ ] Rehearse the full sequence three times under 4:45.
- [ ] Test the local API from a clean browser and confirm all four MP4s play.
- [ ] Clear `equinox-incidents` localStorage before recording, or use a fresh browser profile.
- [ ] Keep a backup recording of the same spoken demo.
- [ ] Keep the final repository URL, commit/tag, evidence URL, and public app URL visible in notes.
- [ ] If YOLO inference fails or weights are unavailable, omit the live inference action and state the integration limitation; never substitute a fabricated metric.
