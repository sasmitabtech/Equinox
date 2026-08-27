# Equinox: State Export (2026-08-27)

## Session / model
- Previous orchestration ran in Codex with delegated Luna/Terra workers. That session's agents are **not reachable** from this session; their work survives only as files on disk (listed below).
- Current session: Claude Code, model Claude Fable 5. `/model luna` is not a valid model name here (options: fable, opus, sonnet, haiku). Nothing changed.
- Active agents in this session: none besides the main model.

## Git
- Repo: `sasmitabtech/Equinox` (JoshinyMaria has WRITE, invitation accepted).
- Branch: `feature/operations-console-redesign` (local only, not pushed), based on `origin/main` @ `1ea4b47`.
- `main` history rewritten: author/committer on the implementation commit is Joshiny Maria (`afc753b`). Merge commit `1ea4b47` by Sasmita S. No Ashlin/Antony references in reachable history.
- PR #1 merged.

## Uncommitted work on this branch
Modified: `.gitignore`, `README.md`, `pipeline/deployment/api.py` (serves Vite dist, dash-free copy), 4 raw MP4s re-encoded to H.264/yuv420p.
New:
- `dashboard/frontend/` React + Vite + Tailwind + shadcn (Button, Badge, Card, Separator, Skeleton, Alert, Tooltip, Tabs), Kokonut carousel + loader, Magic UI magic-card, Source Serif 4 display + Geist UI. Built `dist/` present.
- `pipeline/evaluation/` + `tests/test_visdrone_smoke_evaluation.py`
- `evidence/visdrone_smoke_yolo11n/` (metrics.json, REPORT.md, per_class_metrics.csv, error_summary.csv, 5 annotated images: 3 success, 2 failure)
- `submission/` DEMO_RUNBOOK.md, FORM_DRAFT.md, RELEASE_CHECKLIST.md
- `PRODUCT.md`, `.21st/` (design context)

## Measured results (evidence/visdrone_smoke_yolo11n/metrics.json)
- YOLO11n COCO-pretrained, 50 VisDrone val images, conf 0.25, IoU 0.5, mapped classes only
- Precision 0.773 / Recall 0.196 / F1 0.313 / mAP@0.5 0.181
- 11.057 s wall for 50 images (~221 ms/img CPU); 612 raw detections
- Dominant error: 1671 missed small objects (false_negative_missed)

## Runtime
- Dashboard: FastAPI on http://127.0.0.1:8002 (restarted this session). 8000/8001 are retired.
- Tests: `python -m unittest discover -s tests` passes (22 tests) in this session. pytest is not installed in .venv; unittest is the runner.

## Open items before submission
1. ~~Mobile clip fix~~ verified in this session: no horizontal overflow at 500px, register and evidence stack cleanly.
2. Commit branch + push + PR to main.
3. Cloudflare Pages deploy (static dashboard + Pages Functions for demo API) -> public URL.
4. 5-minute spoken demo video (user).
5. Fill FORM_DRAFT placeholders: video URL, app URL, evidence URL, release tag, team ownership text.
6. Clean-clone rehearsal, release tag, freeze.
