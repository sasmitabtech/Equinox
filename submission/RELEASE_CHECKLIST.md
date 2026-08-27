# Release and submission checklist

## Repository state

- [ ] Do not switch branches during finalization; decide whether the final is `main` or a clearly named release branch.
- [ ] Complete the requested author/history cleanup, including removing the reachable Ashlin commit history.
- [ ] Confirm the final tree contains the implementation, submission files, README, docs, tests, fixtures, and any explicitly promised evidence.
- [ ] Confirm no secrets, local absolute paths, private data, `.venv`, ignored model caches, or unlicensed datasets are included.
- [ ] Freeze the final tree after verification; no code/evaluation/dashboard changes after recording unless the video is re-recorded.

## Clean-clone reproducibility

- [ ] Clone the exact final repository into a clean directory.
- [ ] Follow the README setup from the documented project directory.
- [ ] Confirm the checked-in synthetic fixtures work without a network download.
- [ ] Confirm the dashboard/API starts and all four evidence MP4s play.
- [ ] Confirm the optional VisDrone path is clearly described as optional and does not imply committed data or severity validation.
- [ ] If detector inference is demonstrated, package/document the exact compatible weights source, checksum, command, and input frames; otherwise disclose that weights are external/not tracked.

## Tests and evidence

- [ ] Run `python -m compileall -q pipeline tests`.
- [ ] Run `python -m unittest discover -s tests -v` (currently verified: 14 passing tests).
- [ ] Run `python -m unittest discover -s pipeline/ingestion -p 'test_*.py' -v` (currently verified: 3 passing tests).
- [ ] Inspect `docs/validation_report.md`; preserve the Severe→Critical failure and synthetic-data caveat.
- [ ] Add/commit a dedicated evidence/results folder if the form requires a folder URL, or explicitly use the final docs/tests path.
- [ ] Do not claim detector precision, recall, mAP, false-alert rate, latency, or real-world generalization without an actual measured artifact.

## Links and final identity

- [ ] Test the public GitHub URL from an incognito/unauthenticated browser.
- [ ] Create and verify the immutable final commit hash or release tag after all edits.
- [ ] Confirm the form uses that exact post-cleanup hash/tag, not the current pre-submission main `1ea4b4774492466902bb2bfc1dfc77c501033717`.
- [ ] Deploy the dashboard only if it is safe and reproducible; test the exact public URL from a clean browser.
- [ ] Confirm the video URL is public/unlisted as intended, opens without login, and is no longer than five minutes.
- [ ] Confirm the evidence-folder URL resolves to the final hash/tag.
- [ ] Keep localhost `127.0.0.1:8000` in the runbook only as a local fallback, never as the claimed public URL.

## Dataset, licensing, and claims

- [ ] Identify generated fixtures as synthetic and state that no ELCIA/ELCITA footage is evidenced unless the team adds and documents it.
- [ ] If VisDrone is reported, retain its source URL, attribution/license review, split/limit, provenance, and checksum; do not imply that it labels severity/accessibility.
- [ ] Do not claim Open Images was used as image data merely because a legacy metadata helper exists.
- [ ] State that no team member personally annotated real footage if that remains true.
- [ ] Verify each team member’s actual contribution with them; never infer ownership from commit authorship alone.
- [ ] Keep the declaration unticked until every statement is true for the frozen release.

## Demonstration freeze

- [ ] Rehearse `submission/DEMO_RUNBOOK.md` three times and remain under 4:45.
- [ ] Speak over the actual dashboard/API actions; slides alone are not acceptable.
- [ ] Show one success sequence and the documented Severe→Critical failure.
- [ ] State seeded overlays, deterministic re-check, placeholder geometry, synthetic data, and no live drone feed.
- [ ] Make a backup recording and retain an offline local copy.
- [ ] After recording, compare the video’s UI/hash/URLs against the frozen release one last time.
