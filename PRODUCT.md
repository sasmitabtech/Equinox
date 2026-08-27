# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user: an emergency-corridor operator monitoring road access during an incident. Their job is to inspect evidence, understand the declared incident state, coordinate the next response step, and verify clearance.

Secondary audience: ELCIA Smart City Drone-AI Challenge jury members evaluating a working prototype during a live demonstration. This evaluation context is inferred from the submission brief.

## Product Purpose

Equinox is a prototype operator console for reviewing road-corridor incidents and managing their response status. It makes the current evidence, accessibility score, recommended action, and workflow state visible in one place.

## Positioning

Equinox combines a runnable local incident-management workflow with a separately demonstrable aerial-object-detection pipeline, while explicitly distinguishing seeded prototype intelligence from live model output.

## Operating Context

The operator uses a desktop or tablet browser during an incident review. The jury may ask for a live walkthrough, a status change, a clearance re-check, or an explanation of data provenance.

## Capabilities and Constraints

- Confirmed: FastAPI provides health, incidents, incident status updates, clearance re-checks, and local MP4 evidence routes.
- Confirmed: a React/Vite dashboard consumes those endpoints.
- Confirmed: incident records and clearance re-check outcomes are seeded/simulated in the current prototype; this must be obvious in the UI.
- Confirmed: a local YOLO/VisDrone smoke path exists but does not generate the dashboard records.
- Confirmed: Cloudflare Pages is a future deployment target, not a current deployment claim.
- Inferred: the public deployment needs a configurable API base URL so the frontend can later point at a Worker/Pages Function.

## Brand Commitments

Equinox names an emergency-corridor monitoring product. The user has explicitly requested an editorial operational SaaS visual language: warm off-white canvas, charcoal type, hairline borders, flat asymmetrical grid, desaturated semantic pastels, and no gradients, glass, neon, or fake technical chrome.

## Evidence on Hand

- Four generated MP4 fixtures in `equinox_deliverable_scaffold/equinox/data/raw/`.
- Seeded incident records in `equinox_deliverable_scaffold/equinox/pipeline/deployment/api.py`.
- Local VisDrone preparation and YOLO inference utilities, documented in the repository.
- No public dashboard deployment, real incident footage, or end-to-end model-generated dashboard alerts are available today.

## Product Principles

1. Make the next operational decision visible before secondary detail.
2. Evidence is primary; simulation must never be disguised as live AI output.
3. Keep response state changes explicit, recoverable, and observable.
4. Preserve a credible live-demo path on a modest local machine.

## Accessibility & Inclusion

Keyboard-operable controls, visible focus states, readable contrast, and non-color-only status labels are required for the dashboard.
