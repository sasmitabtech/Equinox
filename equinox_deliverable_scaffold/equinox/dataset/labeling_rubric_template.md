# Labeling Rubric — Equinox (fill in before labeling begins)

Both team members must agree on this rubric in one sitting (plan §3.3) before labeling any frames. Double-label a 10% sample independently afterward and check agreement.

## Detection classes (bounding box)
| Class | Definition | Include / Exclude notes |
|---|---|---|
| `vehicle_stalled` | A vehicle stationary in a traffic lane, not at a signal/junction stop | Exclude vehicles stopped <5s (likely a signal stop) |
| `vehicle_parked_illegal` | A vehicle parked in a no-parking zone or reducing usable lane width | — |
| `debris` | Any object obstructing the road that is not a vehicle | e.g. fallen branch, cargo spill |
| _(add more as needed)_ | | |

## Severity levels (incident-window label)
Define the **explicit, measurable** threshold for each — not a vibe:

| Severity | Definition (fill in numbers your team agrees on) |
|---|---|
| Normal | Road remains passable; clear lane width ≥ ___ m |
| Moderate | Some restriction; clear lane width between ___ and ___ m |
| Severe | Significant blockage; clear lane width between ___ and ___ m for > ___ s |
| Critical | Clear lane width < ___ m across full corridor for > ___ s |

## Accessibility score (0-100)
Describe how the numeric score is derived from severity + factors (occupancy %, dwell time, queue length) so it's reproducible, not eyeballed per-clip.

## Inter-labeler agreement check
- Sample size double-labeled: ___ windows (target ≥10% of total)
- Agreement method: % exact match / Cohen's kappa
- Result: ___
- If agreement is low: revise the thresholds above and re-label the disagreement cases before proceeding.

## Known ambiguous cases (log as you go)
Keep a running list here of edge cases you disagreed on and how you resolved them — this becomes evidence of labeling rigor for the dataset card.
