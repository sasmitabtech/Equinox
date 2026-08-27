# Validation Report

Held-out clips: 4 | windows: 4

## Classification report
```
              precision    recall  f1-score   support

    Critical       0.50      1.00      0.67         1
    Moderate       1.00      1.00      1.00         1
      Normal       1.00      1.00      1.00         1
      Severe       0.00      0.00      0.00         1

    accuracy                           0.75         4
   macro avg       0.62      0.75      0.67         4
weighted avg       0.62      0.75      0.67         4

```

## Confusion matrix (rows=true, cols=pred; Normal/Moderate/Severe/Critical)

```
[[1 0 0 0]
 [0 1 0 0]
 [0 0 0 1]
 [0 0 0 1]]
```

**Critical incidents misclassified as Normal: 0** — this is the failure mode with real safety cost; investigate these rows first.

## TODO before submission

- [ ] Attach 5-10 qualitative example cards (frame + prediction + ground truth)

- [ ] Break down performance by location/time-of-day (plan §3.6 bias check)

- [ ] Report alert latency separately (needs frame-timestamped runs, not just this static eval)
