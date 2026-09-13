---
tags: [results, experiment]
---
# 10 Window Length Experiment

Every other comparison in this vault uses a 24-hour window. That is close to
the best case for a tree ensemble: 24 lag columns with nearly all the signal in
two or three of them. The standard argument for a recurrent model is that it
should pull ahead once the window grows, because a tree receives one column per
timestep and feature while an LSTM folds the same window through one cell.

This is that experiment. Both models see **the same bundle** at every length -
same rows, same split, same scaler - and the only thing that changes is how
many hours they are handed.

| Window | Flat columns for XGBoost | LSTM MAE | XGBoost MAE | Gap | Train time |
| --- | --- | --- | --- | --- | --- |
| 12h | 84 | 322.9 | 157.6 | **+104.9%** | 38s / 16s |
| 24h | 168 | 339.7 | 154.3 | **+120.2%** | 49s / 17s |

Gap is `(LSTM - XGBoost) / XGBoost`; negative means the LSTM is ahead.

![[12_window_sweep.png]]

**The gap does not narrow.** Lengthening the window does not help the
recurrence on this series, so the usual argument - "an LSTM compresses a
long history where a tree drowns in columns" - simply does not apply here.
That is worth stating plainly rather than quietly dropping: a prediction
was made, it was tested, and it failed.

## What this does and does not test

A longer window on this dataset mostly adds **more daily cycles**, not longer
dependencies - the structure in motorway traffic is daily, so hours 25 to 96
largely repeat information hours 1 to 24 already carried. So this measures
whether the recurrence handles a long, largely redundant window better than a
lag table does. It does **not** test genuinely long-range dependencies, which
would need a series that actually has them.

The stronger test - several correlated series sharing one model - is still
unrun. See [[06 Limits and Next Steps]].

Related: [[09 Model Comparison]] - [[05 Results]]
