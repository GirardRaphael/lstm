---
tags: [results, comparison]
---
# 09 Model Comparison

Every model tried on the motorway dataset, scored on **the same test windows**.

![[10_model_comparison.png]]

| Model | Inputs | MAE | RMSE | MAPE | Train time |
| --- | --- | --- | --- | --- | --- |
| XGBoost - traffic + weather + calendar | 264 | **154.5** | 238.6 | 6.6% | 14s |
| XGBoost - gap-guarded windows | 24 | **157.7** | 242.4 | 6.9% | 5s |
| XGBoost - past traffic only | 24 | **173.5** | 264.0 | 7.7% | 2s |
| XGBoost - multi-horizon (+1h head) | 24 | **175.1** | 266.4 | 7.8% | 5s |
| LSTM - past traffic only | 1 | **228.8** | 326.2 | 11.4% | 1,072s |
| LSTM - gap-guarded windows | 1 | **240.6** | 332.8 | 11.6% | 369s |
| LSTM - traffic + weather + calendar | 11 | **241.1** | 349.2 | 10.5% | 204s |
| LSTM - multi-horizon (+1h head) | 1 | **244.5** | 346.5 | 11.0% | 834s |
| Naive - last hour | - | **585.6** | - | - | - |
| Naive - same hour yesterday | - | **594.3** | - | - | - |

The best result on this dataset is **XGBoost - traffic + weather + calendar** at **154.5**.

## The uncomfortable result

> Gradient boosting on the **same tensors** - the same windows, the same
> split, the same scaler, just flattened - scores **155** against the
> best LSTM's **229**. That is **32% more accurate**, trained
> **75x faster**.

This is the single most useful thing in the project, and it should be said
out loud rather than buried. It does not mean the LSTM was a mistake - it
means *this problem* does not need one. One strongly periodic series with a
24-hour window is close to the ideal case for a tree ensemble: the useful
signal is almost entirely in `t-1h`, `t-2h` and `t-24h`, and a tree can
split on those directly without learning a recurrence.

**When the LSTM would start to win:**

- dependencies longer than the input window, where lag columns run out
- many correlated series (several junctions) sharing one model
- irregular or variable-length sequences, which a fixed lag table cannot express
- learning a representation to reuse elsewhere, rather than one number

Answering *"why an LSTM?"* with **"I measured it, and for this dataset it is
not the best tool - here is the number"** is a stronger answer than any
amount of theory.


## Does the verdict hold on a second dataset?

The same code, the same architecture, the same evaluation - pointed at hourly
**bike rentals in Washington DC** instead of motorway traffic. Only two
command-line arguments changed.

| Dataset | Best LSTM | Best XGBoost | Winner | Unit |
| --- | --- | --- | --- | --- |
| Motorway traffic (UCI Metro Interstate) | 228.8 | 154.5 | **XGBoost** by 32.5% | vehicles per hour |
| Bike rentals (UCI Bike Sharing) | 38.5 | 38.6 | **LSTM** by 0.3% | rentals per hour |

**The verdict flips between the two datasets.** That is the single most
useful thing this project found, and it is only visible because the same
pipeline was run on a second, unrelated series.

Anyone who concludes "gradient boosting beats LSTMs on time series" from
the traffic result alone would be wrong on the bike data, and vice versa.
The right conclusion is narrower and more useful: *this* comparison is
cheap to run, so run it on your data instead of inheriting someone else's
answer.

*MAE is in the units of the target, so the two rows must never be compared to
each other - only within a row.*

## Forecasting further ahead

![[11_horizons.png]]

| Horizon | MAE |
| --- | --- |
| +1h | 244.5 |
| +3h | 361.6 |
| +6h | 472.4 |

One model, three outputs. Accuracy decays as the horizon grows, which is
the expected and honest result: the further ahead you look, the less the
last 24 hours determine the answer.

## How the comparison is kept fair

The benchmark does not rebuild features. It takes the exact tensors the LSTM
was trained on, shape `(n, timesteps, features)`, and flattens them to
`(n, timesteps * features)`. Same rows, same chronological split, same scaler,
same targets, same validation cut. The only difference is that the tree model
cannot see the ordering of the timesteps except through column position.

Related: [[05 Results]] - [[06 Limits and Next Steps]] - [[08 Exam Questions]]
