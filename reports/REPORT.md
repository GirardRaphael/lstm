# Traffic volume forecasting with an LSTM

*Generated from the stored run artifacts on 2026-09-12 14:02. Rerun
`python scripts/build_report.py` after any retrain.*

---

## 1. The problem

Motorway traffic follows a strong daily cycle: a morning peak, an evening
peak, an overnight trough. A control system that only reacts to congestion is
always a step behind one that can see the next hour coming.

**Question.** Given the last 24 hours of traffic volume, how many vehicles
will pass during the next hour?

| | |
| --- | --- |
| Learning type | Supervised |
| Task | Regression - the output is a count of vehicles |
| Algorithm | Stacked LSTM (64 -> 32) |
| Metric | MAE, in vehicles per hour |

To be precise about the claim: this model does not control traffic lights. It
produces a forecast a controller could consume.

## 2. The data

**Metro Interstate Traffic Volume**, UCI Machine Learning Repository - hourly
westbound traffic on I-94 between Minneapolis and Saint Paul.

| | |
| --- | --- |
| Rows as delivered | 48,204 |
| Unique hours after cleaning | 40,575 |
| Period | 2012-10-02 to 2018-09-30 |
| Target range | 0 to 7,280 vehicles (mean 3,260) |
| Duplicate timestamps merged | 7,629 |
| Hours missing entirely | 11,976 |

Two imperfections were handled explicitly rather than silently. **Duplicate
timestamps** (the same hour logged twice with different weather text) are
collapsed by averaging the target - neither inventing data nor keeping two
conflicting rows. **Missing hours** are *not* interpolated; fabricating
traffic volumes would flatter every metric that follows.

## 3. Method

```
raw CSV -> sort chronologically -> collapse duplicate hours
        -> chronological split 80/20
        -> MinMaxScaler fitted on TRAIN ONLY
        -> sliding windows of 24 hours
        -> (samples, 24, features)
```

Three decisions carry the credibility of the result:

1. **The split is chronological.** Shuffling would put future hours in the
   training set. The score would look excellent and mean nothing, because in
   production only the past is available.
2. **The scaler is fitted on the training slice only.** Had it seen the test
   set, the model would indirectly know the range of the future - data
   leakage.
3. **The test set keeps its history.** The first test window needs 24 hours
   of context, taken from the end of the training set. That is history, not
   leakage: the model only ever looks backwards.

`tests/test_pipeline.py` verifies all three, so the claims are checkable
rather than asserted.

## 4. Model

```
(24, F) -> LSTM 64 (return_sequences) -> Dropout 0.2
           -> LSTM 32                    -> Dropout 0.2
           -> Dense 16 relu -> Dense 1 linear
```

The final layer has no activation: the output is a count of vehicles, not a
probability. Overfitting is controlled by dropout, a validation split watched
every epoch, and EarlyStopping with the best weights restored.

## 5. Results

Trained for 49 epochs (1,072s) on 32,436 sequences; evaluated
on 8,115 hours the model never saw.

| Metric | Value |
| --- | --- |
| MAE | **228.8 vehicles per hour** |
| RMSE | 326.2 |
| MAPE | 11.4% |

A MAE means nothing on its own, so two naive baselines were scored on the same
windows:

| Predictor | MAE |
| --- | --- |
| LSTM | **228.8** |
| "the same hour yesterday" | 594.3 |
| "the last hour" | 585.6 |

The network is **+60.9%** better than the stronger baseline. Had it not
beaten both, the honest conclusion would have been that an LSTM is the wrong
tool for this problem.

## 6. Is an LSTM the right tool here?

The honest way to answer "why an LSTM?" is to measure the alternative rather
than argue for it. XGBoost was trained on **exactly the same tensors** - the
same windows, the same chronological split, the same scaler - simply flattened
from `(n, 24, F)` to `(n, 24*F)`. The only thing the tree model loses is
the ordering of the timesteps.

| | MAE | RMSE | Training time |
| --- | --- | --- | --- |
| LSTM | 228.8 | 326.2 | 1,072s |
| XGBoost | 177.3 | 268.8 | 10s |

**XGBoost wins, by 22%, and XGBoost trains 113x faster.**

This does not make the LSTM a mistake; it makes it the wrong tool *for this
problem*. A single strongly periodic series with a 24-hour window is close to
the ideal case for a tree ensemble: nearly all the signal sits in `t-1h`,
`t-2h` and `t-24h`, and a tree can split on those columns directly instead of
learning a recurrence. An LSTM starts to pay for itself when dependencies run
longer than the input window, when many correlated series share one model,
when sequence lengths vary, or when the learned representation is reused.

## 7. Every variant tried

| Model | Inputs | MAE | Training time |
| --- | --- | --- | --- |
| XGBoost - traffic + calendar (no weather) | 168 | 154.3 | 14s |
| XGBoost - traffic + weather + calendar | 264 | 154.5 | 14s |
| XGBoost - all 11 inputs, dropout 0.35 | 264 | 154.5 | 23s |
| XGBoost - gap-guarded windows | 24 | 157.7 | 5s |
| XGBoost - multi-horizon (+1h head) | 24 | 175.1 | 5s |
| XGBoost - past traffic only, dropout 0.35 | 24 | 177.3 | 8s |
| XGBoost - past traffic only | 24 | 177.3 | 10s |
| LSTM - traffic + calendar (no weather) | 7 | 201.4 | 688s |
| LSTM - all 11 inputs, dropout 0.35 | 11 | 206.0 | 723s |
| LSTM - past traffic only | 1 | 228.8 | 1,072s |
| LSTM - past traffic only, dropout 0.35 | 1 | 229.8 | 630s |
| LSTM - gap-guarded windows | 1 | 240.6 | 369s |
| LSTM - traffic + weather + calendar | 11 | 241.1 | 204s |
| LSTM - multi-horizon (+1h head) | 1 | 244.5 | 834s |
| Naive - last hour | - | 585.6 | - |
| Naive - same hour yesterday | - | 594.3 | - |
| LSTM - rentals + weather + calendar | 11 | 38.5 | 235s |
| XGBoost - rentals + weather + calendar | 264 | 38.6 | 8s |
| LSTM - past rentals only | 1 | 41.3 | 154s |
| XGBoost - past rentals only | 24 | 43.1 | 4s |
| Naive - same hour yesterday | - | 80.5 | - |
| Naive - last hour | - | 85.2 | - |

### The ablation that explains the table

The first multivariate run scored *worse* than the plain baseline (241.1
against 228.8), which looked like the extra columns being useless. The
training history said something more specific: validation loss bottomed out at
**epoch 7 of 12** and then climbed 7%. That is overfitting, not
under-training, and more patience cannot fix it - `restore_best_weights` hands
back epoch 7 either way. So the follow-up was an ablation, with a control.

| Run | Inputs | Dropout | MAE |
| --- | --- | --- | --- |
| Past traffic only | 1 | 0.20 | 228.8 |
| Past traffic only *(control)* | 1 | 0.35 | 229.8 |
| Traffic **+ calendar**, no weather | 7 | 0.20 | **201.4** |
| Traffic + calendar + weather | 11 | 0.20 | 241.1 |
| Traffic + calendar + weather | 11 | 0.35 | 206.0 |

Read in order, the table settles three questions at once:

1. **The control does not move** (228.8 to 229.8). Stronger dropout on its own
   buys nothing, so none of the gains below are really about regularisation.
2. **The calendar columns are the signal.** Hour and weekday, encoded as
   sine/cosine pairs, take the network from 228.8 to 201.4 - a 12% gain at
   unchanged dropout.
3. **The weather columns are noise that costs.** Adding them to the calendar
   features loses 40 points (201.4 to 241.1). `rain_1h` and `snow_1h` are zero
   for the overwhelming majority of hours; they add parameters the network can
   overfit and no information it can use. Heavier dropout repairs most of the
   damage (206.0) but never recovers the calendar-only score.

The tree model shows the mirror image: XGBoost scores 154.3 with calendar only
and 154.5 with weather added - statistically the same number. **Both models
gain from the calendar and neither uses the weather; the difference is that
gradient boosting simply ignores an irrelevant column while the LSTM
overfits it.** That robustness is a real, practical argument for tree
ensembles that no amount of theory about sequences addresses.

### One more result that contradicts the obvious expectation

**Discarding windows that span a gap in the series did not improve accuracy.**
28.7% of the training windows silently contain a jump in time. Removing
them is more correct, but it costs a third of the training data and the score
did not improve, so the flaw is real and not material.

### Forecasting further ahead

| Horizon | MAE | MAPE |
| --- | --- | --- |
| +1h | 244.5 | 11.0% |
| +3h | 361.6 | 19.4% |
| +6h | 472.4 | 22.6% |

One model, three outputs. Accuracy decays with the horizon, which is
the expected result: the further ahead you look, the less the last
24 hours determine the answer.

## 8. Does any of this generalise? A second dataset

The same code was pointed at a completely unrelated series: **hourly bike
rentals in Washington DC** (UCI Bike Sharing, 17,379 hours, 2011-2012).
Nothing changed but two command-line arguments - a different file and a
different target column. Rentals run from 1 to 977 an hour
against the motorway's 0 to 7,280, and bike demand collapses in
winter in a way motorway traffic never does.

| Dataset and inputs | LSTM MAE | XGBoost MAE | Winner |
| --- | --- | --- | --- |
| Bike rentals, past rentals only | 41.3 | 43.1 | **LSTM** by 4.1% |
| Bike rentals, + weather & calendar | 38.5 | 38.6 | **LSTM** by 0.3% |
| Motorway traffic, past traffic only | 228.8 | 177.3 | XGBoost by 22.5% |

**This is the most interesting result in the project.** The conclusion from
section 6 - "gradient boosting beats the LSTM" - does not survive contact with
a second dataset. On bike rentals the ordering flips, and the extra weather and
calendar features *help* the network here while they *hurt* it on traffic.

A plausible reading, offered as a hypothesis rather than a finding: bike demand
depends much more on the same hour yesterday. In the XGBoost importances,
`t-24h` carries 21% of the signal for rentals against 4%
for traffic. It is also far more weather-driven - nobody cycles in the rain -
so there is real information in the exogenous columns for the recurrence to
exploit.

The honest takeaway is not "LSTMs are good" or "LSTMs are bad". It is that the
question is empirical, the experiment is cheap, and running it on one dataset
is not enough to answer it.

## 9. Looking inside the network

Keras returns a prediction, not an explanation. `introspect.py` re-implements
the LSTM cell in NumPy from the trained weights:

```
z   = x_t . W + h_(t-1) . U + b
i   = sigmoid(z_i)    f = sigmoid(z_f)    g = tanh(z_g)    o = sigmoid(z_o)
c_t = f * c_(t-1) + i * g
h_t = o * tanh(c_t)
```

and is checked against Keras on every export. The maximum absolute difference
is around `1e-07`, i.e. floating-point noise, so every activation exported to
the Obsidian vault is provably the one the network computed. The vault holds
one note per neuron, one canvas per hour of the input window, and the four
gates with this model's real values.

Perturbing each input hour in turn shows the model leans on the last two hours
for about half its answer, but also consults around 12 hours back - the
opposite phase of the daily cycle. It is not simply copying the last value.

## 10. Limits

- The model sees only past traffic. Weather, holidays and accidents are
  invisible to it, and those are precisely the hours when a forecast matters
  most.
- It cannot predict a first-time event.
- It is harder to interpret than a decision tree. The vault narrows that gap;
  it does not close it.
- On this dataset it is also slower and less accurate than gradient boosting.

## 11. Next steps

1. Retrain the multivariate model with a longer EarlyStopping patience, to
   separate "the features do not help" from "the run stopped too early".
2. Benchmark against XGBoost on several junctions at once, where an LSTM's
   shared representation should start to pay off.
3. Feed the forecast into a signal-timing optimiser:
   `sensors -> history -> model -> forecast -> controller -> adaptive lights`.

---

*Code, figures and the full Obsidian vault:
[github.com/GirardRaphael/lstm](https://github.com/GirardRaphael/lstm)*
