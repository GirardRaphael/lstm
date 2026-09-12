---
tags: [results]
---
# 05 Results

## Metrics on the test set (8,115 unseen hours)

| Horizon | MAE | RMSE | MAPE | Same hour yesterday | Last hour | Gain |
| --- | --- | --- | --- | --- | --- | --- |
| +1h | 228.8 | 326.2 | 11.4% | 594.3 | 585.6 | **+60.9%** |

## Say it in words, not in numbers

> On hours it had never seen, the model is wrong by about
> **229 vehicles per hour** on average. Traffic on this road runs
> between 0 and 7,280 vehicles an hour, so that is roughly
> **11%** off.

## Why the baselines are in the table

A MAE of 229 means nothing on its own. Two baselines make it an argument:

- **Last hour** (persistence): predict whatever the previous hour was. MAE 586.
- **Same hour yesterday**: exploit the daily cycle without any learning. MAE 594.

The network lands at 229, i.e. **+60.9%** against the better of
the two. If it had not beaten both, the honest conclusion would have been
that an LSTM is the wrong tool here.

![[09_baselines.png]]

## Learning curve

![[03_training_loss.png]]

Trained for **49 epochs** (1072.1s) before EarlyStopping restored the
best weights. Training and validation loss fall together - the sign that
dropout and early stopping are doing their job.

## Actual vs predicted

![[04_actual_vs_predicted.png]]

The model tracks the shape of the daily cycle closely. It tends to shave the
very top of the sharp peaks, which is expected: MSE rewards being close on
average more than nailing the extremes.

## Error distribution

![[05_error_distribution.png]]

## Which past hours the model actually uses

![[08_input_sensitivity.png]]

Each of the 24 input hours was nudged in turn and the change in the output
measured, on the busiest window of the test set:

- **t-1h** - 26.7% of the movement
- **t-2h** - 22.7% of the movement
- **t-13h** - 6.9% of the movement

The last three hours account for 53% of the movement, but t-9h, t-12h, t-13h, t-14h also rank in the top six. The network is consulting roughly half a day back as well as the immediate past - which is why it beats the persistence baseline instead of tying with it.

## Now look inside

- **[[Neurons - Rush Hour.canvas]]** - every unit on the busiest hour of the test set
- **[[Neurons - Quiet Night.canvas]]** - the same units, near-empty road
- **[[t00.canvas]]** - 24 canvases, one per hour

Next: [[06 Limits and Next Steps]]
