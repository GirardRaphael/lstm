---
tags: [presentation]
---
# 07 Presentation Script

Target: 8 minutes, plus questions. Times are cumulative.

## 0:00 - 0:45 - The question

> "Traffic is a time series: what happens now depends on what happened in the
> hours before. My project asks whether the last 24 hours of traffic volume
> are enough to predict the next hour. I used an LSTM, a neural network built
> for sequences."

## 0:45 - 2:00 - The data

Show [[02 Dataset]] and `01_dataset.png`.

> "48,204 hourly measurements from the UCI repository, on I-94 between
> Minneapolis and Saint Paul, from 2012 to 2018. The target is
> `traffic_volume`: vehicles counted during the hour."

Show `02_daily_profile.png`.

> "This is the pattern the network has to learn: a morning peak, an evening
> peak, an overnight trough."

## 2:00 - 3:15 - The method

Show the diagram in [[03 Pipeline]]. Hit the two points a teacher will probe:

> "The split is chronological, not random - in production you only have the
> past. And the scaler is fitted only on the training set, otherwise the model
> would indirectly know the range of the future. That is data leakage."

## 3:15 - 4:15 - The network

Open **[[Network Architecture.canvas]]**.

> "24 hours in. A first LSTM of 64 units produces one hidden state per
> hour. A second LSTM of 32 units collapses that into a single vector for
> the day. Two dense layers turn it into one number. The last layer is linear
> because the output is a count of vehicles, not a probability."

## 4:15 - 5:30 - **The demo** (the part they will remember)

Open **[[Neurons - Rush Hour.canvas]]**.

> "This is not an illustration. Every colour is a real activation, replayed
> from the trained weights. Red means the unit fired positively, blue
> negatively, grey means it stayed silent. This window ends at 2018-04-12 16:00, the
> busiest hour of the test set: the model predicted 6,164 vehicles,
> the truth was 7,213."

Then open **[[Neurons - Quiet Night.canvas]]**.

> "Same network, an almost empty road. A completely different set of units
> lights up. The network has specialised."

Then open **[[t00.canvas]]** and step through a few hours.

> "And here is the memory forming, hour by hour. Each canvas is one timestep.
> Watch the forget gate: when it stays near 1, the unit is holding on to the
> whole day."

## 5:30 - 6:45 - The results

Show [[05 Results]].

> "On 8,115 hours the model had never seen, it is wrong by
> **229 vehicles per hour** on average. Predicting 'the same hour
> yesterday' gives 594, so the network is 61% better. Without
> that comparison the number would not mean anything."

Show `04_actual_vs_predicted.png`.

> "It follows the daily cycle and slightly flattens the sharpest peaks."

## 6:45 - 8:00 - Limits and what comes next

From [[06 Limits and Next Steps]]:

> "The model has never heard of weather, holidays or accidents - exactly the
> situations where a forecast matters most. The next version adds those
> columns. And to be precise about the claim: this model does not control
> traffic lights. It produces a forecast a controller could use."

Have [[08 Exam Questions]] open in a second tab.

## The live demo, as a fallback

```
python -m traffic_lstm.predict --last-hours

  Window    : 24 hours ending 2018-04-12 16:00
  Prediction: 6,164 vehicles
  Level     : HIGH
```
