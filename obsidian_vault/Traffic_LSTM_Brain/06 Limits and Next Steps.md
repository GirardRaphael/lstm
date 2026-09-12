---
tags: [critique]
---
# 06 Limits and Next Steps

## What this model cannot do

**It has never heard of weather, holidays or accidents.** It only sees past
traffic. A snowstorm, a closed lane or a stadium emptying out are invisible
to it, and those are exactly the hours where a forecast would be most useful.

**It cannot predict a first-time event.** An LSTM extrapolates patterns it
has seen. The first hour of an unprecedented situation will be missed.

**It is hard to interpret.** This vault is an attempt to fix that, but a
decision tree would explain itself in one line. That trade is the price of
modelling temporal dependencies.

**It needs a lot of data and compute.** ~48,000 hours here. On a few hundred
rows, a simpler model would win.

## When an LSTM would be the wrong choice

If the observations had no temporal relationship, the sequence structure
would be wasted and Random Forest or XGBoost on engineered features
(hour, weekday, lag-1, lag-24) would be simpler, faster and often as good.
That is an honest comparison to run, not a weakness to hide.

## Next steps, in order of value

1. **Add the exogenous columns.** `temp`, `rain_1h`, `snow_1h`, `holiday`,
   plus hour-of-day and day-of-week encoded as sine/cosine pairs. Input shape
   becomes `(24, 8)` and nothing else in the pipeline changes.
2. **Compare against a gradient-boosted baseline** on the same split.
3. **Predict several hours ahead.** Already supported:
   `--horizons 1 3 6` trains one model with 24-hour input and three outputs.
4. **Feed a controller.** The forecast becomes an input to a signal-timing
   optimiser - see the diagram in [[01 Problem]].

## The honest summary

> The model learns the daily traffic cycle well and beats both naive
> baselines. It does not understand *why* traffic changes, and it will fail
> on exactly the unusual hours a traffic operator cares about most. Adding
> weather and calendar features is the obvious next move.

Next: [[07 Presentation Script]]
