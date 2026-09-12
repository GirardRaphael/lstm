---
tags: [presentation, qa]
---
# 08 Exam Questions

### Why an LSTM rather than a plain neural network?

A dense network treats the 24 inputs as 24 unrelated numbers. An LSTM
processes them **in order** and carries a memory from one hour to the next,
so it can learn that a rising 16:00 usually leads to a peak at 17:00.

### What does the LSTM actually remember?

The cell state `c`. At each hour the forget gate decides what fraction of `c`
survives, the input gate decides how much new information is written. A unit
whose forget gate stays near 1 carries information across the whole day - see
[[Cell state]] and the timeline canvases.

### What is overfitting, and what did you do about it?

Learning the training set so precisely that performance on new data drops.
Four defences here: **Dropout 0.2**, a **validation split** watched every
epoch, **EarlyStopping** (patience 5, best weights restored), and a
**chronological train/test split** so the test set is genuinely unseen.

### Why MAE rather than MSE for reporting?

MSE is the training loss because it penalises large misses harder. MAE is the
reported metric because it is in the unit of the problem: "wrong by
229 vehicles per hour" is a sentence anyone can check. Both are in
[[05 Results]].

### Is a MAE of 229 good?

On its own, unknowable - which is why the baselines are there. Predicting
"the same hour yesterday" gives 594 and "the last hour" gives
586. The model is **+60.9%** better than the best of them.

### Why 24 timesteps?

One full daily cycle. Shorter cuts the cycle in half; much longer adds
parameters without new information.

### Why is the last layer linear?

Regression. A sigmoid would squash the output into [0, 1] and a softmax would
turn it into class probabilities. We want a count of vehicles.

### What do the 64 units in the first layer do?

Each learns its own feature of the sequence. On the rush-hour window,
22 of 64 units carry most of the signal while the rest stay near
zero - open [[Neurons - Rush Hour.canvas]] and the grey nodes are the quiet
ones. Individual units are documented in `Neurons/`.

### How do you know the numbers in this vault are real?

`introspect.py` re-implements the LSTM cell in NumPy from the trained weights
and is checked against Keras on every export. Current maximum absolute
difference: **1.17e-07**.

### When would an LSTM be the wrong choice?

With no temporal structure, or with little data. Random Forest or XGBoost on
lag features would be simpler and often as strong. See
[[06 Limits and Next Steps]].

### Does this control the traffic lights?

No. It produces a forecast that a signal controller could consume. Claiming
more would be wrong.
