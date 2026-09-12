---
tags: [mechanism, gate]
---
# Forget gate

`f = sigmoid(x_t . W_f + h_(t-1) . U_f + b_f)`

Decides what fraction of the existing memory survives this hour.

- `f` near **1** - keep the memory intact (a unit tracking the whole day)
- `f` near **0** - wipe it and start fresh (a unit that only cares about now)

It multiplies the old cell state: `c_t = f * c_(t-1) + ...`

## What this gate did on the rush-hour window

Averaged over all 64 units of [[lstm_1]], hour by hour:

`█▅▅▆██▇▇▆▅▄▄▄▄▄▅▆▇█▇▅▄▂▁`

| | |
| --- | --- |
| Mean opening | 0.661 |
| Most open at | t-20h (0.678) |
| Most closed at | t-1h (0.633) |

Step through **[[t00.canvas]]** to see this gate hour by hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] -
[[Output gate]] - [[Cell state]] - [[04 Architecture]]
