---
tags: [mechanism, gate]
---
# Candidate memory

`g = tanh(x_t . W_g + h_(t-1) . U_g + b_g)`

The *content* the unit proposes to store, squashed into [-1, 1]. [[Input gate]] then decides how much of it actually lands in [[Cell state]].

## What this gate did on the rush-hour window

Averaged over all 64 units of [[lstm_1]], hour by hour:

`▇▄▅▆▇▆▅▄▃▂▁▁▁▂▄▅▇▇▇██▇▇▇`

| | |
| --- | --- |
| Mean opening | 0.009 |
| Most open at | t-5h (0.026) |
| Most closed at | t-13h (-0.015) |

Step through **[[t00.canvas]]** to see this gate hour by hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] -
[[Output gate]] - [[Cell state]] - [[04 Architecture]]
