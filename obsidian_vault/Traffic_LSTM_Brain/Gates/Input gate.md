---
tags: [mechanism, gate]
---
# Input gate

`i = sigmoid(x_t . W_i + h_(t-1) . U_i + b_i)`

Decides how much of the new candidate information gets written into memory this hour.

- `i` near **1** - this hour is worth remembering
- `i` near **0** - ignore it

It gates the candidate: `c_t = ... + i * g`

## What this gate did on the rush-hour window

Averaged over all 64 units of [[lstm_1]], hour by hour:

`▃▁▃▃▃▄▅▆▆▇███▇▄▂▂▂▃▃▄▅▅▄`

| | |
| --- | --- |
| Mean opening | 0.559 |
| Most open at | t-13h (0.593) |
| Most closed at | t-23h (0.526) |

Step through **[[t00.canvas]]** to see this gate hour by hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] -
[[Output gate]] - [[Cell state]] - [[04 Architecture]]
