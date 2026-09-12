---
tags: [mechanism, gate]
---
# Output gate

`o = sigmoid(x_t . W_o + h_(t-1) . U_o + b_o)`

Decides how much of the memory is exposed to the next layer:

`h_t = o * tanh(c_t)`

A unit can hold information in `c` for hours while keeping `o` closed, then open it exactly when it matters.

## What this gate did on the rush-hour window

Averaged over all 64 units of [[lstm_1]], hour by hour:

`▃▅▅▂▁▂▃▄▅▆████▆▄▃▂▁▂▄▅▆▆`

| | |
| --- | --- |
| Mean opening | 0.565 |
| Most open at | t-13h (0.589) |
| Most closed at | t-20h (0.541) |

Step through **[[t00.canvas]]** to see this gate hour by hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] -
[[Output gate]] - [[Cell state]] - [[04 Architecture]]
