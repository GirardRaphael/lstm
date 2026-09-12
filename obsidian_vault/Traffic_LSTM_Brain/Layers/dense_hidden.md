---
tags: [layer]
---
# dense_hidden

16 units, `relu` activation. Takes the summary vector produced by [[lstm_2]]
and recombines it non-linearly before the output.

`relu(x) = max(0, x)` - anything negative becomes exactly zero, which is why
some units below read `+0.00`.

| | |
| --- | --- |
| Units firing on the rush-hour window | 9 of 16 |
| Largest activation | 0.521 |

`▅█▁▇▁▇▁▁▆▄▁▁▇▁▁█`

Feeds [[traffic_output]]. Related: [[04 Architecture]]
