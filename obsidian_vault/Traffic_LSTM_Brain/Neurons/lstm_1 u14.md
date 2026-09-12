---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 14
---
# lstm_1 u14

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▂▃▂▂▃▃▄▆▇█▇▆▄▂▁▂▂▂▂▂▂▂▂▂` | -0.041 | +0.098 at t-15h |
| Quiet night | `▆▇█▇▅▃▂▂▁▁▁▂▃▃▄▅▆▆▆▅▄▄▃▂` | -0.043 | -0.051 at t-15h |

Reacted most strongly at: **t-15h, t-14h, t-16h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▂▄▅▆▆▇█████▇▄▂▂▃▃▄▄▄▄▃` | 0.641 |
| [[Input gate]] | `▅▁▃▄▅▆▆▇████▇▆▄▂▂▃▃▄▄▄▅▄` | 0.443 |
| [[Output gate]] | `▅▁▂▄▅▅▆▇█████▆▄▂▂▂▃▃▃▃▄▃` | 0.397 |
| [[Cell state]] | `▃▂▂▃▄▅▆▇███▇▅▃▁▁▁▁▂▂▃▃▃▃` | final -0.128 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
