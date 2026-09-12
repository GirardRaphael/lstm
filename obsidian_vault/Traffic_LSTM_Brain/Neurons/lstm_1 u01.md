---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 1
---
# lstm_1 u01

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▁▂▅▆▆▇█████▇▆▄▂▂▂▃▄▄▄▅▄` | -0.055 | -0.120 at t-23h |
| Quiet night | `▅▇█▇▅▃▃▂▁▁▁▂▃▃▅▅▆▇▇▇▇▆▇▆` | -0.006 | -0.030 at t-15h |

Reacted most strongly at: **t-23h, t-9h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄█▃▁▂▃▂▂▂▂▂▃▃▄▅▆▄▄▄▄▄▄▃▃` | 0.701 |
| [[Input gate]] | `▅▃▆▇████▇▆▅▄▃▂▁▂▄▅▇██▇▇▅` | 0.463 |
| [[Output gate]] | `▄█▅▁▁▂▂▂▂▂▂▂▂▃▄▅▃▃▂▃▄▄▄▅` | 0.514 |
| [[Cell state]] | `▅▁▁▄▅▆▇▇███▇▇▆▄▁▁▁▂▃▃▄▄▄` | final -0.095 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
