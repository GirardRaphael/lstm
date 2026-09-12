---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 55
---
# lstm_1 u55

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▅▅▅▅▅▆▇████▇▆▆▅▄▃▂▁▁▂▄▆` | -0.026 | -0.103 at t-5h |
| Quiet night | `▄▃▂▁▁▂▂▂▂▃▄▄▅▇▆███▆▅▄▄▃▂` | +0.002 | +0.012 at t-9h |

Reacted most strongly at: **t-5h, t-4h, t-6h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▇▆▆▇▆▅▄▃▂▁▁▁▂▄▆▇███▇▆▅▄` | 0.680 |
| [[Input gate]] | `▃▄▄▅▆▆▅▄▃▂▂▁▁▁▂▃▄▅▇███▆▅` | 0.537 |
| [[Output gate]] | `▄▆▅▄▅▅▅▄▃▂▂▁▁▂▃▄▄▅▆▇██▇▇` | 0.560 |
| [[Cell state]] | `▄▅▅▅▅▅▆▇████▇▆▅▅▄▃▂▁▁▂▄▅` | final -0.042 |

A mean forget value of **0.68** means this unit keeps roughly
68% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
