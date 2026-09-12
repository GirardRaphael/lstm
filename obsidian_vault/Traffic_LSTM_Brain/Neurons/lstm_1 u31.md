---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 31
---
# lstm_1 u31

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▂▂▅▇▇██▇▇▆▅▄▄▂▁▂▄▅▇███▇` | +0.026 | -0.090 at t-9h |
| Quiet night | `▄▇█▆▃▂▁▁▂▃▄▅▆▆▇▆▆▅▄▂▂▁▁▁` | -0.013 | -0.014 at t-3h |

Reacted most strongly at: **t-9h, t-23h, t-8h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▃█▆▅▆▆▆▅▄▃▂▂▁▁▂▅▅▆▇▇▇▆▅▄` | 0.698 |
| [[Input gate]] | `▄▆██▇▆▆▅▃▂▁▁▁▂▄▇████▇▆▆▅` | 0.479 |
| [[Output gate]] | `▅█▇▅▄▄▄▃▂▂▁▁▁▂▅▇▇▆▆▅▅▆▆▆` | 0.505 |
| [[Cell state]] | `▅▂▂▅▇▇███▇▆▅▄▃▂▁▂▃▅▇███▇` | final +0.048 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
