---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 39
---
# lstm_1 u39

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▆██▇▆▅▅▄▃▂▂▁▁▂▅▇███▇▆▆▆` | +0.045 | +0.062 at t-6h |
| Quiet night | `▆▆▆▅▅▅▆▆▇▇█████▇▇▆▅▅▄▃▂▁` | -0.030 | -0.030 at t-1h |

Reacted most strongly at: **t-6h, t-7h, t-21h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▅▆▇▆▅▅▅▅▅▆▆▇▇▆▇██▇▆▄▃▂▁` | 0.699 |
| [[Input gate]] | `▄▆▇▆▅▄▄▃▄▄▅▅▅▅▆██▇▆▄▃▂▁▁` | 0.449 |
| [[Output gate]] | `▆█▇▅▅▄▃▂▁▁▁▁▂▃▆███▇▇▆▄▃▃` | 0.483 |
| [[Cell state]] | `▄▆███▇▆▅▄▃▂▂▁▁▂▅▇███▇▇▇▇` | final +0.098 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
