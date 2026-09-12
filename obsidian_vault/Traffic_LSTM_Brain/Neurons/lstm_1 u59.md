---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 59
---
# lstm_1 u59

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▆▂▁▂▁▁▁▁▂▂▂▃▅▆▅▄▃▂▂▂▃▃▅` | +0.068 | +0.160 at t-24h |
| Quiet night | `▁▁▄▇█▇▇██▇▆▆▇▅▆▄▄▄▆▅▄▃▂▂` | -0.031 | -0.035 at t-23h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▅▅▆██▇▇▅▄▂▁▁▁▂▄▅▆██▇▆▅▄` | 0.598 |
| [[Input gate]] | `▅▇▅▂▁▂▃▄▆▇█████▆▄▃▂▂▃▄▅▇` | 0.745 |
| [[Output gate]] | `▄▆▅▃▁▂▃▄▅▇████▇▆▄▃▁▁▂▄▅▆` | 0.736 |
| [[Cell state]] | `█▅▃▂▂▁▁▁▂▂▂▂▃▄▆▄▄▃▂▂▂▃▃▅` | final +0.086 |

A mean forget value of **0.60** means this unit keeps roughly
60% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
