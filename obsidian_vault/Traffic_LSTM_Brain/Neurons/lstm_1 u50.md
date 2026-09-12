---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 50
---
# lstm_1 u50

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▆▄▂▂▁▁▁▁▂▃▃▄▆▇▆▅▃▃▃▃▃▄▆` | +0.067 | +0.128 at t-24h |
| Quiet night | `▃▁▂▆█████▇▆▅▅▄▅▃▃▃▅▅▅▃▂▂` | -0.041 | -0.044 at t-23h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▆▃▃▅▆▆▇█████▇▅▅▃▃▃▃▂▁▁▁` | 0.550 |
| [[Input gate]] | `▄▃▃▂▂▃▄▅▆▇████▆▃▂▁▁▂▃▄▅▆` | 0.727 |
| [[Output gate]] | `▄▄▄▃▂▂▄▅▆▇████▇▅▃▂▁▁▂▄▅▆` | 0.741 |
| [[Cell state]] | `█▆▄▂▂▁▁▁▂▂▃▃▄▅▇▆▅▃▃▃▃▃▄▅` | final +0.084 |

A mean forget value of **0.55** means this unit keeps roughly
55% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
