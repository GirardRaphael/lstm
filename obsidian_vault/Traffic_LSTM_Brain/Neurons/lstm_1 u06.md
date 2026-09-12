---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 6
---
# lstm_1 u06

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇██▆▄▃▂▁▁▂▃▃▄▆██▇▆▅▄▄▄▄▆` | +0.026 | +0.052 at t-23h |
| Quiet night | `▃▁▁▃▅▆▇████▇▇▆▆▆▅▄▄▅▄▄▃▂` | -0.013 | -0.020 at t-22h |

Reacted most strongly at: **t-23h, t-22h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇▃▂▃▆▆▇███▇▆▆▄▂▁▁▂▃▄▃▂▂▁` | 0.622 |
| [[Input gate]] | `▄▂▂▃▄▄▅▇▇███▇▆▃▁▁▁▁▂▂▃▃▃` | 0.423 |
| [[Output gate]] | `▄▂▃▃▃▄▅▆▇███▇▆▄▂▁▁▁▁▂▃▄▄` | 0.447 |
| [[Cell state]] | `▆█▇▅▃▂▁▁▁▂▂▃▃▅▇█▇▆▅▄▃▃▄▅` | final +0.059 |

A mean forget value of **0.62** means this unit keeps roughly
62% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
