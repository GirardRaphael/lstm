---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 9
---
# lstm_1 u09

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▄▆██▇▇▇▇▇▇▇▇▇▇▇██▆▄▂▁▁▂` | -0.084 | -0.096 at t-3h |
| Quiet night | `▁▄▄▃▂▂▄▅▆▆▇████▇▇▇▆▅▅▅▅▅` | +0.014 | +0.023 at t-13h |

Reacted most strongly at: **t-3h, t-2h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅█▂▁▄▆▆▆▆▅▅▄▃▂▃▃▁▂▃▄▅▅▅▅` | 0.662 |
| [[Input gate]] | `▄▄▆▇▇▆▅▄▃▂▁▁▁▂▃▄▆▇███▇▆▅` | 0.520 |
| [[Output gate]] | `▄█▆▆▆▆▅▄▃▂▁▁▁▂▃▆▆▇▇███▇▇` | 0.526 |
| [[Cell state]] | `▆▄▆██▇▇▇▇▇▇▇▇▇▆▇██▆▄▂▁▁▁` | final -0.145 |

A mean forget value of **0.66** means this unit keeps roughly
66% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
