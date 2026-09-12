---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 20
---
# lstm_1 u20

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▄▆▇▇▇██▇▇▆▇▆▄▃▅▅▆▆▇▇▇▆▅` | -0.055 | -0.159 at t-24h |
| Quiet night | `█▇▅▂▂▃▂▁▁▂▂▃▂▄▂▄▄▄▂▄▄▆▆▆` | +0.020 | +0.029 at t-24h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▇▆▇██▇▆▅▄▂▁▁▁▃▅▆▇████▇▆` | 0.640 |
| [[Input gate]] | `▅▅▃▂▂▄▅▆▇▇████▆▄▂▁▁▂▄▅▆▇` | 0.742 |
| [[Output gate]] | `▅▄▄▃▂▃▅▆▇▇████▆▄▂▁▁▂▃▅▆▇` | 0.744 |
| [[Cell state]] | `▁▄▆▇▇███▇▇▆▇▆▅▄▄▅▆▆▇▇▇▆▅` | final -0.066 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
