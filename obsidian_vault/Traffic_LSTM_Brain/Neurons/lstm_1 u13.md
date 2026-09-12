---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 13
---
# lstm_1 u13

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▃▁▁▂▁▁▂▂▂▂▂▂▄▅▃▃▃▂▂▂▂▃▄` | +0.047 | +0.174 at t-24h |
| Quiet night | `▁▃▆█▇▆▆▆▆▅▅▅▆▄▅▃▄▄▅▃▃▂▂▂` | -0.019 | -0.027 at t-24h |

Reacted most strongly at: **t-24h, t-10h, t-11h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▄▅▇██▇▇▅▄▂▁▁▁▃▄▅▆▇██▇▇▆` | 0.642 |
| [[Input gate]] | `▃▆▄▁▁▃▄▄▆▇████▇▅▂▁▁▂▄▅▆▇` | 0.760 |
| [[Output gate]] | `▅▄▃▂▂▃▄▅▆▇████▇▄▃▁▁▂▄▅▆▇` | 0.786 |
| [[Cell state]] | `█▃▁▁▂▁▁▂▂▂▂▂▂▄▅▃▃▃▂▂▂▂▃▄` | final +0.055 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
