---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 44
---
# lstm_1 u44

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▄▁▁▂▂▂▂▂▂▃▂▃▄▅▄▃▃▃▄▄▃▄▄` | +0.043 | +0.173 at t-24h |
| Quiet night | `▁▂▅█▇▅▆▆▆▅▄▄▆▃▅▃▄▃▆▄▄▃▃▃` | -0.027 | -0.039 at t-24h |

Reacted most strongly at: **t-24h, t-10h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▆▆▆▆▅▅▄▃▂▁▁▂▄▆▆▆▆▆▅▅▄▅` | 0.590 |
| [[Input gate]] | `▂▆▅▃▁▂▂▃▄▅▇███▇▆▅▄▃▄▅▆▇▇` | 0.849 |
| [[Output gate]] | `▂▃▅▃▁▁▂▃▄▅▇███▇▅▄▃▂▃▅▆▇▇` | 0.848 |
| [[Cell state]] | `█▄▁▁▂▂▂▂▂▂▂▂▃▄▅▃▃▃▃▃▄▃▄▄` | final +0.048 |

A mean forget value of **0.59** means this unit keeps roughly
59% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
