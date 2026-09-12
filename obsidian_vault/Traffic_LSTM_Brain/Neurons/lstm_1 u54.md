---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 54
---
# lstm_1 u54

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▄▄▅▆▇████▇▇▅▄▂▁▁▂▃▄▅▆▆▇` | +0.019 | -0.114 at t-9h |
| Quiet night | `▅▆▇▆▅▅▄▄▄▅▅▆▇▇████▇▆▅▃▂▁` | -0.027 | +0.027 at t-8h |

Reacted most strongly at: **t-9h, t-8h, t-7h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄█▆▄▃▂▂▁▁▂▃▄▄▆▇█▇▇▆▅▄▃▂▃` | 0.723 |
| [[Input gate]] | `▅▇▄▂▁▁▁▁▂▃▄▅▆▇█▇▆▄▃▂▂▂▂▃` | 0.514 |
| [[Output gate]] | `▃█▆▄▃▃▂▁▁▁▂▃▄▅▇█▇▇▆▆▅▅▄▄` | 0.514 |
| [[Cell state]] | `▄▃▄▅▆▇████▇▆▅▄▂▁▁▂▃▄▅▆▆▆` | final +0.037 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
