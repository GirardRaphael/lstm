---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 7
---
# lstm_1 u07

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▁▂▃▄▄▄▃▂▂▁▁▁▁▁▁▂▄▆██▇▅▄` | +0.042 | +0.141 at t-4h |
| Quiet night | `▃▆█▇▅▃▃▄▄▅▆▇█▆▆▄▄▃▃▁▁▁▂▃` | -0.004 | -0.008 at t-5h |

Reacted most strongly at: **t-4h, t-5h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▆▇▇▆▅▄▃▂▁▁▁▂▃▅▇███▇▆▄▄▃` | 0.708 |
| [[Input gate]] | `▄▂▄▇█▇▆▅▄▃▂▁▁▁▁▂▅▆▇██▇▆▄` | 0.484 |
| [[Output gate]] | `▅▆▅▅▆▆▅▄▃▂▁▁▁▂▃▅▅▆▇███▇▆` | 0.519 |
| [[Cell state]] | `▃▂▂▄▅▅▄▄▃▂▁▁▁▁▁▁▃▄▆██▇▆▄` | final +0.075 |

A mean forget value of **0.71** means this unit keeps roughly
71% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
