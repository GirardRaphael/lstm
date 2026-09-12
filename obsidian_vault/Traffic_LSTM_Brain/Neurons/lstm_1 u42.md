---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 42
---
# lstm_1 u42

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▃▂▃▄▅▇███▇▆▅▄▂▂▁▁▂▄▅▅▅▅` | -0.006 | -0.055 at t-8h |
| Quiet night | `▂▆█▇▆▅▃▂▁▁▁▂▃▄▄▅▅▅▄▃▃▃▄▅` | +0.026 | +0.036 at t-22h |

Reacted most strongly at: **t-8h, t-7h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▅▇▇▆▅▅▄▄▅▅▆▆▆▆▇██▇▅▃▂▁▁` | 0.683 |
| [[Input gate]] | `▄▄▂▁▃▄▅▆▇▇███▇▅▃▁▁▁▂▂▃▃▄` | 0.464 |
| [[Output gate]] | `▆▁▄▆▇▇▇███▇▇▆▅▃▂▄▄▅▅▄▄▅▄` | 0.470 |
| [[Cell state]] | `▄▃▂▃▄▆▇███▇▆▅▄▂▂▁▁▃▄▅▆▆▅` | final -0.014 |

A mean forget value of **0.68** means this unit keeps roughly
68% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
