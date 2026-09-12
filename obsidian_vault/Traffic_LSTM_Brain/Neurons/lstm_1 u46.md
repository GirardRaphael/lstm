---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 46
---
# lstm_1 u46

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▁▄███▇▇▇▆▆▆▇▆▄▄▅▇██▇▆▅▄` | -0.091 | -0.191 at t-23h |
| Quiet night | `▆▇▅▃▂▁▁▁▁▁▂▂▂▂▂▃▄▄▄▅▆▇██` | +0.012 | -0.034 at t-16h |

Reacted most strongly at: **t-23h, t-9h, t-24h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆█▃▁▃▅▅▅▆▅▅▄▃▃▄▄▂▂▂▃▃▄▄▄` | 0.622 |
| [[Input gate]] | `▂█▇▄▃▂▂▁▁▁▂▂▂▃▅▇▆▆▅▅▆▆▆▆` | 0.650 |
| [[Output gate]] | `▃█▆▃▂▂▂▁▁▁▂▂▂▃▆▇▆▅▄▃▄▄▅▆` | 0.650 |
| [[Cell state]] | `▃▁▄▇██▇▇▆▆▆▆▆▅▄▃▅▇██▇▆▅▄` | final -0.127 |

A mean forget value of **0.62** means this unit keeps roughly
62% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
