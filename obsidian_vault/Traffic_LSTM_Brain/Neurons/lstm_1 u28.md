---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 28
---
# lstm_1 u28

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▄▆▆▇▆▅▄▃▂▂▁▁▁▂▄▅▇▇█████` | +0.067 | +0.068 at t-4h |
| Quiet night | `▆▆▅▅▅▅▅▆▆▇▇████▇▇▆▅▅▄▃▂▁` | -0.016 | -0.016 at t-1h |

Reacted most strongly at: **t-4h, t-1h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▅▇▆▅▃▂▁▁▁▂▂▃▄▅▇██▆▅▃▂▁▁` | 0.723 |
| [[Input gate]] | `▅▅▆▇█▇▆▅▄▃▂▂▁▁▂▄▆▇██▇▆▅▄` | 0.471 |
| [[Output gate]] | `▅█▅▃▃▄▃▂▂▁▁▁▁▂▄▅▄▄▄▄▄▄▃▃` | 0.467 |
| [[Cell state]] | `▃▄▆▇▇▆▅▄▃▂▂▁▁▁▂▃▅▇▇█████` | final +0.146 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
