---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 2
---
# lstm_1 u02

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅█▇▅▃▂▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▃▃▄` | -0.006 | +0.048 at t-9h |
| Quiet night | `█▃▁▂▅▇███▇▆▅▄▃▂▂▂▂▃▅▆▇▇▇` | -0.008 | -0.023 at t-22h |

Reacted most strongly at: **t-9h, t-17h, t-16h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▃▆▇▇▇██████▆▄▂▃▄▅▅▅▄▄▃` | 0.668 |
| [[Input gate]] | `▇▄▂▂▃▄▅▆▇████▇▆▃▂▁▁▁▁▁▂▃` | 0.459 |
| [[Output gate]] | `▇▃▃▃▃▄▅▇▇████▇▅▄▂▁▁▁▁▂▃▃` | 0.450 |
| [[Cell state]] | `▅█▇▅▃▂▁▁▁▂▃▄▄▆▇█▇▆▄▃▃▃▃▄` | final -0.014 |

A mean forget value of **0.67** means this unit keeps roughly
67% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
