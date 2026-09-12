---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 8
---
# lstm_2 u08

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▆▇▇▆▆▇▇████▇▇▆▇▆▅▄▃▂▁▁▂` | -0.104 | -0.123 at t-2h |
| Quiet night | `▁▃▃▃▃▃▃▃▃▄▅▅▆▇▇███▇▆▅▄▃▂` | +0.030 | +0.076 at t-8h |

Reacted most strongly at: **t-2h, t-3h, t-4h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇▄▄▆▇▇▇▆▅▃▂▁▁▂▃▃▄▅▆▇████` | 0.715 |
| [[Input gate]] | `▆▁▃▆████▇▆▄▄▄▄▂▁▃▄▅▆▅▅▅▅` | 0.456 |
| [[Output gate]] | `█▃▁▂▄▄▅▆▆▆▅▅▅▅▅▂▂▂▂▂▃▃▄▅` | 0.438 |
| [[Cell state]] | `▆▇▇▇▆▆▇█████▇▇▇▇▆▅▄▂▁▁▂▃` | final -0.218 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
