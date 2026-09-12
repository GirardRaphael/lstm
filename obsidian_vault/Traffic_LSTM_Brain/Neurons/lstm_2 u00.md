---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 0
---
# lstm_2 u00

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▂▃▇██▇█▇▇▆▅▅▃▂▁▃▄▅▅▅▄▄▃▃` | -0.040 | -0.079 at t-10h |
| Quiet night | `▁▂▂▁▁▂▂▃▄▅▆▇▇███▇▇▆▅▄▄▄▃` | +0.054 | +0.107 at t-11h |

Reacted most strongly at: **t-10h, t-24h, t-21h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▄▃▄▅▄▄▃▂▂▁▁▂▄▅▄▄▄▄▄▄▄▄▄` | 0.698 |
| [[Input gate]] | `▁▅██▆▅▅▃▂▁▂▃▄▄▄▇▇██▇▆▅▄▄` | 0.473 |
| [[Output gate]] | `▇▇▆▅▄▃▂▂▁▁▁▂▃▅▇██▇▆▅▄▃▃▄` | 0.491 |
| [[Cell state]] | `▁▃▇█████▇▆▅▄▃▂▁▃▄▅▅▄▄▃▃▂` | final -0.084 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
