---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 25
---
# lstm_2 u25

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▄▁▁▃▄▅▅▅▆▆▆▇▇▇▅▃▃▄▆▇██▇` | +0.080 | +0.093 at t-2h |
| Quiet night | `▇▇▇██▇▆▅▄▂▁▁▁▁▂▂▃▄▅▅▅▅▅▅` | -0.040 | -0.092 at t-13h |

Reacted most strongly at: **t-2h, t-3h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▃█▇▄▃▂▁▁▂▂▄▄▅▅▇█▇▆▅▄▄▄▃▃` | 0.763 |
| [[Input gate]] | `▄▆▇▇▆▅▄▃▁▁▁▂▃▄▅▆▇███▇▇▆▆` | 0.480 |
| [[Output gate]] | `▅▇█▇▆▅▄▃▂▁▁▂▂▃▅▇███▇▇▇▆▆` | 0.513 |
| [[Cell state]] | `▆▄▁▁▃▄▅▆▆▇▇███▇▅▃▃▄▅▇▇█▇` | final +0.138 |

A mean forget value of **0.76** means this unit keeps roughly
76% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
