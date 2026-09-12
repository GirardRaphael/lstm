---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 11
---
# lstm_1 u11

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▅▄▄▅▆▇███▇▇▆▆▅▄▂▁▁▃▅▇██` | +0.041 | -0.069 at t-7h |
| Quiet night | `▁▃▄▅▅▅▅▅▅▅▆▆▇████▇▇▅▄▃▃▂` | +0.011 | +0.038 at t-9h |

Reacted most strongly at: **t-7h, t-6h, t-8h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▅▆▇▆▄▃▃▂▁▁▁▂▃▅▇██▇▆▅▄▃▃` | 0.697 |
| [[Input gate]] | `▄▆▆▆▆▅▄▃▂▁▁▁▁▂▃▆▇██▇▆▅▄▃` | 0.441 |
| [[Output gate]] | `▃▃▅▇▇▆▅▄▃▂▁▁▁▁▂▄▆▇██▇▆▄▃` | 0.444 |
| [[Cell state]] | `▅▅▄▃▄▆▇███▇▇▆▅▄▃▂▁▁▃▅▆▇█` | final +0.101 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
