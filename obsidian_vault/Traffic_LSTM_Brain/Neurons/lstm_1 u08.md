---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 8
---
# lstm_1 u08

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▃▁▁▂▄▅▆▆▇▇▇███▅▃▁▁▃▄▅▅▆` | +0.040 | +0.058 at t-11h |
| Quiet night | `▁▃▅▇▇▇▇▇▇▇▇▇▇▇▇▇▇███████` | +0.049 | +0.049 at t-1h |

Reacted most strongly at: **t-11h, t-12h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▁▃▄▅▅▅▅▆▆▆▆▆▆▅▃▄▄▄▄▃▂▂▂` | 0.698 |
| [[Input gate]] | `▅▃▇█▇▆▅▅▅▆▆▆▇▆▅▆▇▇▇▅▃▂▁▁` | 0.443 |
| [[Output gate]] | `█▄▅▄▄▃▁▁▁▁▂▂▃▅▆▇▇▇▅▄▃▂▂▂` | 0.487 |
| [[Cell state]] | `▄▃▁▁▂▄▅▆▇█████▇▅▂▁▁▃▄▅▆▆` | final +0.088 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
