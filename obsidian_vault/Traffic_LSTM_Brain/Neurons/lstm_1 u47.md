---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 47
---
# lstm_1 u47

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇▄▁▁▁▁▂▂▃▃▃▃▅▇▆▅▄▃▃▃▄▅▆` | +0.059 | +0.089 at t-24h |
| Quiet night | `▂▁▃▆█████▇▆▅▅▄▄▃▃▂▄▄▃▃▃▃` | -0.011 | -0.017 at t-23h |

Reacted most strongly at: **t-24h, t-23h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▅▅▇██▇▆▅▃▂▁▁▂▄▄▅▇▇▇▆▅▄` | 0.631 |
| [[Input gate]] | `▃▃▃▂▂▃▄▅▆▇███▇▅▃▂▁▁▂▃▄▅▆` | 0.646 |
| [[Output gate]] | `▄▅▄▂▂▄▅▆▇▇███▇▆▄▂▁▁▂▄▆▇█` | 0.646 |
| [[Cell state]] | `█▇▄▁▁▁▁▂▂▃▃▃▃▅▇▆▅▄▃▃▃▄▅▆` | final +0.083 |

A mean forget value of **0.63** means this unit keeps roughly
63% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
