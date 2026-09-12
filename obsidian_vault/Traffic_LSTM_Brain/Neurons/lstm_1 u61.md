---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 61
---
# lstm_1 u61

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▃▆▇▇▇███▇▇▇▆▄▃▄▅▆▇▆▆▆▆▅` | -0.034 | -0.103 at t-24h |
| Quiet night | `▇▇▅▁▁▂▂▂▂▂▃▃▂▄▃▄▄▄▃▄▄▆▇█` | +0.025 | +0.025 at t-1h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▃▄▆▇███▇▆▅▄▃▂▁▂▃▄▆▆▆▅▄▃` | 0.616 |
| [[Input gate]] | `▄▄▂▁▂▃▄▅▆▇████▆▃▁▁▁▂▃▅▅▆` | 0.661 |
| [[Output gate]] | `▃▁▃▃▂▃▄▅▆▇███▇▅▂▂▁▁▂▃▄▅▆` | 0.661 |
| [[Cell state]] | `▁▃▆▇▇▇███▇▇▇▆▅▃▅▅▆▇▆▆▆▆▅` | final -0.048 |

A mean forget value of **0.62** means this unit keeps roughly
62% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
