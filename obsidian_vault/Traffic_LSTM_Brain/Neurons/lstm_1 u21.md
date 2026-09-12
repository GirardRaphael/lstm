---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 21
---
# lstm_1 u21

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▅▅▅▅▆▆▅▅▄▃▂▁▁▁▂▂▃▅▆▇███` | +0.130 | +0.134 at t-3h |
| Quiet night | `▆▆▆▆▆▆▆▆▆▇▇█████▇▆▅▅▄▃▂▁` | -0.040 | -0.040 at t-1h |

Reacted most strongly at: **t-3h, t-2h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▇▇▆▄▄▃▃▃▄▅▅▆▆▇██▇▆▄▃▂▁▁` | 0.705 |
| [[Input gate]] | `▇▇▆▆▆▅▄▃▂▁▁▁▂▃▅▇▇██▇▆▄▄▃` | 0.481 |
| [[Output gate]] | `▃█▅▃▃▃▃▂▁▁▁▁▁▂▄▆▅▅▅▅▅▅▄▃` | 0.488 |
| [[Cell state]] | `▃▄▅▅▅▆▆▆▅▄▃▂▁▁▁▂▂▃▅▆▇███` | final +0.274 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
