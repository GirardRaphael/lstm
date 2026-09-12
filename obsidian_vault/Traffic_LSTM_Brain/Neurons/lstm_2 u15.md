---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 15
---
# lstm_2 u15

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▆▅▅▄▄▂▁▁▃▄▄▅▆▇▆▆▆▆▇▇▇▇▇` | +0.087 | -0.148 at t-17h |
| Quiet night | `▅▁▂▆█████▇▆▅▄▃▂▁▁▁▄▅▆▆▆▇` | -0.035 | -0.092 at t-8h |

Reacted most strongly at: **t-17h, t-24h, t-16h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇█▇▇▇▆▅▄▃▁▁▁▂▄▆▇▇▇▇█████` | 0.723 |
| [[Input gate]] | `▂▁▂▄▅▆▆▇██▇▇▆▄▂▁▂▂▂▃▃▃▃▃` | 0.499 |
| [[Output gate]] | `▇▂▁▃▅▆▆▇██▇▇▆▅▄▁▁▂▂▃▃▄▄▄` | 0.467 |
| [[Cell state]] | `▇▇▆▅▄▃▁▁▁▃▄▄▅▆▇███████▇▇` | final +0.190 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
