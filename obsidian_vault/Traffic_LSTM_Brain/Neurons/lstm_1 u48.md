---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 48
---
# lstm_1 u48

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇████▇▅▃▂▁▁▂▆█▇████████` | +0.035 | -0.240 at t-14h |
| Quiet night | `▇▅▃▄▅▆▇▇████▇▇▆▅▄▂▂▂▂▁▁▁` | -0.156 | -0.156 at t-1h |

Reacted most strongly at: **t-14h, t-13h, t-15h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▁▂▅▆▆▆▇█████▆▃▂▂▃▄▄▃▃▃▂` | 0.447 |
| [[Input gate]] | `▄▁▂▄▅▆▆▇█████▆▄▂▂▂▃▄▄▄▄▃` | 0.456 |
| [[Output gate]] | `▃▁▂▃▄▅▆▇▇███▇▆▃▁▁▂▂▂▂▃▃▂` | 0.359 |
| [[Cell state]] | `█▇▇▇▇▆▅▄▃▂▁▁▂▄▇█████████` | final +0.165 |

A mean forget value of **0.45** means this unit keeps roughly
45% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
