---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 12
---
# lstm_1 u12

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▆▇████▇▆▄▂▁▁▃▅▆▇▇▇█▇▇▇▇` | +0.054 | -0.299 at t-12h |
| Quiet night | `▇▆▅▅▅▅▆▇▇█████▇▇▆▄▃▃▂▂▁▁` | -0.209 | -0.209 at t-1h |

Reacted most strongly at: **t-12h, t-13h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▄▆▆▆▆▇▇████▇▅▃▄▄▄▄▄▄▄▄` | 0.681 |
| [[Input gate]] | `▅▁▃▅▅▅▅▆▇████▇▄▂▃▃▃▃▃▃▃▃` | 0.404 |
| [[Output gate]] | `▄▁▂▄▅▅▅▇▇████▇▄▂▂▂▃▃▃▃▃▂` | 0.371 |
| [[Cell state]] | `▆▆▇██▇▇▆▅▃▂▁▁▂▄▆▇███████` | final +0.267 |

A mean forget value of **0.68** means this unit keeps roughly
68% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
