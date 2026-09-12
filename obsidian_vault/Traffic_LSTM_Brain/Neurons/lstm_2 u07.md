---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 7
---
# lstm_2 u07

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▂▅▆▇▇██▇▆▆▆▅▃▂▃▄▄▄▄▃▄▄▃` | -0.132 | -0.259 at t-24h |
| Quiet night | `▆█▅▁▁▃▃▃▃▄▄▅▅▆▅▆▆▆▄▅▅▅▅▄` | +0.060 | +0.107 at t-23h |

Reacted most strongly at: **t-24h, t-23h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇██▇▇▆▅▄▂▁▁▁▂▄▇█████▇▇▇▇` | 0.691 |
| [[Input gate]] | `▂▂▂▃▄▅▆▇▇██▇▇▄▂▁▁▂▃▄▆▆▇▆` | 0.645 |
| [[Output gate]] | `▅▂▁▁▃▅▅▇▇██▇▆▄▂▁▂▂▃▅▆▇▇▇` | 0.679 |
| [[Cell state]] | `▁▂▅▇▇▇██▇▆▆▆▅▃▂▃▄▄▄▄▄▄▄▃` | final -0.178 |

A mean forget value of **0.69** means this unit keeps roughly
69% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
