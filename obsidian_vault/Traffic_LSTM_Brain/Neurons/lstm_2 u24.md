---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 24
---
# lstm_2 u24

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▆▃▂▂▂▁▂▃▄▅▄▅▇▇▅▅▄▄▅▅▆▇█` | +0.140 | -0.148 at t-18h |
| Quiet night | `▃▁▄██▇▇▇▆▅▄▃▄▂▃▂▂▃▅▅▅▄▄▄` | -0.069 | -0.112 at t-23h |

Reacted most strongly at: **t-18h, t-1h, t-21h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▃▂▆██▇▇▅▃▂▁▂▃▃▁▂▄▅▇▇▇▇▇` | 0.619 |
| [[Input gate]] | `▁▄▇▇▇▇▇▇▇▇▇█▇▅▂▄▅▆▇▇██▇▇` | 0.571 |
| [[Output gate]] | `▅▃▂▂▄▆▆▇██▇▇▇▅▃▁▁▁▂▃▅▆▇█` | 0.604 |
| [[Cell state]] | `█▇▃▁▁▂▁▂▃▄▅▅▆▇█▆▅▅▄▅▆▆▇█` | final +0.197 |

A mean forget value of **0.62** means this unit keeps roughly
62% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
