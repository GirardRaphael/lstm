---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 5
---
# lstm_2 u05

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇█▆▄▃▃▂▂▁▁▁▁▁▃▆▇▇▆▅▅▅▄▄▅` | +0.126 | +0.265 at t-23h |
| Quiet night | `▅▄▄▆▇▇▇███▇▆▆▄▄▃▂▂▃▂▂▁▁▁` | -0.052 | -0.052 at t-1h |

Reacted most strongly at: **t-23h, t-9h, t-8h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▆▅▄▃▂▂▁▁▂▂▄▆██▇▇▆▅▅▄▄▄` | 0.666 |
| [[Input gate]] | `▇▇▄▂▁▁▁▁▂▃▄▅▅▇█▆▅▄▄▄▄▄▄▅` | 0.523 |
| [[Output gate]] | `▇█▆▄▃▃▂▁▁▁▂▂▂▅▇▇▆▆▅▅▅▅▅▆` | 0.545 |
| [[Cell state]] | `▇█▆▅▄▃▂▂▁▁▁▁▁▃▆▇▇▇▆▆▅▅▅▅` | final +0.203 |

A mean forget value of **0.67** means this unit keeps roughly
67% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
