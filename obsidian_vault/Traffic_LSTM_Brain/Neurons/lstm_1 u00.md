---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 0
---
# lstm_1 u00

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▂▅▆▅▅▅▅▄▃▂▁▁▁▂▃▅▆▇████▇▇` | +0.123 | +0.142 at t-4h |
| Quiet night | `▂▁▁▂▃▄▅▆▇███▇▇▆▆▄▃▂▃▂▂▂▂` | +0.018 | +0.060 at t-14h |

Reacted most strongly at: **t-4h, t-5h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅█▄▁▁▁▁▁▂▃▄▄▅▆▇▇▅▄▃▂▂▁▁▂` | 0.720 |
| [[Input gate]] | `▆▁▃▅▇▇▇███▇▇▇▅▃▂▃▃▄▅▅▅▅▅` | 0.452 |
| [[Output gate]] | `▃█▄▁▂▃▄▄▆▆▇▇▇▆▅▅▂▂▂▂▃▃▃▄` | 0.471 |
| [[Cell state]] | `▂▄▆▆▅▅▄▃▂▂▁▁▁▂▃▅▆▇████▇▇` | final +0.273 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
