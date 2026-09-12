---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 6
---
# lstm_2 u06

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▃▆█▇▇█▇▇▆▆▇▄▁▁▃▄▅▆▆▅▅▃▁` | -0.134 | -0.136 at t-24h |
| Quiet night | `██▄▁▂▃▃▃▂▃▄▄▃▅▄▅▅▅▃▅▅▆▆▆` | +0.060 | +0.085 at t-24h |

Reacted most strongly at: **t-24h, t-1h, t-11h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▆▆▇▇▆▅▄▃▁▁▁▂▃▆▆▇███▇▆▅▄` | 0.601 |
| [[Input gate]] | `▂▁▂▃▄▅▆▇▇███▇▅▁▁▁▂▃▃▄▅▅▅` | 0.557 |
| [[Output gate]] | `▁▂▂▂▃▄▅▆▇███▇▄▁▁▁▂▂▃▄▅▅▅` | 0.601 |
| [[Cell state]] | `▁▃▇█▇▇▇▇▇▆▆▆▄▂▂▃▅▆▆▆▅▅▄▂` | final -0.216 |

A mean forget value of **0.60** means this unit keeps roughly
60% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
