---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 13
---
# lstm_2 u13

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▅▄▄▅▄▃▂▁▁▂▂▃▄▅▅▅▆▆▇███▇` | +0.081 | +0.108 at t-4h |
| Quiet night | `█▆▆▇▇▆▆▆▅▅▄▃▃▂▂▁▁▁▃▄▅▆▆▇` | -0.041 | -0.099 at t-8h |

Reacted most strongly at: **t-4h, t-24h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▆▆▇▆▅▄▃▂▁▁▂▄▆▆▇▇▇▇████` | 0.748 |
| [[Input gate]] | `▄▁▂▃▄▅▆▇███▇▇▅▃▁▁▁▁▁▁▂▃▃` | 0.478 |
| [[Output gate]] | `█▃▁▂▃▄▅▆▇█▇▇▇▇▅▃▂▂▂▂▃▃▄▅` | 0.477 |
| [[Cell state]] | `▅▅▄▄▄▃▂▂▁▁▂▂▂▃▄▄▅▆▇██▇▇▅` | final +0.157 |

A mean forget value of **0.75** means this unit keeps roughly
75% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
