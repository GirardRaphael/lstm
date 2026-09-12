---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 12
---
# lstm_2 u12

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▇▆▅▄▃▂▁▁▁▁▁▁▂▃▅▆▇▇██▇▇▇` | +0.204 | +0.236 at t-4h |
| Quiet night | `█▄▂▂▂▁▁▂▂▃▃▃▃▂▂▁▁▁▃▃▃▃▄▄` | -0.058 | -0.070 at t-9h |

Reacted most strongly at: **t-4h, t-5h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▅▅▅▄▃▂▁▁▁▂▃▅▇▇▆▆▆▆▆▆▆▇` | 0.724 |
| [[Input gate]] | `▄▆▇▇▇▆▅▄▂▁▁▁▁▂▄▆▇███▇▆▆▅` | 0.515 |
| [[Output gate]] | `██▆▅▄▃▃▂▁▁▁▂▂▅██▇▆▆▆▆▆▆▇` | 0.526 |
| [[Cell state]] | `▅▆▆▆▅▄▃▂▁▁▁▁▂▃▄▅▆▇▇███▇▇` | final +0.337 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
