---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 3
---
# lstm_2 u03

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆█▆▅▄▃▂▂▁▁▁▁▁▂▆███▇▇▆▅▅▅` | +0.110 | +0.265 at t-8h |
| Quiet night | `▇▅▃▃▃▄▅▆▇██▇▇▆▅▄▃▃▂▂▂▁▁▁` | -0.082 | -0.082 at t-1h |

Reacted most strongly at: **t-8h, t-7h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▇▇▇▅▄▃▁▁▂▃▄▆▇▇████▇▆▅▅` | 0.702 |
| [[Input gate]] | `▇█▆▄▃▂▂▁▁▁▂▃▃▆██▇▆▆▅▅▄▄▅` | 0.501 |
| [[Output gate]] | `██▅▃▃▂▂▁▁▁▁▂▂▅█▇▆▅▅▅▅▅▅▆` | 0.505 |
| [[Cell state]] | `▆▇▆▅▄▄▃▂▁▁▁▁▁▃▅▇███▇▇▆▅▅` | final +0.191 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
