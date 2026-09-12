---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 36
---
# lstm_1 u36

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▆▇▆▅▄▂▁▁▁▂▂▃▃▅▆▇██▇▆▄▃▂` | -0.010 | +0.204 at t-7h |
| Quiet night | `▅▃▂▂▄▆▇███▇▆▄▃▂▂▁▁▂▄▅▆▇▇` | +0.010 | -0.020 at t-8h |

Reacted most strongly at: **t-7h, t-6h, t-8h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▇▇▆▅▄▃▂▁▁▁▂▂▃▆████▇▆▅▄▄` | 0.722 |
| [[Input gate]] | `▄▅▆▆▆▆▆▅▄▃▂▁▁▁▂▄▅▆▆▇███▇` | 0.537 |
| [[Output gate]] | `▄▆▆▅▅▅▅▄▄▃▂▁▁▁▃▄▄▅▅▆███▇` | 0.554 |
| [[Cell state]] | `▃▅▆▆▅▄▂▁▁▁▂▂▃▄▅▆▇██▇▅▄▃▃` | final -0.015 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
