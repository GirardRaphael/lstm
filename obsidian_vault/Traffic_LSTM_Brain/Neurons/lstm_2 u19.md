---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 19
---
# lstm_2 u19

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▅▆▅▅▅▆▆▇███▇▆▅▄▃▃▃▂▂▂▂▁` | -0.210 | -0.210 at t-1h |
| Quiet night | `▁▂▃▄▄▃▃▃▃▃▃▄▄▅▆▆▇▇█▇▇▇▇▇` | +0.034 | +0.040 at t-6h |

Reacted most strongly at: **t-1h, t-2h, t-4h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▄▁▂▃▃▂▂▁▁▁▁▂▄▅▃▃▃▃▄▄▄▅▆` | 0.724 |
| [[Input gate]] | `▇▄▂▂▂▂▂▄▅▆▇███▇▅▅▃▂▁▁▁▂▂` | 0.476 |
| [[Output gate]] | `█▃▁▂▂▂▂▃▃▄▄▄▅▆▆▃▃▂▂▃▃▃▄▄` | 0.454 |
| [[Cell state]] | `█▅▄▃▄▄▅▆▇███▇▇▆▄▂▂▁▁▁▁▁▁` | final -0.448 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
