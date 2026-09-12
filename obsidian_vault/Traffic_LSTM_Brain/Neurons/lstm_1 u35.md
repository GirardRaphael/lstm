---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 35
---
# lstm_1 u35

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▁▁▃▅▅▄▃▂▂▂▂▂▂▁▁▂▄▆██▇▅▄` | +0.043 | +0.219 at t-4h |
| Quiet night | `▆██▆▂▁▁▂▂▃▄▅▆▄▅▃▃▂▂▁▂▂▃▄` | -0.023 | -0.030 at t-19h |

Reacted most strongly at: **t-4h, t-5h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▄▅▇▇▆▅▄▃▂▁▁▁▂▃▅▆███▇▆▅▃` | 0.687 |
| [[Input gate]] | `▃▃▅▇▇▆▅▄▃▂▁▁▁▁▂▄▆▇██▇▇▅▄` | 0.544 |
| [[Output gate]] | `▅▇▅▅▆▆▅▄▃▂▁▁▁▂▄▅▅▆▇███▇▇` | 0.571 |
| [[Cell state]] | `▃▁▁▄▅▅▄▄▃▂▁▁▁▁▁▁▃▅▆██▇▅▄` | final +0.065 |

A mean forget value of **0.69** means this unit keeps roughly
69% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
