---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 18
---
# lstm_2 u18

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅█▆▅▄▃▂▁▁▁▁▁▂▃▆███▇▇▇▆▆▆` | +0.161 | +0.228 at t-8h |
| Quiet night | `█▆▅▅▆▆▅▅▅▄▃▂▂▁▂▁▁▁▂▂▂▂▁▁` | -0.123 | -0.123 at t-1h |

Reacted most strongly at: **t-8h, t-9h, t-7h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇▇▆▆▆▅▃▂▁▁▁▂▃▅▇▇████▇▆▅▄` | 0.720 |
| [[Input gate]] | `▄▆▆▄▃▂▂▂▃▅▆▇▇▇██▇▆▅▄▃▃▂▁` | 0.485 |
| [[Output gate]] | `▅█▆▃▂▂▁▁▁▂▃▃▄▆██▇▆▅▄▄▄▃▄` | 0.525 |
| [[Cell state]] | `▅▇▇▅▄▃▂▁▁▁▁▁▂▃▆▇█████▇▇▇` | final +0.327 |

A mean forget value of **0.72** means this unit keeps roughly
72% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
