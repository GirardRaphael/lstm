---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 11
---
# lstm_2 u11

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `██▇▆▅▄▃▂▁▁▁▃▄▆██▇▆▆▅▅▅▆▇` | +0.063 | -0.173 at t-15h |
| Quiet night | `▅▃▁▁▂▄▆▇███▇▇▆▆▅▄▄▄▄▅▅▆▆` | +0.010 | -0.052 at t-21h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▃▁▂▄▅▆▆▇▇▇▆▆▆▅▂▁▁▂▃▄▅▆▆` | 0.653 |
| [[Input gate]] | `▅▁▁▄▆▆▇██▇▆▆▆▄▂▁▁▂▃▄▄▄▅▄` | 0.408 |
| [[Output gate]] | `█▇▃▁▁▂▂▂▂▃▃▄▄▆▇▆▄▂▁▂▂▃▅▆` | 0.474 |
| [[Cell state]] | `██▇▆▅▄▃▂▁▁▂▃▅▇███▇▇▆▆▆▇█` | final +0.103 |

A mean forget value of **0.65** means this unit keeps roughly
65% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
