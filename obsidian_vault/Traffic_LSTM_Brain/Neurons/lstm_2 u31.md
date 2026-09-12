---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 31
---
# lstm_2 u31

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▃▂▂▃▄▅▆██▇▆▅▄▃▃▃▂▂▂▂▁▁▁` | -0.074 | +0.142 at t-15h |
| Quiet night | `▁▃▆▇▆▅▄▄▃▃▃▃▃▄▄▅▆▇▇▇▇▇██` | +0.098 | +0.098 at t-1h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▁▃▇████▆▅▄▅▅▄▂▂▃▄▆▆▅▅▅▄` | 0.595 |
| [[Input gate]] | `▄▁▂▅▆▇▇████▇▇▅▂▁▂▃▄▅▆▆▅▄` | 0.427 |
| [[Output gate]] | `▆▁▁▄▆▆▇███▇▇▇▅▃▁▂▃▄▄▅▅▅▄` | 0.394 |
| [[Cell state]] | `▄▂▂▃▄▅▅▇██▇▇▆▅▄▃▃▃▃▃▂▂▁▁` | final -0.221 |

A mean forget value of **0.60** means this unit keeps roughly
60% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
