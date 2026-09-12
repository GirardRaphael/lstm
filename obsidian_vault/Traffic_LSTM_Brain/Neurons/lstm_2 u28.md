---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 28
---
# lstm_2 u28

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▂▂▁▂▃▄▆██▇▆▅▃▂▂▂▂▁▁▁▁▁▁` | -0.028 | +0.310 at t-15h |
| Quiet night | `▁▅██▆▅▄▃▂▂▂▂▂▃▃▄▆▇█▇▇▇██` | +0.145 | +0.149 at t-21h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▂▅▇▇███▇▆▆▆▅▃▁▂▃▄▄▄▅▅▅` | 0.602 |
| [[Input gate]] | `▃▁▁▄▅▆▇███▇▇▆▄▂▁▁▂▃▃▃▃▃▃` | 0.383 |
| [[Output gate]] | `▅▁▁▃▅▆▇███▇▆▆▄▂▁▁▂▂▃▃▃▃▃` | 0.342 |
| [[Cell state]] | `▁▁▁▂▂▃▄▆▇█▇▇▅▄▂▂▂▂▂▁▁▁▁▁` | final -0.116 |

A mean forget value of **0.60** means this unit keeps roughly
60% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
