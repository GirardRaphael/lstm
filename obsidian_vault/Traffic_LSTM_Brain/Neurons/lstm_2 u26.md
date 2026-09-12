---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 26
---
# lstm_2 u26

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇▇▇▇▆▄▃▁▁▂▃▄▇█▇███▇▇▇██` | +0.033 | -0.289 at t-15h |
| Quiet night | `▇▃▁▂▄▅▆▇███▇▇▆▆▅▃▂▂▃▃▃▃▃` | -0.107 | -0.141 at t-22h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▁▃▅▆▇▇██████▆▃▂▃▃▄▄▄▄▄▃` | 0.572 |
| [[Input gate]] | `▃▁▂▄▅▆▇███▇▇▆▄▂▁▂▂▃▃▃▃▃▂` | 0.416 |
| [[Output gate]] | `▄▁▁▄▅▆▇███▇▇▇▅▂▁▂▂▃▃▃▃▃▃` | 0.394 |
| [[Cell state]] | `██▇▇▆▅▄▂▁▁▂▃▄▆████▇▇▇▇▇█` | final +0.119 |

A mean forget value of **0.57** means this unit keeps roughly
57% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
