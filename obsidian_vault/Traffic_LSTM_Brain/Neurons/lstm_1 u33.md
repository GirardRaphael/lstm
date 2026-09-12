---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 33
---
# lstm_1 u33

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▆█▆▄▃▂▂▁▁▁▁▂▃▅██▆▅▃▃▃▃▃` | -0.003 | +0.207 at t-8h |
| Quiet night | `▇▅▂▁▂▄▅▇▇██▇▆▆▅▄▃▂▂▂▂▃▂▂` | -0.043 | -0.050 at t-21h |

Reacted most strongly at: **t-8h, t-22h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▄▆▅▅▅▆▇▇▇██▇▅▄▅▄▄▄▃▂▃▂` | 0.573 |
| [[Input gate]] | `▃▇▇▄▂▁▁▁▂▃▄▅▅▆▇█▇▅▃▂▁▁▁▂` | 0.503 |
| [[Output gate]] | `▅▅█▆▃▂▂▂▃▄▅▆▆▇▇██▆▄▂▁▁▂▃` | 0.500 |
| [[Cell state]] | `▃▆█▆▄▃▂▂▁▁▁▁▂▃▅██▇▅▄▃▃▃▃` | final -0.007 |

A mean forget value of **0.57** means this unit keeps roughly
57% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
