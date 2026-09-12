---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 41
---
# lstm_1 u41

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▃▃▃▄▅▆▇███▇▆▄▃▂▁▁▁▂▃▄▅▅` | +0.051 | +0.142 at t-15h |
| Quiet night | `▃▄▅▅▅▄▄▃▃▃▄▅▅▆▇▇██▇▆▅▄▃▁` | -0.018 | +0.065 at t-7h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▅██▇▇█▇▇▇▆▆▅▃▂▅▆▇▇▇▇▇▆` | 0.745 |
| [[Input gate]] | `▃▁▂▄▅▆▆▇████▇▅▂▁▁▂▃▃▃▄▃▃` | 0.486 |
| [[Output gate]] | `▄▁▄▇▇▇▇██████▆▃▂▄▄▅▅▄▄▄▃` | 0.419 |
| [[Cell state]] | `▄▃▃▃▄▅▆▇███▇▆▅▃▂▁▁▁▂▃▅▆▆` | final +0.154 |

A mean forget value of **0.75** means this unit keeps roughly
75% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
