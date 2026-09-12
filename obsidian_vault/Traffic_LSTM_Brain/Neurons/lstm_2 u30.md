---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 30
---
# lstm_2 u30

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▆▅▄▃▃▂▃▄▄▅▅▅▆▆▇███▆▅▃▁▁` | -0.172 | -0.172 at t-1h |
| Quiet night | `▁▁▂▃▄▅▆▆▇███▇▇▆▆▆▅▅▆▆▇▇█` | +0.103 | +0.103 at t-1h |

Reacted most strongly at: **t-1h, t-2h, t-7h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▂▇█▇▆▅▅▃▂▁▁▂▂▃▅▇▇▇▇▇▇▇▆▇` | 0.747 |
| [[Input gate]] | `▅██▇▆▅▄▃▂▁▁▂▂▃▆▇▇▇▇▇▇█▇█` | 0.502 |
| [[Output gate]] | `▂▂▄▇███▇▆▅▄▄▃▂▁▁▃▄▆▆▆▆▆▅` | 0.400 |
| [[Cell state]] | `▆▆▅▄▄▃▃▃▄▄▄▅▅▆▇███▇▆▄▃▂▁` | final -0.459 |

A mean forget value of **0.75** means this unit keeps roughly
75% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
