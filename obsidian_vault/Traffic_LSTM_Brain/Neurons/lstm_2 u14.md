---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 14
---
# lstm_2 u14

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▂▄▆▆▇██████▇▄▁▂▃▄▅▅▆▆▆▆` | -0.059 | -0.274 at t-10h |
| Quiet night | `▄▆▄▁▂▄▄▄▄▄▅▅▄▆▅▆▆▆▄▆▆▇██` | +0.072 | +0.072 at t-1h |

Reacted most strongly at: **t-10h, t-24h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇██▇▅▄▃▂▁▁▂▂▃▅▇███▇▇▇▆▆▆` | 0.709 |
| [[Input gate]] | `▆▃▁▁▁▂▂▄▅▇████▇▅▃▂▁▁▁▁▁▁` | 0.422 |
| [[Output gate]] | `▇▆▃▁▁▁▁▂▄▅▇▇▇██▇▆▄▃▃▃▂▂▃` | 0.557 |
| [[Cell state]] | `▁▁▃▅▆▇███▇▇█▆▄▁▂▂▃▄▅▅▆▆▅` | final -0.121 |

A mean forget value of **0.71** means this unit keeps roughly
71% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
