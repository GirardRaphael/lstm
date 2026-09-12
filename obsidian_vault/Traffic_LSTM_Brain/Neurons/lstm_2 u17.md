---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 17
---
# lstm_2 u17

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▇▄▃▂▂▁▁▁▂▃▄▅▇█▇▅▄▃▃▄▄▅▆` | +0.059 | +0.126 at t-10h |
| Quiet night | `▂▁▂▄▅▇▇███▇▇▆▆▆▅▅▅▆▆▇▇▇█` | +0.080 | +0.080 at t-1h |

Reacted most strongly at: **t-10h, t-17h, t-16h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▁▄▅▆▆▇███▇▇▆▃▁▁▂▂▃▃▃▄▃` | 0.613 |
| [[Input gate]] | `▁▃▅▇▇▇████▇▇▆▄▂▃▄▅▆▅▅▅▄▃` | 0.479 |
| [[Output gate]] | `▁▆▇▅▅▅▆▇▇▇▇██▆▅▇▅▄▃▂▂▃▃▃` | 0.470 |
| [[Cell state]] | `▇▇▄▂▂▂▁▁▁▂▃▄▅▇█▇▅▄▃▃▃▄▅▆` | final +0.140 |

A mean forget value of **0.61** means this unit keeps roughly
61% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
