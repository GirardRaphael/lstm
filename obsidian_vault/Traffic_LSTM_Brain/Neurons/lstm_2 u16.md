---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 16
---
# lstm_2 u16

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇▇▇▆▅▄▂▁▁▂▃▅▇██▇▇▇▇▇▇▇█` | +0.064 | -0.224 at t-15h |
| Quiet night | `▅▃▁▂▃▅▆▇███▇▇▆▅▄▂▁▁▂▂▃▃▃` | -0.039 | -0.079 at t-6h |

Reacted most strongly at: **t-15h, t-16h, t-14h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▂▁▄▅▆▇████▇▇▆▄▁▁▁▂▂▃▄▅▅` | 0.592 |
| [[Input gate]] | `▃▁▂▄▅▆▇███▇▇▆▄▂▁▁▁▂▂▃▃▄▄` | 0.420 |
| [[Output gate]] | `▆▂▁▂▄▅▆▇█████▇▅▂▁▁▁▁▂▃▃▄` | 0.399 |
| [[Cell state]] | `██▇▆▆▅▄▂▁▁▂▃▅▇███▇▇▆▆▇▇█` | final +0.176 |

A mean forget value of **0.59** means this unit keeps roughly
59% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
