---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 49
---
# lstm_1 u49

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▂▃▃▃▄▅▆▇▇███▇▅▄▂▁▁▁▃▄▆▇█` | +0.076 | +0.080 at t-14h |
| Quiet night | `▁▂▄▅▅▅▅▅▅▅▆▆▆▇▇████▇▇▆▄▃` | +0.021 | +0.050 at t-7h |

Reacted most strongly at: **t-14h, t-15h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▅▇▇▇▇████▇▇▆▃▃▄▅▅▅▄▃▃▃` | 0.695 |
| [[Input gate]] | `▇▄▆██▇▇▇▇▇▆▅▅▄▃▅▇▇▇▅▃▂▁▁` | 0.447 |
| [[Output gate]] | `█▅▆▆▆▅▄▄▄▅▆▇███▇███▇▅▃▂▁` | 0.469 |
| [[Cell state]] | `▂▃▃▃▄▅▆▇▇██▇▆▅▄▂▁▁▁▃▄▆▇█` | final +0.175 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
