---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 5
---
# lstm_1 u05

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▅▆▇▇▇▆▅▄▂▁▁▁▂▄▅▆▇███▇▆▆` | +0.042 | +0.099 at t-5h |
| Quiet night | `▆▅▄▃▄▄▅▆▇███▇▇▆▅▄▃▂▁▁▁▁▁` | -0.044 | -0.047 at t-4h |

Reacted most strongly at: **t-5h, t-13h, t-6h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▅▇▇▆▆▆▇▇▇██▇▅▄▆▆▆▆▄▃▃▂` | 0.661 |
| [[Input gate]] | `▅▁▃▅▆▆▆▇▇████▇▄▂▃▃▄▄▄▄▄▃` | 0.419 |
| [[Output gate]] | `▆▁▃▅▆▆▇▇█████▇▄▂▃▃▄▅▅▅▅▄` | 0.440 |
| [[Cell state]] | `▄▅▆▇▇▆▅▄▃▂▁▁▁▂▄▅▇▇███▇▆▆` | final +0.107 |

A mean forget value of **0.66** means this unit keeps roughly
66% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
