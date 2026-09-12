---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 1
---
# lstm_2 u01

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▄▄▄▄▅▆▇████▇▆▅▃▂▁▁▁▁▂▃▃` | -0.117 | -0.179 at t-5h |
| Quiet night | `▁▃▄▄▄▃▃▃▃▃▄▅▆▆▇████▇▇▆▆▆` | +0.058 | +0.081 at t-7h |

Reacted most strongly at: **t-5h, t-6h, t-4h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▇▇▇▆▅▄▃▂▁▁▂▄▆▆▇▇███▇▇▇` | 0.736 |
| [[Input gate]] | `▇█▇▆▅▄▄▃▂▁▁▁▂▄▇██▇▆▆▅▄▄▃` | 0.466 |
| [[Output gate]] | `█▆▄▄▄▄▃▃▂▁▁▁▁▄▇▅▄▄▅▆▆▆▆▇` | 0.482 |
| [[Cell state]] | `▅▄▃▄▄▅▆▇████▇▆▅▃▂▁▁▁▁▂▃▃` | final -0.232 |

A mean forget value of **0.74** means this unit keeps roughly
74% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
