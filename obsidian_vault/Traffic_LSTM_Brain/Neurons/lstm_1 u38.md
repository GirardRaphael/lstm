---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 38
---
# lstm_1 u38

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▅▄▄▄▅▆▇█████▇▆▅▃▂▁▁▂▃▅▆` | -0.047 | -0.186 at t-6h |
| Quiet night | `▂▃▄▄▃▃▂▁▁▁▁▂▃▅▆▇███▇▆▅▄▃` | +0.008 | +0.025 at t-7h |

Reacted most strongly at: **t-6h, t-5h, t-7h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▃▆▇▆▄▃▃▂▂▂▃▄▄▄▆███▆▅▃▂▁` | 0.704 |
| [[Input gate]] | `▃▆▇▆▅▄▃▂▁▁▁▁▂▃▄▇██▇▆▅▄▃▃` | 0.494 |
| [[Output gate]] | `▅▅▆▆▆▆▅▄▃▂▁▁▁▂▃▅▆▇▇███▇▆` | 0.518 |
| [[Cell state]] | `▆▅▄▃▄▅▆▇████▇▇▆▄▃▁▁▁▂▃▅▅` | final -0.086 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
