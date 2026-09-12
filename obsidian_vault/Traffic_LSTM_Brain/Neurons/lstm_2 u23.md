---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 23
---
# lstm_2 u23

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▂▇██████▇▇█▆▄▁▃▄▆▆▆▆▆▅▃` | -0.127 | -0.195 at t-24h |
| Quiet night | `▇█▅▂▁▂▁▁▁▂▃▃▃▅▄▅▅▆▅▆▆▇██` | +0.053 | +0.053 at t-1h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▆▅▇▇▆▅▄▄▃▃▃▄▆██▇██▇▆▄▃▁` | 0.629 |
| [[Input gate]] | `▂▄▄▂▁▁▁▂▃▅▇█▇▆▄▄▃▃▃▃▃▄▃▃` | 0.533 |
| [[Output gate]] | `▇█▅▂▁▁▁▁▃▄▅▅▅▆██▇▅▄▄▄▅▅▆` | 0.569 |
| [[Cell state]] | `▁▂▆██▇███▇▇▇▆▄▁▃▄▆▆▆▅▅▄▃` | final -0.216 |

A mean forget value of **0.63** means this unit keeps roughly
63% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
