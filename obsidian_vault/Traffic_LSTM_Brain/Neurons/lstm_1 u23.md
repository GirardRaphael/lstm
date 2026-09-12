---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 23
---
# lstm_1 u23

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▇██▇▆▅▄▃▂▁▁▁▃▅▇████▇▇▆▆` | -0.023 | -0.201 at t-13h |
| Quiet night | `█▆▅▄▄▄▄▅▅▅▅▅▅▄▄▃▂▂▁▁▁▁▁▂` | -0.129 | -0.144 at t-6h |

Reacted most strongly at: **t-13h, t-14h, t-12h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▃▅▆▆▇▇█████▇▄▂▃▃▄▄▄▄▄▄` | 0.642 |
| [[Input gate]] | `▅▁▂▃▄▄▅▆▇▇███▇▄▂▂▂▂▂▂▂▂▂` | 0.443 |
| [[Output gate]] | `▅▁▃▅▆▆▆▇▇████▇▄▂▃▃▄▄▃▃▃▃` | 0.397 |
| [[Cell state]] | `▆▇▇▇▇▆▅▄▃▂▁▁▁▂▄▆███▇▇▆▆▅` | final -0.076 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
