---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 26
---
# lstm_1 u26

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▅▅▄▃▃▂▂▂▃▃▄▄▅▇███▆▅▃▂▁▁` | -0.112 | -0.112 at t-1h |
| Quiet night | `▄▃▃▃▃▃▄▄▄▄▃▃▂▂▁▁▁▁▂▃▄▅▇█` | +0.026 | -0.032 at t-9h |

Reacted most strongly at: **t-1h, t-2h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄█▇▅▄▃▂▁▁▁▂▂▃▄▆███▆▅▄▃▂▃` | 0.739 |
| [[Input gate]] | `▆█▆▅▄▄▃▂▁▁▁▁▂▃▆█▇█▇▆▆▄▃▃` | 0.496 |
| [[Output gate]] | `▄█▇▅▅▄▃▂▁▁▁▂▂▃▅▇██▇▇▆▅▄▄` | 0.500 |
| [[Cell state]] | `▅▅▅▄▄▃▂▂▂▂▃▄▅▆▇███▇▅▃▂▁▁` | final -0.238 |

A mean forget value of **0.74** means this unit keeps roughly
74% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
