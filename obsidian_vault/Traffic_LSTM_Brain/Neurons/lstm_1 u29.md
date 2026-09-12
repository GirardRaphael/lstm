---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 29
---
# lstm_1 u29

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▄▄▅▆▆▆▅▄▄▃▂▁▁▁▂▃▅▇██▇▆▅` | +0.022 | +0.134 at t-4h |
| Quiet night | `▇██▇▅▅▅▅▆▇▇███▇▇▆▅▄▂▂▁▁▁` | -0.053 | -0.053 at t-2h |

Reacted most strongly at: **t-4h, t-5h, t-11h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▅█▇▆▅▅▅▆▆▇██▆▆███▇▅▄▂▁` | 0.701 |
| [[Input gate]] | `▅▁▆██▆▆▆▅▆▆▆▇▆▅▄▆▇█▇▆▅▄▃` | 0.472 |
| [[Output gate]] | `▅▃▄▆▆▄▃▂▁▁▁▁▂▃▄▅▇███▇▆▅▄` | 0.522 |
| [[Cell state]] | `▄▄▅▆▆▇▆▆▅▄▃▂▁▁▁▂▄▅▇██▇▆▅` | final +0.043 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
