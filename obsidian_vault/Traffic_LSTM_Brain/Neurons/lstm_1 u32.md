---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 32
---
# lstm_1 u32

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▅▅▂▁▂▃▃▄▄▅▆▇▇██▆▄▂▁▁▂▃▄` | -0.015 | +0.143 at t-9h |
| Quiet night | `▁▁▁▂▄▅▅▅▄▄▃▃▂▂▂▂▂▃▄▅▆▆▇█` | +0.120 | +0.120 at t-1h |

Reacted most strongly at: **t-9h, t-10h, t-11h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▂▂▄▄▄▅▅▆▇▇███▆▄▄▄▄▄▃▂▂▁` | 0.659 |
| [[Input gate]] | `▄▁▆█▇▅▅▄▄▄▃▄▄▄▄▅▇██▇▅▄▃▂` | 0.453 |
| [[Output gate]] | `▄▅▇▇▅▃▂▂▁▁▁▂▃▃▅▇██▇▆▅▄▃▃` | 0.525 |
| [[Cell state]] | `▄▅▅▂▁▁▂▃▃▄▅▆▇███▆▄▂▁▁▂▃▃` | final -0.034 |

A mean forget value of **0.66** means this unit keeps roughly
66% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
