---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 52
---
# lstm_1 u52

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▅▄▄▄▄▄▅▆▇▇███▇▆▄▃▂▁▁▂▂▃` | -0.080 | -0.128 at t-4h |
| Quiet night | `▂▃▃▃▄▄▃▃▂▂▁▁▁▁▂▃▄▅▅▆▇▇██` | +0.022 | +0.022 at t-1h |

Reacted most strongly at: **t-4h, t-5h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▄▅▇▆▅▄▄▃▃▄▄▅▅▅▆▇██▇▅▃▂▁` | 0.704 |
| [[Input gate]] | `▅▆▆▆▆▆▄▃▂▂▁▁▁▂▄▆▇██▇▆▅▄▃` | 0.468 |
| [[Output gate]] | `▃█▆▅▅▅▅▃▃▂▁▁▁▁▃▅▅▆▇▇▆▅▄▃` | 0.454 |
| [[Cell state]] | `▆▅▄▄▃▄▄▅▆▇▇███▇▆▄▃▂▁▁▁▂▃` | final -0.185 |

A mean forget value of **0.70** means this unit keeps roughly
70% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
