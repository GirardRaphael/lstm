---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 22
---
# lstm_2 u22

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▅▄▄▄▄▃▂▁▁▁▁▁▁▂▃▃▄▄▅▅▆▇█` | +0.436 | +0.436 at t-1h |
| Quiet night | `█▅▂▁▂▃▄▅▆▇██▇▇▆▅▃▂▁▂▂▂▂▃` | -0.090 | -0.112 at t-6h |

Reacted most strongly at: **t-1h, t-2h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `██▆▅▅▄▃▂▁▁▂▂▃▆▇▇▇▇▆▇▇▇▇█` | 0.746 |
| [[Input gate]] | `▅▂▃▅▇▇██▇▇▅▅▄▃▂▁▂▃▄▅▆▆▇▇` | 0.470 |
| [[Output gate]] | `█▄▁▁▂▃▃▃▃▃▂▂▃▄▅▃▂▁▁▂▃▄▆▇` | 0.491 |
| [[Cell state]] | `▄▅▆▅▅▄▃▂▁▁▁▁▁▂▃▃▄▄▅▅▆▇▇█` | final +0.714 |

A mean forget value of **0.75** means this unit keeps roughly
75% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
