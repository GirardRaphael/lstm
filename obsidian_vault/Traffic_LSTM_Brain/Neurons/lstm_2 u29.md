---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 29
---
# lstm_2 u29

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▆▃▂▂▂▁▁▂▃▄▄▆██▆▅▄▄▄▄▅▆█` | +0.154 | +0.162 at t-10h |
| Quiet night | `▃▁▃▇█▇████▇▆▇▅▆▅▅▄▆▅▅▄▄▃` | -0.079 | -0.116 at t-23h |

Reacted most strongly at: **t-10h, t-18h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▁▂▅▇███▇▆▅▄▅▄▃▁▂▃▄▅▅▅▅▄` | 0.523 |
| [[Input gate]] | `▁▆▇▆▅▅▆▆▇███▇▅▄▅▅▆▆▆▆▇▆▆` | 0.631 |
| [[Output gate]] | `▂▂▁▁▂▄▅▆▇███▇▅▂▂▁▁▁▂▃▄▅▅` | 0.595 |
| [[Cell state]] | `▇▆▂▁▁▂▁▂▂▃▄▄▅▇█▆▅▄▃▃▄▄▆▇` | final +0.250 |

A mean forget value of **0.52** means this unit keeps roughly
52% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
