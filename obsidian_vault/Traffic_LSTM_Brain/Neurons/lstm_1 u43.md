---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 43
---
# lstm_1 u43

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇▃▁▂▂▂▂▁▁▂▂▃▆▇▅▄▃▄▆▇███` | +0.061 | +0.065 at t-2h |
| Quiet night | `▅▄▄▆▇▆▆▅▄▄▃▂▂▁▂▁▁▁▃▄▅▆▇█` | -0.001 | -0.037 at t-9h |

Reacted most strongly at: **t-2h, t-3h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆█▂▂▅▇███▇▆▆▅▄▃▃▁▂▄▄▄▃▂▂` | 0.626 |
| [[Input gate]] | `▁▂▃▂▁▁▂▂▃▅▆███▆▄▃▂▂▁▂▂▂▃` | 0.605 |
| [[Output gate]] | `▃▆▄▂▁▂▂▃▄▅▆▇██▇▆▄▃▂▂▃▄▅▆` | 0.629 |
| [[Cell state]] | `█▇▃▁▂▂▂▂▁▁▂▂▃▆▇▅▄▄▄▆▇███` | final +0.093 |

A mean forget value of **0.63** means this unit keeps roughly
63% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
