---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 17
---
# lstm_1 u17

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `█▇▃▁▁▁▁▁▁▂▂▂▂▄▆▅▄▃▃▃▃▃▄▅` | +0.052 | +0.105 at t-24h |
| Quiet night | `▃▃▅▇█▇▇▇▇▆▅▄▅▃▄▃▃▂▄▃▃▂▁▁` | -0.017 | -0.017 at t-1h |

Reacted most strongly at: **t-24h, t-23h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▅▃▄▅▆▅▆▅▄▂▁▁▁▃▄▃▃▄▄▃▃▃▃` | 0.659 |
| [[Input gate]] | `▂▆▄▁▁▂▂▂▂▂▃▃▃▃▄▄▃▃▃▄▆▇██` | 0.650 |
| [[Output gate]] | `▄▆▆▃▂▂▂▁▁▁▁▂▂▃▅▅▅▄▄▅▆▇██` | 0.655 |
| [[Cell state]] | `█▇▃▁▁▁▁▁▁▂▂▂▂▄▆▅▄▃▃▃▃▃▄▅` | final +0.072 |

A mean forget value of **0.66** means this unit keeps roughly
66% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
