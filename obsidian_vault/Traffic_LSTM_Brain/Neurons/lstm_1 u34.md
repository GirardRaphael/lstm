---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 34
---
# lstm_1 u34

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅█▄▂▁▁▁▁▁▁▂▂▂▄▆▆▄▂▂▁▂▂▃▅` | +0.081 | +0.194 at t-23h |
| Quiet night | `▆▂▁▄▇████▇▆▄▅▄▃▃▂▂▃▄▄▃▂▁` | -0.035 | -0.035 at t-1h |

Reacted most strongly at: **t-23h, t-9h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅█▃▁▃▄▄▄▄▄▄▄▄▄▅▅▃▃▃▄▄▄▄▄` | 0.641 |
| [[Input gate]] | `▄▇▆▃▁▁▂▂▃▄▆▇▇██▇▆▄▃▂▃▄▅▆` | 0.684 |
| [[Output gate]] | `▄█▆▂▁▂▂▃▄▅▆▆▆▇▇▇▅▄▂▂▂▃▄▆` | 0.644 |
| [[Cell state]] | `▆█▅▂▁▁▁▁▁▂▂▂▃▄▆▇▅▃▂▂▂▃▄▅` | final +0.115 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
