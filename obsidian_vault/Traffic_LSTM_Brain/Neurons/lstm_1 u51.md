---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 51
---
# lstm_1 u51

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `██▇▆▅▄▃▂▁▁▂▃▄▆████▇▇▇▆▅▆` | +0.042 | +0.074 at t-10h |
| Quiet night | `▁▁▂▅▆▇▇███▇▆▆▅▅▄▃▂▃▃▄▄▄▅` | +0.023 | +0.042 at t-16h |

Reacted most strongly at: **t-10h, t-24h, t-8h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▂▄▆▇▇▆▇▇▇▇██▇▅▄▅▆▇▆▄▃▂▁` | 0.647 |
| [[Input gate]] | `▅▁▃▅▆▆▇████▇▇▅▃▁▂▂▄▅▅▆▆▆` | 0.519 |
| [[Output gate]] | `▅▂▃▃▄▅▆▇▇███▇▆▄▂▁▁▂▃▄▅▆▆` | 0.539 |
| [[Cell state]] | `▇▇▇▅▄▄▂▁▁▁▂▂▃▅▇███▇▇▆▅▄▅` | final +0.073 |

A mean forget value of **0.65** means this unit keeps roughly
65% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
