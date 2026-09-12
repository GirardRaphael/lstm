---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 60
---
# lstm_1 u60

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▂▅▇▇████▇▇▆▅▃▂▃▅▆▇▇▆▆▅▃` | -0.056 | -0.109 at t-24h |
| Quiet night | `▇█▆▃▁▁▁▁▁▂▃▄▃▄▄▅▅▅▄▃▄▅▅▄` | +0.015 | +0.032 at t-23h |

Reacted most strongly at: **t-24h, t-10h, t-23h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▆▂▃▆████▆▅▃▂▂▂▂▁▂▄▅▅▄▄▃` | 0.569 |
| [[Input gate]] | `▄▄▃▂▂▂▃▄▆▇████▆▄▂▁▁▁▂▃▄▅` | 0.684 |
| [[Output gate]] | `▄▅▄▂▁▂▃▄▆▇████▇▅▄▂▁▁▂▃▄▆` | 0.702 |
| [[Cell state]] | `▁▂▅▇█████▇▆▆▅▄▂▃▅▆▇▇▆▆▅▄` | final -0.076 |

A mean forget value of **0.57** means this unit keeps roughly
57% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
