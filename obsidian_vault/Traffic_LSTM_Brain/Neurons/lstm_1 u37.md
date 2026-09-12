---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 37
---
# lstm_1 u37

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▅▇█▇██▇▇▇▆▇▆▄▄▅▆▇▇▇▇▆▅▄` | -0.067 | -0.154 at t-24h |
| Quiet night | `█▆▃▁▂▃▃▃▃▄▅▅▄▆▄▆▅▆▄▅▅▆▆▅` | +0.019 | +0.034 at t-24h |

Reacted most strongly at: **t-24h, t-10h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄▅▄▅▇███▇▆▄▃▂▁▁▂▂▄▅▆▆▅▅▄` | 0.557 |
| [[Input gate]] | `▃▃▃▂▂▄▅▆▆▇███▇▆▃▁▁▁▂▄▅▆▆` | 0.762 |
| [[Output gate]] | `▂▃▃▃▃▄▅▆▇▇███▇▅▂▁▁▁▂▄▅▆▆` | 0.750 |
| [[Cell state]] | `▁▅█████▇▇▇▇▇▆▅▄▆▆▇▇▇▇▇▆▅` | final -0.082 |

A mean forget value of **0.56** means this unit keeps roughly
56% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
