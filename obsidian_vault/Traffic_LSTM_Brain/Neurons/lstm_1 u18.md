---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 18
---
# lstm_1 u18

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▄▂▁▃▄▅▆▇███▇▆▄▂▁▁▂▂▃▃▄▄▃` | -0.046 | -0.083 at t-9h |
| Quiet night | `▅▇█▇▅▃▂▂▁▁▁▂▃▄▅▅▆▆▆▅▅▅▅▅` | -0.001 | -0.019 at t-15h |

Reacted most strongly at: **t-9h, t-8h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▃▅▄▄▄▅▅▅▆▆▇██▇▄▃▃▃▄▃▃▂▂▁` | 0.685 |
| [[Input gate]] | `▆▂▂▄▆▇▇███▇▆▅▄▃▁▁▂▃▄▅▅▆▆` | 0.479 |
| [[Output gate]] | `▇▃▅▄▃▄▅▇▇███▇▇▅▃▃▁▁▂▂▄▅▆` | 0.480 |
| [[Cell state]] | `▄▂▂▃▄▅▆▇███▇▆▅▂▁▁▁▂▃▃▄▄▄` | final -0.093 |

A mean forget value of **0.68** means this unit keeps roughly
68% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
