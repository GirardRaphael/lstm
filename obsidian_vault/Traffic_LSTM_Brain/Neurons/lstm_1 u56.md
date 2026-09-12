---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 56
---
# lstm_1 u56

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▂▅██▇▆▆▆▆▆▆▇▆▅▄▆▆▆▅▃▁▁▁▃` | -0.045 | -0.066 at t-3h |
| Quiet night | `▄▄▂▁▂▃▃▄▄▅▆▆▆▇▇█▇▇▅▅▄▄▂▁` | +0.007 | +0.025 at t-9h |

Reacted most strongly at: **t-3h, t-4h, t-2h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▅▆▇▇▇▆▅▄▃▂▁▁▁▂▅▆▇██▇▆▅▄` | 0.657 |
| [[Input gate]] | `▆▁▄▇▇▆▇▇▇▇███▇▅▃▅▆▇▇▇▇▇▆` | 0.586 |
| [[Output gate]] | `▃▄▂▁▃▄▅▅▆▇███▇▅▂▁▂▃▅▇█▇▇` | 0.588 |
| [[Cell state]] | `▂▅██▇▆▆▅▆▆▆▆▆▅▄▆▆▆▅▃▁▁▁▂` | final -0.074 |

A mean forget value of **0.66** means this unit keeps roughly
66% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
