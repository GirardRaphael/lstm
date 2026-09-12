---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 27
---
# lstm_1 u27

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▂▂▄▅▆▇▇███▇▆▅▃▁▁▂▂▃▃▃▃▂` | -0.063 | -0.071 at t-9h |
| Quiet night | `▇██▆▃▂▂▁▁▁▁▂▂▃▃▄▄▅▄▄▃▃▃▃` | -0.020 | -0.029 at t-16h |

Reacted most strongly at: **t-9h, t-8h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▇▅▃▃▃▂▂▃▄▅▅▆▇▇▇▅▅▄▃▂▁▁▂` | 0.710 |
| [[Input gate]] | `▅▁▄▇█▇▇▇▇▆▆▅▅▄▃▃▅▆▇▇▅▄▄▃` | 0.415 |
| [[Output gate]] | `▅█▇▄▃▃▂▁▁▁▂▃▃▃▅▇▇▇▆▄▃▂▁▁` | 0.463 |
| [[Cell state]] | `▆▂▂▄▅▆▆▇███▇▆▅▂▁▁▂▂▂▂▂▂▁` | final -0.142 |

A mean forget value of **0.71** means this unit keeps roughly
71% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
