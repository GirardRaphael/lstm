---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 57
---
# lstm_1 u57

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▃▁▄▇████▇▇▆▆▆▅▃▃▅▇██▇▆▅▃` | -0.068 | -0.117 at t-23h |
| Quiet night | `▇█▇▄▁▁▁▁▁▂▃▄▄▄▄▅▅▅▅▄▅▅▆▆` | +0.003 | -0.018 at t-19h |

Reacted most strongly at: **t-23h, t-24h, t-10h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▇█▃▁▄▅▆▆▆▅▄▃▂▂▃▄▁▁▂▃▄▅▅▆` | 0.621 |
| [[Input gate]] | `▃▆▅▂▁▂▂▂▂▂▃▃▃▃▄▄▃▃▂▃▅▆▇█` | 0.653 |
| [[Output gate]] | `▄█▆▃▁▁▂▂▃▃▄▅▅▅▇▇▅▄▂▂▃▄▅▆` | 0.625 |
| [[Cell state]] | `▂▁▄▇███▇▇▆▆▆▆▄▃▃▅▇██▇▆▅▃` | final -0.100 |

A mean forget value of **0.62** means this unit keeps roughly
62% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
