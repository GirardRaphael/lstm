---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 9
---
# lstm_2 u09

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▁▄▇█▇▇▇▇████▆▄▂▅▆▆▆▆▅▅▄▃` | -0.063 | -0.103 at t-24h |
| Quiet night | `▅▅▄▃▄▄▃▂▁▁▂▃▃▅▄▆▇█▇███▇▇` | +0.051 | +0.055 at t-5h |

Reacted most strongly at: **t-24h, t-10h, t-1h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `█▃▁▃▄▄▅▆▆▆▆▅▆▆▆▃▂▂▂▂▂▃▃▃` | 0.584 |
| [[Input gate]] | `▃▁▂▃▃▃▃▃▄▅▇██▆▄▃▃▄▄▃▃▂▂▂` | 0.424 |
| [[Output gate]] | `▆▆▄▂▁▁▁▁▂▃▅▅▆▇█▇▅▄▂▁▁▁▂▃` | 0.457 |
| [[Cell state]] | `▁▄▇█▇▇██████▆▄▃▅▅▆▆▅▄▄▃▂` | final -0.143 |

A mean forget value of **0.58** means this unit keeps roughly
58% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
