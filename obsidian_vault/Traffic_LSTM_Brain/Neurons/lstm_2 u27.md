---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 27
---
# lstm_2 u27

**Role on the rush-hour window:** Dormant - contributes almost nothing on this sample

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▄▁▁▃▃▄▄▅▅▆▇▇▇█▆▄▂▂▃▄▄▄▅` | +0.007 | +0.083 at t-10h |
| Quiet night | `▆▆▇██▇▆▄▃▂▁▁▂▂▃▃▄▄▅▅▅▄▄▃` | -0.067 | -0.090 at t-13h |

Reacted most strongly at: **t-10h, t-11h, t-22h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▆▅▄▄▃▃▄▅▆▇█████▇▆▄▃▂▁▁▁▂` | 0.643 |
| [[Input gate]] | `▁▅█▇▅▅▅▅▅▅▇██▇▆█▇▇▆▄▃▂▂▁` | 0.350 |
| [[Output gate]] | `▃▇█▆▄▃▃▂▁▁▁▂▃▄▆███▇▅▄▄▃▃` | 0.444 |
| [[Cell state]] | `▅▄▁▁▂▃▃▄▅▆▇███▇▅▃▂▂▃▃▄▄▄` | final +0.021 |

A mean forget value of **0.64** means this unit keeps roughly
64% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
