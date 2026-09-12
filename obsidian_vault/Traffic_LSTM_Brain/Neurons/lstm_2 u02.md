---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 2
---
# lstm_2 u02

**Role on the rush-hour window:** Long memory - carries information across the whole day

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▆▅▅▄▄▄▅▆▇█████▇▆▆▅▅▄▄▃▂▁` | -0.293 | -0.293 at t-1h |
| Quiet night | `▄▅▇█▇▆▅▄▃▂▂▁▁▁▁▂▃▄▅▅▆▆▇▇` | +0.031 | -0.063 at t-12h |

Reacted most strongly at: **t-1h, t-2h, t-3h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `██▆▄▄▄▃▂▁▁▁▂▃▅▇▇▆▆▆▆▆▇▇█` | 0.770 |
| [[Input gate]] | `▅▁▁▃▅▆▇███▇▇▆▄▂▁▁▂▂▃▃▄▄▄` | 0.433 |
| [[Output gate]] | `█▂▁▃▄▅▅▆▆▆▅▅▅▅▄▂▂▂▃▃▄▄▅▅` | 0.511 |
| [[Cell state]] | `▆▅▄▄▄▅▅▆▇█████▇▆▆▅▅▄▄▃▂▁` | final -0.575 |

A mean forget value of **0.77** means this unit keeps roughly
77% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
