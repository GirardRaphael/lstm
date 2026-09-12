---
tags: [neuron, lstm_1]
layer: lstm_1
unit: 30
---
# lstm_1 u30

**Role on the rush-hour window:** Long memory - carries information across the whole day

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▅▃▃▄▅▇▇███▇▇▆▅▃▂▁▁▃▄▆▇▇▇` | +0.037 | -0.234 at t-8h |
| Quiet night | `▄▇█▇▅▄▃▂▂▂▃▄▅▆▇███▇▅▄▂▂▁` | -0.009 | +0.029 at t-8h |

Reacted most strongly at: **t-8h, t-7h, t-9h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▄█▆▄▃▃▂▁▁▁▂▂▃▄▆█▇▆▆▅▄▄▄▄` | 0.755 |
| [[Input gate]] | `▃█▇▃▂▂▁▁▁▂▃▄▅▆▇█▇▆▅▄▄▅▅▅` | 0.577 |
| [[Output gate]] | `▄██▅▄▄▃▂▁▁▁▁▂▃▅▇▇▇▇▇▇█▇▇` | 0.612 |
| [[Cell state]] | `▅▃▃▄▅▆▇███▇▇▆▅▃▂▁▁▃▄▅▆▇▇` | final +0.054 |

A mean forget value of **0.76** means this unit keeps roughly
76% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_1]]
