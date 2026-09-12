---
tags: [neuron, lstm_2]
layer: lstm_2
unit: 10
---
# lstm_2 u10

**Role on the rush-hour window:** Mixed - balances recent hours against the daily pattern

## Hidden state across the 24-hour window

| Window | Trace | final `h` | peak |
| --- | --- | --- | --- |
| Rush hour | `▇▆▅▃▂▂▁▁▂▄▆▆▆▆▆▆▆▅▅▅▆▆▇█` | +0.164 | -0.308 at t-18h |
| Quiet night | `▄▁▂▅▇████▇▇▆▆▄▄▃▃▃▄▅▅▅▄▄` | -0.095 | -0.171 at t-23h |

Reacted most strongly at: **t-18h, t-17h, t-19h**

## Gates on the rush-hour window

| Gate | Trace | mean |
| --- | --- | --- |
| [[Forget gate]] | `▅▁▂▅▇▇██▇▇▅▅▅▄▂▁▂▃▄▅▅▅▅▄` | 0.465 |
| [[Input gate]] | `▁▃▅▇▇███▇▇▆▅▅▂▁▂▃▅▆▇███▇` | 0.583 |
| [[Output gate]] | `▄▂▃▅▇▇███▇▆▆▅▃▁▁▂▃▄▆▇▇▇▆` | 0.556 |
| [[Cell state]] | `▇▆▄▂▂▂▁▁▂▄▅▆▆▇▇▆▆▅▅▅▆▆▇█` | final +0.251 |

A mean forget value of **0.47** means this unit keeps roughly
47% of its memory from one hour to the next.

## Where to see it

- [[Neurons - Rush Hour.canvas]] and [[Neurons - Quiet Night.canvas]]
- [[t00.canvas]] onwards, for this hour by hour
- Part of [[lstm_2]]
