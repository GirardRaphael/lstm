---
tags: [mechanism, gate]
---
# Cell state

`c_t = f * c_(t-1) + i * g`

The memory itself - the reason an LSTM can connect an hour to something that
happened a day earlier. It is never shown to the next layer directly:
[[Output gate]] decides how much of it leaks out as `h_t = o * tanh(c_t)`.

## How the memory built up on the rush-hour window

Mean `|c|` across the 64 units of [[lstm_1]], hour by hour:

`▆▇▆▅▃▂▃▃▃▂▂▁▁▃▇▇████▇▆▇▇`

| | |
| --- | --- |
| Mean magnitude at the start | 0.094 |
| Mean magnitude at the end | 0.103 |
| Units saturated at the end (`|c| > 0.9`) | 0 of 64 |

A rising line means the layer is accumulating information as it walks through
the day rather than resetting at every hour.

Related: [[Forget gate]] - [[Input gate]] - [[Candidate memory]] - [[Output gate]]
