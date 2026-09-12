---
tags: [layer]
---
# lstm_2

32 LSTM units. Returns **only the final summary vector**.

## Behaviour on the rush-hour window

| | |
| --- | --- |
| Units carrying signal (`mean |h| > 0.05`) | 27 of 32 |
| Near-silent units | 5 |
| Long-memory units (`mean f > 0.75`) | 2 |
| Strongest activation | +0.436 |

That 5 units stay near zero on this window is normal and useful: dropout
during training pushes the layer to spread its representation, so different
subsets specialise in different regimes. Compare
[[Neurons - Rush Hour.canvas]] with [[Neurons - Quiet Night.canvas]] - the
lit-up subsets differ.

## Every unit, every hour

![[06_heatmap_lstm_2.png]]

Rows are units, columns are the 24 hours of the window. Red is a positive
activation, blue negative, white silent. The vertical stripes are hours where
most of the layer reacted at once.

## How the gates behaved

![[07_gates_lstm_2.png]]

## Most active units here

| Unit | mean \|h\| | final h | peaks at | role |
| --- | --- | --- | --- | --- |
| [[lstm_2 u22\|u22]] | +0.137 | +0.436 | t-1h | Mixed |
| [[lstm_2 u03\|u03]] | +0.128 | +0.110 | t-8h | Mixed |
| [[lstm_2 u12\|u12]] | +0.117 | +0.204 | t-4h | Mixed |
| [[lstm_2 u18\|u18]] | +0.117 | +0.161 | t-8h | Mixed |
| [[lstm_2 u05\|u05]] | +0.105 | +0.126 | t-23h | Mixed |
| [[lstm_2 u19\|u19]] | +0.102 | -0.210 | t-1h | Mixed |
| [[lstm_2 u07\|u07]] | +0.101 | -0.132 | t-24h | Mixed |
| [[lstm_2 u14\|u14]] | +0.095 | -0.059 | t-10h | Mixed |

Every unit has its own note in `Neurons/`.

Related: [[04 Architecture]] - [[Forget gate]] - [[Input gate]] - [[Output gate]] - [[Cell state]]
