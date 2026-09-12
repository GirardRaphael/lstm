---
tags: [layer]
---
# lstm_1

64 LSTM units. Returns **one hidden state per hour**.

## Behaviour on the rush-hour window

| | |
| --- | --- |
| Units carrying signal (`mean |h| > 0.05`) | 22 of 64 |
| Near-silent units | 42 |
| Long-memory units (`mean f > 0.75`) | 1 |
| Strongest activation | +0.219 |

That 42 units stay near zero on this window is normal and useful: dropout
during training pushes the layer to spread its representation, so different
subsets specialise in different regimes. Compare
[[Neurons - Rush Hour.canvas]] with [[Neurons - Quiet Night.canvas]] - the
lit-up subsets differ.

## Every unit, every hour

![[06_heatmap_lstm_1.png]]

Rows are units, columns are the 24 hours of the window. Red is a positive
activation, blue negative, white silent. The vertical stripes are hours where
most of the layer reacted at once.

## How the gates behaved

![[07_gates_lstm_1.png]]

## Most active units here

| Unit | mean \|h\| | final h | peaks at | role |
| --- | --- | --- | --- | --- |
| [[lstm_1 u16\|u16]] | +0.118 | +0.072 | t-8h | Mixed |
| [[lstm_1 u12\|u12]] | +0.095 | +0.054 | t-12h | Mixed |
| [[lstm_1 u30\|u30]] | +0.085 | +0.037 | t-8h | Long memory |
| [[lstm_1 u36\|u36]] | +0.083 | -0.010 | t-7h | Mixed |
| [[lstm_1 u45\|u45]] | +0.079 | +0.167 | t-1h | Mixed |
| [[lstm_1 u00\|u00]] | +0.078 | +0.123 | t-4h | Mixed |
| [[lstm_1 u38\|u38]] | +0.072 | -0.047 | t-6h | Mixed |
| [[lstm_1 u48\|u48]] | +0.070 | +0.035 | t-14h | Mixed |

Every unit has its own note in `Neurons/`.

Related: [[04 Architecture]] - [[Forget gate]] - [[Input gate]] - [[Output gate]] - [[Cell state]]
