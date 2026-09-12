"""Traffic volume forecasting with an LSTM neural network.

A small, readable reference implementation built around three ideas:

1. `data`       - turn a raw hourly CSV into supervised sequences, leak-free.
2. `model`      - a stacked LSTM regressor (24 hours in, N horizons out).
3. `introspect` - replay the trained LSTM cell by hand so every gate and every
                  neuron activation can be inspected, exported and visualised.
"""

__version__ = "1.0.0"
