# Present today — Traffic LSTM

## Open the deck

```bash
cd presentation
npm install
python3 server/sidecar.py          # 0.0.0.0:43126 — Keras-weight replay
npm run dev -- --host 0.0.0.0 --port 43125
```

The sidecar needs NumPy + h5py (`python3 -m pip install h5py`). TensorFlow is **not** required: it loads `models/baseline_univariate.keras` weights and runs the same NumPy LSTM cell as `introspect.py`.

Open **[http://127.0.0.1:43125](http://127.0.0.1:43125)**

If the sidecar is down, the live panel still uses `public/hour_lookup.json` (real forward passes, nearest Metro Interstate hour) and labels that as replayed Keras output — not a quiet/rush blend.

## Demo flow (about 5 minutes)

1. **Slideshow** — advance with → or **Next**. Stop on **Say this out loud**: XGBoost won on motorway MAE; LSTM won narrowly on bikes; verdict flips; keep both.
2. On the talk-track slide, say the four lines, then click **Open live intersection**.
3. **Live intersection**
   - Click **Randomize & spawn** (or press **R**) to generate random cars / trucks / pedestrians.
   - Click **Create traffic jam** or **Random jam** (or press **J**).
   - Point at the **Decision brain**: Forget / Input / Output gates, neuron grid, and the Keras forecast for the closest real hour.
   - Recommended green is a **heuristic** driven by that forecast — the LSTM did not choose the light.
   - Toggle **Adaptive green** off/on once to show the contrast.

## Keys

| Key | Action |
| --- | --- |
| ← → / Space | Prev / next slide |
| L | Live intersection |
| S | Slideshow |
| R | Randomize & spawn (live) |
| J | Random jam (live) |

## Numbers on the slides (do not round prettier)

From `reports/model_comparison.json` and `reports/REPORT.md`:

| Dataset | Model | MAE |
| --- | --- | --- |
| Metro Interstate | XGBoost + calendar | **154.3** |
| Metro Interstate | LSTM + calendar | **201.4** |
| Metro Interstate | LSTM univariate | **228.8** |
| Metro Interstate | Naive last-hour | **585.6** |
| Bike sharing | LSTM univariate | **41.3** |
| Bike sharing | XGBoost univariate | **43.1** |
| Bike sharing | LSTM + weather/calendar | **38.5** |
| Bike sharing | XGBoost + weather/calendar | **38.6** |

XGBoost won on the motorway. LSTM won (narrowly) on bikes. That is why both stay.

## Boundary line (say out loud)

This intersection is a **simulation**. The cartoon lights are **not a closed-loop controller** and are **not connected to city signals**. The LSTM forecasts demand on stored motorway hours; it does not control real traffic lights.
