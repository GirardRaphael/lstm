# Present today — Traffic LSTM

## Open the deck

```bash
cd presentation
npm install
npm run dev -- --host 0.0.0.0 --port 43125
```

Open **[http://127.0.0.1:43125](http://127.0.0.1:43125)**

## Demo flow (about 5 minutes)

1. **Slideshow** — advance with → or **Next** through the story slides.
2. On the talk-track slide, say the four lines, then click **Open live intersection**.
3. **Live intersection**
   - Click **Randomize & spawn** (or press **R**) to generate random cars / trucks / pedestrians.
   - Click **Create traffic jam** or **Random jam** (or press **J**).
   - Point at the response banner: heaviest queue + NS/EW green timers stretching when adaptive is on.
   - Toggle **Adaptive green** off/on once to show the contrast.

## Keys

| Key | Action |
| --- | --- |
| ← → / Space | Prev / next slide |
| L | Live intersection |
| S | Slideshow |
| R | Randomize & spawn (live) |
| J | Random jam (live) |

## Boundary line (if asked)

This intersection is a **simulation**. The LSTM forecasts demand; it does not control real traffic lights.
