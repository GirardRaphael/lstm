# Traffic LSTM — Presentation Deck

Slideshow of the Traffic LSTM / Observatory findings, plus a live **2D intersection** simulator for demos.

## Run

```bash
cd presentation
npm install
npm run dev -- --host 0.0.0.0 --port 43125
```

Open [http://127.0.0.1:43125](http://127.0.0.1:43125).

- **Slideshow** — arrow keys or on-screen Previous / Next
- **Live intersection** — spawn cars / trucks / pedestrians, create a traffic jam, toggle adaptive green

This simulator is a **simulation**. Cartoon lights are not a closed-loop controller and are not connected to city sensors or signal hardware. The live forecast is a real Keras-weight replay for the closest Metro Interstate hour; recommended green is a heuristic.
