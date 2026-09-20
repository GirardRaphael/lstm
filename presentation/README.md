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

This simulator is read-only theatre: queue pressure stretches green time in the browser. It is not connected to city sensors or signal hardware.
