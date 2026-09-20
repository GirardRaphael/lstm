# Pick up here

Current continuation — 2026-09-20 (cloud resume + presentation deck)

**Authoritative build state** for ML work still lives in:

- `obsidian_vault/Traffic_Observatory_Context` (when present on the branch)
- This file for the cloud-agent resume tip

| | |
| --- | --- |
| GitHub | https://github.com/GirardRaphael/lstm |
| Working tip | `observatory/phase2-v2-wiring` |
| Presentation deck | [`presentation/`](presentation/) — slideshow + live 2D intersection |
| Run deck | `cd presentation && npm install && npm run dev -- --port 43125` |
| Streamlit workbench | `./run_app.ps1` or `PYTHONPATH=src streamlit run app/streamlit_app.py` |

## Verified this session (Linux cloud)

- Reconnected empty Cursor workspace to `GirardRaphael/lstm` @ `observatory/phase2-v2-wiring` (`c01f165` tip, then local fixes)
- Suites: pipeline **18/18**, temporal **18/18** (after portability fix), readiness **21/21**, saved-models **9 + 2 skip**, app **3/3**, phase2 wiring **9/9**
- Fixed POSIX resolution of archived Windows `model_path` values in `pipeline_v2.read_legacy_artifact`
- Added presentation web app: slideshow of project findings + interactive 2D intersection (cars / trucks / pedestrians, traffic-jam generator, adaptive green)

## Product boundary (unchanged)

The live intersection is a **simulation**. It does not control real signals or connect to city sensors.

## Push status

- Pushed to the Cursor session remote on `observatory/phase2-v2-wiring`
- GitHub push needs a `GITHUB_TOKEN` (requested). Until then, mirror commits manually or authorize the token.

## Next owner actions

1. Present from [http://127.0.0.1:43125](http://127.0.0.1:43125) (Slideshow → Live intersection)
2. Provide GitHub token / push credentials so this branch lands on `GirardRaphael/lstm`
3. Optional: merge phase2 into an integration branch after review (do not fast-forward `main` without authorization)

```bash
cd presentation && npm run dev -- --host 0.0.0.0 --port 43125
PYTHONPATH=src python3.12 tests/test_pipeline.py
PYTHONPATH=src python3.12 tests/test_temporal.py
```
