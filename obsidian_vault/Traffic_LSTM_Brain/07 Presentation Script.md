# Presentation script: hourly forecasting, evidence first

This project forecasts next-hour counts from historical hourly data. It is a research workbench and does not control traffic lights or measure congestion.

The old v1 scores are exploratory: validation contaminated preprocessing, some windows bridged outages, and the test results influenced historical model choices. I recovered a causal pipeline, added stronger baselines, and reran the comparison. The test period was previously inspected, so this remains retrospective evidence.

Show the generated `reports/REPORT.md` or the app's Forecast evidence page. Read the exact numbers from the linked manifests. XGBoost was selected using validation on both datasets. Its motorway advantage over the LSTM is substantial. On bikes the paired MAE interval crosses zero; MAPE favors the tree while peak-hour MAE and RMSE favor the LSTM. There is no universal winner demonstrated here.

The direct tree sees seasonal lags and the target clock. The LSTM sees a 24-hour sequence. This compares practical candidates, not recurrence in isolation. Both used fixed budgets; neither result establishes the optimal tuned architecture.

The useful deliverable is the testable causal contract: split before fitting, exclude incomplete windows, use timestamp-true baselines, save portable model packages and report uncertainty and failures. The LSTM remains useful for inspecting gates and hidden states.

Before a customer pilot, we need authorized local observations, rolling-origin and untouched holdouts, calibrated prediction intervals, persistent jobs, access controls, monitoring and recovery tests. We have not shown reduced road delays or validated street control.

Open the educational demo last. Explain that it replays historical model computations; it is not a live control system.
