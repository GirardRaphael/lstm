---
id: toc-02-product-scope
type: scope
status: active
owner: principal
updated_utc: 2026-09-13T14:05:00Z
verified_commit: a652a04fbaee8579fdf20fbe4efe162d66155373
---

# Product Scope

## Mission

Build a **read-only Traffic Observatory**: explore observations, check quality, train/compare forecasting models, review forecasts and measure errors. Keep the LSTM educational workspace.

## Operator questions

1. Which streams are available, current and trustworthy enough?
2. Where and when does recorded demand change?
3. Which forecasts exist, when issued, with what evidence?
4. How accurate were models when actuals arrived?
5. Why is a forecast unavailable, or why did a job fail?

## Hard boundaries

- No connection to or control of traffic signals / roadside hardware.
- No controller-command endpoints or live actuation instructions.
- Authorized observation exports and labelled synthetic/replay data only.
- Volume bands are not measured congestion and not safety evidence.
- Hourly motorway results do not validate five-minute intersection forecasting.

## Educational retention

Preserve notebooks, neuron introspection, and `obsidian_vault/Traffic_LSTM_Brain`.
Exporters must never delete or overwrite this project-context vault.
