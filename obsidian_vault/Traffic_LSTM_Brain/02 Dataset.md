---
tags: [data]
---
# 02 Dataset

**Metro Interstate Traffic Volume** - UCI Machine Learning Repository.
Hourly westbound traffic on I-94 between Minneapolis and Saint Paul.

| Property | Value |
| --- | --- |
| Rows | 48,204 |
| Period | 2012-10-02 09:00 to 2018-09-30 23:00 |
| Target | `traffic_volume` - vehicles counted during the hour |
| Range | 0 to 7,280 vehicles (mean 3,260) |
| Duplicate timestamps | 7629 |
| Missing target values | 0 |
| Hours missing from the range | 11,976 |

## What we do about the imperfections

**Duplicate timestamps.** The same hour is sometimes logged twice with
different weather descriptions. We keep one row per hour and average the
target - we neither invent data nor silently keep two conflicting rows.

**Gaps.** There are 11,976 hours with no record at all, mostly a long
outage in 2014-2015. We do *not* interpolate them. Inventing traffic
volumes would make the metrics look better than the model deserves. The
sequence builder simply slides over the rows that exist.

## Columns we deliberately ignore for now

`temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `holiday`.

Version 1 is univariate on purpose: it isolates the question "can past traffic
alone predict future traffic?". Adding weather is the first item in
[[06 Limits and Next Steps]].

![[01_dataset.png]]

![[02_daily_profile.png]]

The second figure is the pattern the network has to learn: two sharp peaks,
a deep overnight trough, and a spread that widens during the day.

Next: [[03 Pipeline]]
