---
tags: [layer]
---
# traffic_output

1 linear unit(s) - horizon(s) +1h.

**No activation function.** This is a regression: the output is a number of
vehicles per hour. A sigmoid would clamp it to [0, 1]; a softmax would turn it
into class probabilities. Neither is what we want.

## The two windows documented in this vault

| Window | Ends at | Predicted | Actual | Error |
| --- | --- | --- | --- | --- |
| Rush Hour | 2018-04-12 16:00 | 6,164 | 7,213 | -1,049 |
| Night | 2018-04-15 03:00 | 520 | 151 | +369 |

## Turning the number into a decision

| Predicted volume | Level |
| --- | --- |
| below 2,000 | LOW - free flow |
| 2,000 to 4,000 | MODERATE - steady traffic |
| above 4,000 | HIGH - congestion risk |

Fed by [[dense_hidden]]. Related: [[05 Results]]
