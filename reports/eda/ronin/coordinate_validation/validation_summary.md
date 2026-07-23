# Sensor Coordinate & Orientation Validation Summary

This report summarizes the coordinate consistency, quaternion orientation continuity, and physical cross-sensor coherence checks over canonical SensorRecord streams.

## Validation Criteria

The checking processes use standard thresholds (abnormal angular step defined as > 15.0° or 0.2618 rad):

| Criterion              | Warning (REVIEW)   | Failure (FAIL)   |
|:-----------------------|:-------------------|:-----------------|
| Quaternion norm error  | > 0.01             | > 0.05           |
| Gravity magnitude bias | > 0.1 m/s^2        | > 1.0 m/s^2      |
| Abnormal angular jumps | > 0                | > 20             |
| Axis polarity flips    | Detected           | -                |


## Dataset Summary

| Metric                         | Value        |
|:-------------------------------|:-------------|
| Total recordings               | 152          |
| PASS status count              | 0            |
| REVIEW status count            | 152          |
| FAIL status count              | 0            |
| Average gravity magnitude bias | 0.0034 m/s^2 |


## Recording-Level Validation Status

| Recording   | Status   |   Gravity Bias |   Axis Consistency |
|:------------|:---------|---------------:|-------------------:|
| unknown     | REVIEW   |    -0.00337813 |                  0 |
| unknown     | REVIEW   |    -0.00337746 |                  0 |
| unknown     | REVIEW   |    -0.00337433 |                  0 |
| unknown     | REVIEW   |    -0.00341321 |                  0 |
| unknown     | REVIEW   |    -0.00340304 |                  0 |
| unknown     | REVIEW   |    -0.0033656  |                  0 |
| unknown     | REVIEW   |    -0.00338809 |                  0 |
| unknown     | REVIEW   |    -0.00338062 |                  0 |
| unknown     | REVIEW   |    -0.00337049 |                  0 |
| unknown     | REVIEW   |    -0.00336301 |                  0 |
| unknown     | REVIEW   |    -0.00336502 |                  0 |
| unknown     | REVIEW   |    -0.00339163 |                  0 |
| unknown     | REVIEW   |    -0.00336373 |                  0 |
| unknown     | REVIEW   |    -0.00338376 |                  0 |
| unknown     | REVIEW   |    -0.00338131 |                  0 |
| unknown     | REVIEW   |    -0.0033836  |                  0 |
| unknown     | REVIEW   |    -0.00341251 |                  0 |
| unknown     | REVIEW   |    -0.00337068 |                  0 |
| unknown     | REVIEW   |    -0.0033649  |                  0 |
| unknown     | REVIEW   |    -0.00338659 |                  0 |
| unknown     | REVIEW   |    -0.0033943  |                  0 |
| unknown     | REVIEW   |    -0.0033626  |                  0 |
| unknown     | REVIEW   |    -0.00336825 |                  0 |
| unknown     | REVIEW   |    -0.00337243 |                  0 |
| unknown     | REVIEW   |    -0.00339703 |                  0 |
| unknown     | REVIEW   |    -0.0033612  |                  0 |
| unknown     | REVIEW   |    -0.00338219 |                  0 |
| unknown     | REVIEW   |    -0.00340628 |                  0 |
| unknown     | REVIEW   |    -0.00338086 |                  0 |
| unknown     | REVIEW   |    -0.00340646 |                  0 |
| unknown     | REVIEW   |    -0.00337322 |                  0 |
| unknown     | REVIEW   |    -0.00337316 |                  0 |
| unknown     | REVIEW   |    -0.0033896  |                  0 |
| unknown     | REVIEW   |    -0.0033822  |                  0 |
| unknown     | REVIEW   |    -0.00338077 |                  0 |
| unknown     | REVIEW   |    -0.00337011 |                  0 |
| unknown     | REVIEW   |    -0.0033661  |                  0 |
| unknown     | REVIEW   |    -0.00337809 |                  0 |
| unknown     | REVIEW   |    -0.00337961 |                  0 |
| unknown     | REVIEW   |    -0.00337288 |                  0 |
| unknown     | REVIEW   |    -0.00337954 |                  0 |
| unknown     | REVIEW   |    -0.00338711 |                  0 |
| unknown     | REVIEW   |    -0.00338938 |                  0 |
| unknown     | REVIEW   |    -0.00336773 |                  0 |
| unknown     | REVIEW   |    -0.00336885 |                  0 |
| unknown     | REVIEW   |    -0.00338642 |                  0 |
| unknown     | REVIEW   |    -0.00337053 |                  0 |
| unknown     | REVIEW   |    -0.00337234 |                  0 |
| unknown     | REVIEW   |    -0.00338691 |                  0 |
| unknown     | REVIEW   |    -0.00341959 |                  0 |
| unknown     | REVIEW   |    -0.00339572 |                  0 |
| unknown     | REVIEW   |    -0.00335828 |                  0 |
| unknown     | REVIEW   |    -0.00336776 |                  0 |
| unknown     | REVIEW   |    -0.00337127 |                  0 |
| unknown     | REVIEW   |    -0.00338641 |                  0 |
| unknown     | REVIEW   |    -0.00340298 |                  0 |
| unknown     | REVIEW   |    -0.00337964 |                  0 |
| unknown     | REVIEW   |    -0.00337561 |                  0 |
| unknown     | REVIEW   |    -0.00336617 |                  0 |
| unknown     | REVIEW   |    -0.00335337 |                  0 |
| unknown     | REVIEW   |    -0.00336642 |                  0 |
| unknown     | REVIEW   |    -0.00336297 |                  0 |
| unknown     | REVIEW   |    -0.00338266 |                  0 |
| unknown     | REVIEW   |    -0.0033969  |                  0 |
| unknown     | REVIEW   |    -0.00338666 |                  0 |
| unknown     | REVIEW   |    -0.00336859 |                  0 |
| unknown     | REVIEW   |    -0.00340091 |                  0 |
| unknown     | REVIEW   |    -0.0033864  |                  0 |
| unknown     | REVIEW   |    -0.00338103 |                  0 |
| unknown     | REVIEW   |    -0.00338143 |                  0 |
| unknown     | REVIEW   |    -0.00338249 |                  0 |
| unknown     | REVIEW   |    -0.00336236 |                  0 |
| unknown     | REVIEW   |    -0.00337198 |                  0 |
| unknown     | REVIEW   |    -0.00340012 |                  0 |
| unknown     | REVIEW   |    -0.00338101 |                  0 |
| unknown     | REVIEW   |    -0.00339588 |                  0 |
| unknown     | REVIEW   |    -0.00341455 |                  0 |
| unknown     | REVIEW   |    -0.00336402 |                  0 |
| unknown     | REVIEW   |    -0.00337096 |                  0 |
| unknown     | REVIEW   |    -0.00337913 |                  0 |
| unknown     | REVIEW   |    -0.00336639 |                  0 |
| unknown     | REVIEW   |    -0.00339224 |                  0 |
| unknown     | REVIEW   |    -0.00338623 |                  0 |
| unknown     | REVIEW   |    -0.00339554 |                  0 |
| unknown     | REVIEW   |    -0.00340016 |                  0 |
| unknown     | REVIEW   |    -0.00337203 |                  0 |
| unknown     | REVIEW   |    -0.00338499 |                  0 |
| unknown     | REVIEW   |    -0.00335826 |                  0 |
| unknown     | REVIEW   |    -0.00335335 |                  0 |
| unknown     | REVIEW   |    -0.00337049 |                  0 |
| unknown     | REVIEW   |    -0.00338185 |                  0 |
| unknown     | REVIEW   |    -0.00336855 |                  0 |
| unknown     | REVIEW   |    -0.00339666 |                  0 |
| unknown     | REVIEW   |    -0.00339429 |                  0 |
| unknown     | REVIEW   |    -0.00336519 |                  0 |
| unknown     | REVIEW   |    -0.00335936 |                  0 |
| unknown     | REVIEW   |    -0.0033984  |                  0 |
| unknown     | REVIEW   |    -0.00339544 |                  0 |
| unknown     | REVIEW   |    -0.00340045 |                  0 |
| unknown     | REVIEW   |    -0.00337646 |                  0 |
| unknown     | REVIEW   |    -0.00341141 |                  0 |
| unknown     | REVIEW   |    -0.00342356 |                  0 |
| unknown     | REVIEW   |    -0.00338737 |                  0 |
| unknown     | REVIEW   |    -0.00339516 |                  0 |
| unknown     | REVIEW   |    -0.00337916 |                  0 |
| unknown     | REVIEW   |    -0.00338149 |                  0 |
| unknown     | REVIEW   |    -0.00338484 |                  0 |
| unknown     | REVIEW   |    -0.0033719  |                  0 |
| unknown     | REVIEW   |    -0.00336701 |                  0 |
| unknown     | REVIEW   |    -0.00338451 |                  0 |
| unknown     | REVIEW   |    -0.00337982 |                  0 |
| unknown     | REVIEW   |    -0.00337603 |                  0 |
| unknown     | REVIEW   |    -0.00339038 |                  0 |
| unknown     | REVIEW   |    -0.00339472 |                  0 |
| unknown     | REVIEW   |    -0.00338005 |                  0 |
| unknown     | REVIEW   |    -0.00337973 |                  0 |
| unknown     | REVIEW   |    -0.00336823 |                  0 |
| unknown     | REVIEW   |    -0.00337263 |                  0 |
| unknown     | REVIEW   |    -0.00339204 |                  0 |
| unknown     | REVIEW   |    -0.00339686 |                  0 |
| unknown     | REVIEW   |    -0.00338052 |                  0 |
| unknown     | REVIEW   |    -0.00343552 |                  0 |
| unknown     | REVIEW   |    -0.00336486 |                  0 |
| unknown     | REVIEW   |    -0.00336509 |                  0 |
| unknown     | REVIEW   |    -0.00339189 |                  0 |
| unknown     | REVIEW   |    -0.0033807  |                  0 |
| unknown     | REVIEW   |    -0.00336151 |                  0 |
| unknown     | REVIEW   |    -0.00338905 |                  0 |
| unknown     | REVIEW   |    -0.00338063 |                  0 |
| unknown     | REVIEW   |    -0.00339043 |                  0 |
| unknown     | REVIEW   |    -0.00340669 |                  0 |
| unknown     | REVIEW   |    -0.0034065  |                  0 |
| unknown     | REVIEW   |    -0.00336507 |                  0 |
| unknown     | REVIEW   |    -0.00336204 |                  0 |
| unknown     | REVIEW   |    -0.00336869 |                  0 |
| unknown     | REVIEW   |    -0.00336641 |                  0 |
| unknown     | REVIEW   |    -0.00337138 |                  0 |
| unknown     | REVIEW   |    -0.00341452 |                  0 |
| unknown     | REVIEW   |    -0.00339134 |                  0 |
| unknown     | REVIEW   |    -0.00339854 |                  0 |
| unknown     | REVIEW   |    -0.00339728 |                  0 |
| unknown     | REVIEW   |    -0.00340515 |                  0 |
| unknown     | REVIEW   |    -0.00339046 |                  0 |
| unknown     | REVIEW   |    -0.00340334 |                  0 |
| unknown     | REVIEW   |    -0.00338279 |                  0 |
| unknown     | REVIEW   |    -0.00339461 |                  0 |
| unknown     | REVIEW   |    -0.00339387 |                  0 |
| unknown     | REVIEW   |    -0.00336825 |                  0 |
| unknown     | REVIEW   |    -0.00338022 |                  0 |
| unknown     | REVIEW   |    -0.00339059 |                  0 |
| unknown     | REVIEW   |    -0.00338334 |                  0 |
| unknown     | REVIEW   |    -0.0033862  |                  0 |


## Key Findings

- Out of 152 recordings: 0 passed, 152 require review, and 0 failed consistency checks.
- Average absolute gravity bias is 0.0034 m/s^2.
- No recordings failed. The dataset is geometrically sound for navigation.
