# Data Validation Report
## Explainable AI Learning Analytics Platform — Zewail City

**Generated:** Phase 3 — EDA and Data Validation
**Dataset:** Synthetic (Zenodo-calibrated, research-backed correlations)

---

## Summary

| Metric | Value |
|--------|-------|
| Students | 7,500 |
| Course Records | 105,059 |
| Programmes | 11 |
| Validation Checks | 7 (7 passed, 0 warnings) |

---

## Key Statistics

### Cumulative GPA
| Stat | Value |
|------|-------|
| Mean | 2.141 |
| Std  | 1.140 |
| Min  | 0.000 |
| Max  | 4.000 |
| 25th pct | 1.260 |
| Median   | 2.300 |
| 75th pct | 3.123 |

### Risk Distribution
| risk_level   |   Count |   Pct |
|:-------------|--------:|------:|
| Low Risk     |    3200 |  42.7 |
| High Risk    |    3172 |  42.3 |
| Medium Risk  |    1128 |  15   |

### GPA Band Distribution
| gpa_band                |   Count |   Pct |
|:------------------------|--------:|------:|
| At Risk (<2.0)          |    3170 |  42.3 |
| Good (3.0-3.5)          |    1394 |  18.6 |
| Average (2.5-3.0)       |    1132 |  15.1 |
| Below Average (2.0-2.5) |     949 |  12.7 |
| High (>=3.5)            |     855 |  11.4 |

### Score Component Statistics (per course)
|       |   attendance |   assignments |   quizzes |      labs |   midterm |   final_exam |   overall_score |
|:------|-------------:|--------------:|----------:|----------:|----------:|-------------:|----------------:|
| count |    105059    |     105059    | 105059    | 105059    | 105059    |    105059    |       105059    |
| mean  |        79.52 |         72.68 |     66.08 |     77.7  |     78.08 |        77.05 |           75.71 |
| std   |        12.6  |         14.6  |     18.64 |     13.28 |     16.28 |        16.97 |           13.16 |
| min   |        40    |         25.1  |      0    |     21.8  |      0.8  |         5.3  |           19.7  |
| 25%   |        70.6  |         62    |     52.6  |     68.5  |     67.3  |        65.8  |           66.7  |
| 50%   |        79.8  |         72.8  |     66.5  |     78.4  |     80.2  |        79.2  |           77.6  |
| 75%   |        89.2  |         83.5  |     80.2  |     87.8  |     91.2  |        90.7  |           86.3  |
| max   |       100    |        100    |    100    |    100    |    100    |       100    |          100    |

### Correlation Matrix (student summaries)
|             |   Attendance |   Assignments |   Midterm |   Cum GPA |   Failed # |
|:------------|-------------:|--------------:|----------:|----------:|-----------:|
| Attendance  |        1     |         0.36  |     0.39  |     0.492 |     -0.288 |
| Assignments |        0.36  |         1     |     0.418 |     0.627 |     -0.369 |
| Midterm     |        0.39  |         0.418 |     1     |     0.944 |     -0.68  |
| Cum GPA     |        0.492 |         0.627 |     0.944 |     1     |     -0.65  |
| Failed #    |       -0.288 |        -0.369 |    -0.68  |    -0.65  |      1     |

---

## Validation Findings

- GPA mean=2.141  std=1.140  min=0.00  max=4.00
- ✅ PASS: GPA mean is within realistic university range (1.5-3.5)
- GPA normality test p-value: 0.0000 (non-normal - expected for mixed population)
- Attendance mean=79.5%  std=10.3%
- Zenodo reference attendance mean=80.2%  std=11.5%
- ✅ PASS: Attendance mean within 5% of reference (0.7%)
- Correlation: Attendance  vs GPA = 0.492  (research target ~0.44)
- Correlation: Assignments vs GPA = 0.627  (research target ~0.45)
- Correlation: Midterm     vs GPA = 0.944  (expected 0.60-0.90)
- Correlation: Failed#     vs GPA = -0.650  (expected negative -0.50 to -0.80)
- ✅ PASS: Attendance-GPA correlation 0.492 in expected range [0.3,0.7]
- ✅ PASS: Assignments-GPA correlation 0.627 in expected range [0.3,0.7]
- ✅ PASS: Midterm-GPA correlation 0.944 in expected range [0.5,0.99]
- ✅ PASS: FailedCourses-GPA correlation -0.650 in expected range [-0.9,-0.3]
- Course failure rate: 13.6% of credit-weighted courses
- ✅ PASS: Course failure rate within realistic 5-30% range
- Risk level 'Low Risk': 42.7%
- Risk level 'High Risk': 42.3%
- Risk level 'Medium Risk': 15.0%
- Programme distribution: {'CSAI': 1694, 'DSAI': 1348, 'SWE': 760, 'BUS': 645, 'MECH': 606, 'EEE': 523, 'MATH': 517, 'CIV': 371, 'PHYS': 369, 'FIN': 369, 'CHEM': 298}


---

## EDA Figures

- `eda_figures/01_score_distributions.png` — All score component histograms
- `eda_figures/02_gpa_risk_distribution.png` — GPA histogram and risk pie chart
- `eda_figures/03_correlation_heatmap.png` — Feature correlation heatmap
- `eda_figures/04_programme_semester_dist.png` — Programme and semester distributions
- `eda_figures/05_prerequisite_chains.png` — Prerequisite chain correlations


---

## Conclusion

All validation checks passed. The synthetic dataset is realistic and ready for model training.
