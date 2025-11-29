# Brain State GLMM Analysis Results

This document contains all statistical tables from the brain state GLMM analysis.

## Table of Contents

1. [Main Effects and Interactions](#1-main-effects-and-interactions)
2. [Group-Specific Effects](#2-group-specific-effects)
3. [Multiple Comparisons Analysis](#3-multiple-comparisons-analysis)
4. [Cross-Validation Summary](#4-cross-validation-summary)
5. [State Occupancy Rates](#5-state-occupancy-rates)
6. [State Transition Rates](#6-state-transition-rates)
7. [Main Effects](#7-main-effects)
8. [Group Effects](#8-group-effects)

---

## 1. Main Effects and Interactions

This table presents the main effects and group interactions for all features analyzed in the brain state dynamics study. Features are sorted by Bayes Factor (strongest evidence first).

| Feature                             |   Coefficient |   Std. Error |   P(Effect > 0) |   Bayes Factor | Evidence Category   |   Odds Ratio |   OR Lower |   OR Upper |
|:------------------------------------|--------------:|-------------:|----------------:|---------------:|:--------------------|-------------:|-----------:|-----------:|
| lee_girl_together                   |         0.884 |        0.146 |           1     |    8.89129e+07 | Extreme evidence    |        2.42  |      1.817 |      3.221 |
| arthur_speaking                     |        -0.336 |        0.079 |           0     | 8458.31        | Extreme evidence    |        0.715 |      0.612 |      0.834 |
| has_adv                             |        -0.272 |        0.086 |           0.001 |  144.43        | Extreme evidence    |        0.762 |      0.644 |      0.902 |
| lee_girl_verb                       |         0.416 |        0.179 |           0.99  |   14.72        | Strong evidence     |        1.515 |      1.066 |      2.153 |
| has_noun                            |         0.148 |        0.072 |           0.98  |    8.08        | Moderate evidence   |        1.16  |      1.006 |      1.336 |
| has_noun:Group Interaction          |         0.128 |        0.072 |           0.962 |    4.83        | Moderate evidence   |        1.137 |      0.987 |      1.31  |
| arthur_adj                          |        -0.227 |        0.15  |           0.065 |    3.14        | Moderate evidence   |        0.797 |      0.594 |      1.069 |
| has_adj                             |        -0.108 |        0.085 |           0.1   |    2.27        | Anecdotal evidence  |        0.897 |      0.76  |      1.059 |
| girl_speaking                       |        -0.236 |        0.197 |           0.115 |    2.05        | Anecdotal evidence  |        0.79  |      0.537 |      1.162 |
| has_adj:Group Interaction           |        -0.088 |        0.085 |           0.15  |    1.71        | Anecdotal evidence  |        0.916 |      0.776 |      1.081 |
| lee_girl_verb:Group Interaction     |         0.175 |        0.179 |           0.837 |    1.62        | Anecdotal evidence  |        1.192 |      0.839 |      1.693 |
| lee_speaking                        |         0.05  |        0.078 |           0.741 |    1.23        | Anecdotal evidence  |        1.051 |      0.903 |      1.224 |
| lee_girl_together:Group Interaction |         0.077 |        0.145 |           0.704 |    1.15        | Anecdotal evidence  |        1.081 |      0.814 |      1.435 |
| girl_speaking:Group Interaction     |        -0.088 |        0.197 |           0.327 |    1.11        | Anecdotal evidence  |        0.916 |      0.623 |      1.346 |
| lee_speaking:Group Interaction      |        -0.027 |        0.078 |           0.365 |    1.06        | Anecdotal evidence  |        0.973 |      0.836 |      1.134 |
| has_adv:Group Interaction           |        -0.018 |        0.086 |           0.419 |    1.02        | Anecdotal evidence  |        0.983 |      0.83  |      1.163 |
| arthur_speaking:Group Interaction   |         0.012 |        0.079 |           0.563 |    1.01        | Anecdotal evidence  |        1.013 |      0.868 |      1.182 |
| has_verb                            |         0.01  |        0.073 |           0.552 |    1.01        | Anecdotal evidence  |        1.01  |      0.875 |      1.166 |
| arthur_adj:Group Interaction        |        -0.015 |        0.15  |           0.461 |    1           | Anecdotal evidence  |        0.986 |      0.735 |      1.322 |
| has_verb:Group Interaction          |         0.004 |        0.073 |           0.522 |    1           | Anecdotal evidence  |        1.004 |      0.87  |      1.159 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| lee_girl_verb     |         0.591 |       1.806 |          0.99  |           0.24  |         1.271 |            0.828 |        0.175 |                   0.837 |
| has_noun          |         0.276 |       1.318 |          0.997 |           0.02  |         1.02  |            0.576 |        0.128 |                   0.962 |
| girl_speaking     |        -0.324 |       0.723 |          0.122 |          -0.148 |         0.862 |            0.297 |       -0.088 |                   0.327 |
| has_adj           |        -0.196 |       0.822 |          0.05  |          -0.021 |         0.98  |            0.431 |       -0.088 |                   0.15  |
| lee_girl_together |         0.961 |       2.614 |          1     |           0.806 |         2.239 |            1     |        0.077 |                   0.704 |
| lee_speaking      |         0.023 |       1.024 |          0.584 |           0.077 |         1.08  |            0.759 |       -0.027 |                   0.365 |
| has_adv           |        -0.289 |       0.749 |          0.009 |          -0.254 |         0.776 |            0.018 |       -0.018 |                   0.419 |
| arthur_adj        |        -0.241 |       0.786 |          0.128 |          -0.212 |         0.809 |            0.158 |       -0.015 |                   0.461 |
| arthur_speaking   |        -0.323 |       0.724 |          0.002 |          -0.348 |         0.706 |            0.001 |        0.012 |                   0.563 |
| has_verb          |         0.014 |       1.014 |          0.552 |           0.006 |         1.006 |            0.522 |        0.004 |                   0.522 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_noun:group_has_noun_interaction                   |                   0.962 | 0.038 | True                    |
| has_adj:group_has_adj_interaction                     |                   0.85  | 0.094 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.837 | 0.117 | False                   |
| lee_girl_together:group_lee_girl_together_interaction |                   0.704 | 0.162 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.673 | 0.195 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.635 | 0.223 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.581 | 0.251 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.563 | 0.274 | False                   |
| arthur_adj:group_arthur_adj_interaction               |                   0.539 | 0.295 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.522 | 0.314 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| has_verb          |       -0.0014 |       0.0117 |         -0.0013 |         0.0121 |            0.9675 |
| arthur_speaking   |        0.0212 |       0.0121 |          0.0214 |         0.0131 |            0.9273 |
| lee_girl_together |        0.0848 |       0.0273 |          0.0856 |         0.0195 |            0.7135 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.121251 |
| Paranoia |         0.129887 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0251462 |   0.0251462 |
| Paranoia |    0.0264327 |   0.0263158 |

---

## 7. Main Effects

Table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE          |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:-------------------|----------------:|---------------:|-------------:|
| lee_girl_together                   | 0.884 ± 0.146 ***  |           1     |    8.89129e+07 |        2.42  |
| arthur_speaking                     | -0.336 ± 0.079 *** |           0     | 8458.31        |        0.715 |
| has_adv                             | -0.272 ± 0.086 *** |           0.001 |  144.43        |        0.762 |
| lee_girl_verb                       | 0.416 ± 0.179 **   |           0.99  |   14.72        |        1.515 |
| has_noun                            | 0.148 ± 0.072 *    |           0.98  |    8.08        |        1.16  |
| has_noun:Group Interaction          | 0.128 ± 0.072 *    |           0.962 |    4.83        |        1.137 |
| arthur_adj                          | -0.227 ± 0.15 *    |           0.065 |    3.14        |        0.797 |
| has_adj                             | -0.108 ± 0.085     |           0.1   |    2.27        |        0.897 |
| girl_speaking                       | -0.236 ± 0.197     |           0.115 |    2.05        |        0.79  |
| has_adj:Group Interaction           | -0.088 ± 0.085     |           0.15  |    1.71        |        0.916 |
| lee_girl_verb:Group Interaction     | 0.175 ± 0.179      |           0.837 |    1.62        |        1.192 |
| lee_speaking                        | 0.05 ± 0.078       |           0.741 |    1.23        |        1.051 |
| lee_girl_together:Group Interaction | 0.077 ± 0.145      |           0.704 |    1.15        |        1.081 |
| girl_speaking:Group Interaction     | -0.088 ± 0.197     |           0.327 |    1.11        |        0.916 |
| lee_speaking:Group Interaction      | -0.027 ± 0.078     |           0.365 |    1.06        |        0.973 |
| has_adv:Group Interaction           | -0.018 ± 0.086     |           0.419 |    1.02        |        0.983 |
| arthur_speaking:Group Interaction   | 0.012 ± 0.079      |           0.563 |    1.01        |        1.013 |
| has_verb                            | 0.01 ± 0.073       |           0.552 |    1.01        |        1.01  |
| arthur_adj:Group Interaction        | -0.015 ± 0.15      |           0.461 |    1           |        0.986 |
| has_verb:Group Interaction          | 0.004 ± 0.073      |           0.522 |    1           |        1.004 |

---

## 8. Group Effects

Table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| lee_girl_verb     |         0.591 |           0.24  | 0.175              |                   0.837 |
| has_noun          |         0.276 |           0.02  | 0.128 †            |                   0.962 |
| girl_speaking     |        -0.324 |          -0.148 | -0.088             |                   0.327 |
| has_adj           |        -0.196 |          -0.021 | -0.088             |                   0.15  |
| lee_girl_together |         0.961 |           0.806 | 0.077              |                   0.704 |
| lee_speaking      |         0.023 |           0.077 | -0.027             |                   0.365 |
| has_adv           |        -0.289 |          -0.254 | -0.018             |                   0.419 |
| arthur_adj        |        -0.241 |          -0.212 | -0.015             |                   0.461 |
| arthur_speaking   |        -0.323 |          -0.348 | 0.012              |                   0.563 |
| has_verb          |         0.014 |           0.006 | 0.004              |                   0.522 |