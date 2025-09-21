# Brain State GLMM Analysis Results

This document contains all statistical tables from the brain state GLMM analysis.

## Table of Contents

1. [Main Effects and Interactions](#1-main-effects-and-interactions)
2. [Group-Specific Effects](#2-group-specific-effects)
3. [Multiple Comparisons Analysis](#3-multiple-comparisons-analysis)
4. [Cross-Validation Summary](#4-cross-validation-summary)
5. [State Occupancy Rates](#5-state-occupancy-rates)
6. [State Transition Rates](#6-state-transition-rates)
7. [Publication-Ready Main Effects](#7-publication-ready-main-effects)
8. [Publication-Ready Group Effects](#8-publication-ready-group-effects)

---

## 1. Main Effects and Interactions

This table presents the main effects and group interactions for all features analyzed in the brain state dynamics study. Features are sorted by Bayes Factor (strongest evidence first).

| Feature                             |   Coefficient |   Std. Error |   P(Effect > 0) |   Bayes Factor | Evidence Category   |   Odds Ratio |   OR Lower |   OR Upper |
|:------------------------------------|--------------:|-------------:|----------------:|---------------:|:--------------------|-------------:|-----------:|-----------:|
| arthur_speaking                     |        -0.241 |        0.059 |           0     |        4358.08 | Extreme evidence    |        0.786 |      0.7   |      0.882 |
| lee_girl_verb                       |         0.376 |        0.146 |           0.995 |          27.5  | Strong evidence     |        1.456 |      1.094 |      1.938 |
| lee_girl_together                   |         0.299 |        0.13  |           0.989 |          13.99 | Strong evidence     |        1.348 |      1.045 |      1.739 |
| has_adj:Group Interaction           |         0.139 |        0.063 |           0.986 |          11.42 | Strong evidence     |        1.149 |      1.016 |      1.3   |
| has_adj                             |         0.136 |        0.063 |           0.984 |          10.17 | Strong evidence     |        1.145 |      1.012 |      1.295 |
| has_verb                            |        -0.107 |        0.056 |           0.029 |           6.07 | Moderate evidence   |        0.899 |      0.805 |      1.003 |
| arthur_adj:Group Interaction        |         0.163 |        0.108 |           0.935 |           3.14 | Moderate evidence   |        1.177 |      0.953 |      1.454 |
| has_noun                            |         0.073 |        0.055 |           0.906 |           2.38 | Anecdotal evidence  |        1.075 |      0.965 |      1.199 |
| arthur_adj                          |        -0.126 |        0.108 |           0.121 |           1.98 | Anecdotal evidence  |        0.882 |      0.714 |      1.089 |
| lee_girl_together:Group Interaction |        -0.15  |        0.13  |           0.124 |           1.94 | Anecdotal evidence  |        0.861 |      0.668 |      1.11  |
| lee_girl_verb:Group Interaction     |        -0.129 |        0.145 |           0.188 |           1.48 | Anecdotal evidence  |        0.879 |      0.661 |      1.169 |
| girl_speaking:Group Interaction     |         0.118 |        0.15  |           0.784 |           1.36 | Anecdotal evidence  |        1.125 |      0.838 |      1.51  |
| has_verb:Group Interaction          |        -0.032 |        0.056 |           0.283 |           1.18 | Anecdotal evidence  |        0.968 |      0.868 |      1.081 |
| arthur_speaking:Group Interaction   |        -0.032 |        0.059 |           0.292 |           1.16 | Anecdotal evidence  |        0.968 |      0.863 |      1.087 |
| has_adv:Group Interaction           |         0.015 |        0.063 |           0.592 |           1.03 | Anecdotal evidence  |        1.015 |      0.896 |      1.149 |
| lee_speaking:Group Interaction      |         0.008 |        0.06  |           0.556 |           1.01 | Anecdotal evidence  |        1.008 |      0.897 |      1.134 |
| girl_speaking                       |        -0.02  |        0.15  |           0.446 |           1.01 | Anecdotal evidence  |        0.98  |      0.73  |      1.315 |
| lee_speaking                        |        -0.007 |        0.06  |           0.452 |           1.01 | Anecdotal evidence  |        0.993 |      0.883 |      1.116 |
| has_adv                             |        -0.005 |        0.063 |           0.47  |           1    | Anecdotal evidence  |        0.995 |      0.879 |      1.127 |
| has_noun:Group Interaction          |        -0     |        0.055 |           0.499 |           1    | Anecdotal evidence  |        1     |      0.897 |      1.114 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_adj        |         0.037 |       1.038 |          0.596 |          -0.289 |         0.749 |            0.029 |        0.163 |                   0.935 |
| lee_girl_together |         0.149 |       1.161 |          0.792 |           0.448 |         1.566 |            0.993 |       -0.15  |                   0.124 |
| has_adj           |         0.274 |       1.316 |          0.999 |          -0.003 |         0.997 |            0.486 |        0.139 |                   0.986 |
| lee_girl_verb     |         0.247 |       1.28  |          0.885 |           0.505 |         1.656 |            0.993 |       -0.129 |                   0.188 |
| girl_speaking     |         0.098 |       1.103 |          0.677 |          -0.138 |         0.871 |            0.258 |        0.118 |                   0.784 |
| has_verb          |        -0.139 |       0.87  |          0.04  |          -0.075 |         0.928 |            0.174 |       -0.032 |                   0.283 |
| arthur_speaking   |        -0.273 |       0.761 |          0.001 |          -0.209 |         0.812 |            0.006 |       -0.032 |                   0.292 |
| has_adv           |         0.01  |       1.01  |          0.545 |          -0.019 |         0.981 |            0.414 |        0.015 |                   0.592 |
| lee_speaking      |         0.001 |       1.001 |          0.506 |          -0.016 |         0.984 |            0.426 |        0.008 |                   0.556 |
| has_noun          |         0.073 |       1.075 |          0.823 |           0.073 |         1.076 |            0.825 |       -0     |                   0.499 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_adj:group_has_adj_interaction                     |                   0.986 | 0.014 | True                    |
| arthur_adj:group_arthur_adj_interaction               |                   0.935 | 0.039 | True                    |
| lee_girl_together:group_lee_girl_together_interaction |                   0.876 | 0.068 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.812 | 0.098 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.784 | 0.121 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.717 | 0.148 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.708 | 0.169 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.592 | 0.199 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.556 | 0.226 | False                   |
| has_noun:group_has_noun_interaction                   |                   0.501 | 0.253 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| lee_girl_together |       -0.1587 |       0.022  |         -0.1585 |         0.0201 |            0.9151 |
| arthur_speaking   |       -0.0419 |       0.0075 |         -0.0418 |         0.0061 |            0.8088 |
| has_verb          |       -0.0248 |       0.0087 |         -0.0247 |         0.0069 |            0.787  |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.381842 |
| Paranoia |         0.462481 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.040117  |   0.0410526 |
| Paranoia |    0.0421053 |   0.0425731 |

---

## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE          |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:-------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | -0.241 ± 0.059 *** |           0     |        4358.08 |        0.786 |
| lee_girl_verb                       | 0.376 ± 0.146 **   |           0.995 |          27.5  |        1.456 |
| lee_girl_together                   | 0.299 ± 0.13 **    |           0.989 |          13.99 |        1.348 |
| has_adj:Group Interaction           | 0.139 ± 0.063 **   |           0.986 |          11.42 |        1.149 |
| has_adj                             | 0.136 ± 0.063 **   |           0.984 |          10.17 |        1.145 |
| has_verb                            | -0.107 ± 0.056 *   |           0.029 |           6.07 |        0.899 |
| arthur_adj:Group Interaction        | 0.163 ± 0.108 *    |           0.935 |           3.14 |        1.177 |
| has_noun                            | 0.073 ± 0.055      |           0.906 |           2.38 |        1.075 |
| arthur_adj                          | -0.126 ± 0.108     |           0.121 |           1.98 |        0.882 |
| lee_girl_together:Group Interaction | -0.15 ± 0.13       |           0.124 |           1.94 |        0.861 |
| lee_girl_verb:Group Interaction     | -0.129 ± 0.145     |           0.188 |           1.48 |        0.879 |
| girl_speaking:Group Interaction     | 0.118 ± 0.15       |           0.784 |           1.36 |        1.125 |
| has_verb:Group Interaction          | -0.032 ± 0.056     |           0.283 |           1.18 |        0.968 |
| arthur_speaking:Group Interaction   | -0.032 ± 0.059     |           0.292 |           1.16 |        0.968 |
| has_adv:Group Interaction           | 0.015 ± 0.063      |           0.592 |           1.03 |        1.015 |
| lee_speaking:Group Interaction      | 0.008 ± 0.06       |           0.556 |           1.01 |        1.008 |
| girl_speaking                       | -0.02 ± 0.15       |           0.446 |           1.01 |        0.98  |
| lee_speaking                        | -0.007 ± 0.06      |           0.452 |           1.01 |        0.993 |
| has_adv                             | -0.005 ± 0.063     |           0.47  |           1    |        0.995 |
| has_noun:Group Interaction          | -0.0 ± 0.055       |           0.499 |           1    |        1     |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_adj        |         0.037 |          -0.289 | 0.163 †            |                   0.935 |
| lee_girl_together |         0.149 |           0.448 | -0.15              |                   0.124 |
| has_adj           |         0.274 |          -0.003 | 0.139 †            |                   0.986 |
| lee_girl_verb     |         0.247 |           0.505 | -0.129             |                   0.188 |
| girl_speaking     |         0.098 |          -0.138 | 0.118              |                   0.784 |
| has_verb          |        -0.139 |          -0.075 | -0.032             |                   0.283 |
| arthur_speaking   |        -0.273 |          -0.209 | -0.032             |                   0.292 |
| has_adv           |         0.01  |          -0.019 | 0.015              |                   0.592 |
| lee_speaking      |         0.001 |          -0.016 | 0.008              |                   0.556 |
| has_noun          |         0.073 |           0.073 | -0.0               |                   0.499 |