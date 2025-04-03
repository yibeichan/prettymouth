# Neural State Dynamics Analysis Supplementary Material

This document contains all statistical tables from the brain state dynamics analysis.

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
| arthur_speaking                     |        -0.247 |        0.059 |           0     |        6530.16 | Extreme evidence    |        0.781 |      0.696 |      0.877 |
| lee_girl_verb                       |         0.374 |        0.146 |           0.995 |          26.89 | Strong evidence     |        1.454 |      1.092 |      1.936 |
| lee_girl_together                   |         0.297 |        0.13  |           0.989 |          13.67 | Strong evidence     |        1.346 |      1.043 |      1.737 |
| has_adj:Group Interaction           |         0.142 |        0.063 |           0.988 |          12.79 | Strong evidence     |        1.152 |      1.019 |      1.304 |
| has_adj                             |         0.133 |        0.063 |           0.982 |           9.21 | Moderate evidence   |        1.142 |      1.009 |      1.292 |
| has_verb                            |        -0.114 |        0.056 |           0.021 |           7.95 | Moderate evidence   |        0.892 |      0.799 |      0.996 |
| arthur_adj:Group Interaction        |         0.164 |        0.108 |           0.937 |           3.21 | Moderate evidence   |        1.179 |      0.954 |      1.456 |
| has_noun                            |         0.075 |        0.055 |           0.914 |           2.53 | Anecdotal evidence  |        1.078 |      0.968 |      1.202 |
| arthur_adj                          |        -0.127 |        0.108 |           0.119 |           2.01 | Anecdotal evidence  |        0.881 |      0.713 |      1.087 |
| lee_girl_together:Group Interaction |        -0.148 |        0.13  |           0.127 |           1.92 | Anecdotal evidence  |        0.862 |      0.669 |      1.112 |
| lee_girl_verb:Group Interaction     |        -0.128 |        0.145 |           0.19  |           1.47 | Anecdotal evidence  |        0.88  |      0.662 |      1.171 |
| girl_speaking:Group Interaction     |         0.107 |        0.15  |           0.761 |           1.29 | Anecdotal evidence  |        1.112 |      0.829 |      1.493 |
| arthur_speaking:Group Interaction   |        -0.034 |        0.059 |           0.284 |           1.18 | Anecdotal evidence  |        0.967 |      0.862 |      1.085 |
| has_verb:Group Interaction          |        -0.031 |        0.056 |           0.293 |           1.16 | Anecdotal evidence  |        0.97  |      0.869 |      1.083 |
| lee_speaking:Group Interaction      |         0.017 |        0.06  |           0.615 |           1.04 | Anecdotal evidence  |        1.018 |      0.905 |      1.144 |
| has_adv:Group Interaction           |         0.013 |        0.063 |           0.581 |           1.02 | Anecdotal evidence  |        1.013 |      0.895 |      1.147 |
| girl_speaking                       |        -0.009 |        0.15  |           0.476 |           1    | Anecdotal evidence  |        0.991 |      0.738 |      1.33  |
| has_noun:Group Interaction          |        -0.003 |        0.055 |           0.48  |           1    | Anecdotal evidence  |        0.997 |      0.895 |      1.111 |
| lee_speaking                        |        -0.002 |        0.06  |           0.484 |           1    | Anecdotal evidence  |        0.998 |      0.888 |      1.121 |
| has_adv                             |        -0.002 |        0.063 |           0.487 |           1    | Anecdotal evidence  |        0.998 |      0.882 |      1.13  |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_adj        |         0.037 |       1.038 |          0.596 |          -0.292 |         0.747 |            0.028 |        0.164 |                   0.937 |
| lee_girl_together |         0.149 |       1.161 |          0.792 |           0.446 |         1.561 |            0.992 |       -0.148 |                   0.127 |
| has_adj           |         0.275 |       1.316 |          0.999 |          -0.009 |         0.991 |            0.458 |        0.142 |                   0.988 |
| lee_girl_verb     |         0.247 |       1.28  |          0.885 |           0.502 |         1.652 |            0.993 |       -0.128 |                   0.19  |
| girl_speaking     |         0.098 |       1.103 |          0.677 |          -0.116 |         0.891 |            0.293 |        0.107 |                   0.761 |
| arthur_speaking   |        -0.28  |       0.756 |          0     |          -0.213 |         0.808 |            0.005 |       -0.034 |                   0.284 |
| has_verb          |        -0.145 |       0.865 |          0.034 |          -0.084 |         0.92  |            0.145 |       -0.031 |                   0.293 |
| lee_speaking      |         0.015 |       1.015 |          0.571 |          -0.02  |         0.98  |            0.407 |        0.017 |                   0.615 |
| has_adv           |         0.011 |       1.011 |          0.549 |          -0.015 |         0.985 |            0.433 |        0.013 |                   0.581 |
| has_noun          |         0.073 |       1.075 |          0.823 |           0.078 |         1.081 |            0.841 |       -0.003 |                   0.48  |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_adj:group_has_adj_interaction                     |                   0.988 | 0.012 | True                    |
| arthur_adj:group_arthur_adj_interaction               |                   0.937 | 0.038 | True                    |
| lee_girl_together:group_lee_girl_together_interaction |                   0.873 | 0.067 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.81  | 0.098 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.761 | 0.126 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.716 | 0.153 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.707 | 0.173 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.615 | 0.199 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.581 | 0.224 | False                   |
| has_noun:group_has_noun_interaction                   |                   0.52  | 0.249 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| lee_girl_together |       -0.1582 |       0.022  |         -0.158  |         0.02   |            0.9118 |
| has_verb          |       -0.0232 |       0.0086 |         -0.0232 |         0.0071 |            0.827  |
| arthur_speaking   |       -0.0433 |       0.0074 |         -0.0432 |         0.006  |            0.8064 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.381842 |
| Paranoia |         0.463531 |

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
| arthur_speaking                     | -0.247 ± 0.059 *** |           0     |        6530.16 |        0.781 |
| lee_girl_verb                       | 0.374 ± 0.146 **   |           0.995 |          26.89 |        1.454 |
| lee_girl_together                   | 0.297 ± 0.13 **    |           0.989 |          13.67 |        1.346 |
| has_adj:Group Interaction           | 0.142 ± 0.063 **   |           0.988 |          12.79 |        1.152 |
| has_adj                             | 0.133 ± 0.063 *    |           0.982 |           9.21 |        1.142 |
| has_verb                            | -0.114 ± 0.056 *   |           0.021 |           7.95 |        0.892 |
| arthur_adj:Group Interaction        | 0.164 ± 0.108 *    |           0.937 |           3.21 |        1.179 |
| has_noun                            | 0.075 ± 0.055      |           0.914 |           2.53 |        1.078 |
| arthur_adj                          | -0.127 ± 0.108     |           0.119 |           2.01 |        0.881 |
| lee_girl_together:Group Interaction | -0.148 ± 0.13      |           0.127 |           1.92 |        0.862 |
| lee_girl_verb:Group Interaction     | -0.128 ± 0.145     |           0.19  |           1.47 |        0.88  |
| girl_speaking:Group Interaction     | 0.107 ± 0.15       |           0.761 |           1.29 |        1.112 |
| arthur_speaking:Group Interaction   | -0.034 ± 0.059     |           0.284 |           1.18 |        0.967 |
| has_verb:Group Interaction          | -0.031 ± 0.056     |           0.293 |           1.16 |        0.97  |
| lee_speaking:Group Interaction      | 0.017 ± 0.06       |           0.615 |           1.04 |        1.018 |
| has_adv:Group Interaction           | 0.013 ± 0.063      |           0.581 |           1.02 |        1.013 |
| girl_speaking                       | -0.009 ± 0.15      |           0.476 |           1    |        0.991 |
| has_noun:Group Interaction          | -0.003 ± 0.055     |           0.48  |           1    |        0.997 |
| lee_speaking                        | -0.002 ± 0.06      |           0.484 |           1    |        0.998 |
| has_adv                             | -0.002 ± 0.063     |           0.487 |           1    |        0.998 |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_adj        |         0.037 |          -0.292 | 0.164 †            |                   0.937 |
| lee_girl_together |         0.149 |           0.446 | -0.148             |                   0.127 |
| has_adj           |         0.275 |          -0.009 | 0.142 †            |                   0.988 |
| lee_girl_verb     |         0.247 |           0.502 | -0.128             |                   0.19  |
| girl_speaking     |         0.098 |          -0.116 | 0.107              |                   0.761 |
| arthur_speaking   |        -0.28  |          -0.213 | -0.034             |                   0.284 |
| has_verb          |        -0.145 |          -0.084 | -0.031             |                   0.293 |
| lee_speaking      |         0.015 |          -0.02  | 0.017              |                   0.615 |
| has_adv           |         0.011 |          -0.015 | 0.013              |                   0.581 |
| has_noun          |         0.073 |           0.078 | -0.003             |                   0.48  |