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
| arthur_speaking                     |        -0.302 |        0.071 |           0     |        8443.04 | Extreme evidence    |        0.739 |      0.643 |      0.85  |
| has_adj                             |         0.119 |        0.075 |           0.945 |           3.56 | Moderate evidence   |        1.126 |      0.973 |      1.304 |
| girl_speaking:Group Interaction     |        -0.217 |        0.175 |           0.108 |           2.15 | Anecdotal evidence  |        0.805 |      0.571 |      1.135 |
| arthur_adj:Group Interaction        |        -0.156 |        0.132 |           0.119 |           2.01 | Anecdotal evidence  |        0.855 |      0.66  |      1.108 |
| arthur_speaking:Group Interaction   |        -0.066 |        0.071 |           0.175 |           1.55 | Anecdotal evidence  |        0.936 |      0.814 |      1.076 |
| girl_speaking                       |        -0.152 |        0.175 |           0.193 |           1.46 | Anecdotal evidence  |        0.859 |      0.609 |      1.211 |
| lee_girl_verb                       |         0.13  |        0.176 |           0.77  |           1.31 | Anecdotal evidence  |        1.138 |      0.807 |      1.607 |
| has_adv:Group Interaction           |        -0.053 |        0.075 |           0.239 |           1.29 | Anecdotal evidence  |        0.948 |      0.819 |      1.098 |
| arthur_adj                          |        -0.094 |        0.132 |           0.239 |           1.29 | Anecdotal evidence  |        0.911 |      0.703 |      1.18  |
| has_verb:Group Interaction          |         0.045 |        0.066 |           0.754 |           1.27 | Anecdotal evidence  |        1.047 |      0.919 |      1.191 |
| lee_girl_together:Group Interaction |         0.092 |        0.159 |           0.718 |           1.18 | Anecdotal evidence  |        1.096 |      0.802 |      1.498 |
| lee_speaking:Group Interaction      |        -0.036 |        0.07  |           0.305 |           1.14 | Anecdotal evidence  |        0.965 |      0.842 |      1.107 |
| has_noun:Group Interaction          |         0.029 |        0.065 |           0.671 |           1.1  | Anecdotal evidence  |        1.029 |      0.906 |      1.17  |
| has_adj:Group Interaction           |         0.033 |        0.074 |           0.669 |           1.1  | Anecdotal evidence  |        1.033 |      0.893 |      1.195 |
| has_verb                            |        -0.028 |        0.066 |           0.335 |           1.1  | Anecdotal evidence  |        0.972 |      0.854 |      1.107 |
| lee_girl_verb:Group Interaction     |         0.064 |        0.176 |           0.642 |           1.07 | Anecdotal evidence  |        1.066 |      0.755 |      1.504 |
| has_noun                            |        -0.022 |        0.065 |           0.369 |           1.06 | Anecdotal evidence  |        0.978 |      0.861 |      1.112 |
| lee_girl_together                   |         0.018 |        0.159 |           0.546 |           1.01 | Anecdotal evidence  |        1.018 |      0.745 |      1.392 |
| has_adv                             |         0.008 |        0.075 |           0.541 |           1.01 | Anecdotal evidence  |        1.008 |      0.87  |      1.167 |
| lee_speaking                        |        -0.007 |        0.07  |           0.461 |           1    | Anecdotal evidence  |        0.993 |      0.866 |      1.139 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| girl_speaking     |        -0.369 |       0.691 |          0.068 |           0.065 |         1.067 |            0.603 |       -0.217 |                   0.108 |
| arthur_adj        |        -0.25  |       0.779 |          0.091 |           0.063 |         1.065 |            0.631 |       -0.156 |                   0.119 |
| lee_girl_together |         0.11  |       1.116 |          0.687 |          -0.074 |         0.929 |            0.372 |        0.092 |                   0.718 |
| arthur_speaking   |        -0.369 |       0.692 |          0     |          -0.236 |         0.79  |            0.009 |       -0.066 |                   0.175 |
| lee_girl_verb     |         0.194 |       1.214 |          0.782 |           0.066 |         1.068 |            0.604 |        0.064 |                   0.642 |
| has_adv           |        -0.045 |       0.956 |          0.334 |           0.061 |         1.063 |            0.718 |       -0.053 |                   0.239 |
| has_verb          |         0.017 |       1.017 |          0.573 |          -0.074 |         0.929 |            0.215 |        0.045 |                   0.754 |
| lee_speaking      |        -0.042 |       0.958 |          0.334 |           0.029 |         1.029 |            0.614 |       -0.036 |                   0.305 |
| has_adj           |         0.152 |       1.164 |          0.925 |           0.086 |         1.09  |            0.794 |        0.033 |                   0.669 |
| has_noun          |         0.007 |       1.007 |          0.53  |          -0.051 |         0.951 |            0.291 |        0.029 |                   0.671 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| girl_speaking:group_girl_speaking_interaction         |                   0.892 | 0.108 | False                   |
| arthur_adj:group_arthur_adj_interaction               |                   0.881 | 0.113 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.825 | 0.134 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.761 | 0.16  | False                   |
| has_verb:group_has_verb_interaction                   |                   0.754 | 0.177 | False                   |
| lee_girl_together:group_lee_girl_together_interaction |                   0.718 | 0.195 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.695 | 0.211 | False                   |
| has_noun:group_has_noun_interaction                   |                   0.671 | 0.225 | False                   |
| has_adj:group_has_adj_interaction                     |                   0.669 | 0.237 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.642 | 0.249 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| lee_girl_together |        0.0623 |       0.0264 |          0.0632 |         0.0256 |            0.9712 |
| arthur_speaking   |       -0.0659 |       0.0137 |         -0.0659 |         0.0128 |            0.9369 |
| has_verb          |        0.046  |       0.0115 |          0.0463 |         0.0099 |            0.8604 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.136655 |
| Paranoia |         0.171199 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0288889 |   0.0278363 |
| Paranoia |    0.0355556 |   0.0346199 |

---

## 7. Main Effects

Table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE          |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:-------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | -0.302 ± 0.071 *** |           0     |        8443.04 |        0.739 |
| has_adj                             | 0.119 ± 0.075 *    |           0.945 |           3.56 |        1.126 |
| girl_speaking:Group Interaction     | -0.217 ± 0.175     |           0.108 |           2.15 |        0.805 |
| arthur_adj:Group Interaction        | -0.156 ± 0.132     |           0.119 |           2.01 |        0.855 |
| arthur_speaking:Group Interaction   | -0.066 ± 0.071     |           0.175 |           1.55 |        0.936 |
| girl_speaking                       | -0.152 ± 0.175     |           0.193 |           1.46 |        0.859 |
| lee_girl_verb                       | 0.13 ± 0.176       |           0.77  |           1.31 |        1.138 |
| has_adv:Group Interaction           | -0.053 ± 0.075     |           0.239 |           1.29 |        0.948 |
| arthur_adj                          | -0.094 ± 0.132     |           0.239 |           1.29 |        0.911 |
| has_verb:Group Interaction          | 0.045 ± 0.066      |           0.754 |           1.27 |        1.047 |
| lee_girl_together:Group Interaction | 0.092 ± 0.159      |           0.718 |           1.18 |        1.096 |
| lee_speaking:Group Interaction      | -0.036 ± 0.07      |           0.305 |           1.14 |        0.965 |
| has_noun:Group Interaction          | 0.029 ± 0.065      |           0.671 |           1.1  |        1.029 |
| has_adj:Group Interaction           | 0.033 ± 0.074      |           0.669 |           1.1  |        1.033 |
| has_verb                            | -0.028 ± 0.066     |           0.335 |           1.1  |        0.972 |
| lee_girl_verb:Group Interaction     | 0.064 ± 0.176      |           0.642 |           1.07 |        1.066 |
| has_noun                            | -0.022 ± 0.065     |           0.369 |           1.06 |        0.978 |
| lee_girl_together                   | 0.018 ± 0.159      |           0.546 |           1.01 |        1.018 |
| has_adv                             | 0.008 ± 0.075      |           0.541 |           1.01 |        1.008 |
| lee_speaking                        | -0.007 ± 0.07      |           0.461 |           1    |        0.993 |

---

## 8. Group Effects

Table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef |   Group Difference |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|-------------------:|------------------------:|
| girl_speaking     |        -0.369 |           0.065 |             -0.217 |                   0.108 |
| arthur_adj        |        -0.25  |           0.063 |             -0.156 |                   0.119 |
| lee_girl_together |         0.11  |          -0.074 |              0.092 |                   0.718 |
| arthur_speaking   |        -0.369 |          -0.236 |             -0.066 |                   0.175 |
| lee_girl_verb     |         0.194 |           0.066 |              0.064 |                   0.642 |
| has_adv           |        -0.045 |           0.061 |             -0.053 |                   0.239 |
| has_verb          |         0.017 |          -0.074 |              0.045 |                   0.754 |
| lee_speaking      |        -0.042 |           0.029 |             -0.036 |                   0.305 |
| has_adj           |         0.152 |           0.086 |              0.033 |                   0.669 |
| has_noun          |         0.007 |          -0.051 |              0.029 |                   0.671 |