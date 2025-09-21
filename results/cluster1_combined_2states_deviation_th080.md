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
| arthur_speaking                     |         0.209 |        0.059 |           1     |         514.27 | Extreme evidence    |        1.232 |      1.097 |      1.383 |
| has_verb                            |         0.142 |        0.057 |           0.994 |          23.59 | Strong evidence     |        1.153 |      1.032 |      1.288 |
| has_adj:Group Interaction           |        -0.139 |        0.063 |           0.014 |          11.23 | Strong evidence     |        0.87  |      0.769 |      0.985 |
| arthur_adj:Group Interaction        |        -0.164 |        0.108 |           0.064 |           3.18 | Moderate evidence   |        0.848 |      0.686 |      1.049 |
| lee_girl_verb                       |         0.209 |        0.149 |           0.92  |           2.68 | Anecdotal evidence  |        1.232 |      0.921 |      1.649 |
| lee_girl_together:Group Interaction |         0.157 |        0.131 |           0.885 |           2.05 | Anecdotal evidence  |        1.17  |      0.905 |      1.511 |
| lee_girl_together                   |         0.149 |        0.131 |           0.873 |           1.92 | Anecdotal evidence  |        1.161 |      0.899 |      1.501 |
| arthur_adj                          |         0.103 |        0.108 |           0.829 |           1.57 | Anecdotal evidence  |        1.108 |      0.897 |      1.37  |
| lee_girl_verb:Group Interaction     |         0.137 |        0.149 |           0.822 |           1.53 | Anecdotal evidence  |        1.147 |      0.857 |      1.535 |
| girl_speaking:Group Interaction     |        -0.118 |        0.151 |           0.216 |           1.36 | Anecdotal evidence  |        0.888 |      0.661 |      1.193 |
| has_adj                             |        -0.043 |        0.063 |           0.246 |           1.27 | Anecdotal evidence  |        0.958 |      0.846 |      1.084 |
| has_verb:Group Interaction          |         0.033 |        0.056 |           0.719 |           1.18 | Anecdotal evidence  |        1.033 |      0.925 |      1.154 |
| has_noun                            |        -0.031 |        0.055 |           0.291 |           1.16 | Anecdotal evidence  |        0.97  |      0.87  |      1.081 |
| arthur_speaking:Group Interaction   |         0.032 |        0.059 |           0.706 |           1.16 | Anecdotal evidence  |        1.033 |      0.92  |      1.159 |
| lee_speaking                        |        -0.024 |        0.06  |           0.344 |           1.08 | Anecdotal evidence  |        0.976 |      0.868 |      1.098 |
| has_adv                             |        -0.024 |        0.064 |           0.35  |           1.08 | Anecdotal evidence  |        0.976 |      0.862 |      1.105 |
| has_adv:Group Interaction           |        -0.015 |        0.064 |           0.404 |           1.03 | Anecdotal evidence  |        0.985 |      0.869 |      1.115 |
| lee_speaking:Group Interaction      |        -0.009 |        0.06  |           0.44  |           1.01 | Anecdotal evidence  |        0.991 |      0.881 |      1.114 |
| has_noun:Group Interaction          |         0.001 |        0.055 |           0.505 |           1    | Anecdotal evidence  |        1.001 |      0.898 |      1.116 |
| girl_speaking                       |        -0.001 |        0.151 |           0.498 |           1    | Anecdotal evidence  |        0.999 |      0.744 |      1.343 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_adj        |        -0.062 |       0.94  |          0.343 |           0.267 |         1.306 |            0.96  |       -0.164 |                   0.064 |
| lee_girl_together |         0.306 |       1.358 |          0.951 |          -0.007 |         0.993 |            0.484 |        0.157 |                   0.885 |
| has_adj           |        -0.182 |       0.833 |          0.021 |           0.096 |         1.1   |            0.858 |       -0.139 |                   0.014 |
| lee_girl_verb     |         0.346 |       1.414 |          0.95  |           0.072 |         1.074 |            0.633 |        0.137 |                   0.822 |
| girl_speaking     |        -0.119 |       0.888 |          0.288 |           0.118 |         1.125 |            0.71  |       -0.118 |                   0.216 |
| has_verb          |         0.175 |       1.191 |          0.986 |           0.109 |         1.116 |            0.915 |        0.033 |                   0.719 |
| arthur_speaking   |         0.241 |       1.272 |          0.998 |           0.177 |         1.193 |            0.983 |        0.032 |                   0.706 |
| has_adv           |        -0.04  |       0.961 |          0.329 |          -0.009 |         0.991 |            0.46  |       -0.015 |                   0.404 |
| lee_speaking      |        -0.033 |       0.967 |          0.348 |          -0.015 |         0.985 |            0.429 |       -0.009 |                   0.44  |
| has_noun          |        -0.03  |       0.971 |          0.352 |          -0.031 |         0.969 |            0.345 |        0.001 |                   0.505 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_adj:group_has_adj_interaction                     |                   0.986 | 0.014 | True                    |
| arthur_adj:group_arthur_adj_interaction               |                   0.936 | 0.039 | True                    |
| lee_girl_together:group_lee_girl_together_interaction |                   0.885 | 0.064 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.822 | 0.093 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.784 | 0.117 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.719 | 0.145 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.706 | 0.166 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.596 | 0.196 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.56  | 0.223 | False                   |
| has_noun:group_has_noun_interaction                   |                   0.505 | 0.25  | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| lee_girl_together |        0.1651 |       0.0225 |          0.1652 |         0.0206 |            0.9183 |
| arthur_speaking   |        0.0419 |       0.0076 |          0.0419 |         0.0061 |            0.8096 |
| has_verb          |        0.0249 |       0.0088 |          0.0249 |         0.0069 |            0.789  |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.618158 |
| Paranoia |         0.537519 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0410526 |   0.040117  |
| Paranoia |    0.0425731 |   0.0421053 |

---

## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE         |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | 0.209 ± 0.059 *** |           1     |         514.27 |        1.232 |
| has_verb                            | 0.142 ± 0.057 **  |           0.994 |          23.59 |        1.153 |
| has_adj:Group Interaction           | -0.139 ± 0.063 ** |           0.014 |          11.23 |        0.87  |
| arthur_adj:Group Interaction        | -0.164 ± 0.108 *  |           0.064 |           3.18 |        0.848 |
| lee_girl_verb                       | 0.209 ± 0.149     |           0.92  |           2.68 |        1.232 |
| lee_girl_together:Group Interaction | 0.157 ± 0.131     |           0.885 |           2.05 |        1.17  |
| lee_girl_together                   | 0.149 ± 0.131     |           0.873 |           1.92 |        1.161 |
| arthur_adj                          | 0.103 ± 0.108     |           0.829 |           1.57 |        1.108 |
| lee_girl_verb:Group Interaction     | 0.137 ± 0.149     |           0.822 |           1.53 |        1.147 |
| girl_speaking:Group Interaction     | -0.118 ± 0.151    |           0.216 |           1.36 |        0.888 |
| has_adj                             | -0.043 ± 0.063    |           0.246 |           1.27 |        0.958 |
| has_verb:Group Interaction          | 0.033 ± 0.056     |           0.719 |           1.18 |        1.033 |
| has_noun                            | -0.031 ± 0.055    |           0.291 |           1.16 |        0.97  |
| arthur_speaking:Group Interaction   | 0.032 ± 0.059     |           0.706 |           1.16 |        1.033 |
| lee_speaking                        | -0.024 ± 0.06     |           0.344 |           1.08 |        0.976 |
| has_adv                             | -0.024 ± 0.064    |           0.35  |           1.08 |        0.976 |
| has_adv:Group Interaction           | -0.015 ± 0.064    |           0.404 |           1.03 |        0.985 |
| lee_speaking:Group Interaction      | -0.009 ± 0.06     |           0.44  |           1.01 |        0.991 |
| has_noun:Group Interaction          | 0.001 ± 0.055     |           0.505 |           1    |        1.001 |
| girl_speaking                       | -0.001 ± 0.151    |           0.498 |           1    |        0.999 |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_adj        |        -0.062 |           0.267 | -0.164 †           |                   0.064 |
| lee_girl_together |         0.306 |          -0.007 | 0.157              |                   0.885 |
| has_adj           |        -0.182 |           0.096 | -0.139 †           |                   0.014 |
| lee_girl_verb     |         0.346 |           0.072 | 0.137              |                   0.822 |
| girl_speaking     |        -0.119 |           0.118 | -0.118             |                   0.216 |
| has_verb          |         0.175 |           0.109 | 0.033              |                   0.719 |
| arthur_speaking   |         0.241 |           0.177 | 0.032              |                   0.706 |
| has_adv           |        -0.04  |          -0.009 | -0.015             |                   0.404 |
| lee_speaking      |        -0.033 |          -0.015 | -0.009             |                   0.44  |
| has_noun          |        -0.03  |          -0.031 | 0.001              |                   0.505 |